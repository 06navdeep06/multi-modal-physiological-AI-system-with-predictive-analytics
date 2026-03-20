"""Multi-Modal Physiological Monitoring System - Main Entry Point

Production-grade entry point that orchestrates:
  - CHROM rPPG → heart rate, HRV (RMSSD/SDNN/pNN50), SpO2 estimate
  - Landmark-based respiration (Welch PSD)
  - PERCLOS + microsleep + drowsiness (both eyes averaged)
  - Multi-backend emotion detection (DeepFace → FER → heuristic)
  - Statistical feature fusion → stress classification
  - SQLite persistence with WAL mode
  - Rotating log files
  - REST API: /api/metrics, /api/history, /api/stats, /api/export,
              /api/alerts, /api/alerts/summary, /api/health
"""

from __future__ import annotations

import csv
import io
import json
import os
import signal
import sqlite3
import sys
import threading
import time
import logging
from collections import Counter
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from typing import Optional

import cv2
import numpy as np
from flask import Flask, jsonify, request, Response, send_from_directory
from flask_cors import CORS

from config import *
from vision.face_detection import FaceDetector
from vision.landmarks import FaceLandmarks
from signal_processing.rppg import RPPG
from signal_processing.respiration import Respiration
from signal_processing.blink_fatigue import BlinkFatigue
from signal_processing.emotion import EmotionDetector
from features.fusion import FeatureFusion
from models.classical_model import ClassicalModel

# ===== LOGGING =====
# On Render (or any env where LOG_FILE='stdout') log only to stdout to avoid
# writing to an ephemeral filesystem and cluttering the build cache.
_handlers: list = [logging.StreamHandler()]
if LOG_FILE and LOG_FILE.lower() != 'stdout':
    try:
        _log_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
        )
        _handlers.append(_log_handler)
    except OSError:
        pass  # filesystem not writable – stdout only is fine
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, handlers=_handlers)
logger = logging.getLogger(__name__)

# ===== GLOBAL STATE =====
metrics: dict = {
    # Core vitals
    'hr': None, 'hrv': None, 'rmssd': None,
    'resp': None, 'blink': None, 'fatigue': None,
    'perclos': None, 'drowsiness': None, 'microsleeps': None,
    'emotion': None, 'emotion_scores': None,
    'stress': None, 'stress_score': None,
    'spo2': None, 'signal_quality': None,
    # Frequency-domain HRV (populated after warm-up)
    'hrv_freq': None,        # full freq-domain dict
    'lf_hf_ratio': None,     # sympathovagal balance
    'coherence': None,       # cardiac coherence %
    'lf_power': None,
    'hf_power': None,
    # Autonomic stress
    'stress_index': None,    # Baevsky SI (normalised 0–1)
    # Waveform & Poincaré data
    'ppg_signal': [],        # last ~150 normalised PPG samples
    'rr_intervals': [],      # RR series (ms) for Poincaré plot
    # Housekeeping
    'alert': None, 'alerts': [],
    'timestamp': None,
    'warmup': True,
    'warmup_pct': 0,
    'face_detected': False,
}
metrics_lock: threading.Lock = threading.Lock()
stop_event:   threading.Event = threading.Event()
processing_error: Optional[str] = None
alert_history:    list = []
session_start:    Optional[datetime] = None
total_frames:     int = 0

# ===== MJPEG FRAME BUFFER =====
_latest_frame: Optional[bytes] = None   # JPEG-encoded bytes of the latest annotated frame
_frame_lock:   threading.Lock  = threading.Lock()

# ===== BROWSER CAMERA MODE =====
# When no physical camera is found (e.g. Render cloud), the browser streams
# frames via POST /api/frame and processing happens here server-side.
_browser_mode:    bool           = False
_procs:           dict           = {}      # shared processor instances
_procs_lock:      threading.Lock = threading.Lock()
_bc_hr_ema:       Optional[float] = None
_bc_prev_nose:    Optional[tuple] = None
_bc_frames:       int            = 0
_BC_LEFT_EYE  = [33, 160, 158, 133, 153, 144]
_BC_RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ===== FLASK APP =====
app = Flask(__name__)
CORS(app, origins=CORS_ORIGINS)


# ------------------------------------------------------------------
# API endpoints
# ------------------------------------------------------------------

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    with metrics_lock:
        return jsonify({
            'data':   metrics.copy(),
            'status': 'error' if processing_error else 'ok',
            'error':  processing_error,
        })


@app.route('/api/health', methods=['GET'])
def health_check():
    uptime = (datetime.now() - session_start).total_seconds() if session_start else 0
    return jsonify({
        'status':           'healthy',
        'timestamp':        datetime.now().isoformat(),
        'uptime_sec':       round(uptime),
        'frames_processed': total_frames,
    }), 200


@app.route('/api/history', methods=['GET'])
def get_history():
    limit = min(request.args.get('limit', HISTORY_LIMIT, type=int), 2000)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(
                'SELECT * FROM metrics ORDER BY id DESC LIMIT ?', (limit,)
            )]
        return jsonify({'data': list(reversed(rows)), 'count': len(rows)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    uptime = (datetime.now() - session_start).total_seconds() if session_start else 0
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute('''
                SELECT AVG(hr), MIN(hr), MAX(hr),
                       AVG(resp), AVG(fatigue), AVG(spo2),
                       AVG(hrv_rmssd), AVG(perclos), COUNT(*)
                FROM metrics
                WHERE timestamp > datetime('now', '-1 hour')
            ''').fetchone()

        def _r(v, n=1):
            return round(v, n) if v is not None else None

        return jsonify({
            'session_uptime_sec': round(uptime),
            'total_frames':       total_frames,
            'averages_last_hour': {
                'hr':       _r(row[0]), 'hr_min': _r(row[1]), 'hr_max': _r(row[2]),
                'resp':     _r(row[3]), 'fatigue': _r(row[4], 3),
                'spo2':     _r(row[5]), 'rmssd':   _r(row[6], 2),
                'perclos':  _r(row[7]), 'readings': row[8],
            },
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export', methods=['GET'])
def export_csv():
    limit = min(request.args.get('limit', 1000, type=int), 10_000)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(
                'SELECT * FROM metrics ORDER BY id DESC LIMIT ?', (limit,)
            )]
        if not rows:
            return jsonify({'error': 'No data to export'}), 404

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(reversed(rows))
        return Response(
            buf.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition':
                     'attachment; filename=physiological_metrics.csv'},
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts', methods=['GET'])
def get_alert_history_ep():
    limit = min(request.args.get('limit', 50, type=int), 500)
    with metrics_lock:
        recent = list(reversed(alert_history[-limit:]))
    return jsonify({'alerts': recent, 'total': len(alert_history)})


@app.route('/api/camera-info', methods=['GET'])
def camera_info():
    """Diagnostic: probe available cameras and report their status.

    Scans indices 0–3 with all backends and returns which ones are
    openable and deliver frames.  Useful for debugging camera issues
    without starting the full processing thread.
    """
    results = []
    backends_to_try: list[tuple[str, int]]
    if sys.platform == 'win32':
        backends_to_try = [
            ('DirectShow',      cv2.CAP_DSHOW),
            ('MediaFoundation', cv2.CAP_MSMF),
            ('Auto',            cv2.CAP_ANY),
        ]
    else:
        backends_to_try = [('Auto', cv2.CAP_ANY)]

    for bname, bid in backends_to_try:
        for idx in range(4):
            try:
                cap = cv2.VideoCapture(idx, bid)
                opened = cap.isOpened()
                frame_ok = False
                width = height = fps_reported = 0
                if opened:
                    ret, frame = cap.read()
                    frame_ok   = ret and frame is not None and frame.size > 0
                    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps_reported = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if opened:
                    results.append({
                        'index':   idx,
                        'backend': bname,
                        'opens':   opened,
                        'frames':  frame_ok,
                        'width':   width,
                        'height':  height,
                        'fps':     fps_reported,
                    })
            except Exception as e:
                results.append({'index': idx, 'backend': bname, 'error': str(e)})

    working = [r for r in results if r.get('frames')]
    return jsonify({
        'platform':      sys.platform,
        'demo_mode':     DEMO_MODE,
        'working_cameras': working,
        'all_probed':    results,
        'recommendation': (
            f"Use WEBCAM_INDEX={working[0]['index']} with backend {working[0]['backend']}"
            if working else
            'No working camera found. Set DEMO_MODE=true or check USB/permissions.'
        ),
    })


@app.route('/api/waveform', methods=['GET'])
def get_waveform():
    """Return real-time PPG signal and RR intervals for waveform / Poincaré rendering."""
    with metrics_lock:
        return jsonify({
            'ppg_signal':   metrics.get('ppg_signal', []),
            'rr_intervals': metrics.get('rr_intervals', []),
            'signal_quality': metrics.get('signal_quality'),
            'timestamp':    metrics.get('timestamp'),
        })


@app.route('/api/alerts/summary', methods=['GET'])
def get_alert_summary():
    limit      = min(request.args.get('limit', 200, type=int), 2000)
    window_sec = request.args.get('window_sec', type=int)

    with metrics_lock:
        recent = list(alert_history[-limit:])

    if window_sec:
        cutoff   = datetime.now() - timedelta(seconds=max(window_sec, 0))
        filtered = []
        for a in recent:
            try:
                alert_time = datetime.fromisoformat(a['timestamp']) if a.get('timestamp') else None
            except Exception:
                alert_time = None
            if alert_time is None or alert_time >= cutoff:
                filtered.append(a)
        recent = filtered

    return jsonify({
        'total':       len(recent),
        'by_severity': dict(Counter(a.get('severity', 'warning') for a in recent)),
        'latest':      recent[-1] if recent else None,
        'window_sec':  window_sec,
        'inspected':   limit,
    })


@app.route('/api/frame', methods=['POST'])
def receive_frame():
    """Accept a JPEG frame from the browser camera and run the full processing pipeline.

    The browser captures its local camera via getUserMedia, encodes each frame as
    JPEG and POSTs the raw bytes here.  All signal-processing (face, rPPG, HRV,
    respiration, blink/fatigue, emotion, stress) runs server-side just as it does
    in process_video(), and the global metrics dict is updated so every SSE/poll
    client sees live data.  Returns the annotated JPEG so the browser can optionally
    show it, but displaying the local <video> (zero latency) is fine too.
    """
    global _browser_mode, _bc_hr_ema, _bc_prev_nose, _bc_frames
    global total_frames, session_start, processing_error

    if not _ensure_processors():
        return jsonify({'error': 'processors not ready'}), 503

    # ── Decode incoming JPEG ──────────────────────────────────────
    data = request.get_data()
    if not data:
        return jsonify({'error': 'no frame data'}), 400
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'invalid frame'}), 400

    # Serialise: processors are stateful ring-buffers, not thread-safe
    with _procs_lock:
        if session_start is None:
            session_start = datetime.now()
        total_frames += 1
        _bc_frames   += 1

        face_detector      = _procs['face']
        landmarks_detector = _procs['lm']
        rppg_proc          = _procs['rppg']
        resp_proc          = _procs['resp']
        blink_proc         = _procs['blink']
        emotion_det        = _procs['emotion']
        fusion             = _procs['fusion']
        clf_model          = _procs['clf']
        RESP_LM            = Respiration.RESP_LANDMARK_INDICES

        # ── Face detection ────────────────────────────────────────
        try:
            bboxes = face_detector.detect(frame)
            if not bboxes:
                with metrics_lock:
                    metrics['face_detected'] = False
                return jsonify({'face': False}), 200
            x, y, w, h = bboxes[0]
            if w <= 0 or h <= 0:
                return jsonify({'face': False}), 200
            face_roi = frame[y:y + h, x:x + w]
        except Exception as e:
            logger.debug(f'BC face: {e}')
            return jsonify({'face': False}), 200

        # ── Landmarks ─────────────────────────────────────────────
        try:
            lm_list = landmarks_detector.get_landmarks(frame)
            if not lm_list:
                return jsonify({'face': False}), 200
            lm = lm_list[0]
        except Exception as e:
            logger.debug(f'BC landmarks: {e}')
            return jsonify({'face': False}), 200

        # ── Motion gating ─────────────────────────────────────────
        _motion_px = 0.0
        if len(lm) > 6:
            nose_xy = lm[6]
            if _bc_prev_nose is not None:
                dx = nose_xy[0] - _bc_prev_nose[0]
                dy = nose_xy[1] - _bc_prev_nose[1]
                _motion_px = float((dx * dx + dy * dy) ** 0.5)
            _bc_prev_nose = nose_xy

        # ── rPPG (CHROM) ──────────────────────────────────────────
        hr = hrv_metrics = spo2 = signal_quality = rmssd = None
        hrv_freq = stress_idx = None
        ppg_signal: list = []
        rr_intervals: list = []
        try:
            fh_y1 = y + max(0, int(h * 0.08))
            fh_y2 = y + min(h, int(h * 0.38))
            fh_x1 = x + max(0, int(w * 0.18))
            fh_x2 = x + min(w, int(w * 0.82))
            forehead_roi = frame[fh_y1:fh_y2, fh_x1:fh_x2]
            if forehead_roi.size > 0 and _motion_px < 6.0:
                rgb_means = (
                    float(np.mean(forehead_roi[:, :, 2])),
                    float(np.mean(forehead_roi[:, :, 1])),
                    float(np.mean(forehead_roi[:, :, 0])),
                )
                rppg_proc.update(rgb_means)
            hr_raw, hrv_metrics = rppg_proc.compute_hr()
            signal_quality = round(rppg_proc.signal_quality, 2)
            spo2           = rppg_proc.estimate_spo2()
            if hr_raw is not None and signal_quality >= SIGNAL_QUALITY_MIN:
                clipped    = float(np.clip(hr_raw, 30, 220))
                _bc_hr_ema = clipped if _bc_hr_ema is None \
                             else _HR_ALPHA * clipped + (1 - _HR_ALPHA) * _bc_hr_ema
                hr = round(_bc_hr_ema, 1)
            elif _bc_hr_ema is not None:
                hr = round(_bc_hr_ema, 1)
            rmssd        = hrv_metrics.get('rmssd') if hrv_metrics else None
            hrv_freq     = rppg_proc.compute_hrv_frequency_domain()
            stress_idx   = rppg_proc.baevsky_stress_index()
            ppg_signal   = rppg_proc.get_ppg_signal(150)
            rr_intervals = rppg_proc.rr_intervals[-50:]
        except Exception as e:
            logger.debug(f'BC rPPG: {e}')

        # ── Respiration ───────────────────────────────────────────
        resp_rate = None
        try:
            if RESP_LM and max(RESP_LM) < len(lm):
                resp_proc.update([lm[i][1] for i in RESP_LM])
                resp_rate = resp_proc.compute_resp_rate()
        except Exception as e:
            logger.debug(f'BC resp: {e}')

        # ── Blink / Fatigue / PERCLOS ─────────────────────────────
        blink_rate = fatigue_score = perclos = drowsiness = microsleeps = None
        try:
            all_eye = _BC_LEFT_EYE + _BC_RIGHT_EYE
            if max(all_eye) < len(lm):
                left_ear  = BlinkFatigue.compute_ear([lm[i] for i in _BC_LEFT_EYE])
                right_ear = BlinkFatigue.compute_ear([lm[i] for i in _BC_RIGHT_EYE])
                blink_proc.update((left_ear, right_ear))
                result = blink_proc.compute_blink_fatigue()
                if result[0] is not None:
                    blink_rate, fatigue_score, perclos, drowsiness, microsleeps = result
        except Exception as e:
            logger.debug(f'BC blink: {e}')

        # ── Emotion ───────────────────────────────────────────────
        emotion = emotion_probs = None
        try:
            emotion_probs = emotion_det.predict(face_roi, lm)
            emotion = max(emotion_probs, key=emotion_probs.get) if emotion_probs else None
        except Exception as e:
            logger.debug(f'BC emotion: {e}')

        # ── Stress ────────────────────────────────────────────────
        stress_level = stress_score_val = None
        try:
            fusion.update([
                hr            or 0.0,
                rmssd         or 0.0,
                resp_rate     or 0.0,
                blink_rate    or 0.0,
                fatigue_score or 0.0,
            ])
            fused = fusion.get_fused_features()
            if fused is not None:
                proba = clf_model.predict(fused.reshape(1, -1))
                if proba is not None and len(proba) > 0:
                    p = proba[0]
                    stress_score_val = round(
                        float(p[1] * 0.5 + p[2]) if len(p) >= 3 else float(p[-1]), 3
                    )
                    if stress_score_val < STRESS_THRESHOLDS['low']:
                        stress_level = 'Low'
                    elif stress_score_val < STRESS_THRESHOLDS['medium']:
                        stress_level = 'Medium'
                    else:
                        stress_level = 'High'
        except Exception as e:
            logger.debug(f'BC stress: {e}')

        # ── Warmup ────────────────────────────────────────────────
        warmup_done = (
            len(rppg_proc.r_window)   >= rppg_proc.window_size and
            len(blink_proc.ear_window) >= blink_proc.window_size and
            len(resp_proc.window)     >= resp_proc.window_size
        )
        warmup_pct = min(100, int(min(
            len(rppg_proc.r_window)   / rppg_proc.window_size,
            len(blink_proc.ear_window) / blink_proc.window_size,
            len(resp_proc.window)     / resp_proc.window_size,
        ) * 100))

        # ── Alerts ────────────────────────────────────────────────
        _ts = datetime.now().isoformat()
        new_alerts: list[dict] = []

        def _al(msg: str, sev: str = 'warning') -> None:
            new_alerts.append({'message': msg, 'severity': sev, 'timestamp': _ts})

        if stress_level == 'High':
            _al('High stress detected', 'critical')
        if fatigue_score and fatigue_score > FATIGUE_THRESHOLD:
            _al('High fatigue detected')
        if drowsiness and drowsiness in DROWSINESS_ALERT_LEVELS:
            _al(f'Drowsiness: {drowsiness}', 'critical')
        if hr and (hr < HR_NORMAL_RANGE[0] or hr > HR_NORMAL_RANGE[1]):
            _al(f'Abnormal HR: {hr:.0f} BPM')
        if rmssd and rmssd < HRV_RMSSD_LOW:
            _al('Low HRV – elevated autonomic stress')
        if spo2 and spo2 < SPO2_LOW_THRESHOLD:
            _al(f'Low SpO2: {spo2:.1f}%', 'critical')

        # ── Metrics update ────────────────────────────────────────
        with metrics_lock:
            metrics.update({
                'hr':            hr,
                'hrv':           hrv_metrics,
                'rmssd':         round(rmssd, 2) if rmssd else None,
                'resp':          resp_rate,
                'blink':         round(blink_rate, 1) if blink_rate is not None else None,
                'fatigue':       round(fatigue_score, 3) if fatigue_score is not None else None,
                'perclos':       perclos,
                'drowsiness':    drowsiness,
                'microsleeps':   microsleeps,
                'emotion':       emotion,
                'emotion_scores': emotion_probs,
                'stress':        stress_level,
                'stress_score':  stress_score_val,
                'spo2':          spo2,
                'signal_quality': signal_quality,
                'hrv_freq':      hrv_freq,
                'lf_hf_ratio':   hrv_freq.get('lf_hf_ratio') if hrv_freq else None,
                'coherence':     hrv_freq.get('coherence')    if hrv_freq else None,
                'lf_power':      hrv_freq.get('lf_power')     if hrv_freq else None,
                'hf_power':      hrv_freq.get('hf_power')     if hrv_freq else None,
                'stress_index':  stress_idx,
                'ppg_signal':    ppg_signal,
                'rr_intervals':  rr_intervals,
                'alert':         new_alerts[0]['message'] if new_alerts else None,
                'alerts':        new_alerts,
                'timestamp':     _ts,
                'warmup':        not warmup_done,
                'warmup_pct':    warmup_pct,
                'face_detected': True,
            })
            processing_error = None
            for a in new_alerts:
                alert_history.append(a)
            if len(alert_history) > MAX_ALERT_HISTORY:
                del alert_history[:-MAX_ALERT_HISTORY]

        if _bc_frames % DB_SAVE_INTERVAL == 0 and not metrics['warmup']:
            _save_metrics_db(metrics.copy())

        # ── Annotate frame and store for /video_feed ──────────────
        try:
            ann   = frame.copy()
            sq    = signal_quality or 0.0
            b_col = (0, 220, 80) if sq >= 0.6 else (0, 165, 255) if sq >= 0.3 else (0, 60, 220)
            cv2.rectangle(ann, (x, y), (x + w, y + h), b_col, 2)
            fh_col = (0, 255, 255) if _motion_px < 6.0 else (0, 80, 255)
            cv2.rectangle(ann,
                          (x + int(w * 0.18), y + int(h * 0.08)),
                          (x + int(w * 0.82), y + int(h * 0.38)),
                          fh_col, 1)
            _, buf = cv2.imencode('.jpg', ann, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if _:
                with _frame_lock:
                    global _latest_frame
                    _latest_frame = buf.tobytes()
                return Response(buf.tobytes(), mimetype='image/jpeg')
        except Exception as e:
            logger.debug(f'BC annotation: {e}')

    return ('', 204)


@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f'API error: {error}', exc_info=True)
    return jsonify({'status': 'error', 'message': str(error)}), 500


# ------------------------------------------------------------------
# MJPEG live-video stream  (serves annotated frames from the worker)
# ------------------------------------------------------------------

def _make_placeholder_frame() -> bytes:
    """Return a JPEG placeholder when no camera frame is available yet."""
    try:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = (18, 22, 34)  # dark blue-grey
        cv2.putText(frame, 'Initialising...', (170, 220),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (100, 150, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Please position your face in frame', (90, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 140, 200), 1, cv2.LINE_AA)
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes()
    except Exception:
        return b''


@app.route('/video_feed')
def video_feed():
    """MJPEG stream of the processed (annotated) video feed."""
    def generate():
        while not stop_event.is_set():
            with _frame_lock:
                jpeg = _latest_frame
            if jpeg is None:
                jpeg = _make_placeholder_frame()
            if jpeg:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            time.sleep(1.0 / 25)   # cap stream at 25 FPS
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={'Cache-Control': 'no-cache, no-store, must-revalidate',
                 'Pragma': 'no-cache'},
    )


# ------------------------------------------------------------------
# Server-Sent Events  – real-time metrics push at ~4 Hz
# ------------------------------------------------------------------

@app.route('/api/stream')
def metrics_stream_sse():
    """SSE endpoint: pushes a JSON metrics object whenever it changes."""
    def generate():
        last_ts: Optional[str] = None
        while not stop_event.is_set():
            with metrics_lock:
                m = metrics.copy()
            ts = m.get('timestamp')
            if ts != last_ts:
                last_ts = ts
                try:
                    payload = json.dumps(m, default=str)
                    yield f'data: {payload}\n\n'
                except Exception:
                    pass
            time.sleep(0.25)   # 4 updates / second
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control':    'no-cache',
            'X-Accel-Buffering':'no',
            'Connection':       'keep-alive',
        },
    )


# ------------------------------------------------------------------
# Frontend static files
# ------------------------------------------------------------------

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend')


@app.route('/')
@app.route('/index.html')
def serve_index():
    return send_from_directory(_FRONTEND_DIR, 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(_FRONTEND_DIR, filename)


# ------------------------------------------------------------------
# Database helpers
# ------------------------------------------------------------------

def _init_db() -> None:
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS metrics (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT    NOT NULL,
                hr             REAL,
                hrv_rmssd      REAL,
                hrv_sdnn       REAL,
                hrv_pnn50      REAL,
                resp           REAL,
                blink          REAL,
                fatigue        REAL,
                perclos        REAL,
                drowsiness     TEXT,
                emotion        TEXT,
                stress         TEXT,
                stress_score   REAL,
                spo2           REAL,
                signal_quality REAL,
                microsleeps    INTEGER
            );
            CREATE TABLE IF NOT EXISTS alerts (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                message   TEXT NOT NULL,
                severity  TEXT DEFAULT 'warning'
            );
        ''')
        # Prune rows beyond cap
        conn.execute(f'''
            DELETE FROM metrics
            WHERE id NOT IN (SELECT id FROM metrics ORDER BY id DESC LIMIT {DB_MAX_ROWS})
        ''')
    logger.info(f'Database ready: {DB_PATH}')


def _save_metrics_db(m: dict) -> None:
    try:
        hrv = m.get('hrv') or {}
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''
                INSERT INTO metrics
                (timestamp, hr, hrv_rmssd, hrv_sdnn, hrv_pnn50, resp, blink, fatigue,
                 perclos, drowsiness, emotion, stress, stress_score, spo2,
                 signal_quality, microsleeps)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                m.get('timestamp'),
                m.get('hr'),
                hrv.get('rmssd') if hrv else m.get('rmssd'),
                hrv.get('sdnn'),
                hrv.get('pnn50'),
                m.get('resp'),
                m.get('blink'),
                m.get('fatigue'),
                m.get('perclos'),
                m.get('drowsiness'),
                m.get('emotion'),
                m.get('stress'),
                m.get('stress_score'),
                m.get('spo2'),
                m.get('signal_quality'),
                m.get('microsleeps'),
            ))
    except Exception as e:
        logger.debug(f'DB save error: {e}')


# ------------------------------------------------------------------
# Browser-camera processor pool
# ------------------------------------------------------------------

def _ensure_processors() -> bool:
    """Lazily initialise all signal-processing modules (once, thread-safe)."""
    global _procs
    with _procs_lock:
        if _procs:
            return True
        try:
            _procs['face']    = FaceDetector(min_detection_confidence=FACE_DETECTION_CONFIDENCE)
            _procs['lm']      = FaceLandmarks(
                min_detection_confidence=FACE_MESH_CONFIDENCE,
                min_tracking_confidence=FACE_MESH_TRACKING_CONFIDENCE,
            )
            _procs['rppg']    = RPPG(fs=FPS, window_sec=RPPG_WINDOW_SEC)
            _procs['resp']    = Respiration(fs=FPS, window_sec=RESPIRATION_WINDOW_SEC)
            _procs['blink']   = BlinkFatigue(fs=FPS, window_sec=BLINK_WINDOW_SEC,
                                              perclos_window_sec=PERCLOS_WINDOW_SEC)
            _procs['emotion'] = EmotionDetector()
            _procs['fusion']  = FeatureFusion(window_sec=FUSION_WINDOW_SEC, fs=FPS)
            _procs['clf']     = ClassicalModel(model_type='logreg',
                                               model_path=CLASSICAL_MODEL_PATH)
            logger.info('Browser-camera processors initialised')
            return True
        except Exception as e:
            logger.error(f'Processor init failed: {e}', exc_info=True)
            _procs = {}
            return False


# ------------------------------------------------------------------
# Camera helpers
# ------------------------------------------------------------------

def _open_camera(index: int) -> cv2.VideoCapture:
    """Open camera with multi-backend, multi-index retry logic.

    Strategy (Windows-aware):
      1. Prefer DirectShow (CAP_DSHOW) — most reliable on Windows laptops.
      2. Fall back to Media Foundation (CAP_MSMF).
      3. Finally try CAP_ANY (OpenCV default auto-select).
      4. If requested index fails, automatically probe 0, 1, 2.
    """
    backends: list[tuple[str, int]]
    if sys.platform == 'win32':
        backends = [
            ('DirectShow',         cv2.CAP_DSHOW),
            ('MediaFoundation',    cv2.CAP_MSMF),
            ('Auto',               cv2.CAP_ANY),
        ]
    elif sys.platform == 'darwin':
        backends = [('AVFoundation', cv2.CAP_AVFOUNDATION), ('Auto', cv2.CAP_ANY)]
    else:
        backends = [('V4L2', cv2.CAP_V4L2), ('Auto', cv2.CAP_ANY)]

    # If user set index=0 (default), also probe 1 and 2 automatically
    indices = [index] if index > 0 else [0, 1, 2]

    for backend_name, backend_id in backends:
        for idx in indices:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    cap = cv2.VideoCapture(idx, backend_id)
                    if not cap.isOpened():
                        cap.release()
                        break   # this index/backend combo won't work; next backend

                    # Verify the camera actually delivers frames
                    ret, frame = cap.read()
                    if not ret or frame is None or frame.size == 0:
                        cap.release()
                        break

                    # Apply capture properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS,          FPS)
                    # Keep auto-exposure ON — rPPG runs on normalised channels
                    # so absolute brightness changes are compensated by CHROM
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimise latency

                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    logger.info(
                        f'Camera {idx} opened via {backend_name} '
                        f'({actual_w}×{actual_h}) on attempt {attempt}'
                    )
                    return cap

                except Exception as e:
                    logger.debug(f'Camera {idx}/{backend_name} attempt {attempt}: {e}')
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY_SEC)

    raise RuntimeError(
        f'Cannot open any camera. Tried indices={indices}. '
        'Check that no other app is using the webcam, '
        'or set DEMO_MODE=true to run without a camera.'
    )


# ------------------------------------------------------------------
# Video processing thread
# ------------------------------------------------------------------

_HR_ALPHA = 0.25          # EMA weight for new HR reading (lower = smoother)
_MAX_CONSEC_FAILURES = 30  # reconnect after this many consecutive frame failures


# ------------------------------------------------------------------
# Demo mode – synthetic physiological data (no camera required)
# ------------------------------------------------------------------

def _demo_loop() -> None:
    """Generate realistic synthetic metrics when running in headless/demo mode.

    Simulates a person at rest with mild stress variation so every dashboard
    panel shows live, meaningful data without a physical webcam.
    """
    global session_start, total_frames
    import math, random

    session_start = datetime.now()

    # Baseline physiological values
    hr_base      = 72.0
    rmssd_base   = 42.0
    resp_base    = 15.0
    blink_base   = 18.0
    spo2_base    = 98.0

    t = 0.0
    logger.info('Demo mode active – generating synthetic physiological data')

    emotions = ['neutral', 'happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise']

    def _demo_frame(hr_v, spo2_v, resp_v, rmssd_v, fatigue_v,
                    stress_v, sq_v, t_v) -> None:
        """Generate and store a synthetic DEMO frame for the MJPEG stream."""
        try:
            import math as _m
            frm = np.zeros((480, 640, 3), dtype=np.uint8)
            frm[:, :] = (12, 18, 32)

            # Pulsing circle simulating a heartbeat
            pulse = int(abs(_m.sin(t_v * _m.pi * hr_v / 60 / 20)) * 28)
            cv2.circle(frm, (320, 200), 60 + pulse, (20, 60, 20), -1)
            cv2.circle(frm, (320, 200), 60 + pulse, (0, 180, 60), 2)
            cv2.putText(frm, 'DEMO MODE', (215, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(frm, '(No camera – synthetic data)', (130, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 140, 180), 1, cv2.LINE_AA)

            lines = [
                (f'HR:    {hr_v:.0f} BPM',   (80, 290),  (100, 255, 100)),
                (f'SpO2:  {spo2_v:.1f}%',     (80, 318),  (100, 210, 255)),
                (f'Resp:  {resp_v:.0f} BPM',  (80, 346),  (200, 220, 100)),
                (f'HRV:   {rmssd_v:.0f} ms',  (80, 374),  (200, 160, 255)),
                (f'Fat:   {fatigue_v:.2f}',    (360, 290), (255, 190, 80)),
                (f'Stress:{stress_v}',         (360, 318), (255, 100, 100) if stress_v == 'High'
                                                else (255, 200, 80) if stress_v == 'Medium'
                                                else (100, 255, 100)),
                (f'SQ:    {int(sq_v * 100)}%', (360, 346), (100, 255, 100)),
            ]
            for text, pos, col in lines:
                cv2.putText(frm, text, pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 1, cv2.LINE_AA)

            _, buf = cv2.imencode('.jpg', frm, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if _:
                with _frame_lock:
                    global _latest_frame
                    _latest_frame = buf.tobytes()
        except Exception as _e:
            logger.debug(f'Demo frame error: {_e}')

    while not stop_event.is_set():
        t += 1.0
        total_frames += 1

        # Sinusoidal drift + Gaussian noise to mimic real signals
        hr       = round(hr_base   + 8 * math.sin(t / 30) + random.gauss(0, 1.5), 1)
        rmssd    = round(rmssd_base + 10 * math.sin(t / 60) + random.gauss(0, 3), 2)
        sdnn     = round(rmssd * 1.2 + random.gauss(0, 2), 2)
        pnn50    = round(max(0, 25 + 10 * math.sin(t / 45) + random.gauss(0, 4)), 1)
        resp     = round(resp_base  + 2 * math.sin(t / 20) + random.gauss(0, 0.5), 1)
        blink    = round(blink_base + 3 * math.sin(t / 50) + random.gauss(0, 1), 1)
        fatigue  = round(min(1.0, max(0.0, 0.2 + 0.1 * math.sin(t / 90) + random.gauss(0, 0.02))), 3)
        perclos  = round(min(100, max(0, fatigue * 15 + random.gauss(0, 1))), 1)
        spo2     = round(min(100, spo2_base + random.gauss(0, 0.3)), 1)
        sq       = round(min(1.0, max(0.0, 0.75 + random.gauss(0, 0.05))), 2)
        microsleeps = 0

        # Drowsiness based on fatigue level
        if fatigue > 0.7:
            drowsiness = 'Moderate'
        elif fatigue > 0.85:
            drowsiness = 'Severe'
        else:
            drowsiness = 'Alert'

        # Stress score
        stress_score = round(min(1.0, max(0.0,
            0.35 * max(0, (hr - 72) / 40) +
            0.35 * max(0, (42 - rmssd) / 40) +
            0.20 * fatigue + random.gauss(0, 0.02)
        )), 3)
        if stress_score < 0.3:
            stress_level = 'Low'
        elif stress_score < 0.6:
            stress_level = 'Medium'
        else:
            stress_level = 'High'

        # Frequency-domain HRV simulation (Mayer wave ~0.1 Hz + RSA ~0.25 Hz)
        lf_p   = round(max(0.0001, 0.0035 + 0.002  * math.sin(t / 55) + random.gauss(0, 0.0003)), 4)
        hf_p   = round(max(0.0001, 0.0022 + 0.0012 * math.cos(t / 28) + random.gauss(0, 0.0002)), 4)
        vlf_p  = round(max(0.0001, 0.0015 + 0.0008 * math.sin(t / 120)), 4)
        lf_hf  = round(lf_p / (hf_p + 1e-9), 2)
        coher  = round(max(0, min(100, 40 + 25 * math.sin(t / 40) + random.gauss(0, 5))), 1)
        si_raw = round(max(0, min(1.0, 0.38 + 0.15 * math.sin(t / 70) + random.gauss(0, 0.03))), 3)

        # Synthetic RR intervals for Poincaré (10–20 intervals, ~800 ms mean)
        n_rr   = random.randint(14, 22)
        mean_rr_demo = 60000.0 / max(hr, 40)
        sdnn_demo    = max(5, rmssd * 1.2)
        rr_demo = [round(max(400, min(1400,
                    mean_rr_demo + random.gauss(0, sdnn_demo))), 1)
                   for _ in range(n_rr)]

        # PPG waveform (synthetic cardiac pulse shape)
        ppg_demo: list[float] = []
        for k in range(120):
            phase = (k / 120.0) * 2 * math.pi * (hr / 60.0) * 6
            pulse = (math.sin(phase) +
                     0.3 * math.sin(2 * phase + 0.5) +
                     0.15 * math.sin(3 * phase + 1.0))
            ppg_demo.append(round(pulse / 1.45, 4))

        # Emotion scores
        top_emotion = random.choices(
            emotions, weights=[50, 20, 5, 5, 5, 5, 10])[0]
        raw_scores = {e: random.random() * 0.1 for e in emotions}
        raw_scores[top_emotion] += 0.6
        total_s = sum(raw_scores.values())
        emotion_scores = {e: round(v / total_s, 4) for e, v in raw_scores.items()}

        _ts = datetime.now().isoformat()
        new_alerts: list[dict] = []

        def _al(msg: str, sev: str = 'warning') -> None:
            new_alerts.append({'message': msg, 'severity': sev, 'timestamp': _ts})

        if stress_level == 'High':
            _al('High stress detected', 'critical')
        if fatigue > FATIGUE_THRESHOLD:
            _al('High fatigue detected')
        if drowsiness in DROWSINESS_ALERT_LEVELS:
            _al(f'Drowsiness: {drowsiness}', 'critical')
        if not (HR_NORMAL_RANGE[0] <= hr <= HR_NORMAL_RANGE[1]):
            _al(f'Abnormal HR: {hr:.0f} BPM')
        if rmssd < HRV_RMSSD_LOW:
            _al('Low HRV – elevated autonomic stress')
        if spo2 < SPO2_LOW_THRESHOLD:
            _al(f'Low SpO2: {spo2:.1f}%', 'critical')

        with metrics_lock:
            metrics.update({
                'hr':             hr,
                'hrv':            {'rmssd': rmssd, 'sdnn': sdnn, 'pnn50': pnn50},
                'rmssd':          rmssd,
                'resp':           resp,
                'blink':          blink,
                'fatigue':        fatigue,
                'perclos':        perclos,
                'drowsiness':     drowsiness,
                'microsleeps':    microsleeps,
                'emotion':        top_emotion,
                'emotion_scores': emotion_scores,
                'stress':         stress_level,
                'stress_score':   stress_score,
                'spo2':           spo2,
                'signal_quality': sq,
                # Frequency-domain HRV
                'hrv_freq':    {'vlf_power': vlf_p, 'lf_power': lf_p, 'hf_power': hf_p,
                                'lf_hf_ratio': lf_hf, 'coherence': coher,
                                'lf_pct': round(lf_p / (vlf_p + lf_p + hf_p + 1e-9) * 100, 1),
                                'hf_pct': round(hf_p / (vlf_p + lf_p + hf_p + 1e-9) * 100, 1)},
                'lf_hf_ratio': lf_hf,
                'coherence':   coher,
                'lf_power':    lf_p,
                'hf_power':    hf_p,
                'stress_index': si_raw,
                'ppg_signal':   ppg_demo,
                'rr_intervals': rr_demo,
                'alert':          new_alerts[0]['message'] if new_alerts else None,
                'alerts':         new_alerts,
                'timestamp':      _ts,
                'warmup':         t < 5,
                'warmup_pct':     100 if t >= 5 else int(t / 5 * 100),
                'face_detected':  True,
            })
            for a in new_alerts:
                alert_history.append(a)
            if len(alert_history) > MAX_ALERT_HISTORY:
                del alert_history[:-MAX_ALERT_HISTORY]

        if total_frames % DB_SAVE_INTERVAL == 0:
            _save_metrics_db(metrics.copy())

        _demo_frame(hr, spo2, resp, rmssd, fatigue, stress_level, sq, t)

        time.sleep(1.0)

    logger.info('Demo loop stopped')


def process_video() -> None:
    global processing_error, session_start, total_frames, _browser_mode
    cap: Optional[cv2.VideoCapture] = None

    try:
        try:
            cap = _open_camera(WEBCAM_INDEX)
        except RuntimeError as cam_err:
            logger.warning(
                f'No physical camera found – switching to browser-camera mode: {cam_err}'
            )
            _browser_mode = True
            _ensure_processors()
            logger.info('Browser-camera mode active – waiting for frames via POST /api/frame')
            while not stop_event.is_set():
                time.sleep(1.0)
            return

        # ── Module initialisation ─────────────────────────────────────
        logger.info('Initialising signal processing modules...')
        face_detector      = FaceDetector(min_detection_confidence=FACE_DETECTION_CONFIDENCE)
        landmarks_detector = FaceLandmarks(
            min_detection_confidence=FACE_MESH_CONFIDENCE,
            min_tracking_confidence=FACE_MESH_TRACKING_CONFIDENCE,
        )
        rppg_proc     = RPPG(fs=FPS, window_sec=RPPG_WINDOW_SEC)
        resp_proc     = Respiration(fs=FPS, window_sec=RESPIRATION_WINDOW_SEC)
        blink_proc    = BlinkFatigue(fs=FPS, window_sec=BLINK_WINDOW_SEC,
                                     perclos_window_sec=PERCLOS_WINDOW_SEC)
        emotion_det   = EmotionDetector()
        fusion        = FeatureFusion(window_sec=FUSION_WINDOW_SEC, fs=FPS)
        clf_model     = ClassicalModel(model_type='logreg',
                                       model_path=CLASSICAL_MODEL_PATH)

        LEFT_EYE  = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        RESP_LM   = Respiration.RESP_LANDMARK_INDICES

        logger.info('All modules ready')
        processing_error = None
        session_start    = datetime.now()

        hr_ema:            Optional[float] = None
        consec_failures:   int             = 0
        _prev_nose_xy:     Optional[tuple] = None   # for motion gating
        _motion_px:        float           = 0.0    # last frame motion magnitude
        MOTION_GATE_PX:    float           = 6.0    # max pixels/frame before gating rPPG

        # ===== MAIN LOOP =============================================
        while not stop_event.is_set():
            frame_t0 = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                consec_failures += 1
                if consec_failures >= _MAX_CONSEC_FAILURES:
                    logger.warning('Too many frame read failures – reconnecting camera')
                    cap.release()
                    try:
                        cap = _open_camera(WEBCAM_INDEX)
                        consec_failures = 0
                    except RuntimeError as e:
                        processing_error = str(e)
                        logger.error(e)
                        time.sleep(5)
                else:
                    time.sleep(0.05)
                continue
            consec_failures = 0
            total_frames += 1

            # ── Face detection ────────────────────────────────────────
            try:
                bboxes = face_detector.detect(frame)
                if not bboxes:
                    _sleep_remainder(frame_t0)
                    continue
                x, y, w, h = bboxes[0]
                if w <= 0 or h <= 0:
                    continue
                face_roi = frame[y:y + h, x:x + w]
                if face_roi.size == 0:
                    continue
            except Exception as e:
                logger.debug(f'Face detection: {e}')
                continue

            # ── Landmarks ────────────────────────────────────────────
            try:
                lm_list = landmarks_detector.get_landmarks(frame)
                if not lm_list:
                    continue
                lm = lm_list[0]
            except Exception as e:
                logger.debug(f'Landmarks: {e}')
                continue

            # ── Motion gating (landmark velocity) ────────────────────
            # Gate rPPG when head moves — motion injects colour artefacts.
            # Nose-bridge (lm[6]) is a stable anatomical landmark.
            _motion_px = 0.0
            if len(lm) > 6:
                nose_xy = lm[6]
                if _prev_nose_xy is not None:
                    dx = nose_xy[0] - _prev_nose_xy[0]
                    dy = nose_xy[1] - _prev_nose_xy[1]
                    _motion_px = float((dx * dx + dy * dy) ** 0.5)
                _prev_nose_xy = nose_xy

            # ── rPPG (CHROM) ─────────────────────────────────────────
            # FOREHEAD-ONLY ROI: scientifically the best rPPG region —
            # high blood perfusion, minimal hair/beard interference, and
            # least affected by lip/eye muscle movements.
            # Region: rows [8%, 38%] of face height, cols [18%, 82%] width.
            hr = hrv_metrics = spo2 = signal_quality = rmssd = None
            hrv_freq = stress_idx = None
            ppg_signal: list = []
            rr_intervals: list = []
            try:
                fh_y1 = y + max(0, int(h * 0.08))
                fh_y2 = y + min(h, int(h * 0.38))
                fh_x1 = x + max(0, int(w * 0.18))
                fh_x2 = x + min(w, int(w * 0.82))
                forehead_roi = frame[fh_y1:fh_y2, fh_x1:fh_x2]

                # Only update rPPG when head is still enough
                if forehead_roi.size > 0 and _motion_px < MOTION_GATE_PX:
                    rgb_means = (
                        float(np.mean(forehead_roi[:, :, 2])),  # R (BGR order)
                        float(np.mean(forehead_roi[:, :, 1])),  # G
                        float(np.mean(forehead_roi[:, :, 0])),  # B
                    )
                    rppg_proc.update(rgb_means)
                hr_raw, hrv_metrics = rppg_proc.compute_hr()
                signal_quality = round(rppg_proc.signal_quality, 2)
                spo2           = rppg_proc.estimate_spo2()

                if hr_raw is not None and signal_quality >= SIGNAL_QUALITY_MIN:
                    clipped = float(np.clip(hr_raw, 30, 220))
                    hr_ema  = clipped if hr_ema is None \
                        else _HR_ALPHA * clipped + (1 - _HR_ALPHA) * hr_ema
                    hr = round(hr_ema, 1)

                rmssd        = hrv_metrics.get('rmssd') if hrv_metrics else None
                # Advanced HRV — computed every frame (cheap after compute_hr)
                hrv_freq     = rppg_proc.compute_hrv_frequency_domain()
                stress_idx   = rppg_proc.baevsky_stress_index()
                ppg_signal   = rppg_proc.get_ppg_signal(150)
                rr_intervals = rppg_proc.rr_intervals[-50:]
            except Exception as e:
                logger.debug(f'rPPG: {e}')

            # ── Respiration ──────────────────────────────────────────
            resp_rate = None
            try:
                if RESP_LM and max(RESP_LM) < len(lm):
                    resp_proc.update([lm[i][1] for i in RESP_LM])
                    resp_rate = resp_proc.compute_resp_rate()
            except Exception as e:
                logger.debug(f'Respiration: {e}')

            # ── Blink / Fatigue / PERCLOS ─────────────────────────────
            blink_rate = fatigue_score = perclos = drowsiness = microsleeps = None
            try:
                all_eye = LEFT_EYE + RIGHT_EYE
                if max(all_eye) < len(lm):
                    left_ear  = BlinkFatigue.compute_ear([lm[i] for i in LEFT_EYE])
                    right_ear = BlinkFatigue.compute_ear([lm[i] for i in RIGHT_EYE])
                    blink_proc.update((left_ear, right_ear))
                    result = blink_proc.compute_blink_fatigue()
                    if result[0] is not None:
                        blink_rate, fatigue_score, perclos, drowsiness, microsleeps = result
            except Exception as e:
                logger.debug(f'Blink/fatigue: {e}')

            # ── Emotion ──────────────────────────────────────────────
            emotion = emotion_probs = None
            try:
                emotion_probs = emotion_det.predict(face_roi, lm)
                emotion = max(emotion_probs, key=emotion_probs.get) if emotion_probs else None
            except Exception as e:
                logger.debug(f'Emotion: {e}')

            # ── Stress prediction ─────────────────────────────────────
            stress_level = stress_score_val = None
            try:
                fusion.update([
                    hr          or 0.0,
                    rmssd       or 0.0,
                    resp_rate   or 0.0,
                    blink_rate  or 0.0,
                    fatigue_score or 0.0,
                ])
                fused = fusion.get_fused_features()
                if fused is not None:
                    proba = clf_model.predict(fused.reshape(1, -1))
                    if proba is not None and len(proba) > 0:
                        p = proba[0]
                        stress_score_val = round(
                            float(p[1] * 0.5 + p[2]) if len(p) >= 3 else float(p[-1]), 3
                        )
                        if stress_score_val < STRESS_THRESHOLDS['low']:
                            stress_level = 'Low'
                        elif stress_score_val < STRESS_THRESHOLDS['medium']:
                            stress_level = 'Medium'
                        else:
                            stress_level = 'High'
            except Exception as e:
                logger.debug(f'Prediction: {e}')

            # ── Warmup check ──────────────────────────────────────────
            warmup_done = (
                len(rppg_proc.r_window) >= rppg_proc.window_size and
                len(blink_proc.ear_window) >= blink_proc.window_size and
                len(resp_proc.window) >= resp_proc.window_size
            )

            # ── Build alert list ──────────────────────────────────────
            _ts = datetime.now().isoformat()
            new_alerts: list[dict] = []

            def _al(msg: str, sev: str = 'warning') -> None:
                new_alerts.append({'message': msg, 'severity': sev, 'timestamp': _ts})

            if stress_level == 'High':
                _al('High stress detected', 'critical')
            if fatigue_score and fatigue_score > FATIGUE_THRESHOLD:
                _al('High fatigue detected')
            if drowsiness and drowsiness in DROWSINESS_ALERT_LEVELS:
                _al(f'Drowsiness: {drowsiness}', 'critical')
            if hr and (hr < HR_NORMAL_RANGE[0] or hr > HR_NORMAL_RANGE[1]):
                _al(f'Abnormal HR: {hr:.0f} BPM')
            if resp_rate and (resp_rate < RESP_NORMAL_RANGE[0]
                              or resp_rate > RESP_NORMAL_RANGE[1]):
                _al(f'Abnormal RR: {resp_rate:.0f} BPM')
            if rmssd and rmssd < HRV_RMSSD_LOW:
                _al('Low HRV – elevated autonomic stress')
            if spo2 and spo2 < SPO2_LOW_THRESHOLD:
                _al(f'Low SpO2: {spo2:.1f}%', 'critical')

            # ── Warmup progress (0–100 %) ─────────────────────────────
            warmup_pct = min(100, int(
                min(
                    len(rppg_proc.r_window)   / rppg_proc.window_size,
                    len(blink_proc.ear_window) / blink_proc.window_size,
                    len(resp_proc.window)      / resp_proc.window_size,
                ) * 100
            ))

            # ── Thread-safe metrics update ────────────────────────────
            with metrics_lock:
                metrics.update({
                    'hr':            hr,
                    'hrv':           hrv_metrics,
                    'rmssd':         round(rmssd, 2) if rmssd else None,
                    'resp':          resp_rate,
                    'blink':         round(blink_rate, 1) if blink_rate is not None else None,
                    'fatigue':       round(fatigue_score, 3) if fatigue_score is not None else None,
                    'perclos':       perclos,
                    'drowsiness':    drowsiness,
                    'microsleeps':   microsleeps,
                    'emotion':       emotion,
                    'emotion_scores': emotion_probs,
                    'stress':        stress_level,
                    'stress_score':  stress_score_val,
                    'spo2':          spo2,
                    'signal_quality': signal_quality,
                    # Frequency-domain HRV
                    'hrv_freq':    hrv_freq,
                    'lf_hf_ratio': hrv_freq.get('lf_hf_ratio') if hrv_freq else None,
                    'coherence':   hrv_freq.get('coherence')   if hrv_freq else None,
                    'lf_power':    hrv_freq.get('lf_power')    if hrv_freq else None,
                    'hf_power':    hrv_freq.get('hf_power')    if hrv_freq else None,
                    # Autonomic stress index
                    'stress_index': stress_idx,
                    # Waveform & Poincaré
                    'ppg_signal':   ppg_signal,
                    'rr_intervals': rr_intervals,
                    'alert':         new_alerts[0]['message'] if new_alerts else None,
                    'alerts':        new_alerts,
                    'timestamp':     _ts,
                    'warmup':        not warmup_done,
                    'warmup_pct':    warmup_pct,
                    'face_detected': True,
                })
                processing_error = None
                for a in new_alerts:
                    alert_history.append(a)
                if len(alert_history) > MAX_ALERT_HISTORY:
                    del alert_history[:-MAX_ALERT_HISTORY]

            # ── Annotate frame for MJPEG stream ───────────────────────
            try:
                ann = frame.copy()
                sq  = signal_quality or 0.0

                # Face bounding box (colour = signal quality)
                box_col = (0, 220, 80) if sq >= 0.6 else \
                          (0, 165, 255) if sq >= 0.3 else (0, 60, 220)
                cv2.rectangle(ann, (x, y), (x + w, y + h), box_col, 2)

                # Forehead ROI box — shows the region used for rPPG
                fh_col = (0, 255, 255) if _motion_px < MOTION_GATE_PX \
                         else (0, 80, 255)   # cyan=sampling, red=motion gated
                cv2.rectangle(ann,
                              (x + int(w * 0.18), y + int(h * 0.08)),
                              (x + int(w * 0.82), y + int(h * 0.38)),
                              fh_col, 1)
                cv2.putText(ann, 'rPPG',
                            (x + int(w * 0.18), y + int(h * 0.08) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, fh_col, 1, cv2.LINE_AA)

                # Eye landmark dots
                for lm_idx in LEFT_EYE + RIGHT_EYE:
                    if lm_idx < len(lm):
                        cv2.circle(ann,
                                   (int(lm[lm_idx][0]), int(lm[lm_idx][1])),
                                   2, (255, 200, 0), -1)

                # Motion indicator (top-right corner)
                if _motion_px >= MOTION_GATE_PX:
                    cv2.putText(ann, 'MOTION',
                                (ann.shape[1] - 90, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 80, 255), 2, cv2.LINE_AA)

                # Top-left metric overlay
                _oy = 26
                def _ot(text, col=(220, 220, 220)):
                    nonlocal _oy
                    cv2.putText(ann, text, (8, _oy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)
                    _oy += 20

                if hr:
                    _ot(f'HR:   {hr:.0f} BPM',        (100, 255, 100))
                if spo2:
                    _ot(f'SpO2: {spo2:.1f}% ~est',     (100, 210, 255))
                if resp_rate:
                    _ot(f'Resp: {resp_rate:.0f} BPM',  (200, 220, 100))
                if rmssd:
                    _ot(f'HRV:  {rmssd:.0f} ms',       (200, 160, 255))
                if fatigue_score is not None:
                    _ot(f'Fat:  {fatigue_score:.2f}',   (255, 190, 80))
                sq_col = (0, 220, 80) if sq >= 0.6 else \
                         (0, 165, 255) if sq >= 0.3 else (0, 80, 220)
                _ot(f'SQ:   {int(sq * 100)}%',  sq_col)

                if not warmup_done:
                    bar_w = int(warmup_pct / 100 * (w - 4))
                    cv2.rectangle(ann, (x + 2, y + h + 4),
                                  (x + 2 + bar_w, y + h + 14),
                                  (100, 200, 255), -1)
                    cv2.putText(ann, f'Warmup {warmup_pct}%',
                                (x + 2, y + h + 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (160, 220, 255), 1, cv2.LINE_AA)

                _, buf = cv2.imencode('.jpg', ann,
                                      [cv2.IMWRITE_JPEG_QUALITY, 82])
                if _:
                    with _frame_lock:
                        global _latest_frame
                        _latest_frame = buf.tobytes()
            except Exception as _fe:
                logger.debug(f'Frame annotation: {_fe}')

            # ── Periodic DB persist ───────────────────────────────────
            if total_frames % DB_SAVE_INTERVAL == 0 and not metrics['warmup']:
                _save_metrics_db(metrics.copy())

            # ── Periodic console log ──────────────────────────────────
            if total_frames % 100 == 0:
                logger.info(
                    f'Frame {total_frames:6d} | '
                    f'HR={hr} BPM | SpO2={spo2}% | '
                    f'Stress={stress_level} | '
                    f'Fatigue={fatigue_score} | SQ={signal_quality}'
                )

            # ── Precise frame pacing ──────────────────────────────────
            _sleep_remainder(frame_t0)

    except Exception as e:
        logger.critical(f'Fatal error in video processing: {e}', exc_info=True)
        processing_error = str(e)
    finally:
        if cap:
            cap.release()
        logger.info('Video processing thread stopped')


def _sleep_remainder(frame_t0: float) -> None:
    """Sleep for whatever time remains in the current frame budget."""
    elapsed = time.monotonic() - frame_t0
    remaining = 1.0 / FPS - elapsed
    if remaining > 0:
        time.sleep(remaining)


# ------------------------------------------------------------------
# Signal handlers
# ------------------------------------------------------------------

def _signal_handler(signum, _frame) -> None:
    logger.info(f'Signal {signum} received – shutting down gracefully')
    stop_event.set()
    sys.exit(0)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == '__main__':
    logger.info('=' * 60)
    logger.info('Multi-Modal Physiological Monitoring System')
    logger.info(f'Python {sys.version.split()[0]} | PID {os.getpid()}')
    logger.info('=' * 60)

    _init_db()

    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if DEMO_MODE:
        logger.info('DEMO_MODE=true – starting synthetic data generator (no webcam needed)')
        worker_target = _demo_loop
        worker_name   = 'DemoLoop'
    else:
        worker_target = process_video
        worker_name   = 'VideoProcessing'

    vid_thread = threading.Thread(target=worker_target,
                                  daemon=True, name=worker_name)
    vid_thread.start()
    logger.info(f'{worker_name} thread started')

    try:
        logger.info(f'Dashboard → http://localhost:{FLASK_PORT}/index.html')
        logger.info(f'API       → http://localhost:{FLASK_PORT}/api/metrics')
        app.run(host=FLASK_HOST, port=FLASK_PORT,
                debug=FLASK_DEBUG, use_reloader=False)
    except Exception as e:
        logger.error(f'Flask error: {e}', exc_info=True)
    finally:
        stop_event.set()
        logger.info('System stopped')
