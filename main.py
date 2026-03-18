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
    'hr': None, 'hrv': None, 'rmssd': None,
    'resp': None, 'blink': None, 'fatigue': None,
    'perclos': None, 'drowsiness': None, 'microsleeps': None,
    'emotion': None, 'emotion_scores': None,
    'stress': None, 'stress_score': None,
    'spo2': None, 'signal_quality': None,
    'alert': None, 'alerts': [],
    'timestamp': None,
    'warmup': True,   # True until all signal buffers have warmed up
}
metrics_lock: threading.Lock = threading.Lock()
stop_event:   threading.Event = threading.Event()
processing_error: Optional[str] = None
alert_history:    list = []
session_start:    Optional[datetime] = None
total_frames:     int = 0

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


@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f'API error: {error}', exc_info=True)
    return jsonify({'status': 'error', 'message': str(error)}), 500


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
# Camera helpers
# ------------------------------------------------------------------

def _open_camera(index: int) -> cv2.VideoCapture:
    """Open camera with retry logic. Raises RuntimeError if all attempts fail."""
    for attempt in range(1, MAX_RETRIES + 1):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS,          FPS)
            logger.info(f'Camera {index} opened on attempt {attempt}')
            return cap
        cap.release()
        logger.warning(f'Camera {index}: open attempt {attempt}/{MAX_RETRIES} failed')
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SEC)
    raise RuntimeError(f'Cannot open camera {index} after {MAX_RETRIES} attempts')


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
                'alert':          new_alerts[0]['message'] if new_alerts else None,
                'alerts':         new_alerts,
                'timestamp':      _ts,
                'warmup':         t < 5,
            })
            for a in new_alerts:
                alert_history.append(a)
            if len(alert_history) > MAX_ALERT_HISTORY:
                del alert_history[:-MAX_ALERT_HISTORY]

        if total_frames % DB_SAVE_INTERVAL == 0:
            _save_metrics_db(metrics.copy())

        time.sleep(1.0)

    logger.info('Demo loop stopped')


def process_video() -> None:
    global processing_error, session_start, total_frames
    cap: Optional[cv2.VideoCapture] = None

    try:
        cap = _open_camera(WEBCAM_INDEX)

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

        hr_ema:          Optional[float] = None
        consec_failures: int = 0

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

            # ── rPPG (CHROM) ─────────────────────────────────────────
            hr = hrv_metrics = spo2 = signal_quality = rmssd = None
            try:
                rgb_means = (
                    float(np.mean(face_roi[:, :, 2])),  # R  (OpenCV = BGR)
                    float(np.mean(face_roi[:, :, 1])),  # G
                    float(np.mean(face_roi[:, :, 0])),  # B
                )
                rppg_proc.update(rgb_means)
                hr_raw, hrv_metrics = rppg_proc.compute_hr()
                signal_quality = round(rppg_proc.signal_quality, 2)
                spo2 = rppg_proc.estimate_spo2()

                if hr_raw is not None and signal_quality >= SIGNAL_QUALITY_MIN:
                    clipped = float(np.clip(hr_raw, 30, 220))
                    # Exponential moving average for stable display
                    hr_ema = clipped if hr_ema is None \
                        else _HR_ALPHA * clipped + (1 - _HR_ALPHA) * hr_ema
                    hr = round(hr_ema, 1)

                rmssd = hrv_metrics.get('rmssd') if hrv_metrics else None
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
                    'alert':         new_alerts[0]['message'] if new_alerts else None,
                    'alerts':        new_alerts,
                    'timestamp':     _ts,
                    'warmup':        not warmup_done,
                })
                processing_error = None
                for a in new_alerts:
                    alert_history.append(a)
                if len(alert_history) > MAX_ALERT_HISTORY:
                    del alert_history[:-MAX_ALERT_HISTORY]

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
