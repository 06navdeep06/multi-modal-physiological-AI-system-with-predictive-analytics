"""Multi-Modal Physiological Monitoring System - Main Entry Point

Real-time heart rate (CHROM rPPG), HRV, SpO2, respiration, blink/fatigue/PERCLOS,
and emotion detection with stress prediction, SQLite persistence, and a REST API.

New in this version:
  - CHROM-based rPPG (better accuracy)
  - HRV: RMSSD, SDNN, pNN50
  - SpO2 estimation
  - PERCLOS + drowsiness level
  - Both-eyes EAR averaging
  - Microsleep detection
  - SQLite metrics persistence
  - Rotating log files (no unbounded growth)
  - /api/history, /api/stats, /api/export, /api/alerts endpoints
  - Session tracking (uptime, frame count)
  - Signal quality gating for rPPG results
"""

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

import cv2
import numpy as np
from flask import Flask, jsonify, request, Response
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

# ===== LOGGING (rotating file handler) =====
_log_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
)
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[_log_handler, logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ===== GLOBAL STATE =====
metrics = {
    'hr': None, 'hrv': None, 'rmssd': None,
    'resp': None, 'blink': None, 'fatigue': None,
    'perclos': None, 'drowsiness': None, 'microsleeps': None,
    'emotion': None, 'emotion_scores': None,
    'stress': None, 'stress_score': None,
    'spo2': None, 'signal_quality': None,
    'alert': None, 'alerts': [],
    'timestamp': None,
}
metrics_lock   = threading.Lock()
stop_event     = threading.Event()
processing_error: str | None = None
alert_history: list[dict]    = []
session_start: datetime | None = None
total_frames: int = 0

# ===== FLASK APP =====
app = Flask(__name__)
CORS(app)


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
        'status':    'healthy',
        'timestamp': datetime.now().isoformat(),
        'uptime_sec': round(uptime),
        'frames_processed': total_frames,
    }), 200


@app.route('/api/history', methods=['GET'])
def get_history():
    """Return the last N rows from the metrics database."""
    limit = min(request.args.get('limit', HISTORY_LIMIT, type=int), 2000)
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM metrics ORDER BY id DESC LIMIT ?', (limit,))
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return jsonify({'data': list(reversed(rows)), 'count': len(rows)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Session-level statistics (averages over the last hour)."""
    uptime = (datetime.now() - session_start).total_seconds() if session_start else 0
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            SELECT AVG(hr), MIN(hr), MAX(hr),
                   AVG(resp), AVG(fatigue), AVG(spo2),
                   AVG(hrv_rmssd), AVG(perclos), COUNT(*)
            FROM metrics
            WHERE timestamp > datetime('now', '-1 hour')
        ''')
        row = c.fetchone()
        conn.close()

        def _r(v, n=1):
            return round(v, n) if v is not None else None

        return jsonify({
            'session_uptime_sec':    round(uptime),
            'total_frames':          total_frames,
            'averages_last_hour': {
                'hr':        _r(row[0]),
                'hr_min':    _r(row[1]),
                'hr_max':    _r(row[2]),
                'resp':      _r(row[3]),
                'fatigue':   _r(row[4], 3),
                'spo2':      _r(row[5]),
                'rmssd':     _r(row[6], 2),
                'perclos':   _r(row[7]),
                'readings':  row[8],
            },
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export', methods=['GET'])
def export_csv():
    """Download metrics history as a CSV file."""
    limit = min(request.args.get('limit', 1000, type=int), 10_000)
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM metrics ORDER BY id DESC LIMIT ?', (limit,))
        rows = [dict(r) for r in c.fetchall()]
        conn.close()

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
def get_alert_history():
    """Return recent alert history."""
    limit = min(request.args.get('limit', 50, type=int), 500)
    with metrics_lock:
        recent = list(reversed(alert_history[-limit:]))
    return jsonify({'alerts': recent, 'total': len(alert_history)})


@app.route('/api/alerts/summary', methods=['GET'])
def get_alert_summary():
    """Aggregate alert counts and latest entries.

    Query params:
        limit:      max alerts to inspect (default 200, max 2000)
        window_sec: optional lookback window in seconds
    """
    limit = min(request.args.get('limit', 200, type=int), 2000)
    window_sec = request.args.get('window_sec', type=int)

    with metrics_lock:
        recent = list(alert_history[-limit:])

    if window_sec:
        cutoff = datetime.now() - timedelta(seconds=max(window_sec, 0))
        filtered = []
        for alert in recent:
            ts = alert.get('timestamp')
            try:
                alert_time = datetime.fromisoformat(ts) if ts else None
            except Exception:
                alert_time = None
            if alert_time is None or alert_time >= cutoff:
                filtered.append(alert)
        recent = filtered

    severity_counts = Counter(a.get('severity', 'warning') for a in recent)
    latest = recent[-1] if recent else None

    return jsonify({
        'total': len(recent),
        'by_severity': dict(severity_counts),
        'latest': latest,
        'window_sec': window_sec,
        'inspected': limit,
    })


@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f'API error: {error}', exc_info=True)
    return jsonify({'status': 'error', 'message': str(error)}), 500


# ------------------------------------------------------------------
# Database helpers
# ------------------------------------------------------------------

def _init_db():
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
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
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            message   TEXT NOT NULL,
            severity  TEXT DEFAULT 'warning'
        )
    ''')
    # Prune old rows
    c.execute(f'''
        DELETE FROM metrics
        WHERE id NOT IN (SELECT id FROM metrics ORDER BY id DESC LIMIT {DB_MAX_ROWS})
    ''')
    conn.commit()
    conn.close()
    logger.info(f'Database initialised at {DB_PATH}')


def _save_metrics_db(m: dict):
    try:
        hrv = m.get('hrv') or {}
        conn = sqlite3.connect(DB_PATH)
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
        conn.commit()
        conn.close()
    except Exception as e:
        logger.debug(f'DB save error: {e}')


# ------------------------------------------------------------------
# Video processing thread
# ------------------------------------------------------------------

def process_video():
    global processing_error, session_start, total_frames
    cap = None

    try:
        logger.info(f'Opening camera {WEBCAM_INDEX}...')
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open camera {WEBCAM_INDEX}')

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          FPS)
        logger.info('Camera ready')

        # Module init
        logger.info('Initialising signal processing modules...')
        face_detector     = FaceDetector(min_detection_confidence=FACE_DETECTION_CONFIDENCE)
        landmarks_detector = FaceLandmarks(
            min_detection_confidence=FACE_MESH_CONFIDENCE,
            min_tracking_confidence=FACE_MESH_TRACKING_CONFIDENCE,
        )
        rppg         = RPPG(fs=FPS, window_sec=RPPG_WINDOW_SEC)
        respiration  = Respiration(fs=FPS, window_sec=RESPIRATION_WINDOW_SEC)
        blink_fatigue = BlinkFatigue(fs=FPS, window_sec=BLINK_WINDOW_SEC,
                                      perclos_window_sec=PERCLOS_WINDOW_SEC)
        emotion_det  = EmotionDetector()
        fusion       = FeatureFusion(window_sec=FUSION_WINDOW_SEC, fs=FPS)

        try:
            clf_model = ClassicalModel(model_type='logreg',
                                       model_path=CLASSICAL_MODEL_PATH)
        except Exception as e:
            logger.warning(f'Classifier init warning: {e}')
            clf_model = ClassicalModel(model_type='logreg')

        logger.info('All modules ready')
        processing_error = None
        session_start = datetime.now()

        # Eye landmark indices (MediaPipe face mesh)
        LEFT_EYE  = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        RESP_LM   = Respiration.RESP_LANDMARK_INDICES

        # ===== MAIN LOOP =====
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning('Frame read failed')
                time.sleep(0.05)
                continue

            total_frames += 1

            # ── Face detection ──────────────────────────────────────────
            try:
                bboxes = face_detector.detect(frame)
                if not bboxes:
                    time.sleep(1.0 / FPS)
                    continue
                x, y, w, h = bboxes[0]
                if w <= 0 or h <= 0 or x < 0 or y < 0:
                    continue
                face_roi = frame[y:y + h, x:x + w]
                if face_roi.size == 0:
                    continue
            except Exception as e:
                logger.debug(f'Face detection: {e}')
                continue

            # ── Landmark detection ───────────────────────────────────────
            try:
                landmarks_list = landmarks_detector.get_landmarks(frame)
                if not landmarks_list:
                    continue
                lm = landmarks_list[0]
            except Exception as e:
                logger.debug(f'Landmark detection: {e}')
                continue

            # ── rPPG (CHROM) ─────────────────────────────────────────────
            hr = hrv_metrics = spo2 = signal_quality = rmssd = None
            try:
                # OpenCV = BGR → index 2=R, 1=G, 0=B
                rgb_means = (
                    float(np.mean(face_roi[:, :, 2])),  # R
                    float(np.mean(face_roi[:, :, 1])),  # G
                    float(np.mean(face_roi[:, :, 0])),  # B
                )
                rppg.update(rgb_means)
                hr_raw, hrv_metrics = rppg.compute_hr()
                signal_quality = round(rppg.signal_quality, 2)
                spo2 = rppg.estimate_spo2()

                # Gate on signal quality
                if hr_raw is not None and signal_quality >= SIGNAL_QUALITY_MIN:
                    hr = round(float(np.clip(hr_raw, 30, 220)), 1)
                else:
                    hr = None

                if hrv_metrics:
                    rmssd = hrv_metrics.get('rmssd')
            except Exception as e:
                logger.debug(f'rPPG: {e}')

            # ── Respiration ──────────────────────────────────────────────
            resp_rate = None
            try:
                if max(RESP_LM) < len(lm):
                    y_vals = [lm[i][1] for i in RESP_LM]
                    respiration.update(y_vals)
                    resp_rate = respiration.compute_resp_rate()
            except Exception as e:
                logger.debug(f'Respiration: {e}')

            # ── Blink / Fatigue / PERCLOS ────────────────────────────────
            blink_rate = fatigue_score = perclos = drowsiness = microsleeps = None
            try:
                all_eye_idx = LEFT_EYE + RIGHT_EYE
                if max(all_eye_idx) < len(lm):
                    left_ear  = BlinkFatigue.compute_ear([lm[i] for i in LEFT_EYE])
                    right_ear = BlinkFatigue.compute_ear([lm[i] for i in RIGHT_EYE])
                    blink_fatigue.update((left_ear, right_ear))
                    result = blink_fatigue.compute_blink_fatigue()
                    if result[0] is not None:
                        blink_rate, fatigue_score, perclos, drowsiness, microsleeps = result
            except Exception as e:
                logger.debug(f'Blink/fatigue: {e}')

            # ── Emotion ──────────────────────────────────────────────────
            emotion = emotion_probs = None
            try:
                emotion_probs = emotion_det.predict(face_roi, lm)
                if emotion_probs:
                    emotion = max(emotion_probs, key=emotion_probs.get)
            except Exception as e:
                logger.debug(f'Emotion: {e}')

            # ── Feature fusion & stress prediction ───────────────────────
            stress_level = stress_score_val = None
            try:
                feat_vec = [
                    hr       or 0.0,
                    rmssd    or 0.0,
                    resp_rate or 0.0,
                    blink_rate or 0.0,
                    fatigue_score or 0.0,
                ]
                fusion.update(feat_vec)
                fused = fusion.get_fused_features()

                if fused is not None and clf_model is not None:
                    proba = clf_model.predict(fused.reshape(1, -1))
                    if proba is not None and len(proba) > 0:
                        p = proba[0]
                        # p = [p_low, p_medium, p_high]
                        if len(p) >= 3:
                            stress_score_val = round(float(p[1] * 0.5 + p[2]), 3)
                        else:
                            stress_score_val = round(float(p[-1]), 3)

                        if stress_score_val < STRESS_THRESHOLDS['low']:
                            stress_level = 'Low'
                        elif stress_score_val < STRESS_THRESHOLDS['medium']:
                            stress_level = 'Medium'
                        else:
                            stress_level = 'High'
            except Exception as e:
                logger.debug(f'Prediction: {e}')

            # ── Alerts ───────────────────────────────────────────────────
            new_alerts = []
            _ts = datetime.now().isoformat()

            def _alert(msg, sev='warning'):
                new_alerts.append({'message': msg, 'severity': sev, 'timestamp': _ts})

            if stress_level == 'High':
                _alert('High stress detected', 'critical')
            if fatigue_score and fatigue_score > FATIGUE_THRESHOLD:
                _alert('High fatigue detected', 'warning')
            if drowsiness and drowsiness in DROWSINESS_ALERT_LEVELS:
                _alert(f'Drowsiness: {drowsiness}', 'critical')
            if hr and (hr < HR_NORMAL_RANGE[0] or hr > HR_NORMAL_RANGE[1]):
                _alert(f'Abnormal HR: {hr:.0f} BPM', 'warning')
            if resp_rate and (resp_rate < RESP_NORMAL_RANGE[0]
                              or resp_rate > RESP_NORMAL_RANGE[1]):
                _alert(f'Abnormal RR: {resp_rate:.0f} BPM', 'warning')
            if rmssd and rmssd < HRV_RMSSD_LOW:
                _alert('Low HRV – elevated autonomic stress', 'warning')
            if spo2 and spo2 < SPO2_LOW_THRESHOLD:
                _alert(f'Low SpO2: {spo2:.1f}%', 'critical')

            # ── Update shared metrics (thread-safe) ──────────────────────
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
                })
                processing_error = None

                for a in new_alerts:
                    alert_history.append(a)
                if len(alert_history) > MAX_ALERT_HISTORY:
                    del alert_history[:-MAX_ALERT_HISTORY]

            # ── Periodic DB save ─────────────────────────────────────────
            if total_frames % DB_SAVE_INTERVAL == 0:
                _save_metrics_db(metrics.copy())

            # ── Periodic log ─────────────────────────────────────────────
            if total_frames % 100 == 0:
                logger.info(
                    f'Frame {total_frames} | '
                    f'HR={metrics["hr"]} BPM | '
                    f'SpO2={metrics["spo2"]}% | '
                    f'Stress={metrics["stress"]} | '
                    f'Fatigue={metrics["fatigue"]} | '
                    f'SQ={signal_quality}'
                )

            time.sleep(1.0 / FPS)

    except Exception as e:
        logger.critical(f'Critical error in video processing: {e}', exc_info=True)
        processing_error = str(e)
    finally:
        if cap:
            cap.release()
        logger.info('Video processing thread stopped')


# ------------------------------------------------------------------
# Signal handlers
# ------------------------------------------------------------------

def _signal_handler(signum, frame):
    logger.info(f'Signal {signum} received – shutting down')
    stop_event.set()
    sys.exit(0)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == '__main__':
    logger.info('=' * 60)
    logger.info('Multi-Modal Physiological Monitoring System')
    logger.info('=' * 60)

    _init_db()

    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    t = threading.Thread(target=process_video, daemon=True, name='VideoProcessing')
    t.start()
    logger.info('Video processing thread started')

    try:
        logger.info(f'Flask API → http://{FLASK_HOST}:{FLASK_PORT}')
        app.run(host=FLASK_HOST, port=FLASK_PORT,
                debug=FLASK_DEBUG, use_reloader=False)
    except Exception as e:
        logger.error(f'Flask error: {e}', exc_info=True)
    finally:
        stop_event.set()
        logger.info('System stopped')
