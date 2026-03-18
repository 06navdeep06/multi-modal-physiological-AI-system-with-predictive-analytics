"""Configuration for Multi-Modal Physiological Monitoring System"""

import os
import logging

# ===== VIDEO CAPTURE =====
WEBCAM_INDEX = int(os.getenv('WEBCAM_INDEX', 0))
FRAME_WIDTH = int(os.getenv('FRAME_WIDTH', 640))
FRAME_HEIGHT = int(os.getenv('FRAME_HEIGHT', 480))
FPS = int(os.getenv('FPS', 20))
VIDEO_TIMEOUT_SEC = int(os.getenv('VIDEO_TIMEOUT_SEC', 3))

# ===== SIGNAL PROCESSING =====
FUSION_WINDOW_SEC = int(os.getenv('FUSION_WINDOW_SEC', 60))
RPPG_WINDOW_SEC = int(os.getenv('RPPG_WINDOW_SEC', 15))       # Longer window for CHROM stability
RESPIRATION_WINDOW_SEC = int(os.getenv('RESPIRATION_WINDOW_SEC', 15))
BLINK_WINDOW_SEC = int(os.getenv('BLINK_WINDOW_SEC', 10))
PERCLOS_WINDOW_SEC = int(os.getenv('PERCLOS_WINDOW_SEC', 60)) # Full minute for PERCLOS

# ===== MODEL PATHS =====
BASE_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSICAL_MODEL_PATH = os.path.join(BASE_MODEL_DIR, 'models', 'classical_model.pth')
LSTM_MODEL_PATH = os.path.join(BASE_MODEL_DIR, 'models', 'lstm_model.pth')
EMOTION_MODEL_PATH = os.path.join(BASE_MODEL_DIR, 'models', 'emotion_model.pth')

# ===== DATABASE =====
DB_PATH = os.getenv('DB_PATH', os.path.join(BASE_MODEL_DIR, 'data', 'metrics.db'))
DB_MAX_ROWS = int(os.getenv('DB_MAX_ROWS', 100_000))   # Rows kept before auto-cleanup
DB_SAVE_INTERVAL = int(os.getenv('DB_SAVE_INTERVAL', 20))  # Save to DB every N frames
HISTORY_LIMIT = int(os.getenv('HISTORY_LIMIT', 100))   # Default points for /api/history

# ===== ALERT THRESHOLDS =====
STRESS_THRESHOLDS = {'low': 0.3, 'medium': 0.6, 'high': 1.0}
FATIGUE_THRESHOLD = 0.7
DROWSINESS_ALERT_LEVELS = ('Moderate', 'Severe')
HR_NORMAL_RANGE = (50, 120)         # BPM
RESP_NORMAL_RANGE = (10, 25)        # BPM
HRV_RMSSD_LOW = 20.0                # ms – below this flags elevated stress
SPO2_LOW_THRESHOLD = 95.0           # % – below this triggers alert
SIGNAL_QUALITY_MIN = 0.15           # Min acceptable SNR for HR results
MAX_ALERT_HISTORY = 200             # Max alerts kept in memory

# ===== FACE DETECTION =====
FACE_DETECTION_CONFIDENCE = float(os.getenv('FACE_DETECTION_CONFIDENCE', 0.7))
FACE_MESH_CONFIDENCE = float(os.getenv('FACE_MESH_CONFIDENCE', 0.5))
FACE_MESH_TRACKING_CONFIDENCE = float(os.getenv('FACE_MESH_TRACKING_CONFIDENCE', 0.5))

# ===== LOGGING =====
LOG_LEVEL_STR = os.getenv('LOG_LEVEL', 'INFO')
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR.upper(), logging.INFO)
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.getenv('LOG_FILE', 'physiological_system.log')
LOG_MAX_BYTES = int(os.getenv('LOG_MAX_BYTES', 10 * 1024 * 1024))  # 10 MB per file
LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', 5))            # 5 rotated files

# ===== FLASK API =====
FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
MAX_RETRIES = 3
RETRY_DELAY_SEC = 2
