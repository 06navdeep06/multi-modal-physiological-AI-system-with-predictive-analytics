# 🔧 Developer Quick Reference

## System Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│         Multi-Modal Physiological System             │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────┐      ┌─────────────┐      ┌────────┐ │
│  │ Webcam   │─────→│   Vision    │─────→│ Signals│ │
│  │ Input    │      │  Modules    │      │Process │ │
│  └──────────┘      └─────────────┘      └────────┘ │
│       ↓                  ↓                    ↓     │
│    Video              Face Detect        rPPG,Resp │
│    Feed               Landmarks          Blink,Emo │
│                                                     │
│                ┌──────────────────┐               │
│                │  Feature         │               │
│                │  Fusion &        │               │
│                │  Prediction      │               │
│                └──────────────────┘               │
│                        ↓                           │
│                    Metrics                         │
│                        ↓                           │
│    ┌───────────────────┼───────────────────┐     │
│    ↓                   ↓                   ↓     │
│  Flask API        Thread-Safe         Logging    │
│  /api/metrics      Metrics Dict       File & Log │
│  /api/health                                     │
│    ↓                                              │
│  └──────────────────────────────────────────────→ │
│         Frontend Dashboard (Browser)             │
│    - Real-time metrics                           │
│    - Live charts                                 │
│    - Alerts & status                            │
│    - Responsive design                          │
└─────────────────────────────────────────────────────┘
```

---

## Key File Structure

```
project/
├── main.py                    # Entry point (300+ lines)
├── config.py                  # Configuration (40 lines)
│
├── vision/
│   ├── face_detection.py      # Face detection (uses MediaPipe)
│   └── landmarks.py           # Facial landmarks extraction
│
├── signal_processing/
│   ├── rppg.py               # Heart rate & HRV (85 lines)
│   ├── respiration.py        # Breathing rate (75 lines)
│   ├── blink_fatigue.py      # Blink & fatigue (90 lines)
│   ├── emotion.py            # Emotion detection (35 lines)
│   └── filters.py            # Signal filters (65 lines)
│
├── models/
│   ├── classical_model.py     # Stress prediction
│   └── lstm_model.py          # LSTM models (optional)
│
├── features/
│   └── fusion.py              # Feature fusion & buffering
│
├── utils/
│   ├── helpers.py             # Signal utilities (150 lines)
│   └── plotting.py            # Visualization (180 lines)
│
├── frontend/
│   ├── index.html             # Dashboard (120 lines)
│   ├── app.js                 # Frontend logic (390 lines)
│   └── styles.css             # Styling (400 lines)
│
├── requirements.txt           # Dependencies
├── IMPROVEMENTS.md            # Detailed improvements
├── STARTUP_GUIDE.md          # Setup & troubleshooting
└── CHANGES_SUMMARY.md        # This document
```

---

## API Reference

### Endpoints

#### GET /api/metrics
Returns current physiological metrics

**Response:**
```json
{
  "data": {
    "hr": 72.5,          // Heart rate (BPM)
    "hrv": 45.3,         // Heart rate variability (ms)
    "resp": 16.2,        // Respiration rate (BPM)
    "blink": 18.5,       // Blink rate (/min)
    "fatigue": 0.35,     // Fatigue score (0-1)
    "emotion": "neutral", // Detected emotion
    "stress": "Low",     // Stress level
    "alert": null,       // Alert message if any
    "timestamp": "ISO-8601 timestamp"
  },
  "status": "ok",        // "ok" or "error"
  "error": null          // Error message if status is error
}
```

#### GET /api/health
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "ISO-8601 timestamp"
}
```

---

## Configuration

### Main Config (config.py)
```python
# Video input
WEBCAM_INDEX = 0              # Camera index (0 for default)
FRAME_WIDTH = 640             # Resolution
FRAME_HEIGHT = 480
FPS = 20                       # Frames per second

# Signal processing windows
RPPG_WINDOW_SEC = 10
RESPIRATION_WINDOW_SEC = 10
BLINK_WINDOW_SEC = 10
FUSION_WINDOW_SEC = 60        # Feature fusion window

# Thresholds
STRESS_THRESHOLDS = {
    'low': 0.3,               # Stress classification
    'medium': 0.6,
    'high': 1.0
}
FATIGUE_THRESHOLD = 0.7       # Fatigue alert threshold

# Normal ranges
HR_NORMAL_RANGE = (60, 100)   # BPM
RESP_NORMAL_RANGE = (12, 20)  # BPM

# Model paths
CLASSICAL_MODEL_PATH = 'models/classical_model.pth'
LSTM_MODEL_PATH = 'models/lstm_model.pth'

# Flask
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
```

---

## Signal Processing Pipeline

### rPPG (Remote Photoplethysmography)
```
Raw green channel → Detrend → Bandpass filter (0.7-4.0 Hz)
    ↓
FFT → Find peak frequency → Heart rate (Hz × 60 = BPM)
    ↓
Peak detection → RR intervals → HRV (std of intervals)
```

### Respiration
```
Nose landmark Y-coords → Moving average → Normalize
    ↓
FFT → Find peak (0.2-0.5 Hz) → Respiration rate (Hz × 60)
```

### Blink/Fatigue
```
Eye landmarks → Eye Aspect Ratio (EAR)
    ↓
EAR < 0.21 threshold → Blink detection
    ↓
Blink count + mean EAR → Fatigue score
```

### Emotion
```
Face crop → Model inference (placeholder in v2.0)
    ↓
Probability distribution → Max class label
```

### Feature Fusion
```
[HR, HRV, RR, Blink, Fatigue] → Moving average window
    ↓
Buffered features → Classical model
    ↓
Stress probability → Classification (Low/Medium/High)
```

---

## Common Tasks

### Add New Signal Processor
```python
# 1. Create new module: signal_processing/mynew.py
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MyNewProcessor:
    def __init__(self, fs=20):
        self.fs = fs
        self.window = []
    
    def update(self, data):
        self.window.append(data)
    
    def compute_feature(self):
        if len(self.window) < 10:  # Need buffer
            return None
        # Process and return feature
        return feature_value

# 2. Import in main.py
from signal_processing.mynew import MyNewProcessor

# 3. Initialize
processor = MyNewProcessor(fs=FPS)

# 4. Update in loop
processor.update(new_data)
feature = processor.compute_feature()

# 5. Add to metrics
metrics['mynew'] = feature
```

### Modify Alert Logic
```python
# In main.py, update_metrics section
alerts = []

# Add custom alert
if some_condition:
    alerts.append('Your alert message')

# Update metrics
metrics['alert'] = ' | '.join(alerts) if alerts else None
```

### Change Thresholds
```python
# config.py
STRESS_THRESHOLDS = {
    'low': 0.25,        # Lower threshold for more sensitivity
    'medium': 0.5,
}

FATIGUE_THRESHOLD = 0.6   # More sensitive to fatigue
```

### Add New API Endpoint
```python
# In main.py
@app.route('/api/custom', methods=['GET'])
def custom_endpoint():
    """Custom endpoint description"""
    try:
        return jsonify({'custom_data': value})
    except Exception as e:
        logger.error(f'Error in custom endpoint: {e}')
        return jsonify({'error': str(e)}), 500
```

---

## Logging

### View Logs
```bash
# Windows - real-time
Get-Content -Path physiological_system.log -Wait

# Linux/Mac - real-time
tail -f physiological_system.log

# Search for errors
findstr "ERROR" physiological_system.log
grep "ERROR" physiological_system.log
```

### Log Levels
```python
logger.debug('Detailed info')      # Only in debug mode
logger.info('Important info')      # General operation
logger.warning('Watch out')        # Something unexpected
logger.error('Major problem')      # Error occurred
logger.critical('Fatal issue')     # System stopping
```

### Add Logging
```python
import logging
logger = logging.getLogger(__name__)

logger.debug(f'Processing frame {count}')
logger.info(f'HR computed: {hr} BPM')
logger.warning(f'Face detection failed: {error}')
logger.error(f'Signal processing error: {e}', exc_info=True)
```

---

## Error Handling Pattern

### In Signal Processing Modules
```python
try:
    # Your processing logic
    result = process_signal(data)
    return result

except ValueError as e:
    logger.error(f'Value error in process: {e}')
    return None  # Return safe default

except IndexError as e:
    logger.error(f'Index error: {e}')
    return None

except Exception as e:
    logger.error(f'Unexpected error: {e}', exc_info=True)
    return None  # Fail gracefully
```

### In Main Loop
```python
try:
    # Normal operation
    ret, frame = cap.read()
    
    if not ret:
        logger.warning('Failed to read frame')
        continue

except Exception as e:
    logger.error(f'Frame processing error: {e}')
    processing_error = str(e)
    continue  # Skip this frame, try next
```

---

## Performance Tips

### Optimize CPU Usage
1. Reduce FPS if possible
2. Lower resolution (FRAME_WIDTH, FRAME_HEIGHT)
3. Reduce window sizes (FUSION_WINDOW_SEC)
4. Close background applications
5. Monitor in Windows Task Manager

### Optimize Memory
1. Circular buffers automatically limit size
2. Monitor with: `Get-Process python | Measure-Object -Property WorkingSet`
3. Restart system if memory grows unbounded
4. Check for memory leaks in logging

### Optimize Network (for remote access)
1. Use /api/metrics polling instead of WebSocket initially
2. Increase POLL_INTERVAL if frontend lags
3. Compress metrics if needed
4. Consider nginx reverse proxy for load balancing

---

## Debugging Tips

### Print Frame Count
```python
frame_count += 1
if frame_count % 100 == 0:
    logger.info(f'Processed {frame_count} frames')
```

### Test Module Independently
```python
# In Python console
from signal_processing.rppg import RPPG
rppg = RPPG(fs=20)

# Simulate data
for i in range(200):
    rppg.update(100 + 5*np.sin(2*np.pi*i/20))

hr, hrv = rppg.compute_hr()
print(f'HR: {hr}, HRV: {hrv}')
```

### Monitor Metrics in Real-Time
```bash
# Polling loop (Windows)
:loop
powershell -c "curl -s http://localhost:5000/api/metrics | ConvertFrom-Json | Select -ExpandProperty data | Format-Table"
timeout /t 1
goto loop
```

---

## Testing Checklist

- [ ] Backend starts without errors
- [ ] Webcam feed appears
- [ ] Metrics update every second
- [ ] Metrics are reasonable values
- [ ] Alerts appear when expected
- [ ] No crashes after 5+ minutes
- [ ] CPU usage < 70%
- [ ] Memory usage stable
- [ ] Logs show no errors
- [ ] Frontend updates in real-time

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | Mar 2026 | Major improvements, bug fixes, better docs |
| 1.0 | --- | Initial version |

---

## Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start backend
python main.py

# Test API
curl http://localhost:5000/api/metrics
curl http://localhost:5000/api/health

# View logs (Windows)
Get-Content -Path physiological_system.log -Wait

# View logs (Unix)
tail -f physiological_system.log

# Check Python processes
tasklist | findstr python

# Kill server (Ctrl+C works better)
taskkill /PID <pid> /F
```

---

## Resources

- **OpenCV Docs**: https://docs.opencv.org/
- **MediaPipe**: https://mediapipe.dev/
- **Flask**: https://flask.palletsprojects.com/
- **Chart.js**: https://www.chartjs.org/
- **NumPy**: https://numpy.org/doc/

---

**Document Version**: 2.0
**Last Updated**: March 2026
