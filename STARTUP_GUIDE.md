# 🚀 System Startup & Testing Guide

## Prerequisites

- Python 3.8+
- Webcam (USB or integrated)
- 4GB RAM minimum
- Modern browser (Chrome, Firefox, Safari, Edge)

---

## Installation

### 1. Clone/Download the project
```bash
cd "d:\multi-modal physiological AI system with predictive analytics"
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create necessary directories
```bash
mkdir -p models
mkdir -p logs
```

---

## Quick Start

### Step 1: Start the Backend Server
```bash
python main.py
```

**Expected output:**
```
2026-03-18 14:23:45 - __main__ - INFO - ============================================================
2026-03-18 14:23:45 - __main__ - INFO - Multi-Modal Physiological Monitoring System Started
2026-03-18 14:23:45 - __main__ - INFO - ============================================================
2026-03-18 14:23:45 - __main__ - INFO - Initializing video capture from camera 0...
2026-03-18 14:23:46 - __main__ - INFO - Video capture initialized
2026-03-18 14:23:46 - __main__ - INFO - Initializing physiological modules...
2026-03-18 14:23:46 - __main__ - INFO - All modules initialized successfully
2026-03-18 14:23:46 - __main__ - INFO - Video processing thread started
2026-03-18 14:23:46 - __main__ - INFO - Starting Flask API on 0.0.0.0:5000
```

### Step 2: Open the Frontend Dashboard
Open your browser and navigate to:
```
http://localhost:5000
```

Or if accessing from another machine:
```
http://<server-ip>:5000
```

---

## Testing the System

### API Endpoints Test

#### Get Metrics
```bash
curl http://localhost:5000/api/metrics
```

**Expected response:**
```json
{
  "data": {
    "hr": 72.5,
    "hrv": 45.3,
    "resp": 16.2,
    "blink": 18.5,
    "fatigue": 0.35,
    "emotion": "neutral",
    "stress": "Low",
    "alert": null,
    "timestamp": "2026-03-18T14:23:45.123Z"
  },
  "status": "ok",
  "error": null
}
```

#### Health Check
```bash
curl http://localhost:5000/api/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-03-18T14:23:45.123Z"
}
```

### Dashboard Testing

1. **Video Feed**
   - Check that webcam feed is visible
   - Face should be clearly visible for best results
   - Feed updates in real-time

2. **Metrics Display**
   - All values should update every second
   - "--" indicates metric not yet calculated (normal during first 10 seconds)
   - Values should be within expected ranges

3. **Charts**
   - Heart Rate chart shows BPM (40-200 range typical)
   - Respiration chart shows BPM (12-40 range typical)
   - Fatigue chart shows 0-1 scale

4. **Connection Status**
   - Status indicator shows "🟢 Connected" when backend is running
   - Changes to "🔴 Error" if backend is unreachable
   - Auto-reconnects every 3 seconds

5. **Alerts**
   - Warnings appear for high stress, fatigue, or abnormal vitals
   - Auto-dismiss after 5 seconds
   - Multiple alerts can appear simultaneously

---

## Troubleshooting

### Issue: "Webcam access denied"
**Solution:**
1. Grant camera permissions to your browser
2. Check system privacy settings
3. Try in incognito/private mode
4. Use a different browser

### Issue: "Backend not reachable"
**Solution:**
1. Ensure main.py is running
2. Check that port 5000 is available: 
   ```bash
   netstat -ano | findstr :5000
   ```
3. Try accessing http://localhost:5000/api/health in browser
4. Check firewall settings

### Issue: Metrics showing "--"
**Solution:**
- Wait 10-15 seconds for metrics to stabilize
- Ensure face is clearly visible and well-lit
- Keep face steady and within frame
- Check that webcam resolution is acceptable

### Issue: Very high/low heart rate
**Solution:**
- Improve lighting conditions (avoid backlight)
- Keep face still and centered
- Ensure green channel is properly captured
- Try different camera angles

### Issue: High CPU usage
**Solution:**
- Reduce FPS if possible (in config.py)
- Close other applications
- Check system thermal conditions
- Restart the system

### Issue: Backend crashes
**Solution:**
1. Check logs: `tail -f physiological_system.log`
2. Look for module initialization errors
3. Ensure all dependencies are installed
4. Check for out-of-memory errors

---

## Performance Monitoring

### View System Logs
```bash
# Real-time log viewing (Windows)
Get-Content -Path physiological_system.log -Wait

# Real-time log viewing (Linux/Mac)
tail -f physiological_system.log

# Search logs for errors
findstr "ERROR\|CRITICAL" physiological_system.log
```

### Monitor Metrics
```bash
# Poll metrics every 2 seconds
:loop
curl http://localhost:5000/api/metrics
timeout /t 2
goto loop
```

### Check Memory Usage
```bash
# Windows Task Manager
tasklist /FI "IMAGENAME eq python.exe"

# Windows PowerShell
Get-Process python | Select-Object Name, @{Name="Memory (MB)";Expression={[math]::Round($_.WorkingSet / 1MB)}}
```

---

## Configuration Tips

### For Better Accuracy

1. **Lighting**
   - Use bright, even lighting
   - Avoid shadows on face
   - Use diffuse lighting if possible

2. **Camera Positioning**
   - Position at eye level
   - Distance: ~30-60cm from face
   - Keep entire face visible
   - Ensure green channel capture (critical for rPPG)

3. **Environment**
   - Quiet background for audio (if added later)
   - Minimize motion of head/body
   - Avoid rapid orientation changes

4. **Settings** (in config.py)
   ```python
   WEBCAM_INDEX = 0          # 0 for default, 1+ for USB cameras
   FRAME_WIDTH = 640          # Increase for HD
   FRAME_HEIGHT = 480
   FPS = 20                   # Increase for smoother but requires more CPU
   FUSION_WINDOW_SEC = 60     # Increase for more stable stress predictions
   ```

---

## Advanced Usage

### Custom Model Loading

If you have trained models, place them in the `models/` directory and update config.py:

```python
CLASSICAL_MODEL_PATH = 'models/my_custom_model.pth'
LSTM_MODEL_PATH = 'models/my_lstm_model.pth'
EMOTION_MODEL_PATH = 'models/my_emotion_model.pth'
```

### Environment Variables

```bash
# Set custom webcam
set WEBCAM_INDEX=1
python main.py

# Or use in code
import os
webcam_idx = int(os.getenv('WEBCAM_INDEX', 0))
```

### Integration with Other Systems

The REST API can be easily integrated:

```python
import requests

# Get metrics
response = requests.get('http://localhost:5000/api/metrics')
metrics = response.json()
hr = metrics['data']['hr']
stress = metrics['data']['stress']

# Process metrics...
```

---

## Database Logging (Optional Future Enhancement)

To add historical data logging:

```python
# In main.py, add database logging
import sqlite3

db = sqlite3.connect('physiological_data.db')
cursor = db.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS metrics
    (timestamp TEXT, hr REAL, resp REAL, fatigue REAL, stress TEXT)''')

# Log each metric update
cursor.execute('INSERT INTO metrics VALUES (?, ?, ?, ?, ?)',
              (metrics['timestamp'], metrics['hr'], metrics['resp'], 
               metrics['fatigue'], metrics['stress']))
db.commit()
```

---

## Remote Access

### Access from Another Machine

1. Change Flask host in config.py:
   ```python
   FLASK_HOST = '0.0.0.0'  # Listen on all interfaces
   FLASK_PORT = 5000
   ```

2. Find your computer's IP:
   ```bash
   # Windows
   ipconfig | findstr "IPv4"
   
   # Linux/Mac
   ifconfig | grep "inet "
   ```

3. Access from remote machine:
   ```
   http://<your-ip>:5000
   ```

---

## Deployment Checklist

- [ ] All dependencies installed
- [ ] Camera connected and working
- [ ] Backend starts without errors
- [ ] Frontend dashboard loads
- [ ] Metrics update every second
- [ ] Alerts display correctly
- [ ] Logs are being written
- [ ] No crashes after 5+ minutes run
- [ ] CPU usage is reasonable (<70%)
- [ ] Memory usage is stable

---

## Support & Debugging

**For detailed logs**, check `physiological_system.log`:
```
2026-03-18 14:23:45,123 - vision.face_detection - DEBUG - Face detected at (100, 50, 200, 300)
2026-03-18 14:23:46,456 - signal_processing.rppg - INFO - HR computed: 72.5 BPM
```

**Common log messages:**
- `Face detection error` - Face not found in current frame
- `Landmark detection error` - Face landmarks not extracted
- `Error computing HR` - rPPG calculation failed
- `Backend unreachable` - Frontend cannot reach API

---

## Next Steps

1. ✅ Verify system is working
2. 📊 Collect baseline metrics
3. 🎯 Calibrate stress thresholds for your needs
4. 🤖 Consider training custom models
5. 📖 Integrate with other health platforms

---

**System Version**: 2.0
**Last Updated**: March 2026
