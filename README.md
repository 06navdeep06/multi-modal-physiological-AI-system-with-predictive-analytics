# Multi-Modal Physiological Monitoring System with Predictive Analytics

This system estimates physiological signals (heart rate, HRV, respiration, blink/fatigue, emotion) from webcam video and predicts health states in real-time. It features modular Python backend components and a real-time dashboard frontend.

## Setup Instructions

1. **Install Python 3.9+**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the backend:**
   ```bash
   python main.py
   ```
4. **Open the dashboard:**
   Open `frontend/index.html` in your browser (or use a local server for live updates).

---

## How to Run Locally
- Ensure your webcam is connected.
- Run the backend as above.
- Open the dashboard to view real-time physiological metrics and predictions.

---

## Project Structure
- `main.py`: Entry point, orchestrates modules and data flow
- `config.py`: Configuration settings
- `vision/`: Face detection and landmark extraction
- `signal_processing/`: rPPG, respiration, filters
- `features/`: Feature fusion
- `models/`: ML models (classical, LSTM)
- `utils/`: Plotting, helpers
- `frontend/`: Dashboard (HTML, CSS, JS)

---

## Requirements
See `requirements.txt` for all dependencies.

---

## System Flow
1. Capture webcam frames
2. Detect face and landmarks
3. Extract physiological signals (rPPG, respiration, blink, emotion)
4. Fuse features and maintain sliding window
5. Predict health states (stress, fatigue, anomaly)
6. Send results to dashboard for real-time display

---

## Future Improvements
- Replace mock/simulated data with real datasets for model training
- Add user authentication and data logging
- Deploy as a web app or desktop app
- Integrate additional sensors (e.g., wearables)
- Optimize for mobile devices

---

For detailed module explanations, see code comments and module docstrings.
