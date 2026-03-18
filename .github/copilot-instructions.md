# Multi-Modal Physiological Monitoring System with Predictive Analytics

This project implements a real-time system for estimating human physiological signals and predicting health-related states using a webcam. It features modular Python backend components for rPPG, respiration, blink/fatigue, emotion detection, feature fusion, predictive analytics, and a real-time dashboard frontend.

## Project Structure

```
project/
│── main.py
│── config.py
│
├── vision/
│   ├── face_detection.py
│   ├── landmarks.py
│
├── signal_processing/
│   ├── rppg.py
│   ├── respiration.py
│   ├── filters.py
│
├── features/
│   ├── fusion.py
│
├── models/
│   ├── classical_model.py
│   ├── lstm_model.py
│
├── utils/
│   ├── plotting.py
│   ├── helpers.py
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   ├── app.js
```

## Core Technologies
- Python, OpenCV, MediaPipe, NumPy, SciPy, PyTorch
- HTML, CSS, JavaScript, Chart.js

## Modules
- rPPG (Heart Rate, HRV)
- Respiration (Breathing Rate)
- Blink & Fatigue Detection
- Emotion Detection
- Feature Fusion
- Predictive AI (Logistic Regression, Random Forest, LSTM)
- Real-Time Dashboard

---

This file will be updated as each checklist item is completed.
