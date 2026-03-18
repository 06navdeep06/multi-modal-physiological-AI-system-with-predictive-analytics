"""Respiration Rate Detection - Breathing Pattern Analysis

Estimates breathing rate from vertical displacement of multiple facial landmarks
(nose bridge + tip + chin) via Welch's power spectral density method.

Improvements over v1:
  - Uses nose + chin landmarks for larger, more robust excursion signal
  - Welch PSD instead of plain FFT for smoother, more stable frequency peak
  - Detrending + Z-score normalisation before spectral analysis
  - Valid range extended to 6–36 BPM (0.1–0.6 Hz) to capture faster breathing
"""

from __future__ import annotations

import numpy as np
import logging
from scipy.signal import welch

from .filters import SignalFilters

logger = logging.getLogger(__name__)


class Respiration:
    """Respiration rate detector based on facial landmark vertical motion.

    Attributes:
        fs:          Sampling frequency in Hz
        window_size: Number of samples per estimation window
        window:      Circular buffer of mean landmark y-coordinates
    """

    # MediaPipe face mesh landmark indices used for vertical motion tracking
    # Covers: nose bridge (6, 197), nose tip (1, 2, 3, 4), chin (152, 175, 200)
    RESP_LANDMARK_INDICES = [1, 2, 4, 6, 152, 175, 197, 200]

    def __init__(self, fs: int = 20, window_sec: int = 15):
        self.fs = fs
        self.window_size = fs * window_sec
        self.window: list[float] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, y_landmarks):
        """Push new landmark vertical coordinates.

        Args:
            y_landmarks: sequence of y-pixel values from nose/chin landmarks
        """
        try:
            self.window.append(float(np.mean(y_landmarks)))
            if len(self.window) > self.window_size:
                self.window.pop(0)
        except Exception as e:
            logger.debug(f'Respiration update error: {e}')

    def compute_resp_rate(self):
        """Estimate breathing rate using Welch PSD.

        Returns:
            float|None: Respiration rate in breaths per minute, or None
        """
        if len(self.window) < self.window_size:
            return None

        try:
            sig = np.array(self.window, dtype=np.float64)
            sig = SignalFilters.detrend_signal(sig)
            sig = SignalFilters.normalize_signal(sig)

            # Welch PSD: nperseg chosen for ~0.05 Hz frequency resolution
            nperseg = min(len(sig) // 2, self.fs * 8)
            nperseg = max(nperseg, self.fs * 2)   # at least 2 s segment
            freqs, psd = welch(sig, fs=self.fs, nperseg=nperseg,
                               noverlap=nperseg // 2, window='hann')

            # Typical respiratory range: 6–36 BPM → 0.10–0.60 Hz
            valid = (freqs >= 0.10) & (freqs <= 0.60)
            if not np.any(valid):
                return None

            peak_idx = np.argmax(psd[valid])
            resp_hz = freqs[valid][peak_idx]
            resp_bpm = resp_hz * 60.0

            return round(float(resp_bpm), 1)

        except Exception as e:
            logger.error(f'Respiration rate computation error: {e}')
            return None
