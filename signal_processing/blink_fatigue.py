"""Blink & Fatigue Detection - Eye Aspect Ratio + PERCLOS Analysis

Tracks blinks, fatigue and drowsiness using:
- EAR (Eye Aspect Ratio) averaged over both eyes
- PERCLOS (Percentage of Eye Closure) over a 60-second window
- Microsleep detection (eyes closed >= 0.5 s continuously)
- Four-level drowsiness classification: Alert / Mild / Moderate / Severe

Reference: Soukupová & Čech, "Real-Time Eye Blink Detection using Facial Landmarks" (2016)
           Wierwille & Ellsworth, PERCLOS drowsiness metric (1994)
"""

import numpy as np
import logging
from collections import deque

logger = logging.getLogger(__name__)

# PERCLOS → drowsiness level mapping (fraction of time eyes closed)
_PERCLOS_LEVELS = [
    (0.00, 0.15, 'Alert'),
    (0.15, 0.35, 'Mild'),
    (0.35, 0.55, 'Moderate'),
    (0.55, 1.00, 'Severe'),
]


class BlinkFatigue:
    """Blink, fatigue and drowsiness detector using Eye Aspect Ratio.

    Attributes:
        EAR_CLOSED_THRESHOLD:  EAR below which an eye is considered closed
        EAR_BLINK_THRESHOLD:   EAR threshold used for blink-onset detection
        MICROSLEEP_FRAMES_MIN: Minimum consecutive closed frames = microsleep
    """

    EAR_CLOSED_THRESHOLD:  float = 0.20
    EAR_BLINK_THRESHOLD:   float = 0.21
    MICROSLEEP_FRAMES_MIN: int   = 10   # 0.5 s at 20 FPS

    def __init__(self, fs: int = 20, window_sec: int = 10,
                 perclos_window_sec: int = 60):
        self.fs = fs
        self.window_size = fs * window_sec
        self.perclos_window: deque[float] = deque(maxlen=fs * perclos_window_sec)
        self.ear_window: list[float] = []

        # Microsleep tracking
        self._consecutive_closed: int = 0
        self.microsleep_count: int = 0

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_ear(eye: list) -> float:
        """Compute Eye Aspect Ratio from 6 eye-corner landmarks.

        EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
        Landmarks:
            0 = left corner, 1 = top-left, 2 = top-right,
            3 = right corner, 4 = bottom-right, 5 = bottom-left

        Returns:
            float: EAR value (typical open ~ 0.25–0.35, closed < 0.20)
        """
        try:
            A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
            B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
            C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
            return float((A + B) / (2.0 * C + 1e-6))
        except Exception as e:
            logger.debug(f'EAR computation error: {e}')
            return 0.30  # neutral open-eye default

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, ear):
        """Push a new EAR sample.

        Args:
            ear: float, or (left_ear, right_ear) tuple – averaged if tuple
        """
        if isinstance(ear, (tuple, list)) and len(ear) == 2:
            ear = (ear[0] + ear[1]) / 2.0
        ear = float(ear)

        self.ear_window.append(ear)
        if len(self.ear_window) > self.window_size:
            self.ear_window.pop(0)

        self.perclos_window.append(ear)

        # Microsleep: count runs of closed frames >= MICROSLEEP_FRAMES_MIN
        if ear <= self.EAR_CLOSED_THRESHOLD:
            self._consecutive_closed += 1
        else:
            if self._consecutive_closed >= self.MICROSLEEP_FRAMES_MIN:
                self.microsleep_count += 1
            self._consecutive_closed = 0

    def compute_blink_fatigue(self):
        """Compute blink rate, fatigue, PERCLOS and drowsiness.

        Returns:
            tuple: (blink_rate, fatigue_score, perclos, drowsiness, microsleeps)
                   All None if the window is not yet full.
            - blink_rate:   blinks / minute
            - fatigue_score: 0–1  (higher = more fatigued)
            - perclos:      % of time eyes closed (0–100)
            - drowsiness:   'Alert' | 'Mild' | 'Moderate' | 'Severe'
            - microsleeps:  cumulative microsleep events this session
        """
        if len(self.ear_window) < self.window_size:
            return None, None, None, None, None

        try:
            arr = np.array(self.ear_window)

            # Blink rate: count EAR open→closed transitions per minute
            transitions = np.where(
                (arr[:-1] > self.EAR_BLINK_THRESHOLD) &
                (arr[1:] <= self.EAR_BLINK_THRESHOLD)
            )[0]
            window_min = self.window_size / self.fs / 60.0
            blink_rate = float(len(transitions) / window_min)

            # Fatigue: deviation of mean EAR from a typical open value (0.30)
            mean_ear = float(np.mean(arr))
            fatigue_score = float(np.clip(1.0 - mean_ear / 0.30, 0.0, 1.0))

            # PERCLOS over the longer window
            p_arr = np.array(self.perclos_window)
            perclos_frac = float(np.sum(p_arr <= self.EAR_CLOSED_THRESHOLD)
                                 / len(p_arr))
            perclos_pct = perclos_frac * 100.0

            # Drowsiness classification
            drowsiness = 'Alert'
            for low, high, label in _PERCLOS_LEVELS:
                if low <= perclos_frac < high:
                    drowsiness = label
                    break

            return (
                round(blink_rate, 1),
                round(fatigue_score, 3),
                round(perclos_pct, 1),
                drowsiness,
                self.microsleep_count,
            )

        except Exception as e:
            logger.error(f'Blink/fatigue computation error: {e}')
            return None, None, None, None, None
