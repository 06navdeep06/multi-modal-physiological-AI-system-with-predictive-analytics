"""Feature Fusion - Multi-metric Statistical Feature Extraction

Maintains a sliding window over 5 physiological channels and produces a 30-dim
feature vector (6 statistics × 5 channels) suitable for both classical ML models
and the LSTM.

Per-channel statistics extracted:
  norm_mean  – mean normalised to known physiological range [0, 1]
  cv         – coefficient of variation (std / |mean| + ε)
  norm_min   – min, z-scored within window
  norm_max   – max, z-scored within window
  norm_med   – median, z-scored within window
  slope      – linear trend slope, scale-invariant (polyfit / std)

FIX: previous version computed (mean - mean) / std which was always 0, making
the mean feature useless. Now uses physiological-range normalisation.
"""

from __future__ import annotations

import numpy as np
import logging

logger = logging.getLogger(__name__)

# Per-channel physiological bounds used to normalise the mean (0 → low, 1 → high)
_CHANNEL_NORMS: list[tuple[float, float]] = [
    (40.0, 180.0),   # HR: 40–180 BPM
    (0.0,  100.0),   # HRV RMSSD: 0–100 ms
    (5.0,   40.0),   # Respiration: 5–40 BPM
    (0.0,   30.0),   # Blink rate: 0–30 /min
    (0.0,    1.0),   # Fatigue: 0–1
]

FEATURE_CHANNELS = ['hr', 'hrv_rmssd', 'resp', 'blink', 'fatigue']
FEATURE_STATS    = ['norm_mean', 'cv', 'norm_min', 'norm_max', 'norm_med', 'slope']
FEATURE_NAMES    = [f'{ch}_{st}' for ch in FEATURE_CHANNELS for st in FEATURE_STATS]
FEATURE_DIM      = len(FEATURE_NAMES)   # 30


class FeatureFusion:
    """Sliding-window statistical feature extractor for physiological signals.

    Args:
        window_sec: Rolling window length in seconds
        fs:         Sampling frequency in Hz
    """

    def __init__(self, window_sec: int = 60, fs: int = 20):
        self.fs = fs
        self.window_size = window_sec * fs
        self._min_samples = max(10, self.window_size // 10)   # warmup requirement
        self.features: list[list[float]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, feature_vec: list[float]) -> None:
        """Append one frame's feature vector [hr, rmssd, resp, blink, fatigue]."""
        self.features.append([float(v) for v in feature_vec])
        if len(self.features) > self.window_size:
            self.features.pop(0)

    def get_fused_features(self) -> np.ndarray | None:
        """Return a (30,) float32 feature vector, or None if buffer is not ready.

        Feature layout (30 values = 5 channels × 6 stats):
          [hr_norm_mean, hr_cv, hr_norm_min, hr_norm_max, hr_norm_med, hr_slope,
           rmssd_norm_mean, ..., fatigue_slope]
        All values clipped to [-5, 5] to prevent exploding gradients.
        """
        if len(self.features) < self._min_samples:
            return None

        try:
            arr = np.array(self.features, dtype=np.float64)   # (T, 5)
            T = arr.shape[0]
            t = np.linspace(0, 1, T)
            stats: list[float] = []

            for ch in range(arr.shape[1]):
                col = arr[:, ch]
                mean_v = float(np.mean(col))
                std_v  = float(np.std(col)) + 1e-9
                min_v  = float(np.min(col))
                max_v  = float(np.max(col))
                med_v  = float(np.median(col))

                # 1. Physiologically-normalised mean (0 = low end, 1 = high end)
                lo, hi = _CHANNEL_NORMS[ch]
                norm_mean = (mean_v - lo) / (hi - lo + 1e-9)

                # 2. Coefficient of variation
                cv = std_v / (abs(mean_v) + 1e-6)

                # 3–5. Z-score of min / max / median relative to window distribution
                norm_min = (min_v - mean_v) / std_v
                norm_max = (max_v - mean_v) / std_v
                norm_med = (med_v - mean_v) / std_v

                # 6. Scale-invariant linear slope
                try:
                    slope = float(np.polyfit(t, col, 1)[0]) / std_v
                except Exception:
                    slope = 0.0

                stats.extend([norm_mean, cv, norm_min, norm_max, norm_med, slope])

            out = np.array(stats, dtype=np.float32)
            return np.clip(out, -5.0, 5.0)

        except Exception as e:
            logger.error(f'Feature fusion error: {e}')
            return None

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM

    @property
    def feature_names(self) -> list[str]:
        return FEATURE_NAMES

    @property
    def is_ready(self) -> bool:
        return len(self.features) >= self._min_samples
