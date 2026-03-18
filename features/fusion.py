"""Feature Fusion - Multi-metric Statistical Feature Extraction

Maintains a sliding window over the 5 physiological channels and exposes
a richer feature vector containing per-channel statistics that give
classical ML models (and the LSTM) much more signal than raw values alone.

Features extracted per channel (5 channels × 6 stats = 30-dim vector):
  mean, std, min, max, median, linear trend (slope via polyfit)
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# Feature names for documentation / model introspection
FEATURE_CHANNELS = ['hr', 'hrv_rmssd', 'resp', 'blink', 'fatigue']
FEATURE_STATS = ['mean', 'std', 'min', 'max', 'median', 'slope']
FEATURE_NAMES = [f'{ch}_{st}' for ch in FEATURE_CHANNELS for st in FEATURE_STATS]
FEATURE_DIM = len(FEATURE_NAMES)   # 30


class FeatureFusion:
    """Sliding-window statistical feature extractor for physiological signals.

    Attributes:
        window_size: Samples retained per channel
        features:    Circular buffer of per-frame feature vectors
    """

    def __init__(self, window_sec: int = 60, fs: int = 20):
        self.fs = fs
        self.window_size = window_sec * fs
        self.features: list[list[float]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, feature_vec: list):
        """Add one frame's feature vector to the buffer.

        Args:
            feature_vec: list of 5 floats [hr, hrv_rmssd, resp, blink, fatigue]
                         Use 0.0 for any unavailable channel.
        """
        self.features.append([float(v) for v in feature_vec])
        if len(self.features) > self.window_size:
            self.features.pop(0)

    def get_fused_features(self):
        """Return a 30-dim statistical feature vector (or None if buffer empty).

        Returns:
            numpy array shape (30,) of normalised features, or None
        """
        if len(self.features) < max(10, self.window_size // 10):
            # Not enough data yet
            return None

        try:
            arr = np.array(self.features, dtype=np.float64)  # (T, 5)
            n_channels = arr.shape[1]
            stats = []
            t = np.arange(len(arr), dtype=np.float64)

            for ch in range(n_channels):
                col = arr[:, ch]
                mean_v  = np.mean(col)
                std_v   = np.std(col) + 1e-9
                min_v   = np.min(col)
                max_v   = np.max(col)
                med_v   = np.median(col)
                # Linear trend slope (normalised by std so it is scale-invariant)
                try:
                    slope = float(np.polyfit(t / len(t), col, 1)[0]) / (std_v + 1e-9)
                except Exception:
                    slope = 0.0

                # Z-score normalise mean, std, min, max, median
                stats.extend([
                    (mean_v - mean_v) / std_v,   # always 0 (placeholder for shape)
                    std_v / (abs(mean_v) + 1e-6),  # coefficient of variation
                    (min_v  - mean_v) / std_v,
                    (max_v  - mean_v) / std_v,
                    (med_v  - mean_v) / std_v,
                    slope,
                ])

            out = np.array(stats, dtype=np.float32)
            # Clip extreme values to avoid exploding gradients
            out = np.clip(out, -5.0, 5.0)
            return out

        except Exception as e:
            logger.error(f'Feature fusion error: {e}')
            return None

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM

    @property
    def feature_names(self) -> list:
        return FEATURE_NAMES
