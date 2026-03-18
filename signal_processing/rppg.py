"""Remote Photoplethysmography (rPPG) - Heart Rate, HRV & SpO2 Estimation

Implements the CHROM (chrominance-based) rPPG algorithm by de Haan & Jeanne (2013),
which is significantly more robust to motion artefacts and lighting variation than
the simple single-channel green mean approach.

Also provides:
- Comprehensive HRV metrics: RMSSD, SDNN, pNN50, mean RR
- Approximate SpO2 estimation via red/green ratio-of-ratios
- Signal quality (SNR-based) for gating unreliable estimates
"""

import numpy as np
import logging
from .filters import SignalFilters

logger = logging.getLogger(__name__)


class RPPG:
    """Remote PPG using the CHROM algorithm for heart rate estimation.

    Attributes:
        fs:          Sampling frequency in Hz
        window_size: Samples per estimation window
        r/g/b_window: Circular buffers for each colour channel mean
    """

    def __init__(self, fs=20, window_sec=15):
        self.fs = fs
        self.window_size = fs * window_sec
        self.r_window: list[float] = []
        self.g_window: list[float] = []
        self.b_window: list[float] = []
        self._signal_quality: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, rgb_means):
        """Push new per-channel face-ROI means (R, G, B order).

        Args:
            rgb_means: sequence of three floats (r_mean, g_mean, b_mean)
        """
        r, g, b = float(rgb_means[0]), float(rgb_means[1]), float(rgb_means[2])
        self.r_window.append(r)
        self.g_window.append(g)
        self.b_window.append(b)
        if len(self.r_window) > self.window_size:
            self.r_window.pop(0)
            self.g_window.pop(0)
            self.b_window.pop(0)

    def compute_hr(self):
        """Estimate heart rate and HRV from the buffered signal.

        Returns:
            tuple: (hr_bpm: float|None, hrv_metrics: dict|None)
                   hrv_metrics keys: rmssd, sdnn, pnn50, mean_rr (all in ms)
        """
        if len(self.r_window) < self.window_size:
            return None, None

        try:
            chrom = self._compute_chrom_signal()
            chrom = SignalFilters.detrend_signal(chrom)
            filtered = SignalFilters.bandpass_filter(chrom, self.fs,
                                                     lowcut=0.7, highcut=4.0, order=4)

            self._signal_quality = self._compute_signal_quality(filtered)

            fft_mag = np.abs(np.fft.rfft(filtered))
            freqs = np.fft.rfftfreq(len(filtered), 1.0 / self.fs)

            # Cardiac range 40–180 BPM → 0.667–3.0 Hz
            valid = (freqs >= 0.667) & (freqs <= 3.0)
            if not np.any(valid):
                return None, None

            peak_idx = np.argmax(fft_mag[valid])
            hr_hz = freqs[valid][peak_idx]
            hr_bpm = hr_hz * 60.0

            hrv_metrics = self._compute_hrv(filtered, hr_hz)
            return hr_bpm, hrv_metrics

        except Exception as e:
            logger.error(f'Error computing HR: {e}')
            return None, None

    def estimate_spo2(self):
        """Rough SpO2 estimate from red/green ratio-of-ratios.

        NOTE: Not clinical grade – requires calibrated IR hardware for accuracy.
        Provides a plausible proxy (typical range 94–100 %) for display purposes.

        Returns:
            float|None: Estimated SpO2 % (clipped 85–100)
        """
        if len(self.r_window) < self.window_size:
            return None
        try:
            R = np.array(self.r_window)
            G = np.array(self.g_window)

            r_ac = np.std(SignalFilters.bandpass_filter(R, self.fs, 0.7, 4.0))
            r_dc = np.mean(R) + 1e-6
            g_ac = np.std(SignalFilters.bandpass_filter(G, self.fs, 0.7, 4.0))
            g_dc = np.mean(G) + 1e-6

            ros = (r_ac / r_dc) / (g_ac / g_dc + 1e-6)
            spo2 = float(np.clip(110.0 - 25.0 * ros, 85.0, 100.0))
            return round(spo2, 1)
        except Exception as e:
            logger.debug(f'SpO2 estimation error: {e}')
            return None

    @property
    def signal_quality(self) -> float:
        return self._signal_quality

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_chrom_signal(self):
        """CHROM algorithm (de Haan & Jeanne, 2013).

        Cancels specular noise by projecting two chrominance axes:
            Xs = 3R' - 2G'
            Ys = 1.5R' + G' - 1.5B'
        where R'/G'/B' are illumination-normalised channels.
        Final signal S = Xs - α·Ys  (α = std(Xs)/std(Ys)).
        """
        R = np.array(self.r_window)
        G = np.array(self.g_window)
        B = np.array(self.b_window)

        R_n = R / (np.mean(R) + 1e-6)
        G_n = G / (np.mean(G) + 1e-6)
        B_n = B / (np.mean(B) + 1e-6)

        Xs = 3.0 * R_n - 2.0 * G_n
        Ys = 1.5 * R_n + G_n - 1.5 * B_n

        alpha = np.std(Xs) / (np.std(Ys) + 1e-6)
        return Xs - alpha * Ys

    def _compute_signal_quality(self, filtered_signal) -> float:
        """SNR-based quality score (0–1).

        Ratio of peak power in the cardiac band to total spectral power.
        """
        try:
            fft_mag = np.abs(np.fft.rfft(filtered_signal))
            freqs = np.fft.rfftfreq(len(filtered_signal), 1.0 / self.fs)
            band = (freqs >= 0.7) & (freqs <= 3.5)
            if not np.any(band):
                return 0.0
            peak_power = float(np.max(fft_mag[band]) ** 2)
            total_power = float(np.sum(fft_mag ** 2)) + 1e-6
            return float(np.clip(peak_power / total_power * 10.0, 0.0, 1.0))
        except Exception:
            return 0.0

    def _compute_hrv(self, filtered_signal, hr_hz):
        """Compute RMSSD, SDNN, pNN50, and mean RR from detected peaks.

        Args:
            filtered_signal: bandpass-filtered CHROM signal
            hr_hz:           dominant cardiac frequency (Hz)

        Returns:
            dict with keys rmssd, sdnn, pnn50, mean_rr (ms), or None
        """
        try:
            min_distance = max(1, int(self.fs / (hr_hz * 1.6)))

            # Peak detection with minimum inter-peak distance
            peaks = []
            for i in range(1, len(filtered_signal) - 1):
                if (filtered_signal[i] > filtered_signal[i - 1] and
                        filtered_signal[i] > filtered_signal[i + 1]):
                    if not peaks or (i - peaks[-1]) >= min_distance:
                        peaks.append(i)

            if len(peaks) < 3:
                return None

            rr_ms = np.diff(peaks) / self.fs * 1000.0
            # Keep physiologically valid RR intervals (300–2000 ms = 30–200 BPM)
            rr_ms = rr_ms[(rr_ms >= 300) & (rr_ms <= 2000)]
            if len(rr_ms) < 2:
                return None

            diffs = np.diff(rr_ms)
            rmssd = float(np.sqrt(np.mean(diffs ** 2)))
            sdnn = float(np.std(rr_ms))
            pnn50 = float(np.sum(np.abs(diffs) > 50) / len(diffs) * 100.0)
            mean_rr = float(np.mean(rr_ms))

            return {
                'rmssd':   round(rmssd, 2),
                'sdnn':    round(sdnn, 2),
                'pnn50':   round(pnn50, 1),
                'mean_rr': round(mean_rr, 1),
            }
        except Exception as e:
            logger.debug(f'HRV computation error: {e}')
            return None
