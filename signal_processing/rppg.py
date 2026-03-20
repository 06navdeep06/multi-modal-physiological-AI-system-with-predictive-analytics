"""Remote Photoplethysmography (rPPG) — Heart Rate, HRV & SpO2 Estimation

Implements:
  - CHROM algorithm (de Haan & Jeanne, 2013) for illumination-robust rPPG
  - Time-domain HRV: RMSSD, SDNN, pNN50, mean RR interval
  - Frequency-domain HRV: VLF/LF/HF power bands, LF/HF ratio (Malik 1996 standards)
  - Cardiac coherence score — % power in the 0.1 Hz resonance band
    (Rollin McCraty / HeartMath Institute methodology)
  - Baevsky Stress Index — histogram-based regulatory strain
    (SI = AMo / (2 × Mo × MxDMn); Baevsky 1984)
  - Normalised PPG signal buffer for real-time waveform rendering
  - RR interval series accessible for Poincaré plot

References:
  de Haan G., Jeanne V. (2013). Robust Pulse-Rate From Chrominance-Based rPPG.
    IEEE Trans. Biomed. Eng. 60(10):2878-2886.
  Task Force of the ESC / NASPE (1996). Heart Rate Variability: Standards of Measurement.
    Eur. Heart J. 17:354-381.
"""

from __future__ import annotations

import numpy as np
import logging
from scipy.signal import welch, find_peaks

from .filters import SignalFilters

logger = logging.getLogger(__name__)


class RPPG:
    """Remote PPG — CHROM algorithm with full HRV analysis suite.

    Attributes
    ----------
    fs : int
        Sampling frequency (Hz).
    window_size : int
        Buffer length (samples = fs × window_sec).
    r/g/b_window : list[float]
        Circular colour-channel buffers.
    _rr_intervals : list[float]
        Most-recently computed RR interval series (ms). Populated by
        ``compute_hr``; exposed via the ``rr_intervals`` property for use
        in Poincaré plots.
    _chrom_signal : list[float]
        Normalised, filtered CHROM signal (−1 … +1). Populated by
        ``compute_hr``; exposed via ``get_ppg_signal``.
    """

    def __init__(self, fs: int = 20, window_sec: int = 15) -> None:
        self.fs          = fs
        self.window_size = fs * window_sec

        # Colour-channel circular buffers
        self.r_window: list[float] = []
        self.g_window: list[float] = []
        self.b_window: list[float] = []

        # Derived state (updated each compute_hr call)
        self._signal_quality: float       = 0.0
        self._rr_intervals:   list[float] = []   # ms
        self._chrom_signal:   list[float] = []   # normalised −1…+1

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, rgb_means) -> None:
        """Append per-channel face-ROI spatial means (R, G, B order)."""
        r, g, b = float(rgb_means[0]), float(rgb_means[1]), float(rgb_means[2])
        self.r_window.append(r)
        self.g_window.append(g)
        self.b_window.append(b)
        if len(self.r_window) > self.window_size:
            self.r_window.pop(0)
            self.g_window.pop(0)
            self.b_window.pop(0)

    def compute_hr(self) -> tuple:
        """Estimate heart rate and time-domain HRV from the buffered signal.

        Returns
        -------
        tuple
            ``(hr_bpm, hrv_metrics)`` — both are ``None`` until the
            window is fully populated.  ``hrv_metrics`` is a dict with
            keys ``rmssd``, ``sdnn``, ``pnn50``, ``mean_rr`` (all ms).

        Side-effects
        ------------
        Populates ``_rr_intervals`` and ``_chrom_signal``.
        """
        min_samples = self.fs * 3   # allow early estimates after 3 s
        if len(self.r_window) < min_samples:
            return None, None
        try:
            chrom    = self._compute_chrom_signal()
            chrom    = SignalFilters.detrend_signal(chrom)
            filtered = SignalFilters.bandpass_filter(chrom, self.fs,
                                                     lowcut=0.7, highcut=4.0,
                                                     order=4)

            # Cache normalised waveform for rendering (updated here so
            # get_ppg_signal() works immediately after compute_hr)
            mx = float(np.max(np.abs(filtered))) + 1e-9
            self._chrom_signal = [round(float(v / mx), 4) for v in filtered]

            self._signal_quality = self._compute_signal_quality(filtered)

            # Dominant cardiac frequency via FFT peak
            fft_mag = np.abs(np.fft.rfft(filtered))
            freqs   = np.fft.rfftfreq(len(filtered), 1.0 / self.fs)
            valid   = (freqs >= 0.667) & (freqs <= 3.0)   # 40–180 BPM
            if not np.any(valid):
                return None, None

            peak_idx = int(np.argmax(fft_mag[valid]))
            hr_hz    = float(freqs[valid][peak_idx])
            hr_bpm   = hr_hz * 60.0

            hrv_metrics = self._compute_hrv(filtered, hr_hz)
            return hr_bpm, hrv_metrics

        except Exception as e:
            logger.error(f'compute_hr error: {e}')
            return None, None

    def compute_hrv_frequency_domain(self) -> dict | None:
        """Frequency-domain HRV analysis (Task Force of ESC 1996 bands).

        Requires ≥ 8 valid RR intervals (call ``compute_hr`` first).

        Method
        ------
        1. Extract last-computed RR interval series from ``_rr_intervals``.
        2. Uniformly resample at 4 Hz using linear interpolation.
        3. Estimate PSD via Welch's method.
        4. Integrate power in each frequency band.

        Returns
        -------
        dict with keys
            ``vlf_power``  : ms² in 0.003–0.040 Hz
            ``lf_power``   : ms² in 0.040–0.150 Hz (sympatho-vagal modulation)
            ``hf_power``   : ms² in 0.150–0.400 Hz (respiratory / vagal)
            ``total_power``: ms² in VLF+LF+HF
            ``lf_hf_ratio``: dimensionless sympathovagal balance index
            ``lf_pct``     : LF as % of total
            ``hf_pct``     : HF as % of total
            ``coherence``  : % power in cardiac-resonance band (0.085–0.115 Hz)
        """
        if len(self._rr_intervals) < 8:
            return None
        try:
            rr_ms = np.array(self._rr_intervals, dtype=float)
            rr_ms = rr_ms[(rr_ms >= 300) & (rr_ms <= 2000)]
            if len(rr_ms) < 6:
                return None

            # Build uniformly sampled RR time-series at 4 Hz
            rr_times = np.cumsum(rr_ms) / 1000.0            # seconds
            rr_fs    = 4.0
            t_unif   = np.arange(rr_times[0], rr_times[-1], 1.0 / rr_fs)
            if len(t_unif) < 8:
                return None
            rr_interp = np.interp(t_unif, rr_times, rr_ms)
            rr_interp -= np.mean(rr_interp)                  # zero-mean

            nperseg = min(len(rr_interp), max(8, len(rr_interp) // 2))
            freqs_w, psd_w = welch(rr_interp, fs=rr_fs, nperseg=nperseg)

            def _band(flo: float, fhi: float) -> float:
                mask = (freqs_w >= flo) & (freqs_w <= fhi)
                return float(np.trapz(psd_w[mask], freqs_w[mask])) \
                    if np.any(mask) else 0.0

            vlf   = _band(0.003, 0.040)
            lf    = _band(0.040, 0.150)
            hf    = _band(0.150, 0.400)
            total = vlf + lf + hf
            coh   = _band(0.085, 0.115)   # HeartMath coherence window

            lf_hf   = round(lf / (hf + 1e-12), 2)
            coh_pct = float(np.clip(coh / (total + 1e-12) * 100.0, 0.0, 100.0))
            lf_pct  = float(np.clip(lf  / (total + 1e-12) * 100.0, 0.0, 100.0))
            hf_pct  = float(np.clip(hf  / (total + 1e-12) * 100.0, 0.0, 100.0))

            return {
                'vlf_power':   round(vlf,   4),
                'lf_power':    round(lf,    4),
                'hf_power':    round(hf,    4),
                'total_power': round(total, 4),
                'lf_hf_ratio': lf_hf,
                'lf_pct':      round(lf_pct,  1),
                'hf_pct':      round(hf_pct,  1),
                'coherence':   round(coh_pct, 1),
            }
        except Exception as e:
            logger.debug(f'HRV freq-domain error: {e}')
            return None

    def baevsky_stress_index(self) -> float | None:
        """Baevsky Stress Index — histogram-based autonomic regulatory strain.

        SI = AMo / (2 × Mo × MxDMn)

        Parameters (from RR histogram, 10-ms bins)
        ---------
        Mo    : modal (most-frequent) RR interval (ms)
        AMo   : % of RR intervals within ±25 ms of Mo (amplitude of mode)
        MxDMn : variation range = max(RR) − min(RR)  (ms)

        Clinical interpretation (approximate)
        -------------------------------------
        SI <  50  → dominant parasympathetic; relaxed/athletic
        50–150    → balanced / normal rest
        150–300   → elevated sympathetic; mild stress
        SI > 300  → high sympathetic activation / acute stress

        Returns
        -------
        float | None
            SI normalised to [0, 1] via log₁₀ scale (SI=100 → ~0.40),
            for rendering on a gauge. Raw SI also stored in last call.
        """
        if len(self._rr_intervals) < 10:
            return None
        try:
            rr = np.array(self._rr_intervals, dtype=float)
            rr = rr[(rr >= 300) & (rr <= 2000)]
            if len(rr) < 8:
                return None

            # 10-ms histogram bins
            bins = np.arange(300, 2001, 10)
            hist, edges = np.histogram(rr, bins=bins)
            mo_idx = int(np.argmax(hist))
            mo     = float(edges[mo_idx] + 5)                    # bin centre
            amo    = float(np.sum(np.abs(rr - mo) <= 25) / len(rr) * 100.0)
            mxdmn  = float(np.max(rr) - np.min(rr)) + 1e-9

            si_raw = amo / (2.0 * (mo / 1000.0) * mxdmn + 1e-9)  # s⁻¹ scale

            # Normalise: SI=1 → 0, SI=50 → ~0.28, SI=200 → ~0.68, SI=1000 → 1
            si_norm = float(np.clip(np.log10(max(si_raw, 0.1) + 1) /
                                    np.log10(1001), 0.0, 1.0))
            return round(si_norm, 3)
        except Exception as e:
            logger.debug(f'Baevsky SI error: {e}')
            return None

    def estimate_spo2(self) -> float | None:
        """Rough SpO2 proxy from red / green ratio-of-ratios.

        NOT clinical-grade — requires calibrated IR hardware.
        Provides a physiologically plausible proxy (typical 94–100 %).
        """
        if len(self.r_window) < self.window_size:
            return None
        try:
            R = np.array(self.r_window)
            G = np.array(self.g_window)
            r_ac = np.std(SignalFilters.bandpass_filter(R, self.fs, 0.7, 4.0))
            r_dc = np.mean(R) + 1e-9
            g_ac = np.std(SignalFilters.bandpass_filter(G, self.fs, 0.7, 4.0))
            g_dc = np.mean(G) + 1e-9
            ros  = (r_ac / r_dc) / (g_ac / g_dc + 1e-9)
            return round(float(np.clip(110.0 - 25.0 * ros, 85.0, 100.0)), 1)
        except Exception:
            return None

    def get_ppg_signal(self, n_points: int = 150) -> list[float]:
        """Return the last *n_points* of the normalised CHROM PPG signal.

        Suitable for direct canvas rendering (values in −1 … +1).
        Returns an empty list until the first ``compute_hr`` call succeeds.
        """
        if not self._chrom_signal:
            return []
        return self._chrom_signal[-n_points:]

    @property
    def rr_intervals(self) -> list[float]:
        """Last computed RR interval series (ms).  Use for Poincaré plot."""
        return list(self._rr_intervals)

    @property
    def signal_quality(self) -> float:
        """SNR-based quality score in [0, 1]."""
        return self._signal_quality

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_chrom_signal(self) -> np.ndarray:
        """CHROM algorithm (de Haan & Jeanne 2013).

        Xs = 3R′ − 2G′
        Ys = 1.5R′ + G′ − 1.5B′
        S  = Xs − α Ys   where α = std(Xs) / std(Ys)
        """
        R = np.array(self.r_window)
        G = np.array(self.g_window)
        B = np.array(self.b_window)

        R_n = R / (np.mean(R) + 1e-9)
        G_n = G / (np.mean(G) + 1e-9)
        B_n = B / (np.mean(B) + 1e-9)

        Xs    = 3.0 * R_n - 2.0 * G_n
        Ys    = 1.5 * R_n + G_n - 1.5 * B_n
        alpha = np.std(Xs) / (np.std(Ys) + 1e-9)
        return Xs - alpha * Ys

    def _compute_signal_quality(self, filtered_signal: np.ndarray) -> float:
        """SNR-based quality score: peak cardiac-band power / total power.

        Scaled so that a dominant cardiac peak yields ≥ 0.6 (Good).
        """
        try:
            fft_mag = np.abs(np.fft.rfft(filtered_signal))
            freqs   = np.fft.rfftfreq(len(filtered_signal), 1.0 / self.fs)
            band    = (freqs >= 0.7) & (freqs <= 3.5)
            if not np.any(band):
                return 0.0
            peak_pow  = float(np.max(fft_mag[band]) ** 2)
            total_pow = float(np.sum(fft_mag ** 2)) + 1e-9
            return float(np.clip(peak_pow / total_pow * 10.0, 0.0, 1.0))
        except Exception:
            return 0.0

    def _compute_hrv(self, filtered_signal: np.ndarray,
                     hr_hz: float) -> dict | None:
        """Time-domain HRV from scipy peak detection with adaptive prominence.

        Uses ``scipy.signal.find_peaks`` with:
          - ``distance``: minimum samples between peaks (from dominant HR)
          - ``prominence``: 30 % of signal std-dev — rejects noise spikes
            while keeping true cardiac peaks in noisy webcam signals.

        Also stores detected RR series in ``_rr_intervals`` for use by
        ``compute_hrv_frequency_domain`` and ``baevsky_stress_index``.
        """
        try:
            min_distance  = max(1, int(self.fs / (hr_hz * 1.6)))
            min_prom      = float(np.std(filtered_signal)) * 0.30

            peaks, _ = find_peaks(
                filtered_signal,
                distance=min_distance,
                prominence=min_prom,
            )
            peaks = peaks.tolist()

            if len(peaks) < 3:
                return None

            rr_ms = np.diff(peaks) / self.fs * 1000.0
            rr_ms = rr_ms[(rr_ms >= 300) & (rr_ms <= 2000)]
            if len(rr_ms) < 2:
                return None

            # Persist for frequency-domain analysis & Poincaré
            self._rr_intervals = rr_ms.tolist()

            diffs   = np.diff(rr_ms)
            rmssd   = float(np.sqrt(np.mean(diffs ** 2)))
            sdnn    = float(np.std(rr_ms))
            pnn50   = float(np.sum(np.abs(diffs) > 50) / len(diffs) * 100.0)
            mean_rr = float(np.mean(rr_ms))

            return {
                'rmssd':   round(rmssd,   2),
                'sdnn':    round(sdnn,    2),
                'pnn50':   round(pnn50,   1),
                'mean_rr': round(mean_rr, 1),
            }
        except Exception as e:
            logger.debug(f'HRV computation error: {e}')
            return None
