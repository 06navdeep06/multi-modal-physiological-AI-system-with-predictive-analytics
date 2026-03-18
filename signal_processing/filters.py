"""Signal Processing Filters for Physiological Signal Analysis

Implements bandpass filtering and detrending for heart rate, respiration,
and motion signals.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, detrend
import logging

logger = logging.getLogger(__name__)


class SignalFilters:
    """Collection of signal processing filters."""

    @staticmethod
    def bandpass_filter(signal, fs, lowcut=0.7, highcut=4.0, order=4):
        """Apply a Butterworth bandpass filter.

        Args:
            signal:  1-D numpy array
            fs:      Sampling frequency in Hz
            lowcut:  Lower cutoff (Hz)
            highcut: Upper cutoff (Hz)
            order:   Filter order (default 4)

        Returns:
            numpy array: Filtered signal
        """
        try:
            nyq  = 0.5 * fs
            low  = float(np.clip(lowcut  / nyq, 0.001, 0.999))
            high = float(np.clip(highcut / nyq, 0.001, 0.999))
            if low >= high:
                logger.warning(f'Invalid bandpass range: {low=:.4f}, {high=:.4f}')
                return signal
            b, a = butter(order, [low, high], btype='band')
            return filtfilt(b, a, signal)
        except Exception as e:
            logger.error(f'Bandpass filter error: {e}')
            return signal

    @staticmethod
    def detrend_signal(signal):
        """Remove linear trend from signal."""
        try:
            return detrend(signal)
        except Exception as e:
            logger.error(f'Detrend error: {e}')
            return signal

    @staticmethod
    def normalize_signal(signal):
        """Normalise signal to zero mean and unit variance."""
        try:
            mean = np.mean(signal)
            std  = np.std(signal)
            return (signal - mean) / std if std != 0 else signal - mean
        except Exception as e:
            logger.error(f'Normalisation error: {e}')
            return signal
