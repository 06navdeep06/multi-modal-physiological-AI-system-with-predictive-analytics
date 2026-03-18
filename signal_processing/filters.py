\"\"\"Signal Processing Filters for Physiological Signal Analysis

Implements bandpass filtering and detrending for heart rate, respiration, and motion signals.
\"\"\"

import numpy as np
from scipy.signal import butter, filtfilt, detrend
import logging

logger = logging.getLogger(__name__)

class SignalFilters:
    \"\"\"Collection of signal processing filters.\"\"\"\n    
    @staticmethod
    def bandpass_filter(signal, fs, lowcut=0.7, highcut=4.0, order=4):
        \"\"\"Apply Butterworth bandpass filter to signal.
        
        Args:
            signal: 1D numpy array of signal values
            fs: Sampling frequency in Hz
            lowcut: Lower cutoff frequency in Hz (default 0.7 Hz = 42 BPM)
            highcut: Upper cutoff frequency in Hz (default 4.0 Hz = 240 BPM)
            order: Filter order (default 4)
        
        Returns:
            numpy array: Filtered signal
        \"\"\"
        try:
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            
            # Clip to valid range
            low = np.clip(low, 0.001, 0.999)
            high = np.clip(high, 0.001, 0.999)
            
            if low >= high:
                logger.warning(f'Invalid filter range: low={low}, high={high}')
                return signal
            
            b, a = butter(order, [low, high], btype='band')
            return filtfilt(b, a, signal)
        except Exception as e:
            logger.error(f'Bandpass filter error: {e}')
            return signal

    @staticmethod
    def detrend_signal(signal):
        \"\"\"Remove linear trend from signal.
        
        Args:
            signal: 1D numpy array of signal values
        
        Returns:
            numpy array: Detrended signal
        \"\"\"
        try:
            return detrend(signal)
        except Exception as e:
            logger.error(f'Detrend error: {e}')
            return signal

    @staticmethod
    def normalize_signal(signal):
        \"\"\"Normalize signal to zero mean and unit variance.
        
        Args:
            signal: 1D numpy array of signal values
        
        Returns:
            numpy array: Normalized signal
        \"\"\"
        try:
            mean = np.mean(signal)
            std = np.std(signal)
            if std == 0:
                return signal - mean
            return (signal - mean) / std
        except Exception as e:
            logger.error(f'Normalization error: {e}')
            return signal
