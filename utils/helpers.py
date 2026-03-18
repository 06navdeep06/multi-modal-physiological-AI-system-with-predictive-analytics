\"\"\"Helper utilities for signal processing and data manipulation

Includes normalization, windowing, statistical functions, and data validation.
\"\"\"

import numpy as np
import logging

logger = logging.getLogger(__name__)

def normalize(arr):
    \"\"\"Normalize array to zero mean and unit variance.
    
    Args:
        arr: Array-like data
    
    Returns:
        numpy array: Normalized data
    \"\"\"
    arr = np.array(arr)
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-10:  # Avoid division by zero
        return arr - mean
    return (arr - mean) / std

def sliding_window(arr, window_size):
    \"\"\"Extract latest window_size elements from array.
    
    Args:
        arr: Array-like data
        window_size: Number of recent elements to keep
    
    Returns:
        numpy array: Last window_size elements or entire array if smaller
    \"\"\"
    arr = np.array(arr)
    if len(arr) < window_size:
        return arr
    return arr[-window_size:]

def validate_data(arr, min_len=1, dtype=float):
    \"\"\"Validate array data format and length.
    
    Args:
        arr: Array-like data
        min_len: Minimum required length
        dtype: Expected data type
    
    Returns:
        tuple: (is_valid: bool, error_msg: str or None)
    \"\"\"
    try:
        data = np.array(arr, dtype=dtype)
        if len(data) < min_len:
            return False, f'Data too short: {len(data)} < {min_len}'
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            return False, 'Data contains NaN or Inf'
        return True, None
    except Exception as e:
        return False, str(e)

def smooth_signal(arr, window_size=5):
    \"\"\"Apply moving average smoothing.
    
    Args:
        arr: Array-like data
        window_size: Width of smoothing kernel
    
    Returns:
        numpy array: Smoothed signal (padded with original edges)
    \"\"\"
    arr = np.array(arr)
    if len(arr) < window_size:
        return arr
    
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(arr, kernel, mode='same')
    return smoothed

def peak_detection(signal, threshold=None, min_distance=1):
    """Detect peaks in signal.
    
    Args:
        signal: 1D array of signal values
        threshold: Minimum peak value (auto if None)
        min_distance: Minimum samples between peaks (>=1)
    
    Returns:
        numpy array: Indices of detected peaks
    """
    signal = np.array(signal, dtype=np.float64)

    if signal.size < 3:
        return np.array([], dtype=int)

    signal_range = np.max(signal) - np.min(signal)
    if signal_range < 1e-9:
        return np.array([], dtype=int)

    # Auto threshold at 20% of signal range if not specified
    if threshold is None:
        threshold = np.min(signal) + 0.2 * signal_range

    min_distance = max(int(min_distance), 1)

    # Simple peak detection: value > neighbors
    peaks = np.where((signal[1:-1] > signal[:-2]) & 
                     (signal[1:-1] > signal[2:]) &
                     (signal[1:-1] > threshold))[0] + 1
    
    # Remove peaks closer than min_distance
    if len(peaks) > 1:
        keep_mask = np.concatenate(([True], np.diff(peaks) >= min_distance))
        peaks = peaks[keep_mask]
    
    return peaks

def calculate_stats(arr):
    """Calculate basic statistics.
    
    Args:
        arr: Array-like data
    
    Returns:
        dict: Statistics (mean, std, min, max, median)
    \"\"\"
    arr = np.array(arr)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr)),
        'size': len(arr)
    }
