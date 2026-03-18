\"\"\"Plotting utilities for signal visualization and analysis

Includes functions for plotting signals, FFT analysis, and time-domain data.
\"\"\"

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Plotting:
    \"\"\"Collection of plotting utilities for signal analysis.\"\"\"\n    
    @staticmethod
    def plot_signal(signal, title='Signal', xlabel='Time', ylabel='Value', figsize=(10, 4)):
        \"\"\"Plot time-domain signal.
        
        Args:
            signal: 1D array of signal values
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size (width, height)
        \"\"\"
        try:
            plt.figure(figsize=figsize)
            plt.plot(signal, linewidth=1.5)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel(xlabel, fontsize=11)
            plt.ylabel(ylabel, fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f'Error plotting signal: {e}')

    @staticmethod
    def plot_fft(signal, fs, title='Frequency Domain', figsize=(10, 4)):
        \"\"\"Plot frequency domain (FFT magnitude spectrum).
        
        Args:
            signal: 1D array of signal values
            fs: Sampling frequency in Hz
            title: Plot title
            figsize: Figure size (width, height)
        \"\"\"
        try:
            fft = np.fft.rfft(signal)
            freqs = np.fft.rfftfreq(len(signal), 1/fs)
            magnitude = np.abs(fft)
            
            plt.figure(figsize=figsize)
            plt.plot(freqs, magnitude, linewidth=1.5)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Frequency (Hz)', fontsize=11)
            plt.ylabel('Magnitude', fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f'Error plotting FFT: {e}')

    @staticmethod
    def plot_signal_and_fft(signal, fs, title='Signal Analysis', figsize=(14, 5)):
        \"\"\"Plot signal and its FFT side by side.
        
        Args:
            signal: 1D array of signal values
            fs: Sampling frequency in Hz
            title: Plot title
            figsize: Figure size (width, height)
        \"\"\"
        try:
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(1, 2, figure=fig)
            
            # Time domain
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(signal, linewidth=1.5, color='blue')
            ax1.set_title('Time Domain', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Sample', fontsize=10)
            ax1.set_ylabel('Amplitude', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Frequency domain
            ax2 = fig.add_subplot(gs[1])
            fft = np.fft.rfft(signal)
            freqs = np.fft.rfftfreq(len(signal), 1/fs)
            magnitude = np.abs(fft)
            ax2.plot(freqs, magnitude, linewidth=1.5, color='red')
            ax2.set_title('Frequency Domain (FFT)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Frequency (Hz)', fontsize=10)
            ax2.set_ylabel('Magnitude', fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f'Error plotting signal and FFT: {e}')

    @staticmethod
    def plot_signals_comparison(signals_dict, fs=20, title='Signal Comparison', figsize=(14, 6)):
        \"\"\"Plot multiple signals for comparison.
        
        Args:
            signals_dict: Dict of {name: signal_array}
            fs: Sampling frequency in Hz
            title: Plot title
            figsize: Figure size (width, height)
        \"\"\"
        try:
            fig, ax = plt.subplots(figsize=figsize)
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
            
            for (name, signal), color in zip(signals_dict.items(), colors):
                ax.plot(signal, label=name, linewidth=1.5, color=color, alpha=0.8)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Time (samples)', fontsize=11)
            ax.set_ylabel('Amplitude', fontsize=11)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f'Error plotting signal comparison: {e}')

    @staticmethod
    def plot_peaks(signal, peaks, title='Peak Detection', figsize=(12, 5)):
        \"\"\"Plot signal with detected peaks highlighted.
        
        Args:
            signal: 1D array of signal values
            peaks: Array of peak indices
            title: Plot title
            figsize: Figure size (width, height)
        \"\"\"
        try:
            plt.figure(figsize=figsize)
            plt.plot(signal, 'b-', linewidth=1.5, label='Signal')
            plt.plot(peaks, signal[peaks], 'r^', markersize=10, label='Peaks')
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Sample', fontsize=11)
            plt.ylabel('Value', fontsize=11)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f'Error plotting peaks: {e}')

    @staticmethod
    def plot_histogram(data, bins=50, title='Data Distribution', figsize=(10, 5)):
        \"\"\"Plot histogram of data distribution.
        
        Args:
            data: 1D array of values
            bins: Number of histogram bins
            title: Plot title
            figsize: Figure size (width, height)
        \"\"\"
        try:
            plt.figure(figsize=figsize)
            plt.hist(data, bins=bins, color='blue', alpha=0.7, edgecolor='black')
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Value', fontsize=11)
            plt.ylabel('Frequency', fontsize=11)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f'Error plotting histogram: {e}')
