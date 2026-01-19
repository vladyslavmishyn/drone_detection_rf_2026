"""Generalized Cross-Correlation (GCC) algorithms for TDoA estimation."""
from __future__ import annotations

import numpy as np


def _estimate_signal_duration(sig: np.ndarray, threshold_factor: float = 0.1) -> int:
    """Estimate active signal duration by energy threshold.
    
    Args:
        sig: Complex or real signal
        threshold_factor: Fraction of peak energy for threshold
        
    Returns:
        Estimated signal duration in samples
    """
    energy = np.abs(sig) ** 2
    smoothed = np.convolve(energy, np.ones(100) / 100, mode='same')
    threshold = threshold_factor * np.max(smoothed)
    active = smoothed > threshold
    
    if not np.any(active):
        return len(sig)
    
    first_idx = np.argmax(active)
    last_idx = len(active) - np.argmax(active[::-1]) - 1
    
    return max(last_idx - first_idx, 1000)


def gcc_phat(sig1: np.ndarray, sig2: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Cross-Correlation with Phase Transform (GCC-PHAT).
    
    Args:
        sig1: First signal (complex or real)
        sig2: Second signal (complex or real)
        fs: Sample rate in Hz
        
    Returns:
        Tuple of (correlation, lags) where lags are in samples
    """
    n = min(len(sig1), len(sig2))
    
    # Estimate signal duration from energy analysis
    duration1 = _estimate_signal_duration(sig1)
    duration2 = _estimate_signal_duration(sig2)
    signal_duration = max(duration1, duration2)
    
    # Process 4x signal duration or 50k samples max
    process_length = min(int(signal_duration * 4), 50000, n)
    
    s1 = sig1[:process_length] - np.mean(sig1[:process_length])
    s2 = sig2[:process_length] - np.mean(sig2[:process_length])
    
    n_fft = 2 ** int(np.ceil(np.log2(2 * process_length - 1)))
    
    X1 = np.fft.fft(s1, n=n_fft)
    X2 = np.fft.fft(s2, n=n_fft)
    
    cross_spectrum = X1 * np.conj(X2)
    weight = 1.0 / (np.abs(cross_spectrum) + 1e-12)
    
    correlation = np.fft.ifft(cross_spectrum * weight)
    correlation = np.fft.fftshift(correlation).real
    
    lags = np.arange(-n_fft//2, n_fft//2)
    
    # Constrain lags to ±10% of processed length
    max_lag = int(process_length * 0.1)
    mask = np.abs(lags) <= max_lag
    
    return correlation[mask], lags[mask]


def compute_tdoa_from_gcc(sig1: np.ndarray, sig2: np.ndarray, fs: float) -> float:
    """Compute TDoA between two signals using GCC-PHAT.
    
    Args:
        sig1: First receiver signal
        sig2: Second receiver signal
        fs: Sample rate in Hz
        
    Returns:
        Time difference of arrival in seconds (positive if sig1 arrives first)
    """
    corr, lags = gcc_phat(sig1, sig2, fs)
    peak_idx = np.argmax(np.abs(corr))
    delay_samples = lags[peak_idx]
    tdoa_sec = delay_samples / fs
    return float(tdoa_sec)
