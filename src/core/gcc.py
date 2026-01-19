"""Generalized Cross-Correlation (GCC) algorithms for TDoA estimation."""
from __future__ import annotations

import numpy as np


def gcc_phat(sig1: np.ndarray, sig2: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Cross-Correlation with Phase Transform (GCC-PHAT).
    
    Args:
        sig1: First signal (complex or real)
        sig2: Second signal (complex or real)
        fs: Sample rate in Hz
        
    Returns:
        Tuple of (correlation, lags) where lags are in samples
    """
    len1 = len(sig1)
    len2 = len(sig2)
    n_corr = len1 + len2 - 1
    n_fft = 2 ** int(np.ceil(np.log2(n_corr)))
    
    X1 = np.fft.fft(sig1, n=n_fft)
    X2 = np.fft.fft(sig2, n=n_fft)
    
    cross_spectrum = X1 * np.conj(X2)
    weight = 1.0 / (np.abs(cross_spectrum) + 1e-12)
    cross_spectrum_weighted = cross_spectrum * weight
    
    correlation = np.fft.ifft(cross_spectrum_weighted)
    correlation = np.fft.fftshift(correlation).real
    
    lags = np.arange(-n_fft//2, n_fft//2)
    
    return correlation, lags


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
