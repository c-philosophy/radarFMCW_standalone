"""Doppler FFT: convert slow-time (chirp-to-chirp) phase to velocity."""

import numpy as np
from fmcw.utils.window import create_window


def doppler_fft(
    s_r: np.ndarray,
    window_type: str = None,
) -> np.ndarray:
    """Apply Doppler FFT along the chirp axis (axis=1).

    Args:
        s_r: Range-FFT result of shape (A, D, R_bins), complex.
        window_type: Window function name or None.

    Returns:
        Complex Range-Doppler result of shape (A, D, R_bins).
    """
    sig = s_r.astype(np.complex64)

    if window_type is not None:
        win = create_window(sig.shape[1], window_type)
        sig = sig * win[np.newaxis, :, np.newaxis]

    return np.fft.fft(sig, axis=1).astype(np.complex64)
