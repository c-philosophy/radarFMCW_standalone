"""Range FFT: convert fast-time samples to range frequency bins."""

import numpy as np
from fmcw.utils.window import create_window


def range_fft(
    signal: np.ndarray,
    downsample: int = 1,
    window_type: str = None,
) -> np.ndarray:
    """Apply Range FFT along the sample axis (axis=2).

    Truncates to half-spectrum due to real-valued input symmetry.

    Args:
        signal: Real IF signal of shape (A, D, S).
        downsample: Stride for downsampling before FFT.
        window_type: Window function name or None.

    Returns:
        Complex FFT result of shape (A, D, S_eff // 2).
    """
    sig = signal[:, :, ::downsample].astype(np.float64)

    if window_type is not None:
        win = create_window(sig.shape[2], window_type)
        sig = sig * win[np.newaxis, np.newaxis, :]

    s_r = np.fft.fft(sig, axis=2)
    half = s_r.shape[2] // 2
    return s_r[:, :, :half].astype(np.complex64)
