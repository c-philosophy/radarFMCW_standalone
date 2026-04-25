"""Angle FFT: convert spatial phase across antennas to angle of arrival."""

import numpy as np
from fmcw.utils.window import create_window


def angle_fft(
    s_rd: np.ndarray,
    n_angle_bins: int = 180,
    window_type: str = None,
) -> np.ndarray:
    """Apply Angle FFT (beamforming) along the antenna axis (axis=0).

    Args:
        s_rd: Range-Doppler result of shape (A, D, R_bins), complex.
        n_angle_bins: Output angle bins (zero-padded).
        window_type: Window function name or None.

    Returns:
        Complex Range-Doppler-Angle cube of shape (n_angle_bins, D, R_bins).
    """
    sig = s_rd.astype(np.complex64)
    n_ant = sig.shape[0]

    if window_type is not None:
        win = create_window(n_ant, window_type)
        sig = sig * win[:, np.newaxis, np.newaxis]

    return np.fft.fft(sig, axis=0, n=n_angle_bins).astype(np.complex64)
