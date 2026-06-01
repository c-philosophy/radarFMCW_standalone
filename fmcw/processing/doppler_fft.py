"""Doppler FFT: convert slow-time (chirp-to-chirp) phase to velocity.

Output is fftshifted so that DC (zero velocity) is at the center of the
spectrum. Bin indexing after shift:
    bin 0 = -v_max, bin D/2 = DC, bin D-1 = +v_max - dv
"""

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
        Spectrum is fftshifted: DC at center, negative velocities left,
        positive velocities right.
    """
    sig = s_r.astype(np.complex64)

    if window_type is not None:
        win = create_window(sig.shape[1], window_type)
        sig = sig * win[np.newaxis, :, np.newaxis]

    result = np.fft.fft(sig, axis=1).astype(np.complex64)
    return np.fft.fftshift(result, axes=1)
