"""RF impairments: IQ imbalance, phase noise, inter-element coupling."""

from typing import Optional, Tuple
import numpy as np


class IQImbalance:
    """IQ (In-phase / Quadrature) imbalance model.

    Models amplitude and phase mismatch between I and Q channels.
    The ideal complex signal becomes: I' = I * (1 + eps/2), Q' = Q * (1 - eps/2) + phi*I
    """

    def __init__(
        self,
        amplitude_db: float = 0.0,   # amplitude imbalance in dB (0 = perfect)
        phase_deg: float = 0.0,       # phase imbalance in degrees (0 = perfect)
    ):
        self.eps = 10.0 ** (amplitude_db / 20.0) - 1.0  # amplitude error
        self.phi = np.deg2rad(phase_deg)

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Apply IQ imbalance to complex-valued signal.

        Args:
            signal: Complex IF signal after Hilbert or baseband conversion.

        Returns:
            Imbalanced complex signal.
        """
        I = np.real(signal)
        Q = np.imag(signal)

        I_out = I * (1.0 + self.eps / 2.0)
        Q_out = Q * (1.0 - self.eps / 2.0) + self.phi * I

        return I_out + 1j * Q_out


class PhaseNoise:
    """Local oscillator phase noise model.

    Simplified model: white frequency noise integrated over chirp duration.
    More sophisticated models (flicker, Leeson) can be added.
    """

    def __init__(
        self,
        phase_noise_dbc_hz: float = -100.0,  # dBc/Hz at 1 kHz offset (typical)
        seed: Optional[int] = None,
    ):
        # Convert dBc/Hz to phase error variance per sample
        # This is a simplified flat-spectrum model
        self.phase_noise_linear = 10.0 ** (phase_noise_dbc_hz / 10.0)
        self._rng = np.random.default_rng(seed)

    def apply(self, signal: np.ndarray, bandwidth_hz: float) -> np.ndarray:
        """Apply phase noise to signal.

        Args:
            signal: Real or complex signal array.
            bandwidth_hz: Signal bandwidth (determines noise power).

        Returns:
            Signal with phase noise added.
        """
        noise_power = self.phase_noise_linear * bandwidth_hz
        phase_noise = np.sqrt(noise_power) * self._rng.standard_normal(signal.shape)
        # Rotate signal by phase noise (small-angle approx for real)
        if np.iscomplexobj(signal):
            return signal * np.exp(1j * phase_noise)
        else:
            return signal * np.cos(phase_noise) + np.imag(
                signal + 1j * 0
            ) * np.sin(phase_noise)


class ThermalNoise:
    """Receiver thermal noise model."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    def complex_awgn(
        self,
        signal: np.ndarray,
        snr_db: float,
        signal_power: Optional[float] = None,
    ) -> np.ndarray:
        """Add complex AWGN to achieve target SNR.

        Each quadrature gets half the noise power.

        Args:
            signal: Real or complex signal.
            snr_db: Target SNR in dB.
            signal_power: If None, computed from signal.

        Returns:
            Noisy signal with same shape and dtype.
        """
        if signal_power is None:
            signal_power = np.mean(np.abs(signal) ** 2)

        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear

        if np.iscomplexobj(signal):
            noise = np.sqrt(noise_power / 2.0) * (
                self._rng.standard_normal(signal.shape)
                + 1j * self._rng.standard_normal(signal.shape)
            )
        else:
            noise = np.sqrt(noise_power) * self._rng.standard_normal(signal.shape)

        return signal + noise
