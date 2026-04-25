"""FMCW chirp waveform parameter calculation."""

from dataclasses import dataclass
import numpy as np

from fmcw.utils.constants import C


@dataclass
class Waveform:
    """FMCW chirp waveform parameters (derived from radar params)."""

    fc: float
    B: float
    Tc: float
    S: float          # chirp slope Hz/s
    wl: float         # wavelength
    Fs: float         # sampling rate

    @classmethod
    def from_radar_params(cls, radar):
        """Create Waveform from a RadarParams instance."""
        return cls(
            fc=radar.fc,
            B=radar.B,
            Tc=radar.Tc,
            S=radar.S,
            wl=radar.wl,
            Fs=radar.Fs,
        )

    def beat_frequency(self, range_m: float) -> float:
        """Beat frequency for a target at given range.

        f_IF = 2 * S * R / c
        """
        return 2.0 * self.S * range_m / C

    def doppler_shift(self, velocity_ms: float) -> float:
        """Doppler frequency shift for radial velocity.

        f_d = 2 * v / lambda
        """
        return 2.0 * velocity_ms / self.wl

    def range_from_beat(self, beat_hz: float) -> float:
        """Convert beat frequency back to range."""
        return beat_hz * C / (2.0 * self.S)

    def velocity_from_doppler(self, doppler_hz: float) -> float:
        """Convert Doppler frequency to velocity."""
        return doppler_hz * self.wl / 2.0

    def phase_rotation_per_chirp(self, velocity_ms: float) -> float:
        """Phase advance per chirp due to target motion.

        delta_phi = 4 * pi * v * Tc / lambda
        """
        return 4.0 * np.pi * velocity_ms * self.Tc / self.wl

    def angle_steering_vector(self, n_antennas: int, angle_deg: float) -> np.ndarray:
        """Steering vector for a given angle.

        a(theta) = exp(-j * 2*pi * d/lambda * sin(theta) * [0, 1, ..., N-1])
                 = exp(-j * pi * sin(theta) * [0, 1, ..., N-1])  for d = lambda/2
        """
        theta_rad = np.deg2rad(angle_deg)
        n = np.arange(n_antennas)
        return np.exp(-1j * np.pi * np.sin(theta_rad) * n)
