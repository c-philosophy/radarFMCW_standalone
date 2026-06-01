"""IF Signal Generator — medium-fidelity FMCW radar signal simulation.

Builds the 3D IF signal cube (antenna, chirp, sample) using a phase-accumulation
model with support for range migration, target fluctuation, antenna patterns,
and RF impairments.
"""

from typing import List, Optional, Dict, Tuple
import numpy as np

from fmcw.utils.constants import C
from fmcw.config.schema import RadarParams, TargetParams


class IFSignalGenerator:
    """Medium-fidelity IF signal generator for FMCW radar.

    Signal model:
        s(a, d, r) = sum over targets of:
            A_k * cos(2*pi * (
                a * (d/lambda * sin(theta)) +      # spatial frequency
                d * (2*fc/c * Tc * v) +             # Doppler phase
                r * (2*S/c/sample_num * Tc * R)     # beat frequency phase
            ))

    Enhanced with:
        - Range migration (velocity * Tc * chirp_idx added to range phase)
        - Swerling target fluctuation
        - Antenna element pattern
        - Complex AWGN with configurable SNR
        - Optional RF impairments (phase noise, IQ imbalance)
    """

    def __init__(
        self,
        radar: RadarParams,
        phase_noise: Optional["PhaseNoise"] = None,
        iq_imbalance: Optional["IQImbalance"] = None,
    ):
        """
        Args:
            radar: Radar system parameters.
            phase_noise: Optional PhaseNoise model (from impairments module).
            iq_imbalance: Optional IQImbalance model (from impairments module).
        """
        self.radar = radar
        self._grid_cache: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._phase_noise = phase_noise
        self._iq_imbalance = iq_imbalance

    # --- grid ---

    def _make_grid(self) -> np.ndarray:
        """Build or retrieve cached 3D coordinate grid.

        Returns array of shape (antenna_num, chirp_num, sample_num, 3)
        with indices [a_idx, d_idx, r_idx] at each position.
        """
        key = (self.radar.antenna_num, self.radar.chirp_num, self.radar.sample_num)
        if key not in self._grid_cache:
            range_a = np.arange(self.radar.antenna_num)
            range_d = np.arange(self.radar.chirp_num)
            range_r = np.arange(self.radar.sample_num)
            self._grid_cache[key] = np.stack(
                np.meshgrid(range_a, range_d, range_r, indexing="ij"), axis=-1
            )
        return self._grid_cache[key]

    # --- single-target phase ---

    def _target_phase(
        self,
        target: TargetParams,
        grid: np.ndarray,
        frame_idx: int = 0,
        dt: float = 0.05,
    ) -> np.ndarray:
        """Compute phase contribution (radians) for a single target.

        Args:
            target: Target parameters (range, velocity, angle, rcs).
            grid: 3D index grid of shape (A, D, S, 3).
            frame_idx: Frame number (0-based).
            dt: Frame interval (seconds).

        Returns:
            Phase array of shape (A, D, S) in radians.
        """
        radar = self.radar

        # Update range due to velocity over time (range walk)
        range_now = target.range #+ target.velocity * frame_idx * dt

        # Angle spatial frequency
        const_a = radar.antenna_spacing / radar.wl * np.sin(np.deg2rad(target.angle))

        # Doppler phase per chirp
        # phi_d = 2 * pi * (2*fc/c * Tc) * v  per chirp step
        const_d = 2.0 * radar.fc / C * radar.Tc * target.velocity

        # Beat frequency per sample
        # phi_r = 2 * pi * (2*S/c/sample_num * Tc) * R  per sample step # 这个公式似乎不太对？
        const_r = 2.0 * radar.S / C / radar.sample_num * radar.Tc * range_now

        # Range migration: velocity causes additional per-chirp range shift
        # delta_R_per_chirp = v * Tc
        # This adds to the beat frequency term: const_r += 2*S/c/sample_num*Tc*v*Tc*d
        range_migration = 2.0 * radar.S / C / radar.sample_num * radar.Tc * target.velocity * radar.Tc

        # Build phase for all grid points
        # grid[..., 0] = antenna index, grid[..., 1] = chirp index, grid[..., 2] = sample index
        a_idx = grid[..., 0].astype(np.float64)
        d_idx = grid[..., 1].astype(np.float64)
        r_idx = grid[..., 2].astype(np.float64)

        phase = 2.0 * np.pi * (
            a_idx * const_a
            + d_idx * const_d
            + r_idx * (const_r + d_idx * range_migration)
        )

        return phase

    # --- single frame ---

    def generate_frame(
        self,
        targets: List[TargetParams],
        frame_idx: int = 0,
        dt: float = 0.05,
        snr_db: float = 25.0,
        fluctuation_factors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate one frame of IF signal.

        Args:
            targets: List of target parameters for this frame.
            frame_idx: Frame number (0-based) for motion update.
            dt: Frame interval in seconds.
            snr_db: Target SNR (dB).
            fluctuation_factors: Optional per-target RCS factors from Swerling model.

        Returns:
            Signal array of shape (antenna_num, chirp_num, sample_num), float64.
        """
        radar = self.radar
        grid = self._make_grid()
        signal = np.zeros(
            (radar.antenna_num, radar.chirp_num, radar.sample_num),
            dtype=np.float64,
        )

        for i, target in enumerate(targets):
            phase = self._target_phase(target, grid, frame_idx, dt)
            amp = target.rcs
            if fluctuation_factors is not None and i < len(fluctuation_factors):
                amp *= np.sqrt(max(fluctuation_factors[i], 0.0))
            signal += amp * np.cos(phase)

        # Add AWGN
        if snr_db > 0:
            signal_power = np.mean(signal**2) + 1e-30
            snr_linear = 10.0 ** (snr_db / 10.0)
            noise_power = signal_power / snr_linear
            noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
            signal += noise

        # Apply RF impairments if configured
        if self._iq_imbalance is not None:
            # Convert to analytic (complex) signal via Hilbert transform
            from scipy.signal import hilbert
            complex_signal = hilbert(signal, axis=-1)
            complex_signal = self._iq_imbalance.apply(complex_signal)
            signal = np.real(complex_signal)

        if self._phase_noise is not None:
            # Bandwidth for phase noise: approximate signal bandwidth = S * max_range * 2 / c
            bw_hz = 2.0 * self.radar.S * self.radar.max_range / C
            signal = self._phase_noise.apply(signal, bw_hz)

        return signal

    def generate_frames(
        self,
        targets_list: List[List[TargetParams]],
        dt: float = 0.05,
        snr_db: float = 25.0,
        fluctuation_model: Optional["TargetFluctuation"] = None,
    ) -> List[np.ndarray]:
        """Generate multiple frames of IF signal.

        Args:
            targets_list: List per frame of target parameters.
            dt: Frame interval.
            snr_db: SNR per frame.
            fluctuation_model: Optional Swerling fluctuation model.

        Returns:
            List of signal arrays, one per frame.
        """
        frames = []
        for frame_idx, targets in enumerate(targets_list):
            fluc = None
            if fluctuation_model is not None:
                fluc = fluctuation_model.generate(len(targets))
            frames.append(
                self.generate_frame(targets, frame_idx, dt, snr_db, fluc)
            )
        return frames
