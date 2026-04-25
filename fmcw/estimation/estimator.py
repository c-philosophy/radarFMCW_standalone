"""Parameter estimator: converts detection peaks to physical units (range, velocity, angle)."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from fmcw.config.schema import RadarParams
from fmcw.utils.constants import C


@dataclass
class TargetEstimate:
    """Single target estimate for one frame."""

    range: float                     # Range (m)
    velocity: float                  # Radial velocity (m/s), + = approaching
    angle: float                     # Azimuth angle (degrees), 0 = boresight
    doppler_bin: int                 # Doppler FFT bin (raw, before unwrapping)
    range_bin: int                   # Range FFT bin
    angle_bin: int                   # Angle FFT bin (raw)
    amplitude: float                 # Signal magnitude at peak
    snr_db: float = 0.0              # Estimated SNR


@dataclass
class FrameEstimates:
    """All target estimates for one processing frame."""

    frame_idx: int
    targets: List[TargetEstimate]
    timestamp: float = 0.0           # Relative to start of scenario (s)


class ParameterEstimator:
    """Convert detection peaks to physical parameters.

    Uses standard FMCW formulas:
        R = (r_peak / N_sample * Fs) * c / (2 * S)
        v = (d_peak / N_chirp * 2*pi) * lambda / (4 * pi * Tc)
        theta = arcsin(lambda * (a_peak / N_angle * 2*pi) / (2 * pi * d))
    """

    def __init__(self, radar: RadarParams):
        self.radar = radar

    def estimate(
        self,
        s_rda: np.ndarray,
        filtered_peaks: np.ndarray,
        frame_idx: int = 0,
        timestamp: float = 0.0,
        ground_truth: Optional[List] = None,
    ) -> FrameEstimates:
        """Convert peak indices to physical parameters.

        Args:
            s_rda: 3D complex FFT cube of shape (angle_bins, D_bins, R_bins).
            filtered_peaks: (N, 2) array of [doppler_idx, range_idx].
            frame_idx: Frame index.
            timestamp: Frame timestamp (s).
            ground_truth: Optional list of TargetParams for SNR calculation.

        Returns:
            FrameEstimates with all estimated targets.
        """
        targets = []
        for d_peak, r_peak in filtered_peaks:
            # Range
            r_est = self._range_from_bin(int(r_peak))

            # Velocity
            v_est, d_unwrapped = self._velocity_from_bin(int(d_peak))

            # Angle
            a_peak, a_est = self._angle_from_peak(s_rda, int(d_unwrapped), int(r_peak))

            # Amplitude
            amp_bin = a_peak % self.radar.angle_num
            amp = float(np.abs(s_rda[amp_bin, int(d_peak), int(r_peak)]))

            # SNR estimate (simple: peak / median background)
            rd_slice = np.abs(s_rda[:, int(d_peak), int(r_peak)])
            noise_floor = np.median(rd_slice)
            snr = 20.0 * np.log10(max(amp / (noise_floor + 1e-30), 1.0))

            targets.append(TargetEstimate(
                range=r_est,
                velocity=v_est,
                angle=a_est,
                doppler_bin=int(d_peak),
                range_bin=int(r_peak),
                angle_bin=a_peak,
                amplitude=amp,
                snr_db=round(snr, 1),
            ))

        return FrameEstimates(
            frame_idx=frame_idx,
            targets=targets,
            timestamp=timestamp,
        )

    # --- internal conversion methods ---

    def _range_from_bin(self, r_bin: int) -> float:
        """Convert range FFT bin to range in meters."""
        delta_f = r_bin / self.radar.sample_num * self.radar.Fs
        return delta_f * C / (2.0 * self.radar.S)

    def _velocity_from_bin(self, d_bin: int) -> Tuple[float, int]:
        """Convert Doppler bin to velocity (m/s).

        Handles negative velocities (spectrum fold-over).
        Returns (velocity, unwrapped_bin).
        """
        unwrapped = d_bin
        if d_bin >= self.radar.chirp_num / 2:
            unwrapped = d_bin - self.radar.chirp_num
        delta_phi = unwrapped / self.radar.chirp_num * 2.0 * np.pi
        v = delta_phi * self.radar.wl / (4.0 * np.pi * self.radar.Tc)
        return v, unwrapped

    def _angle_from_peak(
        self,
        s_rda: np.ndarray,
        d_peak: int,
        r_peak: int,
    ) -> Tuple[int, float]:
        """Find angle bin and estimate angle in degrees."""
        angle_spectrum = np.abs(s_rda[:, d_peak, r_peak])
        a_peak = int(np.argmax(angle_spectrum))

        a_unwrapped = a_peak
        if a_peak >= self.radar.angle_num / 2:
            a_unwrapped = a_peak - self.radar.angle_num

        delta_phi = a_unwrapped / self.radar.angle_num * 2.0 * np.pi
        sin_theta = self.radar.wl * delta_phi / (2.0 * np.pi * self.radar.antenna_spacing)
        sin_theta = np.clip(sin_theta, -1.0, 1.0)
        angle_rad = np.arcsin(sin_theta)
        return a_peak, float(np.degrees(angle_rad))
