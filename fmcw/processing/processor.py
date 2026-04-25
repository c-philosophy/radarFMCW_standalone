"""FMCW Processor: orchestrates the 3-stage FFT chain."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from fmcw.config.schema import RadarParams
from .range_fft import range_fft
from .doppler_fft import doppler_fft
from .angle_fft import angle_fft


@dataclass
class ProcessingResult:
    """Container for all FFT processing outputs."""

    s_r: np.ndarray      # Range FFT (half-spectrum), complex
    s_rd: np.ndarray     # Range-Doppler FFT, complex
    s_rda: np.ndarray    # Range-Doppler-Angle FFT, complex
    rd_map: np.ndarray   # Range-Doppler magnitude map (D, R)
    ra_map: np.ndarray   # Range-Angle magnitude map (A, R)
    radar_params: RadarParams


class FMCWProcessor:
    """Standard 3-stage FFT processing chain for FMCW radar.

    Pipeline:
        1. Range FFT (axis=2) → truncate half (real signal symmetry)
        2. Doppler FFT (axis=1)
        3. Angle FFT (axis=0, zero-padded to n_angle_bins)
        4. RD map = mean(|s_rda|, axis=0)
        5. RA map = mean(|s_rda|, axis=1)
    """

    def __init__(
        self,
        radar: RadarParams,
        downsample: int = 1,
        range_window: Optional[str] = None,
        doppler_window: Optional[str] = None,
        angle_window: Optional[str] = None,
    ):
        self.radar = radar
        self.downsample = downsample
        self.range_window = range_window
        self.doppler_window = doppler_window
        self.angle_window = angle_window

    def process(self, signal: np.ndarray) -> ProcessingResult:
        """Run the full FFT processing chain.

        Args:
            signal: Raw IF signal of shape (A, D, S), float64.

        Returns:
            ProcessingResult with all intermediate and final outputs.
        """
        # Stage 1: Range FFT
        s_r = range_fft(signal, self.downsample, self.range_window)

        # Stage 2: Doppler FFT
        s_rd = doppler_fft(s_r, self.doppler_window)

        # Stage 3: Angle FFT
        s_rda = angle_fft(s_rd, self.radar.angle_num, self.angle_window)

        # Compute magnitude maps
        rd_map = self._compute_rd_map(s_rda)
        ra_map = self._compute_ra_map(s_rda)

        return ProcessingResult(
            s_r=s_r,
            s_rd=s_rd,
            s_rda=s_rda,
            rd_map=rd_map,
            ra_map=ra_map,
            radar_params=self.radar,
        )

    def _compute_rd_map(self, s_rda: np.ndarray) -> np.ndarray:
        """Range-Doppler map: average magnitude over angle dimension."""
        return np.mean(np.abs(s_rda), axis=0).astype(np.float32)

    def _compute_ra_map(self, s_rda: np.ndarray) -> np.ndarray:
        """Range-Angle map: average magnitude over Doppler dimension."""
        return np.mean(np.abs(s_rda), axis=1).astype(np.float32)
