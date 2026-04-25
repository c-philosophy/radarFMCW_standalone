"""Detection pipeline: peak finding + CFAR filtering."""

import numpy as np

from fmcw.core.registry import AlgorithmRegistry
from .peak_finder import PeakFinder
from .cfar import BaseCFAR


class DetectionPipeline:
    """Run peak detection followed by CFAR filtering.

    Usage:
        det = DetectionPipeline(peak_finder=PeakFinder(), cfar_algorithm="ca_cfar")
        peaks = det.run(rd_map, guard_cells=2, reference_cells=8, alpha=20)
    """

    def __init__(
        self,
        peak_finder: PeakFinder = None,
        cfar_algorithm: str = "ca_cfar",
    ):
        self.peak_finder = peak_finder or PeakFinder()
        self._cfar = AlgorithmRegistry.create("cfar", cfar_algorithm)

    def run(
        self,
        rd_map: np.ndarray,
        guard_cells: int = 2,
        reference_cells: int = 8,
        alpha: float = 20.0,
        pfa: float = 1e-4,
    ) -> np.ndarray:
        """Detect targets in the RD map.

        Args:
            rd_map: 2D Range-Doppler magnitude map.
            guard_cells: CFAR guard band half-width.
            reference_cells: CFAR reference window half-width.
            alpha: CFAR threshold factor.
            pfa: Target false-alarm probability.

        Returns:
            Array of shape (N, 2) with filtered [doppler_idx, range_idx].
        """
        # Stage 1: Find candidate peaks
        candidates = self.peak_finder.detect(rd_map)
        if len(candidates) == 0:
            return np.empty((0, 2), dtype=np.int64)

        # Stage 2: CFAR filtering
        filtered = self._cfar.filter(
            rd_map,
            candidates,
            guard_cells=guard_cells,
            reference_cells=reference_cells,
            alpha=alpha,
            pfa=pfa,
        )
        return filtered

    @property
    def cfar_name(self) -> str:
        return self._cfar.__class__.__name__
