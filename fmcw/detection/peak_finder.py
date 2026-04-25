"""2D peak detection using morphological filtering."""

import numpy as np
from scipy.ndimage import maximum_filter, generate_binary_structure, binary_erosion


class PeakFinder:
    """Detect local maxima in a 2D map using morphological filtering.

    Uses scipy.ndimage for efficient peak detection: each point must be
    the maximum within its neighborhood to be a candidate peak.
    Background (zero-valued) regions are excluded via binary erosion.
    """

    def __init__(self, neighborhood_size: int = 2):
        """
        Args:
            neighborhood_size: Size of 2D connectivity (2 = 8-connected).
        """
        self.structure = generate_binary_structure(2, neighborhood_size)

    def detect(self, rd_map: np.ndarray) -> np.ndarray:
        """Find local maxima in the Range-Doppler map.

        Args:
            rd_map: 2D magnitude map of shape (doppler_bins, range_bins).

        Returns:
            Array of shape (N, 2) with [doppler_idx, range_idx] for each peak.
        """
        data = np.asarray(rd_map, dtype=np.float32)

        # Points that equal the local maximum in their neighborhood
        local_max = maximum_filter(data, footprint=self.structure) == data

        # Exclude zero-value background
        background = data == 0
        eroded_bg = binary_erosion(
            background, structure=self.structure, border_value=1
        )

        # Peaks = local maxima that are NOT in the eroded background boundary
        peaks_mask = local_max ^ eroded_bg

        indices = np.nonzero(peaks_mask)
        return np.column_stack([indices[0], indices[1]])
