"""CFAR（恒虚警率）检测器 — 纯 NumPy 实现。

支持的变体：
    - CA-CFAR: 单元平均
    - OS-CFAR: 有序统计
    - GO-CFAR: 最大选择
    - SO-CFAR: 最小选择

所有变体在候选峰值位置而非全滑动窗口上操作，这是雷达检测的标准
"峰值门控"方法。

在 AlgorithmRegistry 的 "cfar" 类别下注册。
"""

from abc import ABC, abstractmethod
import numpy as np
from fmcw.core.registry import register_algorithm


class BaseCFAR(ABC):
    """Abstract base for CFAR detectors."""

    @abstractmethod
    def filter(
        self,
        rd_map: np.ndarray,
        peak_indices: np.ndarray,
        guard_cells: int,
        reference_cells: int,
        alpha: float,
        pfa: float = 1e-4,
    ) -> np.ndarray:
        """Apply CFAR thresholding at each peak location.

        Args:
            rd_map: 2D Range-Doppler magnitude map (D, R).
            peak_indices: Candidate peaks (N, 2) with [doppler_idx, range_idx].
            guard_cells: Half-width of guard band around CUT.
            reference_cells: Half-width of reference window beyond guard.
            alpha: Threshold scaling factor (for CA-CFAR based variants).
            pfa: Desired false-alarm probability (used for threshold derivation).

        Returns:
            Array of shape (M, 2) with peaks that pass the CFAR test.
        """
        ...

    def _extract_windows(
        self,
        rd_map: np.ndarray,
        d_idx: int,
        r_idx: int,
        guard: int,
        ref: int,
    ) -> tuple:
        """Extract reference and guard windows around a CUT.

        Returns:
            (ref_window_sum, ref_cell_count) tuple.
        """
        D, R = rd_map.shape

        r_min = r_idx - guard - ref
        r_max = r_idx + guard + ref
        d_min = d_idx - guard - ref
        d_max = d_idx + guard + ref

        if not (0 <= r_min and r_max < R and 0 <= d_min and d_max < D):
            return -1.0, 0

        # Reference window: entire outer rectangle
        ref_window = rd_map[d_min:d_max + 1, r_min:r_max + 1]
        ref_sum = np.sum(ref_window)
        ref_area = (2 * (guard + ref) + 1) ** 2

        # Guard window: inner rectangle
        g_r_min = r_idx - guard
        g_r_max = r_idx + guard
        g_d_min = d_idx - guard
        g_d_max = d_idx + guard
        guard_sum = np.sum(rd_map[g_d_min:g_d_max + 1, g_r_min:g_r_max + 1])
        guard_area = (2 * guard + 1) ** 2

        # Net reference sum and count (exclude guard + CUT)
        net_sum = ref_sum - guard_sum
        net_count = ref_area - guard_area

        return net_sum, net_count


@register_algorithm("cfar", "ca_cfar")
class CACFAR(BaseCFAR):
    """Cell-Averaging CFAR.

    Threshold = alpha * mean(reference cells).
    Computes net sum by (ref_window - guard_window) to avoid corner
    double-counting bugs present in original demo code.
    """

    def filter(
        self,
        rd_map: np.ndarray,
        peak_indices: np.ndarray,
        guard_cells: int = 2,
        reference_cells: int = 8,
        alpha: float = 20.0,
        pfa: float = 1e-4,
    ) -> np.ndarray:
        filtered = []
        for d_idx, r_idx in peak_indices:
            d_idx = int(d_idx)
            r_idx = int(r_idx)
            net_sum, net_count = self._extract_windows(
                rd_map, d_idx, r_idx, guard_cells, reference_cells,
            )
            if net_count <= 0:
                continue

            noise_mean = net_sum / net_count
            cut_value = rd_map[d_idx, r_idx]

            if cut_value > noise_mean * alpha:
                filtered.append([d_idx, r_idx])

        return np.array(filtered, dtype=np.int64)


@register_algorithm("cfar", "os_cfar")
class OSCFAR(BaseCFAR):
    """Ordered-Statistics CFAR.

    Uses the k-th smallest value from sorted reference cells as the
    noise estimate. More robust to interfering targets in the reference
    window (avoids elevation of the mean by nearby targets).
    """

    def __init__(self, rank_ratio: float = 0.75):
        """
        Args:
            rank_ratio: Fraction of reference cells to use as rank (0.75 = 75th percentile).
        """
        self.rank_ratio = rank_ratio

    def filter(
        self,
        rd_map,
        peak_indices,
        guard_cells=2,
        reference_cells=8,
        alpha=18.0,
        pfa=1e-4,
    ):
        filtered = []
        for d_idx, r_idx in peak_indices:
            d_idx = int(d_idx)
            r_idx = int(r_idx)
            D, R = rd_map.shape
            guard = guard_cells
            ref = reference_cells

            r_min = r_idx - guard - ref
            r_max = r_idx + guard + ref
            d_min = d_idx - guard - ref
            d_max = d_idx + guard + ref

            if not (0 <= r_min and r_max < R and 0 <= d_min and d_max < D):
                continue

            ref_window = rd_map[d_min:d_max + 1, r_min:r_max + 1]
            ref_size = (2 * (guard + ref) + 1) ** 2

            # Exclude guard + CUT
            g_r_min = r_idx - guard
            g_r_max = r_idx + guard
            g_d_min = d_idx - guard
            g_d_max = d_idx + guard

            # Flatten and remove guard cells
            all_cells = ref_window.flatten()
            guard_mask = np.ones((2 * (guard + ref) + 1, 2 * (guard + ref) + 1), dtype=bool)
            guard_mask[guard:guard + 2 * guard + 1, guard:guard + 2 * guard + 1] = False
            ref_cells = all_cells[guard_mask.flatten()]

            k = max(1, int(len(ref_cells) * self.rank_ratio))
            noise_est = np.partition(ref_cells, k - 1)[k - 1]
            cut_value = rd_map[d_idx, r_idx]

            if cut_value > noise_est * alpha:
                filtered.append([d_idx, r_idx])

        return np.array(filtered, dtype=np.int64)


@register_algorithm("cfar", "go_cfar")
class GOCFAR(BaseCFAR):
    """Greatest-Of CFAR.

    Divides reference window into leading and trailing halves (in range
    or Doppler dimension), takes the MAX of their means. Robust to
    clutter edges (prevents elevated false alarms at transitions).
    """

    def filter(
        self,
        rd_map: np.ndarray,
        peak_indices: np.ndarray,
        guard_cells: int = 2,
        reference_cells: int = 8,
        alpha: float = 20.0,
        pfa: float = 1e-4,
    ) -> np.ndarray:
        filtered = []
        for d_idx, r_idx in peak_indices:
            d_idx = int(d_idx)
            r_idx = int(r_idx)
            D, R = rd_map.shape
            guard = guard_cells
            ref = reference_cells

            r_min = r_idx - guard - ref
            r_max = r_idx + guard + ref
            d_min = d_idx - guard - ref
            d_max = d_idx + guard + ref

            if not (0 <= r_min and r_max < R and 0 <= d_min and d_max < D):
                continue

            # Split into left/right halves along range axis
            left = rd_map[d_min:d_max + 1, r_min:r_idx - guard]
            right = rd_map[d_min:d_max + 1, r_idx + guard + 1:r_max + 1]

            left_mean = np.mean(left) if left.size > 0 else 0.0
            right_mean = np.mean(right) if right.size > 0 else 0.0
            noise_est = max(left_mean, right_mean)
            cut_value = rd_map[d_idx, r_idx]

            if noise_est > 0 and cut_value > noise_est * alpha:
                filtered.append([d_idx, r_idx])

        return np.array(filtered, dtype=np.int64)


@register_algorithm("cfar", "so_cfar")
class SOCFAR(BaseCFAR):
    """Smallest-Of CFAR.

    Takes the MIN of leading/trailing means. Better for detecting
    closely-spaced targets (won't suppress the weaker one), but more
    sensitive to clutter edges.
    """

    def filter(
        self,
        rd_map,
        peak_indices,
        guard_cells=2,
        reference_cells=8,
        alpha=15.0,
        pfa=1e-4,
    ):
        filtered = []
        for d_idx, r_idx in peak_indices:
            d_idx = int(d_idx)
            r_idx = int(r_idx)
            D, R = rd_map.shape
            guard = guard_cells
            ref = reference_cells

            r_min = r_idx - guard - ref
            r_max = r_idx + guard + ref
            d_min = d_idx - guard - ref
            d_max = d_idx + guard + ref

            if not (0 <= r_min and r_max < R and 0 <= d_min and d_max < D):
                continue

            left = rd_map[d_min:d_max + 1, r_min:r_idx - guard]
            right = rd_map[d_min:d_max + 1, r_idx + guard + 1:r_max + 1]

            left_mean = np.mean(left) if left.size > 0 else 0.0
            right_mean = np.mean(right) if right.size > 0 else 0.0
            noise_est = min(left_mean, right_mean) if left.size > 0 and right.size > 0 else max(left_mean, right_mean)
            cut_value = rd_map[d_idx, r_idx]

            if noise_est > 0 and cut_value > noise_est * alpha:
                filtered.append([d_idx, r_idx])

        return np.array(filtered, dtype=np.int64)
