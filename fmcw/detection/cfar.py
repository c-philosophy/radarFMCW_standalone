"""CFAR（恒虚警率）检测器。

支持的变体：
    - CA-CFAR: 单元平均（默认，支持向量化全图 2D 检测）
    - OS-CFAR: 有序统计
    - GO-CFAR: 最大选择
    - SO-CFAR: 最小选择

检测模式：
    - detect():  全图 2D 滑动窗 CFAR（标准模式，统计正确）
        - CA-CFAR 使用 scipy uniform_filter 向量化实现，O(n²) 复杂度
        - OS/GO/SO-CFAR 使用 CA 预筛 + 候选点精确验证的两级方案
    - filter():  峰值门控模式（legacy，仅对指定候选点做 CFAR 验证）

在 AlgorithmRegistry 的 "cfar" 类别下注册。

门限因子 alpha：
    当 alpha=None 时自动从虚警概率 Pfa 和参考窗参数计算。
    计算由 fmcw.detection.cfar_alpha 模块完成，支持：
        - 蒙特卡洛仿真 + 预计算查找表（默认，精确建模 2D 参考窗）
        - 解析/数值公式（回退方案）
    若显式传入 alpha 值则直接使用（向后兼容）。

参考：
    - Zhang et al., "A Low-Complexity Target Detection Technique Using
      the Prefix Sum Algorithm", IET Electronics Letters, 2024.
    - TI mmWave SDK, CFAR DPU implementation.
    - Richards, "Fundamentals of Radar Signal Processing".
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from scipy.ndimage import uniform_filter

from fmcw.core.registry import register_algorithm
from fmcw.detection.cfar_alpha import compute_alpha as _compute_alpha


# ---------------------------------------------------------------------------
# 向量化 2D CA-CFAR 核心实现
# ---------------------------------------------------------------------------

def _ca_cfar_2d_vectorized(
    rd_map: np.ndarray,
    guard_cells: int,
    reference_cells: int,
    alpha: float,
) -> np.ndarray:
    """全图 2D CA-CFAR 检测（向量化，scipy uniform_filter）。

    利用 box filter（等价于积分图/前缀和）将传统的 O(n⁴) 逐 cell
    参考窗求和降至 O(n²)。在 Python 中借助 scipy 的 C 实现，对
    512×256 RD 图约耗时 1-3 ms。

    Args:
        rd_map: 2D Range-Doppler 幅度图 (D, R)。
        guard_cells: 保护带半宽。
        reference_cells: 参考窗半宽。
        alpha: CFAR 门限因子。

    Returns:
        (M, 2) 检测点 [doppler_idx, range_idx] 数组。
    """
    D, R = rd_map.shape
    half = guard_cells + reference_cells

    # 窗口尺寸
    ref_size = 2 * half + 1      # 外矩形边长
    guard_size = 2 * guard_cells + 1  # 保护带边长

    # Box filter: 计算每个 cell 的矩形窗口均值（scipy C 实现，极快）
    # mode='nearest' 用边缘值填充，避免填 0 导致边界噪声低估。
    outer_mean = uniform_filter(
        rd_map.astype(np.float64), size=ref_size,
        mode='nearest',
    )
    guard_mean = uniform_filter(
        rd_map.astype(np.float64), size=guard_size,
        mode='nearest',
    )

    # 参考单元均值 = (外矩形和 - 保护带和) / 参考单元数
    outer_area = float(ref_size ** 2)
    guard_area = float(guard_size ** 2)
    ref_area = outer_area - guard_area
    noise_map = (outer_mean * outer_area - guard_mean * guard_area) / ref_area

    # 门限检测（全图向量化比较）
    detection_mask = rd_map > alpha * noise_map

    # 排除保护带边界：guard 区域内的 cell 本身不能作为 CUT
    if guard_cells > 0:
        g = guard_cells
        detection_mask[:g, :] = False
        detection_mask[-g:, :] = False
        detection_mask[:, :g] = False
        detection_mask[:, -g:] = False

    return np.column_stack(np.where(detection_mask))


# ---------------------------------------------------------------------------
# CA 预筛（用于 OS/GO/SO 的两级检测）
# ---------------------------------------------------------------------------

def _ca_prescan(
    rd_map: np.ndarray,
    guard_cells: int,
    reference_cells: int,
    pfa: float = 0.1,
) -> np.ndarray:
    """CA-CFAR 预筛：用宽松门限快速获取候选点集。

    用于 OS/GO/SO-CFAR 的两级检测：先用 CA 快速扫描全图获取候选，
    再用精确变体验证。

    Args:
        rd_map: 2D Range-Doppler 幅度图。
        guard_cells: 保护带半宽。
        reference_cells: 参考窗半宽。
        pfa: 预筛虚警概率（默认 0.1，非常宽松）。

    Returns:
        (M, 2) 候选点坐标数组。
    """
    alpha = _compute_alpha('ca', pfa, guard_cells, reference_cells,
                           method='analytic')
    return _ca_cfar_2d_vectorized(rd_map, guard_cells, reference_cells, alpha)


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------

class BaseCFAR(ABC):
    """CFAR 检测器抽象基类。

    子类必须实现:
        - compute_alpha(pfa, guard_cells, reference_cells, **kwargs)
        - filter(rd_map, peak_indices, ...)  [向后兼容]
        - detect(rd_map, ...)  [标准全图检测]
    """

    @staticmethod
    @abstractmethod
    def compute_alpha(
        pfa: float,
        guard_cells: int,
        reference_cells: int,
        **kwargs,
    ) -> float:
        """从目标 Pfa 推导门限因子 alpha。"""
        ...

    def _resolve_alpha(
        self,
        alpha: Optional[float],
        pfa: float,
        guard_cells: int,
        reference_cells: int,
        **kwargs,
    ) -> float:
        """解析 alpha：显式传入则直接用，否则从 pfa 自动计算。"""
        if alpha is not None:
            return float(alpha)
        return self.compute_alpha(
            pfa, guard_cells, reference_cells, **kwargs,
        )

    def detect(
        self,
        rd_map: np.ndarray,
        guard_cells: int = 2,
        reference_cells: int = 8,
        alpha: Optional[float] = None,
        pfa: float = 1e-4,
    ) -> np.ndarray:
        """全图 2D CFAR 检测（标准模式）。

        对 RD 图的每一个有效 cell 执行 CFAR 检验。
        CA-CFAR 使用向量化实现；OS/GO/SO-CFAR 使用 CA 预筛 + 精确验证。

        Args:
            rd_map: 2D Range-Doppler 幅度图 (D, R)。
            guard_cells: 保护带半宽。
            reference_cells: 参考窗半宽。
            alpha: 门限因子。None = 从 pfa 自动计算。
            pfa: 目标虚警概率。

        Returns:
            (M, 2) 检测点 [doppler_idx, range_idx] 数组。
        """
        # 默认实现：回退到 filter()，由子类覆写
        # 使用 CA 预筛获取候选，然后 filter 验证
        candidates = _ca_prescan(rd_map, guard_cells, reference_cells,
                                 pfa=0.1)
        if len(candidates) == 0:
            return np.empty((0, 2), dtype=np.int64)
        return self.filter(
            rd_map, candidates,
            guard_cells=guard_cells,
            reference_cells=reference_cells,
            alpha=alpha,
            pfa=pfa,
        )

    @abstractmethod
    def filter(
        self,
        rd_map: np.ndarray,
        peak_indices: np.ndarray,
        guard_cells: int,
        reference_cells: int,
        alpha: Optional[float],
        pfa: float = 1e-4,
    ) -> np.ndarray:
        """峰值门控 CFAR（legacy 模式）。

        仅在指定的候选峰值位置执行 CFAR 检验。
        保留此方法以确保向后兼容。

        Args:
            rd_map: 2D Range-Doppler 幅度图 (D, R)。
            peak_indices: 候选峰值 (N, 2) [doppler_idx, range_idx]。
            guard_cells: 保护带半宽。
            reference_cells: 参考窗半宽。
            alpha: 门限因子。None = 从 pfa 自动计算。
            pfa: 目标虚警概率。

        Returns:
            (M, 2) 通过 CFAR 检验的检测点数组。
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
        """提取 CUT 周围的参考窗和保护窗。

        Returns:
            (net_sum, net_count) 元组。net_sum < 0 表示窗越界。
        """
        D, R = rd_map.shape

        r_min = r_idx - guard - ref
        r_max = r_idx + guard + ref
        d_min = d_idx - guard - ref
        d_max = d_idx + guard + ref

        if not (0 <= r_min and r_max < R and 0 <= d_min and d_max < D):
            return -1.0, 0

        ref_window = rd_map[d_min:d_max + 1, r_min:r_max + 1]
        ref_sum = np.sum(ref_window)
        ref_area = (2 * (guard + ref) + 1) ** 2

        g_r_min = r_idx - guard
        g_r_max = r_idx + guard
        g_d_min = d_idx - guard
        g_d_max = d_idx + guard
        guard_sum = np.sum(rd_map[g_d_min:g_d_max + 1, g_r_min:g_r_max + 1])
        guard_area = (2 * guard + 1) ** 2

        net_sum = ref_sum - guard_sum
        net_count = ref_area - guard_area

        return net_sum, net_count


# ---------------------------------------------------------------------------
# CA-CFAR
# ---------------------------------------------------------------------------

@register_algorithm("cfar", "ca_cfar")
class CACFAR(BaseCFAR):
    """单元平均 CFAR（Cell-Averaging CFAR）。

    全图检测使用 scipy uniform_filter 向量化实现，
    复杂度 O(n²)，对 512×256 RD 图约 1-3 ms。
    """

    @staticmethod
    def compute_alpha(
        pfa: float,
        guard_cells: int,
        reference_cells: int,
        method: str = 'auto',
        **kwargs,
    ) -> float:
        """CA-CFAR alpha 计算：精确闭式解 α = N·(Pfa^(-1/N) - 1)。"""
        return _compute_alpha('ca', pfa, guard_cells, reference_cells,
                              method=method)

    def detect(
        self,
        rd_map: np.ndarray,
        guard_cells: int = 2,
        reference_cells: int = 8,
        alpha: Optional[float] = None,
        pfa: float = 1e-4,
    ) -> np.ndarray:
        """全图向量化 2D CA-CFAR 检测。"""
        alpha = self._resolve_alpha(alpha, pfa, guard_cells, reference_cells)
        return _ca_cfar_2d_vectorized(
            rd_map, guard_cells, reference_cells, alpha,
        )

    def filter(
        self,
        rd_map: np.ndarray,
        peak_indices: np.ndarray,
        guard_cells: int = 2,
        reference_cells: int = 8,
        alpha: Optional[float] = None,
        pfa: float = 1e-4,
    ) -> np.ndarray:
        """峰值门控 CA-CFAR（legacy）。"""
        alpha = self._resolve_alpha(alpha, pfa, guard_cells, reference_cells)
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


# ---------------------------------------------------------------------------
# OS-CFAR
# ---------------------------------------------------------------------------

@register_algorithm("cfar", "os_cfar")
class OSCFAR(BaseCFAR):
    """有序统计 CFAR（Ordered-Statistics CFAR）。

    全图检测使用 CA 预筛 + 候选点精确 OS 验证的两级方案。
    """

    def __init__(self, rank_ratio: float = 0.75):
        """Args:
            rank_ratio: 有序统计量的秩比例（0.75 = 75th percentile）。
        """
        self.rank_ratio = rank_ratio

    @staticmethod
    def compute_alpha(
        pfa: float,
        guard_cells: int,
        reference_cells: int,
        rank_ratio: float = 0.75,
        method: str = 'auto',
        **kwargs,
    ) -> float:
        """OS-CFAR alpha 计算：二分法数值求解 Rohling 方程。"""
        return _compute_alpha('os', pfa, guard_cells, reference_cells,
                              method=method, rank_ratio=rank_ratio)

    def filter(
        self,
        rd_map,
        peak_indices,
        guard_cells=2,
        reference_cells=8,
        alpha: Optional[float] = None,
        pfa=1e-4,
    ):
        """峰值门控 OS-CFAR（也用于 detect() 的第二级验证）。"""
        alpha = self._resolve_alpha(
            alpha, pfa, guard_cells, reference_cells,
            rank_ratio=self.rank_ratio,
        )
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
            all_cells = ref_window.flatten()
            guard_mask = np.ones(
                (2 * (guard + ref) + 1, 2 * (guard + ref) + 1), dtype=bool,
            )
            guard_mask[guard:guard + 2 * guard + 1,
                       guard:guard + 2 * guard + 1] = False
            ref_cells = all_cells[guard_mask.flatten()]

            k = max(1, int(len(ref_cells) * self.rank_ratio))
            noise_est = np.partition(ref_cells, k - 1)[k - 1]
            cut_value = rd_map[d_idx, r_idx]

            if cut_value > noise_est * alpha:
                filtered.append([d_idx, r_idx])

        return np.array(filtered, dtype=np.int64)


# ---------------------------------------------------------------------------
# GO-CFAR
# ---------------------------------------------------------------------------

@register_algorithm("cfar", "go_cfar")
class GOCFAR(BaseCFAR):
    """最大选择 CFAR（Greatest-Of CFAR）。

    全图检测使用 CA 预筛 + 候选点精确 GO 验证的两级方案。
    """

    @staticmethod
    def compute_alpha(
        pfa: float,
        guard_cells: int,
        reference_cells: int,
        method: str = 'auto',
        **kwargs,
    ) -> float:
        """GO-CFAR alpha 计算：二分法数值求解 Gandhi & Kassam 方程。"""
        return _compute_alpha('go', pfa, guard_cells, reference_cells,
                              method=method)

    def filter(
        self,
        rd_map: np.ndarray,
        peak_indices: np.ndarray,
        guard_cells: int = 2,
        reference_cells: int = 8,
        alpha: Optional[float] = None,
        pfa: float = 1e-4,
    ) -> np.ndarray:
        """峰值门控 GO-CFAR（也用于 detect() 的第二级验证）。"""
        alpha = self._resolve_alpha(alpha, pfa, guard_cells, reference_cells)
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
            noise_est = max(left_mean, right_mean)
            cut_value = rd_map[d_idx, r_idx]

            if noise_est > 0 and cut_value > noise_est * alpha:
                filtered.append([d_idx, r_idx])

        return np.array(filtered, dtype=np.int64)


# ---------------------------------------------------------------------------
# SO-CFAR
# ---------------------------------------------------------------------------

@register_algorithm("cfar", "so_cfar")
class SOCFAR(BaseCFAR):
    """最小选择 CFAR（Smallest-Of CFAR）。

    全图检测使用 CA 预筛 + 候选点精确 SO 验证的两级方案。
    """

    @staticmethod
    def compute_alpha(
        pfa: float,
        guard_cells: int,
        reference_cells: int,
        method: str = 'auto',
        **kwargs,
    ) -> float:
        """SO-CFAR alpha 计算：近似闭式解 α = (N/2)·(Pfa^(-2/N) - 1)。"""
        return _compute_alpha('so', pfa, guard_cells, reference_cells,
                              method=method)

    def filter(
        self,
        rd_map,
        peak_indices,
        guard_cells=2,
        reference_cells=8,
        alpha: Optional[float] = None,
        pfa=1e-4,
    ):
        """峰值门控 SO-CFAR（也用于 detect() 的第二级验证）。"""
        alpha = self._resolve_alpha(alpha, pfa, guard_cells, reference_cells)
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
            noise_est = (
                min(left_mean, right_mean)
                if left.size > 0 and right.size > 0
                else max(left_mean, right_mean)
            )
            cut_value = rd_map[d_idx, r_idx]

            if noise_est > 0 and cut_value > noise_est * alpha:
                filtered.append([d_idx, r_idx])

        return np.array(filtered, dtype=np.int64)
