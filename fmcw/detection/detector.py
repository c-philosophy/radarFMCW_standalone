"""检测流水线：全图 2D CFAR 检测 + 峰值分组。

标准流程（推荐）：
    1. 全图 2D CFAR 滑动窗检测 — 统计正确，精确控制 Pfa
    2. 角度多峰检测 — 对每个 RD cell 查多个角度峰（方案 A1）
    3. 3D 峰值分组 — 在 (A, D, R) 空间合并 FFT 旁瓣

    - CA-CFAR: 向量化 uniform_filter 实现，O(n²)，~1-3 ms
    - OS/GO/SO-CFAR: CA 预筛 + 候选点精确验证的两级方案

峰值门控模式（legacy，peak_finder 参数非 None 时启用）
"""

from typing import Optional, List, TYPE_CHECKING
import numpy as np
from scipy.ndimage import label, generate_binary_structure

from fmcw.core.registry import AlgorithmRegistry
from .peak_finder import PeakFinder
from .cfar import BaseCFAR

if TYPE_CHECKING:
    from fmcw.estimation.estimator import TargetEstimate


def _peak_grouping(
    rd_map: np.ndarray,
    detections: np.ndarray,
    connectivity: int = 2,
) -> np.ndarray:
    """CFAR 后置峰值分组：合并同一目标的相邻检测点。

    FMCW 雷达经 FFT 处理后，单个点目标在 RD 图上表现为二维 sinc 函数，
    2D CFAR 会检出主瓣及旁瓣内所有超门限的 cell。本函数通过连通域分析
    将相邻检测点归为一组，每组取 RD 幅度最大的 cell 作为代表。

    Args:
        rd_map: 2D Range-Doppler 幅度图 (D, R)。
        detections: (M, 2) 检测点 [doppler_idx, range_idx] 数组。
        connectivity: 连通域结构 (2=8连通, 1=4连通)。

    Returns:
        (N, 2) 分组后的代表性检测点数组，N <= M。
    """
    if len(detections) <= 1:
        return detections

    # 创建二值掩码并标记连通域
    mask = np.zeros(rd_map.shape, dtype=bool)
    d_idx = detections[:, 0].astype(np.intp)
    r_idx = detections[:, 1].astype(np.intp)
    mask[d_idx, r_idx] = True

    structure = generate_binary_structure(2, connectivity)
    labeled, n_groups = label(mask, structure=structure)

    # if n_groups <= 1:
    #     return detections

    # 每组取 RD 幅度最大的 cell
    grouped = np.empty((n_groups, 2), dtype=np.int64)
    for gid in range(1, n_groups + 1):
        gy, gx = np.where(labeled == gid)
        # 子数组中的最大值索引
        local_max_idx = np.argmax(rd_map[gy, gx])
        grouped[gid - 1] = [gy[local_max_idx], gx[local_max_idx]]

    return grouped


def _group_estimates_3d(
    estimates: list,
    rd_map: np.ndarray,
    angle_num: int,
    connectivity: int = 3,
) -> list:
    """3D (Angle, Doppler, Range) 峰值分组。

    在角度多峰检测后，同一 RD cell 可能产生多个 TargetEstimate
    （不同角度）。本函数在 (A, D, R) 三维空间中做连通域分析，
    合并 FFT 旁瓣产生的相邻检测，保留每组的最大幅度代表。

    Args:
        estimates: List of TargetEstimate objects.
        rd_map: 2D Range-Doppler 幅度图（用于取局部最大值时的参考）。
        angle_num: 角度 bin 总数（用于确定 3D mask 尺寸）。
        connectivity: 连通域结构 (1=6连通, 2=18连通, 3=26连通)。

    Returns:
        List of TargetEstimate（分组后）。
    """
    if len(estimates) <= 1:
        return estimates

    D, R = rd_map.shape
    mask_3d = np.zeros((angle_num, D, R), dtype=bool)
    est_by_bin = {}  # (a, d, r) → TargetEstimate

    for est in estimates:
        d = max(0, min(D - 1, est.doppler_bin))
        r = max(0, min(R - 1, est.range_bin))
        # 超分辨方法的 angle_bin = -1：从物理角度反算 bin 位置
        if est.angle_bin >= 0:
            a = est.angle_bin
        else:
            # sin(θ) = 2*(a - N/2)/N → a = N/2 + N*sin(θ)/2
            sin_theta = np.sin(np.deg2rad(est.angle))
            a = int(np.round(angle_num / 2 + angle_num * sin_theta / 2))
        a = max(0, min(angle_num - 1, a))
        mask_3d[a, d, r] = True
        key = (a, d, r)
        if key not in est_by_bin or est.amplitude > est_by_bin[key].amplitude:
            est_by_bin[key] = est

    structure = generate_binary_structure(3, connectivity)
    labeled, n_groups = label(mask_3d, structure=structure)

    grouped = []
    for gid in range(1, n_groups + 1):
        ga, gd, gr = np.where(labeled == gid)
        # 每组取幅度最大的 TargetEstimate
        best_amp = -1.0
        best_est = None
        for ai, di, ri in zip(ga, gd, gr):
            key = (ai, di, ri)
            if key in est_by_bin and est_by_bin[key].amplitude > best_amp:
                best_amp = est_by_bin[key].amplitude
                best_est = est_by_bin[key]
        if best_est is not None:
            grouped.append(best_est)

    return grouped


class DetectionPipeline:
    """CFAR 目标检测流水线。

    Usage:
        # 标准模式：全图 2D CFAR + 峰值分组（推荐）
        det = DetectionPipeline(cfar_algorithm="ca_cfar")
        peaks = det.run(rd_map, pfa=1e-4)
        # peaks 中每个目标只产生 ~1 个检测点

        # 原始检测点（禁用分组，调试用）
        det = DetectionPipeline(cfar_algorithm="ca_cfar", enable_grouping=False)
        raw_peaks = det.run(rd_map)

        # 峰值门控模式（legacy）
        det = DetectionPipeline(cfar_algorithm="ca_cfar", peak_finder=PeakFinder())
        peaks = det.run(rd_map, alpha=2.0)
    """

    def __init__(
        self,
        cfar_algorithm: str = "ca_cfar",
        peak_finder: Optional[PeakFinder] = None,
        enable_grouping: bool = False,
    ):
        """初始化检测流水线。

        Args:
            cfar_algorithm: CFAR 算法名
                            ("ca_cfar", "os_cfar", "go_cfar", "so_cfar")。
            peak_finder: 可选峰值检测器。None = 全图 2D CFAR（推荐）。
                         传入 PeakFinder 实例则启用峰值门控 legacy 模式。
            enable_grouping: 是否启用峰值分组（仅标准模式生效）。
                             合并同一目标因 FFT 旁瓣产生的多个相邻检测点。
        """
        self.peak_finder = peak_finder
        self.enable_grouping = enable_grouping
        self._cfar: BaseCFAR = AlgorithmRegistry.create("cfar", cfar_algorithm)

    def run(
        self,
        rd_map: np.ndarray,
        guard_cells: int = 2,
        reference_cells: int = 8,
        alpha: Optional[float] = None,
        pfa: float = 1e-4,
    ) -> np.ndarray:
        """对 RD 图执行目标检测。

        Args:
            rd_map: 2D Range-Doppler 幅度图 (D, R)。
            guard_cells: CFAR 保护带半宽。
            reference_cells: CFAR 参考窗半宽。
            alpha: CFAR 门限因子。None = 从 pfa 自动计算。
            pfa: 目标虚警概率。

        Returns:
            (M, 2) 检测点 [doppler_idx, range_idx] 数组。
        """
        if self.peak_finder is not None:
            # 峰值门控模式（legacy）：不执行分组
            candidates = self.peak_finder.detect(rd_map)
            if len(candidates) == 0:
                return np.empty((0, 2), dtype=np.int64)
            return self._cfar.filter(
                rd_map, candidates,
                guard_cells=guard_cells,
                reference_cells=reference_cells,
                alpha=alpha,
                pfa=pfa,
            )

        # 标准模式：全图 2D CFAR 检测
        detections = self._cfar.detect(
            rd_map,
            guard_cells=guard_cells,
            reference_cells=reference_cells,
            alpha=alpha,
            pfa=pfa,
        )

        # 峰值分组：合并同一目标的相邻检测点
        if self.enable_grouping and len(detections) > 1:
            connectivity = 2# reference_cells//2
            detections = _peak_grouping(rd_map, detections, connectivity=connectivity)

        return detections

    @property
    def cfar_name(self) -> str:
        return self._cfar.__class__.__name__
