"""CFAR 门限因子 alpha 计算模块。

支持三种计算模式：
    - 'lut':      基于蒙特卡洛仿真的预计算查找表（最快，~微秒级）
    - 'mc':       实时蒙特卡洛仿真（精确但慢，~百毫秒级）
    - 'analytic': 解析/数值公式（瞬时，对非标参数精度略低）
    - 'auto':     优先查 LUT → 未命中则回退解析公式（默认）

核心思路：
    在 H0 假设（纯噪声，平方律检波 → 指数分布）下，通过蒙特卡洛仿真获得
    每种 CFAR 变体检统计量的经验分布。alpha 即该分布的 (1-Pfa) 分位数。

    一次 MC 仿真可服务于任意 Pfa，因此 LUT 只需按 (guard_cells, reference_cells)
    网格预计算，Pfa 维通过排序统计量直接提取分位数。

参考：
    - Richards, "Fundamentals of Radar Signal Processing"
    - Rohling, "Radar CFAR Thresholding in Clutter and Multiple Target Situations" (1983)
    - Gandhi & Kassam, "Analysis of CFAR Processors in Nonhomogeneous Background" (1988)
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 2D 参考窗几何计算
# ---------------------------------------------------------------------------

def _ref_cell_counts(guard_cells: int, reference_cells: int) -> tuple:
    """计算 2D CFAR 矩形参考窗中的单元数。

    参考窗为 (2*(guard+ref)+1) × (2*(guard+ref)+1) 的正方形区域，
    减去内部 (2*guard+1) × (2*guard+1) 的保护带和 CUT 单元。

    Args:
        guard_cells: 保护带半宽
        reference_cells: 参考窗半宽

    Returns:
        (n_total, n_half) 元组：
            n_total: 总参考单元数（CA/OS/SO 用）
            n_half: 每半窗单元数（GO/SO 沿距离维分裂，每侧 ref × 窗高）
    """
    window_size = 2 * (guard_cells + reference_cells) + 1
    guard_size = 2 * guard_cells + 1
    n_total = window_size ** 2 - guard_size ** 2
    # GO/SO 每半窗：窗高 × ref 列
    n_half = reference_cells * window_size
    return n_total, n_half


# ---------------------------------------------------------------------------
# 蒙特卡洛仿真引擎
# ---------------------------------------------------------------------------

def _mc_statistic_ca(n_ref: int, n_trials: int = 1_000_000,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """CA-CFAR 检验统计量 MC 仿真。

    统计量 T = CUT / mean(参考单元)
    返回升序排列的统计量数组，alpha(Pfa) = T[floor((1-Pfa) * n_trials)]
    """
    if rng is None:
        rng = np.random.default_rng()
    # 第一列为 CUT，其余为参考单元
    samples = rng.exponential(1, size=(n_trials, n_ref + 1))
    cut = samples[:, 0]
    ref_mean = samples[:, 1:].mean(axis=1)
    stats = cut / ref_mean
    stats.sort()
    return stats


def _mc_statistic_os(n_ref: int, k: int, n_trials: int = 1_000_000,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """OS-CFAR 检验统计量 MC 仿真。

    统计量 T = CUT / k-th_smallest(参考单元)
    k = floor(rank_ratio × n_ref)
    """
    if rng is None:
        rng = np.random.default_rng()
    samples = rng.exponential(1, size=(n_trials, n_ref + 1))
    cut = samples[:, 0]
    # 部分排序仅取第 k 小值
    ref_kth = np.partition(samples[:, 1:], k - 1, axis=1)[:, k - 1]
    stats = cut / ref_kth
    stats.sort()
    return stats


def _mc_statistic_go(n_half: int, n_trials: int = 1_000_000,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """GO-CFAR 检验统计量 MC 仿真。

    统计量 T = CUT / max(mean_left, mean_right)
    left/right 各含 n_half 个独立指数分布参考单元。
    """
    if rng is None:
        rng = np.random.default_rng()
    samples = rng.exponential(1, size=(n_trials, 2 * n_half + 1))
    cut = samples[:, 0]
    left_mean = samples[:, 1:1 + n_half].mean(axis=1)
    right_mean = samples[:, 1 + n_half:].mean(axis=1)
    noise_est = np.maximum(left_mean, right_mean)
    stats = cut / noise_est
    stats.sort()
    return stats


def _mc_statistic_so(n_half: int, n_trials: int = 1_000_000,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """SO-CFAR 检验统计量 MC 仿真。

    统计量 T = CUT / min(mean_left, mean_right)
    """
    if rng is None:
        rng = np.random.default_rng()
    samples = rng.exponential(1, size=(n_trials, 2 * n_half + 1))
    cut = samples[:, 0]
    left_mean = samples[:, 1:1 + n_half].mean(axis=1)
    right_mean = samples[:, 1 + n_half:].mean(axis=1)
    noise_est = np.minimum(left_mean, right_mean)
    stats = cut / noise_est
    stats.sort()
    return stats


# ---------------------------------------------------------------------------
# 解析 / 数值公式（LUT 回退）
# ---------------------------------------------------------------------------

def _analytic_ca(n_ref: int, pfa: float) -> float:
    """CA-CFAR 精确闭式解。

    α = N × (Pfa^(-1/N) - 1)
    来源：Richards, 雷达信号处理基础（平方律检波，IID 高斯噪声）
    """
    return n_ref * (pfa ** (-1.0 / n_ref) - 1.0)


def _analytic_os(n_ref: int, k: int, pfa: float) -> float:
    """OS-CFAR 二分法数值求解。

    求解方程（Rohling 1983）：
        Pfa = k × C(N,k) × B(k, N-k+1+α) / B(k, N-k+1)

    在 log 域计算避免 Gamma 函数溢出。
    搜索区间 [1e-6, 500]，典型 α 值在此范围内。
    """
    from scipy.special import gammaln
    from scipy.optimize import bisect

    # 预计算 log(k) + log(C(N,k))
    log_comb = (np.log(k)
                + gammaln(n_ref + 1)
                - gammaln(k + 1)
                - gammaln(n_ref - k + 1))

    def pfa_func(a: float) -> float:
        log_val = (log_comb
                   + gammaln(k)
                   + gammaln(n_ref - k + 1 + a)
                   - gammaln(n_ref + 1 + a))
        return float(np.exp(log_val))

    try:
        return float(bisect(lambda a: pfa_func(a) - pfa, 1e-6, 500.0,
                            xtol=1e-8, rtol=1e-6))
    except ValueError:
        # 若解不在 [1e-6, 500] 区间内，扩大搜索范围
        return float(bisect(lambda a: pfa_func(a) - pfa, 1e-6, 5000.0,
                            xtol=1e-8, rtol=1e-6))


def _analytic_go(n_half: int, pfa: float) -> float:
    """GO-CFAR 二分法数值求解。

    求解方程（Gandhi & Kassam 1988，1D 等价形式）：
        Pfa = 2·(1 + α/M)^(-M) - (1 + 2α/M)^(-M)
    其中 M = n_half 为每半窗参考单元数。
    """
    from scipy.optimize import bisect

    M = float(n_half)

    def pfa_func(a: float) -> float:
        t = a / M
        return float(2.0 * (1.0 + t) ** (-M) - (1.0 + 2.0 * t) ** (-M))

    try:
        return float(bisect(lambda a: pfa_func(a) - pfa, 1e-6, 500.0,
                            xtol=1e-8, rtol=1e-6))
    except ValueError:
        return float(bisect(lambda a: pfa_func(a) - pfa, 1e-6, 5000.0,
                            xtol=1e-8, rtol=1e-6))


def _analytic_so(n_ref: int, pfa: float) -> float:
    """SO-CFAR 近似闭式解。

    α = (N/2) × (Pfa^(-2/N) - 1)

    与 CA-CFAR 形式相同，但有效参考单元数约为 N/2（因取 min 操作
    使噪声估计偏低，需更大的 alpha 补偿）。
    """
    n_eff = n_ref / 2.0
    return n_eff * (pfa ** (-1.0 / n_eff) - 1.0)


# ---------------------------------------------------------------------------
# LUT 管理器
# ---------------------------------------------------------------------------

class AlphaLUT:
    """CFAR alpha 预计算查找表。

    从 .npz 文件加载蒙特卡洛仿真生成的 alpha 表。
    支持在 log(Pfa) 空间上线性插值查询。

    Usage:
        lut = AlphaLUT()
        lut.load()                                     # 加载默认 LUT
        alpha = lut.query('ca', 2, 8, 1e-4)            # 查表
        alpha = lut.query('os', 2, 8, 1e-4, rank_ratio=0.75)
    """

    def __init__(self, lut_path: Optional[str] = None):
        """初始化 LUT 管理器。

        Args:
            lut_path: LUT 文件路径，为 None 时使用默认路径。
        """
        self._data: Optional[dict] = None
        self._path = lut_path

    @property
    def is_loaded(self) -> bool:
        """LUT 是否已加载。"""
        return self._data is not None

    def load(self, path: Optional[str] = None) -> bool:
        """加载 LUT 文件。

        Args:
            path: LUT 文件路径。为 None 时使用 __init__ 中的路径，
                  仍为 None 时使用默认路径 fmcw/data/cfar_alpha_lut.npz

        Returns:
            True 若加载成功，False 若文件不存在。
        """
        if path is None:
            path = self._path
        if path is None:
            # 默认路径：fmcw/data/cfar_alpha_lut.npz
            module_dir = Path(__file__).parent.parent
            path = str(module_dir / "data" / "cfar_alpha_lut.npz")

        if os.path.exists(path):
            self._data = dict(np.load(str(path), allow_pickle=True))
            return True
        return False

    def _interp_log_pfa(self, alphas: np.ndarray, pfa_grid: np.ndarray,
                         target_pfa: float) -> float:
        """在 log(Pfa) 空间上线性插值 alpha。"""
        log_pfa = np.log(pfa_grid)
        log_target = np.log(target_pfa)
        return float(np.interp(log_target, log_pfa, alphas))

    def query(self, variant: str, guard: int, ref: int, pfa: float,
              rank_ratio: Optional[float] = None) -> Optional[float]:
        """查询 alpha 值。

        Args:
            variant: CFAR 变体名 ('ca', 'os', 'go', 'so')
            guard: guard_cells 半宽
            ref: reference_cells 半宽
            pfa: 目标虚警概率
            rank_ratio: OS-CFAR 的 rank 比例（仅 variant='os' 时使用）

        Returns:
            alpha 值，若不在 LUT 覆盖范围内则返回 None。
        """
        if self._data is None:
            return None

        prefix = variant
        guard_key = f"{prefix}_guard"
        if guard_key not in self._data:
            return None

        guards = self._data[guard_key]
        refs = self._data[f"{prefix}_ref"]

        # 精确匹配 guard 和 ref
        guard_matches = np.where(guards == guard)[0]
        ref_matches = np.where(refs == ref)[0]

        if len(guard_matches) == 0 or len(ref_matches) == 0:
            return None

        gi, ri = guard_matches[0], ref_matches[0]

        if variant == "os":
            # OS-CFAR 多一个 rank_ratio 维度
            if rank_ratio is None:
                rank_ratio = 0.75
            ranks = self._data["os_rank"]
            alpha_4d = self._data["os_alpha"]  # [n_g, n_r, n_k, n_p]
            pfa_grid = self._data["os_pfa"]

            # 找最近的 rank_ratio
            rk = int(np.argmin(np.abs(ranks - rank_ratio)))
            alphas = alpha_4d[gi, ri, rk, :]
        else:
            alpha_3d = self._data[f"{prefix}_alpha"]  # [n_g, n_r, n_p]
            pfa_grid = self._data[f"{prefix}_pfa"]
            alphas = alpha_3d[gi, ri, :]

        # 检查 Pfa 是否在网格范围内
        if pfa < pfa_grid.min() or pfa > pfa_grid.max():
            return None

        return self._interp_log_pfa(alphas, pfa_grid, pfa)

    @staticmethod
    def generate(
        guards: list,
        refs: list,
        pfa_grid: np.ndarray,
        rank_ratios: Optional[list] = None,
        n_trials: int = 1_000_000,
        seed: int = 42,
        save_path: Optional[str] = None,
    ) -> dict:
        """生成 CFAR alpha 查找表（通过蒙特卡洛仿真）。

        对 (guard, ref) 二维网格的每个组合，运行一次 MC 仿真获得
        检验统计量的经验分布，然后对所有 Pfa 提取对应的分位数。

        Args:
            guards: guard_cells 列表，如 [1, 2, 3, 4]
            refs: reference_cells 列表，如 [4, 6, 8, 10, 12, 16]
            pfa_grid: Pfa 网格 (log-spaced 推荐), shape (n_pfa,)
            rank_ratios: OS-CFAR rank 比例列表，默认 [0.5, 0.6, 0.75, 0.8, 0.9]
            n_trials: 每次 MC 仿真试验次数（越大越精确）
            seed: 随机种子（保证可复现）
            save_path: 保存路径，为 None 则不保存

        Returns:
            dict 包含所有 LUT 数据，可直接传给 np.savez_compressed。
        """
        if rank_ratios is None:
            rank_ratios = [0.5, 0.6, 0.75, 0.8, 0.9]

        rng = np.random.default_rng(seed)
        n_g = len(guards)
        n_r = len(refs)
        n_p = len(pfa_grid)
        n_k = len(rank_ratios)

        # 预分配
        ca_alpha = np.zeros((n_g, n_r, n_p))
        go_alpha = np.zeros((n_g, n_r, n_p))
        so_alpha = np.zeros((n_g, n_r, n_p))
        os_alpha = np.zeros((n_g, n_r, n_k, n_p))

        total_combos = n_g * n_r
        completed = 0

        for gi, guard in enumerate(guards):
            for ri, ref in enumerate(refs):
                n_total, n_half = _ref_cell_counts(guard, ref)

                # CA-CFAR
                stats = _mc_statistic_ca(n_total, n_trials, rng)
                for pi in range(n_p):
                    ca_alpha[gi, ri, pi] = stats[
                        int((1.0 - pfa_grid[pi]) * n_trials)
                    ]

                # GO-CFAR
                stats = _mc_statistic_go(n_half, n_trials, rng)
                for pi in range(n_p):
                    go_alpha[gi, ri, pi] = stats[
                        int((1.0 - pfa_grid[pi]) * n_trials)
                    ]

                # SO-CFAR
                stats = _mc_statistic_so(n_half, n_trials, rng)
                for pi in range(n_p):
                    so_alpha[gi, ri, pi] = stats[
                        int((1.0 - pfa_grid[pi]) * n_trials)
                    ]

                # OS-CFAR（每种 rank_ratio 都需要一次仿真）
                for ki, rank_r in enumerate(rank_ratios):
                    k_val = max(1, int(n_total * rank_r))
                    stats = _mc_statistic_os(n_total, k_val, n_trials, rng)
                    for pi in range(n_p):
                        os_alpha[gi, ri, ki, pi] = stats[
                            int((1.0 - pfa_grid[pi]) * n_trials)
                        ]

                completed += 1
                print(f"  LUT 生成进度: {completed}/{total_combos} "
                      f"(guard={guard}, ref={ref}, N={n_total}, M_half={n_half})")

        data = {
            # CA-CFAR
            "ca_guard": np.array(guards, dtype=np.int32),
            "ca_ref": np.array(refs, dtype=np.int32),
            "ca_pfa": pfa_grid.astype(np.float64),
            "ca_alpha": ca_alpha.astype(np.float64),
            # GO-CFAR
            "go_guard": np.array(guards, dtype=np.int32),
            "go_ref": np.array(refs, dtype=np.int32),
            "go_pfa": pfa_grid.astype(np.float64),
            "go_alpha": go_alpha.astype(np.float64),
            # SO-CFAR
            "so_guard": np.array(guards, dtype=np.int32),
            "so_ref": np.array(refs, dtype=np.int32),
            "so_pfa": pfa_grid.astype(np.float64),
            "so_alpha": so_alpha.astype(np.float64),
            # OS-CFAR
            "os_guard": np.array(guards, dtype=np.int32),
            "os_ref": np.array(refs, dtype=np.int32),
            "os_rank": np.array(rank_ratios, dtype=np.float64),
            "os_pfa": pfa_grid.astype(np.float64),
            "os_alpha": os_alpha.astype(np.float64),
            # 元数据
            "n_trials": np.int64(n_trials),
        }

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            np.savez_compressed(save_path, **data)
            print(f"LUT 已保存至: {save_path}")

        return data


# ---------------------------------------------------------------------------
# 全局 LUT 实例（懒加载）
# ---------------------------------------------------------------------------

_lut: Optional[AlphaLUT] = None


def _get_lut() -> AlphaLUT:
    """获取全局 LUT 实例（懒加载）。"""
    global _lut
    if _lut is None:
        _lut = AlphaLUT()
    if not _lut.is_loaded:
        _lut.load()
    return _lut


# ---------------------------------------------------------------------------
# 统一 alpha 计算入口
# ---------------------------------------------------------------------------

def compute_alpha(
    variant: str,
    pfa: float,
    guard_cells: int,
    reference_cells: int,
    method: str = 'auto',
    rank_ratio: Optional[float] = None,
    **kwargs,
) -> float:
    """计算 CFAR 门限因子 alpha。

    Args:
        variant: CFAR 变体名 ('ca', 'os', 'go', 'so')
        pfa: 目标虚警概率（如 1e-4）
        guard_cells: 保护带半宽
        reference_cells: 参考窗半宽
        method: 计算方法
            - 'auto': 优先查 LUT，未命中回退解析公式（默认）
            - 'lut': 仅查表，未命中抛异常
            - 'mc': 实时蒙特卡洛仿真（~500ms）
            - 'analytic': 解析/数值公式（瞬时）
        rank_ratio: OS-CFAR 的 rank 比例（默认 0.75）
        **kwargs: 传递给 MC 仿真的额外参数（如 n_trials）

    Returns:
        alpha 门限因子 (float)

    Raises:
        ValueError: method='lut' 且 LUT 未覆盖请求参数时
    """
    if variant not in ('ca', 'os', 'go', 'so'):
        raise ValueError(f"未知 CFAR 变体: {variant}，合法值: ca, os, go, so")

    n_total, n_half = _ref_cell_counts(guard_cells, reference_cells)

    if method == 'auto':
        # 优先查 LUT
        lut = _get_lut()
        extra = {}
        if variant == 'os':
            extra['rank_ratio'] = rank_ratio or 0.75
        result = lut.query(variant, guard_cells, reference_cells, pfa, **extra)
        if result is not None:
            return result
        # LUT 未覆盖，回退到解析公式
        method = 'analytic'

    if method == 'analytic':
        if variant == 'ca':
            return _analytic_ca(n_total, pfa)
        elif variant == 'os':
            k_val = max(1, int(n_total * (rank_ratio or 0.75)))
            return _analytic_os(n_total, k_val, pfa)
        elif variant == 'go':
            return _analytic_go(n_half, pfa)
        elif variant == 'so':
            return _analytic_so(n_total, pfa)

    elif method == 'mc':
        n_trials = kwargs.get('n_trials', 500_000)
        rng = np.random.default_rng()
        if variant == 'ca':
            stats = _mc_statistic_ca(n_total, n_trials, rng)
        elif variant == 'os':
            k_val = max(1, int(n_total * (rank_ratio or 0.75)))
            stats = _mc_statistic_os(n_total, k_val, n_trials, rng)
        elif variant == 'go':
            stats = _mc_statistic_go(n_half, n_trials, rng)
        elif variant == 'so':
            stats = _mc_statistic_so(n_half, n_trials, rng)
        return float(stats[int((1.0 - pfa) * n_trials)])

    elif method == 'lut':
        lut = _get_lut()
        extra = {}
        if variant == 'os':
            extra['rank_ratio'] = rank_ratio or 0.75
        result = lut.query(variant, guard_cells, reference_cells, pfa, **extra)
        if result is None:
            raise ValueError(
                f"LUT 未覆盖参数组合: variant={variant}, "
                f"guard={guard_cells}, ref={reference_cells}, pfa={pfa}"
            )
        return result

    raise ValueError(f"未知 method: {method}，合法值: auto, lut, mc, analytic")
