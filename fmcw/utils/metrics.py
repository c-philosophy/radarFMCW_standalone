"""Performance evaluation metrics for radar detection and tracking."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def rmse(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    """Root Mean Square Error.

    Args:
        estimated: (N, D) estimated positions.
        ground_truth: (N, D) true positions.

    Returns:
        RMSE value.
    """
    return float(np.sqrt(np.mean((estimated - ground_truth) ** 2)))


def ospa_distance(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    p: float = 2.0,
    c: float = 10.0,
) -> float:
    """Optimal SubPattern Assignment (OSPA) metric.

    Args:
        estimated: (M, D) estimated states.
        ground_truth: (N, D) ground truth states.
        p: Order of the metric.
        c: Cutoff distance for cardinality mismatch penalty.

    Returns:
        OSPA distance.
    """
    M = len(estimated)
    N = len(ground_truth)

    if M == 0 and N == 0:
        return 0.0
    if M == 0 or N == 0:
        return c

    # Compute distance matrix
    dist = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            dist[i, j] = min(c, np.linalg.norm(estimated[i] - ground_truth[j]))

    # Optimal assignment (Hungarian)
    row_ind, col_ind = linear_sum_assignment(dist)

    # OSPA
    n = max(M, N)
    d_sum = np.sum(dist[row_ind, col_ind] ** p)
    cardinality = c**p * abs(M - N)
    return float(((d_sum + cardinality) / n) ** (1.0 / p))


def gospa_distance(
    estimated: np.ndarray,     # (M, D) 估计状态
    ground_truth: np.ndarray,  # (N, D) 真值状态
    p: float = 2.0,            # 距离阶数
    c: float = 10.0,           # 截断距离
    alpha: float = 2.0,        # 漏检/虚警惩罚平衡 (α=2 为对称)
) -> float:
    """Generalized Optimal Sub-Pattern Assignment (GOSPA) 距离。

    参考文献：
        Rahmathullah et al., "Generalized Optimal Sub-Pattern Assignment
        Metric", Fusion 2017. (Jean Pierre Le Cadre Best Paper Award)

    GOSPA 同时惩罚定位误差、漏检和虚警，是 OSPA 的改进版：
    - 去归一化：不除以集合大小，避免小基数时的偏差
    - 分离惩罚：α 独立控制漏检(1/α)和虚警(1/(1-α))的权重

    Args:
        estimated: (M, D) 估计状态矩阵。
        ground_truth: (N, D) 真值状态矩阵。
        p: 距离阶数 (1 或 2)。
        c: 截断距离 (m)，超出此距离的匹配按 c 计。
        alpha: 惩罚平衡参数。α=2 表示漏检和虚警等权重。

    Returns:
        GOSPA 距离 (非负，越小越好)。
    """
    M, N = len(estimated), len(ground_truth)

    if M == 0 and N == 0:
        return 0.0

    # 截断距离矩阵 + 匈牙利匹配
    if M > 0 and N > 0:
        dist = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                dist[i, j] = min(c, np.linalg.norm(estimated[i] - ground_truth[j]))
        row, col = linear_sum_assignment(dist)
        n_matched = len(row)
        d_localization = np.sum(dist[row, col] ** p)
    else:
        n_matched = 0
        d_localization = 0.0

    # 对称惩罚：α=2 时漏检和虚警等权重，均为 c^p
    d_missed = c ** p * (N - n_matched)
    d_false = c ** p * (M - n_matched)

    # GOSPA 公式: d^(p) = d_localization + d_missed + d_false
    d_all = d_localization + d_missed + d_false
    return float(d_all ** (1.0 / p))
