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
