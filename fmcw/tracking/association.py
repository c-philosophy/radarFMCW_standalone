"""Data association algorithms: NN, GNN, JPDA.

Pure NumPy + SciPy linear_sum_assignment for Hungarian algorithm.
Registered in AlgorithmRegistry under category "associator".
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from fmcw.core.registry import register_algorithm


# ── 关联辅助数据结构与函数 ──────────────────────────────────────────────

@dataclass
class TrackState:
    """关联器所需的航迹状态，从 Track 对象提取。

    Attributes:
        position: 预测位置 (x, y)
        covariance: 位置协方差 (2x2 子矩阵)
        velocity: 预测径向速度（用于速度门控）
        p_velocity: 速度预测方差（用于速度门控）
    """
    position: np.ndarray          # (2,)
    covariance: np.ndarray        # (2, 2)
    velocity: float = 0.0
    p_velocity: float = 1.0


def mahalanobis_distance(z: np.ndarray, track: TrackState) -> float:
    """计算测量值 z 到航迹 track 的马氏距离。

    d² = (z - μ)ᵀ · S⁻¹ · (z - μ),  S = P_pos + R

    Args:
        z: 测量值 (x, y)
        track: 航迹预测状态

    Returns:
        马氏距离（标量）
    """
    innovation = z - track.position
    S = track.covariance + np.eye(2) * 0.01
    return float(np.sqrt(innovation @ np.linalg.solve(S, innovation)))


def velocity_gate(z_velocity: float, track: TrackState, n_sigma: float = 3.0) -> bool:
    """速度门控：检查测量速度与预测速度是否一致。

    Args:
        z_velocity: 测量径向速度
        track: 航迹预测状态
        n_sigma: 门控阈值（标准差倍数）

    Returns:
        True 表示通过门控（速度一致）
    """
    innov = abs(z_velocity - track.velocity)
    std = np.sqrt(track.p_velocity + 0.1)
    return innov < n_sigma * std


# ── Nearest Neighbor ─────────────────────────────────────────────────────

@register_algorithm("associator", "nn")
class NearestNeighbor:
    """Nearest Neighbor association with distance gating.

    Supports Euclidean or Mahalanobis distance, with optional velocity gating.
    """

    def __init__(
        self,
        gate: float = 6.0,
        gate_velocity: float = 3.0,
        use_mahalanobis: bool = True,
        use_velocity_gating: bool = True,
    ):
        self.gate = gate
        self.gate_velocity = gate_velocity
        self.use_mahalanobis = use_mahalanobis
        self.use_velocity_gating = use_velocity_gating

    def associate(
        self,
        track_states: List[TrackState],
        measurements: np.ndarray,
        velocities: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List, List[int]]:
        """Associate measurements to tracks.

        Args:
            track_states: List of TrackState for each track.
            measurements: (M, 2) array of measurement positions.
            velocities: Optional (M,) array of radial velocities.

        Returns:
            (assignments, unassigned_tracks, unassigned_meas)
        """
        n_tracks = len(track_states)
        n_meas = len(measurements)

        if n_tracks == 0 or n_meas == 0:
            return (
                np.full(n_tracks, -1),
                list(range(n_tracks)),
                list(range(n_meas)),
            )

        # Build cost matrix
        cost = np.zeros((n_tracks, n_meas))
        for i in range(n_tracks):
            for j in range(n_meas):
                if self.use_mahalanobis:
                    d = mahalanobis_distance(measurements[j], track_states[i])
                else:
                    d = float(np.linalg.norm(measurements[j] - track_states[i].position))

                if self.use_velocity_gating and velocities is not None:
                    if not velocity_gate(velocities[j], track_states[i], self.gate_velocity):
                        d = 1e10

                cost[i, j] = d if d < self.gate else 1e10

        # Greedy nearest neighbor
        assignments = np.full(n_tracks, -1)
        assigned_meas = set()

        for i in range(n_tracks):
            candidates = [(j, cost[i, j]) for j in range(n_meas)
                          if j not in assigned_meas and cost[i, j] < self.gate]
            if not candidates:
                continue
            best_j = min(candidates, key=lambda x: x[1])[0]
            assignments[i] = best_j
            assigned_meas.add(best_j)

        unassigned_tracks = [i for i in range(n_tracks) if assignments[i] == -1]
        unassigned_meas = [j for j in range(n_meas) if j not in assigned_meas]

        return assignments, unassigned_tracks, unassigned_meas


# ── Global Nearest Neighbor ──────────────────────────────────────────────

@register_algorithm("associator", "gnn")
class GNN:
    """Global Nearest Neighbor via Hungarian (Munkres) algorithm.

    Minimizes total assignment cost subject to gating.
    Supports Euclidean or Mahalanobis distance, with optional velocity gating.
    """

    def __init__(
        self,
        gate: float = 6.0,
        gate_velocity: float = 3.0,
        use_mahalanobis: bool = True,
        use_velocity_gating: bool = True,
    ):
        self.gate = gate
        self.gate_velocity = gate_velocity
        self.use_mahalanobis = use_mahalanobis
        self.use_velocity_gating = use_velocity_gating

    def associate(
        self,
        track_states: List[TrackState],
        measurements: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        delta_factor_pos: float = 1.0,
        delta_factor_vel: float = 1.0,
    ) -> Tuple[np.ndarray, List, List[int]]:
        n_tracks = len(track_states)
        n_meas = len(measurements)

        if n_tracks == 0 or n_meas == 0:
            return (
                np.full(n_tracks, -1),
                list(range(n_tracks)),
                list(range(n_meas)),
            )

        # Build cost matrix
        cost = np.zeros((n_tracks, n_meas))
        for i in range(n_tracks):
            for j in range(n_meas):
                if self.use_mahalanobis:
                    d_pos = mahalanobis_distance(measurements[j], track_states[i])
                else:
                    d_pos = float(np.linalg.norm(measurements[j] - track_states[i].position))

                if self.use_velocity_gating and velocities is not None:
                    if not velocity_gate(velocities[j], track_states[i], self.gate_velocity):
                        d_vel = 1e10
                    else:
                        d_vel = float(np.linalg.norm(velocities[j] - track_states[i].velocity))
                # Cost TODO: 成本函数计算并未按照`tracking_优化方向讨论.md`中的公式计算，待确认
                cost[i, j] = delta_factor_pos * d_pos + delta_factor_vel * (d_vel if self.use_velocity_gating  else 0.0)

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost)

        assignments = np.full(n_tracks, -1)
        assigned_meas = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < self.gate:
                assignments[r] = c
                assigned_meas.add(c)

        unassigned_tracks = [i for i in range(n_tracks) if assignments[i] == -1]
        unassigned_meas = [j for j in range(n_meas) if j not in assigned_meas]

        return assignments, unassigned_tracks, unassigned_meas


# ── Joint Probabilistic Data Association ─────────────────────────────────

@register_algorithm("associator", "jpda")
class JPDA:
    """Joint Probabilistic Data Association.

    Computes marginal association probabilities for each track-measurement
    pair. Updates are weighted by association probability rather than
    hard assignment.

    This is a simplified single-scan JPDA without track hypothesis trees.
    Supports Mahalanobis distance and velocity gating.
    """

    def __init__(
        self,
        gate: float = 6.0,
        gate_velocity: float = 3.0,
        use_mahalanobis: bool = True,
        use_velocity_gating: bool = True,
        Pd: float = 0.9,
        clutter_density: float = 1e-6,
    ):
        self.gate = gate
        self.gate_velocity = gate_velocity
        self.use_mahalanobis = use_mahalanobis
        self.use_velocity_gating = use_velocity_gating
        self.Pd = Pd
        self.clutter_density = clutter_density

    def associate(
        self,
        track_states: List[TrackState],
        measurements: np.ndarray,
        velocities: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List, List[int], Optional[np.ndarray]]:
        """Compute JPDA association.

        Returns:
            (assignments, unassigned_tracks, unassigned_meas, beta)
        """
        n_tracks = len(track_states)
        n_meas = len(measurements)

        assignments = np.full(n_tracks, -1)
        beta = np.zeros((n_tracks, n_meas + 1))

        if n_tracks == 0 or n_meas == 0:
            beta[:, 0] = 1.0
            return (assignments, list(range(n_tracks)), list(range(n_meas)), beta)

        # Build distance matrix
        dist = np.zeros((n_tracks, n_meas))
        for i in range(n_tracks):
            for j in range(n_meas):
                if self.use_mahalanobis:
                    dist[i, j] = mahalanobis_distance(measurements[j], track_states[i])
                else:
                    dist[i, j] = float(np.linalg.norm(measurements[j] - track_states[i].position))

        # Apply velocity gating
        gate_mask = dist < self.gate
        if self.use_velocity_gating and velocities is not None:
            for i in range(n_tracks):
                for j in range(n_meas):
                    if not velocity_gate(velocities[j], track_states[i], self.gate_velocity):
                        gate_mask[i, j] = False

        beta0 = 1.0 - self.Pd

        for i in range(n_tracks):
            valid = gate_mask[i]
            if not np.any(valid):
                beta[i, 0] = 1.0
                continue

            row = dist[i]
            likelihoods = np.exp(-0.5 * row[valid]**2) * self.Pd
            clutter_term = self.clutter_density * (2 * np.pi)**(2 / 2)

            weights = likelihoods / (likelihoods.sum() + clutter_term + beta0)
            beta[i, 0] = beta0 / (likelihoods.sum() + clutter_term + beta0)
            beta[i, 1:][valid] = weights

            best = np.argmax(beta[i])
            if best > 0:
                assignments[i] = best - 1

        assigned_meas = set(assignments[assignments >= 0])
        unassigned_tracks = [i for i in range(n_tracks) if assignments[i] == -1]
        unassigned_meas = [j for j in range(n_meas) if j not in assigned_meas]

        return assignments, unassigned_tracks, unassigned_meas, beta
