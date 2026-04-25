"""Data association algorithms: NN, GNN, JPDA.

Pure NumPy + SciPy linear_sum_assignment for Hungarian algorithm.
Registered in AlgorithmRegistry under category "associator".
"""

from typing import List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from fmcw.core.registry import register_algorithm


@register_algorithm("associator", "nn")
class NearestNeighbor:
    """Nearest Neighbor association with distance gating.

    Simple greedy approach: each track claims the nearest ungated measurement.
    """

    def __init__(self, gate: float = 3.0):
        """
        Args:
            gate: Maximum allowed distance for association.
        """
        self.gate = gate

    def associate(
        self,
        track_positions: List[np.ndarray],
        measurements: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Associate measurements to tracks.

        Args:
            track_positions: List of (2,) arrays [x, y] for each track.
            measurements: (M, 2) array of measurement positions.

        Returns:
            (assignments, unassigned_tracks, unassigned_meas) where
            assignments[i] = measurement_idx or -1 for track i.
        """
        n_tracks = len(track_positions)
        n_meas = len(measurements)

        if n_tracks == 0 or n_meas == 0:
            return (
                np.full(n_tracks, -1),
                list(range(n_tracks)),
                list(range(n_meas)),
            )

        dist = cdist(np.array(track_positions), measurements)

        assignments = np.full(n_tracks, -1)
        assigned_meas = set()

        # Greedy nearest neighbor
        for i in range(n_tracks):
            # Find nearest ungated measurement
            candidates = [(j, dist[i, j]) for j in range(n_meas)
                          if j not in assigned_meas and dist[i, j] < self.gate]
            if not candidates:
                continue
            best_j = min(candidates, key=lambda x: x[1])[0]
            assignments[i] = best_j
            assigned_meas.add(best_j)

        unassigned_tracks = [i for i in range(n_tracks) if assignments[i] == -1]
        unassigned_meas = [j for j in range(n_meas) if j not in assigned_meas]

        return assignments, unassigned_tracks, unassigned_meas


@register_algorithm("associator", "gnn")
class GNN:
    """Global Nearest Neighbor via Hungarian (Munkres) algorithm.

    Minimizes total assignment cost subject to gating.
    """

    def __init__(self, gate: float = 3.0):
        self.gate = gate

    def associate(
        self,
        track_positions: List[np.ndarray],
        measurements: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        n_tracks = len(track_positions)
        n_meas = len(measurements)

        if n_tracks == 0 or n_meas == 0:
            return (
                np.full(n_tracks, -1),
                list(range(n_tracks)),
                list(range(n_meas)),
            )

        dist = cdist(np.array(track_positions), measurements)

        # Gate: large cost for ungated associations
        cost = np.where(dist < self.gate, dist, 1e10)

        row_ind, col_ind = linear_sum_assignment(cost)

        assignments = np.full(n_tracks, -1)
        assigned_meas = set()

        for r, c in zip(row_ind, col_ind):
            if dist[r, c] < self.gate:
                assignments[r] = c
                assigned_meas.add(c)

        unassigned_tracks = [i for i in range(n_tracks) if assignments[i] == -1]
        unassigned_meas = [j for j in range(n_meas) if j not in assigned_meas]

        return assignments, unassigned_tracks, unassigned_meas


@register_algorithm("associator", "jpda")
class JPDA:
    """Joint Probabilistic Data Association.

    Computes marginal association probabilities for each track-measurement
    pair. Updates are weighted by association probability rather than
    hard assignment.

    This is a simplified single-scan JPDA without track hypothesis trees.
    """

    def __init__(
        self,
        gate: float = 3.0,
        Pd: float = 0.9,
        clutter_density: float = 1e-6,
    ):
        self.gate = gate
        self.Pd = Pd
        self.clutter_density = clutter_density

    def associate(
        self,
        track_positions: List[np.ndarray],
        measurements: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[int], Optional[np.ndarray]]:
        """Compute JPDA association.

        Returns:
            (assignments, unassigned_tracks, unassigned_meas, beta)
            where beta[i, j] is the probability that measurement j
            originates from track i. beta[i, -1] = prob not detected.
        """
        n_tracks = len(track_positions)
        n_meas = len(measurements)

        assignments = np.full(n_tracks, -1)
        beta = np.zeros((n_tracks, n_meas + 1))  # +1 for "no detection"

        if n_tracks == 0 or n_meas == 0:
            beta[:, 0] = 1.0  # All tracks "not detected"
            return (
                assignments,
                list(range(n_tracks)),
                list(range(n_meas)),
                beta,
            )

        dist = cdist(np.array(track_positions), measurements)

        # Gate likelihoods
        det_prob = np.where(dist < self.gate, self.Pd, 0.0)
        beta0 = 1.0 - self.Pd  # Not detected

        # For each track, compute marginal association probabilities
        for i in range(n_tracks):
            # Simplification: treat each track independently (not full JPDA)
            # Full JPDA requires enumerating all feasible joint events
            # Here we use the PDA approximation per track
            row = dist[i]
            valid = dist[i] < self.gate

            if not np.any(valid):
                beta[i, 0] = 1.0
                continue

            likelihoods = np.exp(-0.5 * row[valid]**2) * self.Pd
            clutter_term = self.clutter_density * (2 * np.pi)**(2 / 2)

            weights = likelihoods / (likelihoods.sum() + clutter_term + beta0)
            beta[i, 0] = beta0 / (likelihoods.sum() + clutter_term + beta0)
            beta[i, 1:][valid] = weights

            # Hard assignment (for compatibility): use max probability
            best = np.argmax(beta[i])
            if best > 0:
                assignments[i] = best - 1

        assigned_meas = set(assignments[assignments >= 0])
        unassigned_tracks = [i for i in range(n_tracks) if assignments[i] == -1]
        unassigned_meas = [j for j in range(n_meas) if j not in assigned_meas]

        return assignments, unassigned_tracks, unassigned_meas, beta


