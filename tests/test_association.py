"""数据关联算法测试（NN、GNN、JPDA）。"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from fmcw.tracking.association import NearestNeighbor, GNN, JPDA


def test_nn_simple():
    """NN: should associate closest track to measurement."""
    nn = NearestNeighbor(gate=10.0)

    tracks = [np.array([0.0, 0.0]), np.array([10.0, 10.0])]
    measurements = np.array([[1.0, 0.0], [9.0, 10.0]])

    result = nn.associate(tracks, measurements)

    # Track 0 -> measurement 0 (distance 1.0)
    # Track 1 -> measurement 1 (distance 1.0)
    assignments = result[0]
    assert len(assignments) == 2
    # Each track should be assigned to nearest measurement
    assert assignments[0] == 0, f"Track 0 should get meas 0, got {assignments[0]}"


def test_nn_gate():
    """NN: should reject measurements outside gating threshold."""
    nn = NearestNeighbor(gate=2.0)

    tracks = [np.array([0.0, 0.0])]
    measurements = np.array([[100.0, 100.0]])  # far away

    result = nn.associate(tracks, measurements)

    assignments = result[0]
    # The measurement is outside the gate, so track should be unassigned
    assert assignments[0] == -1 or len(result[2]) == 1


def test_gnn_hungarian():
    """GNN: should find optimal global assignment."""
    gnn = GNN(gate=10.0)

    tracks = [np.array([0.0, 0.0]), np.array([5.0, 5.0])]
    measurements = np.array([[1.0, 0.0], [6.0, 5.0], [100.0, 100.0]])

    result = gnn.associate(tracks, measurements)

    assignments = result[0]
    unassigned_meas = result[2]

    assert len(assignments) == 2
    # Track 0 should be assigned to the closest measurement
    assert assignments[0] == 0  # meas 0 is closest to track 0


def test_gnn_more_tracks():
    """GNN: should handle more tracks than measurements."""
    gnn = GNN(gate=10.0)

    tracks = [np.array([0.0, 0.0]), np.array([5.0, 5.0])]
    measurements = np.array([[1.0, 0.0]])

    result = gnn.associate(tracks, measurements)

    assignments = result[0]
    unassigned_tracks = result[1]

    # One track should be assigned, one unassigned
    assert sum(1 for a in assignments if a >= 0) == 1
    assert len(unassigned_tracks) == 1


def test_gnn_empty():
    """GNN: should handle empty inputs."""
    gnn = GNN(gate=10.0)
    result = gnn.associate([], np.empty((0, 2)))
    assert len(result[0]) == 0  # no assignments
    assert len(result[1]) == 0  # no unassigned tracks
    assert len(result[2]) == 0  # no unassigned meas


def test_jpda_probs():
    """JPDA: should compute marginal probabilities."""
    jpda = JPDA(gate=10.0)

    tracks = [np.array([0.0, 0.0]), np.array([5.0, 5.0])]
    measurements = np.array([[1.0, 0.0], [6.0, 5.0]])

    result = jpda.associate(tracks, measurements)

    assignments = result[0]
    unassigned_tracks = result[1]
    unassigned_meas = result[2]
    beta = result[3]

    assert len(assignments) == 2
    assert beta is not None, "JPDA should return beta matrix"
    assert beta.shape == (len(tracks), len(measurements) + 1), \
        f"Beta shape should be ({len(tracks)}, {len(measurements) + 1}), got {beta.shape}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
