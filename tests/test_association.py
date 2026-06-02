"""数据关联算法测试（NN、GNN、JPDA）。"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from numpy.testing import assert_allclose

from fmcw.tracking.association import (
    NearestNeighbor, GNN, JPDA,
    TrackState, mahalanobis_distance, velocity_gate,
)


def _make_states(positions):
    """辅助：从位置列表创建 TrackState 列表。"""
    return [TrackState(position=np.array(p), covariance=np.eye(2)) for p in positions]


def test_nn_simple():
    """NN: should associate closest track to measurement."""
    nn = NearestNeighbor(gate=10.0, use_mahalanobis=False, use_velocity_gating=False)

    track_states = _make_states([[0.0, 0.0], [10.0, 10.0]])
    measurements = np.array([[1.0, 0.0], [9.0, 10.0]])

    result = nn.associate(track_states, measurements)

    assignments = result[0]
    assert len(assignments) == 2
    assert assignments[0] == 0, f"Track 0 should get meas 0, got {assignments[0]}"


def test_nn_gate():
    """NN: should reject measurements outside gating threshold."""
    nn = NearestNeighbor(gate=2.0, use_mahalanobis=False, use_velocity_gating=False)

    track_states = _make_states([[0.0, 0.0]])
    measurements = np.array([[100.0, 100.0]])

    result = nn.associate(track_states, measurements)

    assignments = result[0]
    assert assignments[0] == -1 or len(result[2]) == 1


def test_gnn_hungarian():
    """GNN: should find optimal global assignment."""
    gnn = GNN(gate=10.0, use_mahalanobis=False, use_velocity_gating=False)

    track_states = _make_states([[0.0, 0.0], [5.0, 5.0]])
    measurements = np.array([[1.0, 0.0], [6.0, 5.0], [100.0, 100.0]])

    result = gnn.associate(track_states, measurements)

    assignments = result[0]
    assert len(assignments) == 2
    assert assignments[0] == 0


def test_gnn_more_tracks():
    """GNN: should handle more tracks than measurements."""
    gnn = GNN(gate=10.0, use_mahalanobis=False, use_velocity_gating=False)

    track_states = _make_states([[0.0, 0.0], [5.0, 5.0]])
    measurements = np.array([[1.0, 0.0]])

    result = gnn.associate(track_states, measurements)

    assignments = result[0]
    unassigned_tracks = result[1]

    assert sum(1 for a in assignments if a >= 0) == 1
    assert len(unassigned_tracks) == 1


def test_gnn_empty():
    """GNN: should handle empty inputs."""
    gnn = GNN(gate=10.0, use_mahalanobis=False, use_velocity_gating=False)
    result = gnn.associate([], np.empty((0, 2)))
    assert len(result[0]) == 0
    assert len(result[1]) == 0
    assert len(result[2]) == 0


def test_jpda_probs():
    """JPDA: should compute marginal probabilities."""
    jpda = JPDA(gate=10.0, use_mahalanobis=False, use_velocity_gating=False)

    track_states = _make_states([[0.0, 0.0], [5.0, 5.0]])
    measurements = np.array([[1.0, 0.0], [6.0, 5.0]])

    result = jpda.associate(track_states, measurements)

    assignments = result[0]
    beta = result[3]

    assert len(assignments) == 2
    assert beta is not None
    assert beta.shape == (len(track_states), len(measurements) + 1)


# ── 马氏距离测试 ────────────────────────────────────────────────────────

def test_mahalanobis_equals_euclidean_when_identity():
    """P=I 时马氏距离 ≈ 欧氏距离（含微小噪声项）。"""
    track = TrackState(position=np.array([0.0, 0.0]), covariance=np.eye(2))
    z = np.array([3.0, 4.0])
    d = mahalanobis_distance(z, track)
    # d = sqrt(25 / 1.01) ≈ 4.975（内部加了 0.01 噪声防奇异）
    assert_allclose(d, 5.0, atol=0.1)


def test_mahalanobis_anisotropic():
    """各向异性协方差下，马氏距离和欧氏距离不同。"""
    P = np.diag([1.0, 100.0])
    track = TrackState(position=np.array([0.0, 0.0]), covariance=P)
    z_x = np.array([3.0, 0.0])
    z_y = np.array([0.0, 30.0])

    d_x = mahalanobis_distance(z_x, track)
    d_y = mahalanobis_distance(z_y, track)
    assert abs(d_x - d_y) < 2.0, f"d_x={d_x:.2f}, d_y={d_y:.2f}"


def test_mahalanobis_gating():
    """门控应拒绝远距离测量。"""
    P = np.eye(2) * 0.01
    track = TrackState(position=np.zeros(2), covariance=P)
    z_far = np.array([1.0, 0.0])
    d = mahalanobis_distance(z_far, track)
    assert d > 6.0


def test_velocity_gate_accept():
    """速度一致时应通过门控。"""
    track = TrackState(
        position=np.zeros(2), covariance=np.eye(2),
        velocity=10.0, p_velocity=1.0,
    )
    assert velocity_gate(10.5, track, n_sigma=3.0)


def test_velocity_gate_reject():
    """速度差异大时应被拒绝。"""
    track = TrackState(
        position=np.zeros(2), covariance=np.eye(2),
        velocity=10.0, p_velocity=1.0,
    )
    assert not velocity_gate(-5.0, track, n_sigma=3.0)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
