"""运动模型单元测试。"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from numpy.testing import assert_allclose
from fmcw.tracking.motion_model import CVModel, CAModel, CTRAModel


# ── CV 测试 ──────────────────────────────────────────────────────────────

def test_cv_dim():
    model = CVModel()
    assert model.dim == 4
    assert len(model.state_names) == 4


def test_cv_f_matrix():
    model = CVModel()
    dt = 0.05
    F = model.F(dt)
    assert F.shape == (4, 4)
    assert F[0, 2] == dt
    assert F[1, 3] == dt
    assert_allclose(F[0, 0], 1.0)
    assert_allclose(F[0, 1], 0.0)
    assert_allclose(F[2, 2], 1.0)


def test_cv_q_symmetric():
    model = CVModel()
    Q = model.Q(0.05, 0.1)
    assert Q.shape == (4, 4)
    assert_allclose(Q, Q.T)
    assert np.all(np.linalg.eigvalsh(Q) >= 0)


def test_cv_q_increases_with_dt():
    model = CVModel()
    Q1 = model.Q(0.05, 0.1)
    Q2 = model.Q(0.10, 0.1)
    assert np.trace(Q2) > np.trace(Q1)


# ── CA 测试 ──────────────────────────────────────────────────────────────

def test_ca_dim():
    model = CAModel()
    assert model.dim == 6
    assert len(model.state_names) == 6


def test_ca_f_matrix():
    model = CAModel()
    dt = 0.05
    F = model.F(dt)
    assert F.shape == (6, 6)
    assert_allclose(F[0, 2], dt)
    assert_allclose(F[0, 4], 0.5 * dt**2)
    assert_allclose(F[2, 4], dt)


def test_ca_q_symmetric():
    model = CAModel()
    Q = model.Q(0.05, 0.1)
    assert Q.shape == (6, 6)
    assert_allclose(Q, Q.T)


# ── CTRA 测试 ────────────────────────────────────────────────────────────

def test_ctra_dim():
    model = CTRAModel()
    assert model.dim == 6
    assert len(model.state_names) == 6


def test_ctra_f_is_none():
    model = CTRAModel()
    assert model.F(0.05) is None


def test_ctra_propagate_straight():
    """ω=0 时直线运动。"""
    model = CTRAModel()
    x = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0])
    dt = 1.0
    x1 = model.propagate(x, dt)
    assert_allclose(x1[0], 10.0, atol=1e-6)
    assert_allclose(x1[1], 0.0, atol=1e-6)


def test_ctra_propagate_turn():
    """ω≠0 时画弧。"""
    model = CTRAModel()
    x = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 30.0])
    dt = 1.0
    x1 = model.propagate(x, dt)
    assert_allclose(x1[2], 30.0, atol=1e-6)
    assert x1[0] > 0
    assert x1[1] > 0


def test_ctra_propagate_acceleration():
    """加速度影响速度。"""
    model = CTRAModel()
    x = np.array([0.0, 0.0, 0.0, 10.0, 2.0, 0.0])
    dt = 1.0
    x1 = model.propagate(x, dt)
    assert_allclose(x1[3], 12.0, atol=1e-6)
