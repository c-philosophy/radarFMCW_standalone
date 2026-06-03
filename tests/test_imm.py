"""IMM 交互多模型单元测试。"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from numpy.testing import assert_allclose

from fmcw.tracking.imm import (
    InteractingMultipleModel,
    cv_to_ctra, ctra_to_cv,
    ca_to_ctra, ctra_to_ca,
    cv_to_ca, ca_to_cv,
)
from fmcw.tracking.kalman import ExtendedKalmanFilter, UnscentedKalmanFilter
from fmcw.tracking.motion_model import CVModel, CAModel, CTRAModel


# ── 状态转换测试 ──────────────────────────────────────────────────────

def test_cv_ctra_roundtrip():
    x_cv = np.array([10.0, 5.0, 3.0, 4.0])  # v=5, yaw≈53.13°
    x_ctra = cv_to_ctra(x_cv)
    assert_allclose(x_ctra[0], 10.0)
    assert_allclose(x_ctra[3], 5.0, atol=1e-6)
    x_cv_back = ctra_to_cv(x_ctra)
    assert_allclose(x_cv_back, x_cv, atol=1e-6)


def test_ca_ctra_roundtrip():
    x_ca = np.array([10.0, 5.0, 3.0, 4.0, 0.5, 0.0])
    x_ctra = ca_to_ctra(x_ca)
    x_ca_back = ctra_to_ca(x_ctra)
    assert_allclose(x_ca_back[:4], x_ca[:4], atol=1e-6)


def test_cv_ca_roundtrip():
    x_cv = np.array([10.0, 5.0, 3.0, 4.0])
    x_ca = cv_to_ca(x_cv)
    assert len(x_ca) == 6
    assert_allclose(x_ca[:4], x_cv)
    assert_allclose(x_ca[4:], [0, 0])
    x_cv_back = ca_to_cv(x_ca)
    assert_allclose(x_cv_back, x_cv)


# ── IMM 构建和基本操作测试 ──────────────────────────────────────────

def _make_imm(dt=0.05):
    """辅助: 创建 3 分支 IMM。"""
    ekf_cv = ExtendedKalmanFilter(dt=dt, model=CVModel())
    ekf_ca = ExtendedKalmanFilter(dt=dt, model=CAModel())
    ukf_ctra = UnscentedKalmanFilter(dt=dt, model=CTRAModel())
    return InteractingMultipleModel(
        branches=[("cv", ekf_cv), ("ca", ekf_ca), ("ctra", ukf_ctra)],
    )


def test_imm_three_branches():
    imm = _make_imm()
    assert len(imm.names) == 3
    assert imm.names == ["cv", "ca", "ctra"]
    assert imm.probs.shape == (3,)


def test_imm_predict():
    imm = _make_imm()
    imm.predict()
    for name in imm.names:
        assert imm.branches[name].filter.x is not None


def test_imm_update():
    imm = _make_imm(dt=0.05)
    z = np.array([50.0, 10.0])
    for br in imm.branches.values():
        br.filter.init(z)

    imm.predict()
    imm.update(np.array([51.0, 10.5]))

    assert imm.probs[0] > 0
    pos = imm.position
    assert_allclose(np.linalg.norm(pos), 51.0, atol=5.0)


def test_imm_constant_velocity_track():
    """匀速跟踪场景: IMM 位置误差应小于单帧噪声。"""
    imm = _make_imm(dt=0.1)
    z = np.array([50.0, 0.0])
    for br in imm.branches.values():
        br.filter.init(z)

    true_x = np.array([50.0 + i * 0.5 for i in range(5)])
    true_y = np.zeros(5)

    for i in range(5):
        imm.predict()
        imm.update(np.array([true_x[i], 0.0]))  # 无噪声测量

    pos = imm.position
    error = np.sqrt((pos[0] - true_x[-1])**2 + pos[1]**2)
    assert error < 1.0, f"IMM 匀速跟踪误差过大: {error:.3f} m"
