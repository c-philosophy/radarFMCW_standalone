"""卡尔曼滤波器实现测试（KF、EKF、UKF）。"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from numpy.testing import assert_allclose

from fmcw.tracking.kalman import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter


def test_kf_predict_update_cycle():
    """KF: predict-update cycle should refine state estimate."""
    dt = 0.05
    kf = KalmanFilter(dt=dt)

    # Initialize with first measurement
    kf.init(np.array([0.0, 0.0]))

    # Predict
    kf.predict()
    assert kf.x.shape == (4,), f"State should be (4,), got {kf.x.shape}"
    assert kf.P.shape == (4, 4), f"Covariance should be (4,4), got {kf.P.shape}"

    # Update with measurement
    z = np.array([1.0, 0.0])
    kf.update(z)
    assert kf.x.shape == (4,)
    # After update, position should be closer to measurement
    assert abs(kf.x[0] - z[0]) < 2.0, "KF should move state toward measurement"


def test_kf_constant_velocity():
    """KF: should track a constant velocity target."""
    dt = 0.05
    kf = KalmanFilter(dt=dt, process_noise=0.1, measurement_noise=0.5)
    kf.init(np.array([0.0, 0.0]))

    # Simulate target moving at 10 m/s along x for 20 steps
    true_positions = []
    for i in range(20):
        true_x = 10.0 * i * dt
        true_y = 0.0
        true_positions.append([true_x, true_y])

        kf.predict()
        # Noisy measurement
        z = np.array([true_x + np.random.randn() * 0.2, true_y + np.random.randn() * 0.2])
        kf.update(z)

        # Position estimate should be close to true position
        assert abs(kf.x[0] - true_x) < 1.0, f"Step {i}: x estimate {kf.x[0]:.2f} vs true {true_x:.2f}"


def test_ekf_construction():
    """EKF: should construct and have basic attributes."""
    dt = 0.05
    ekf = ExtendedKalmanFilter(dt=dt)
    assert ekf.dt == dt
    assert ekf.x.shape == (4,)
    assert ekf.P.shape == (4, 4)


def test_ekf_predict_update():
    """EKF: predict-update with range/angle measurements."""
    dt = 0.05
    ekf = ExtendedKalmanFilter(dt=dt)
    ekf.x = np.array([50.0, 10.0, 10.0, 0.0])  # x=50, y=10, vx=10, vy=0

    ekf.predict()
    # Range/angle measurement
    meas = np.array([51.0, 11.31])  # range ~51m, angle ~11.31 deg
    ekf.update(meas)
    assert ekf.x.shape == (4,)
    # State should have been updated (not still at initial)
    assert not np.allclose(ekf.x, [50.0, 10.0, 10.0, 0.0]), "EKF should update state"


def test_ukf_construction():
    """UKF: should construct and have basic attributes."""
    dt = 0.05
    ukf = UnscentedKalmanFilter(dt=dt)
    assert ukf.n == 4
    assert ukf.m == 2
    assert ukf.x.shape == (4,)
    assert ukf.P.shape == (4, 4)


def test_ukf_sigma_points():
    """UKF: sigma points should be centered on state mean."""
    dt = 0.05
    ukf = UnscentedKalmanFilter(dt=dt)
    sigmas, Wm, Wc = ukf.sigma_points()
    assert sigmas.shape == (2 * ukf.n + 1, ukf.n)
    # Weighted mean of sigma points should equal the state
    mean = np.dot(Wm, sigmas)
    assert_allclose(mean, ukf.x, atol=1e-10)


def test_ukf_predict_update():
    """UKF: predict-update should work end-to-end."""
    dt = 0.05
    ukf = UnscentedKalmanFilter(dt=dt)
    ukf.x = np.array([50.0, 10.0, 10.0, 0.0])

    ukf.predict()
    meas = np.array([51.0, 11.31])
    ukf.update(meas)
    assert ukf.x.shape == (4,)
    # State should have moved toward the measurement
    assert ukf.x[0] > 49.0, f"x should be near 50, got {ukf.x[0]:.2f}"


def test_create_tracker():
    """Factory function should return correct filter types."""
    from fmcw.tracking.kalman import create_tracker

    kf = create_tracker("kf", dt=0.05)
    assert isinstance(kf, KalmanFilter)

    ekf = create_tracker("ekf", dt=0.05)
    assert isinstance(ekf, ExtendedKalmanFilter)

    ukf = create_tracker("ukf", dt=0.05)
    assert isinstance(ukf, UnscentedKalmanFilter)


from fmcw.tracking.motion_model import CVModel, CAModel, CTRAModel


def test_kf_with_cv_model():
    """KF 使用显式 CVModel。"""
    from fmcw.tracking.kalman import KalmanFilter
    kf = KalmanFilter(dt=0.05, model=CVModel())
    assert kf.x.shape == (4,)
    assert kf.F.shape == (4, 4)


def test_kf_with_ca_model():
    """KF 使用 CAModel（状态维度 6）。"""
    from fmcw.tracking.kalman import KalmanFilter
    kf = KalmanFilter(dt=0.05, model=CAModel())
    assert kf.x.shape == (6,)
    assert kf.F.shape == (6, 6)
    # H 矩阵应只观测前两维
    assert kf.H.shape == (2, 6)


def test_ekf_with_ca_model():
    """EKF 使用 CAModel。"""
    from fmcw.tracking.kalman import ExtendedKalmanFilter
    ekf = ExtendedKalmanFilter(dt=0.05, model=CAModel())
    assert ekf.x.shape == (6,)
    # hx 只依赖前两维
    ekf.x = np.array([50.0, 10.0, 5.0, 1.0, 0.1, 0.0])
    z = ekf.hx(ekf.x)
    assert len(z) == 2
    assert z[0] > 0  # range > 0


def test_ctra_ukf_predict():
    """UKF + CTRAModel 的 predict 应正确传播。"""
    from fmcw.tracking.kalman import UnscentedKalmanFilter
    ukf = UnscentedKalmanFilter(dt=1.0, model=CTRAModel())
    ukf.x = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0])
    ukf.predict()
    # 直线运动 1s，x 方向移动 ~10m
    assert abs(ukf.x[0] - 10.0) < 1.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
