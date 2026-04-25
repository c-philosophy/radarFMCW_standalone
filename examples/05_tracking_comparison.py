"""Compare KF vs EKF vs UKF tracking on the same trajectory."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import matplotlib.pyplot as plt

from fmcw.tracking.kalman import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter

# Generate a target trajectory (constant velocity)
dt = 0.1
n_frames = 100
true_x = np.arange(n_frames) * dt * 2.0       # 2 m/s in x
true_y = np.ones(n_frames) * 5.0               # constant y
true_state = np.column_stack([true_x, true_y])

# Simulate radar measurements in polar coordinates
np.random.seed(42)
true_range = np.sqrt(true_x**2 + true_y**2)
true_angle = np.rad2deg(np.arctan2(true_y, true_x))
meas_range = true_range + np.random.normal(0, 0.15, n_frames)
meas_angle = true_angle + np.random.normal(0, 0.5, n_frames)

# Initialize filters
kf = KalmanFilter(dt=dt, process_noise=0.01, measurement_noise=0.15)
ekf = ExtendedKalmanFilter(dt=dt, process_noise=0.01)
ukf = UnscentedKalmanFilter(dt=dt, process_noise=0.01)

# Init with first measurement (convert polar to Cartesian)
r0, a0 = meas_range[0], meas_angle[0]
x0 = r0 * np.cos(np.deg2rad(a0))
y0 = r0 * np.sin(np.deg2rad(a0))

kf.init(np.array([x0, y0]))
ekf.init(np.array([r0, a0]))
ukf.init(np.array([r0, a0]))

# Run filters
kf_hist, ekf_hist, ukf_hist = [], [], []
for i in range(1, n_frames):
    polar_meas = np.array([meas_range[i], meas_angle[i]])
    cart_meas = np.array([
        meas_range[i] * np.cos(np.deg2rad(meas_angle[i])),
        meas_range[i] * np.sin(np.deg2rad(meas_angle[i])),
    ])

    kf.predict()
    kf.update(cart_meas)
    kf_hist.append(kf.position.copy())

    ekf.predict()
    ekf.update(polar_meas)
    ekf_hist.append(ekf.position.copy())

    ukf.predict()
    ukf.update(polar_meas)
    ukf_hist.append(ukf.position.copy())

kf_hist = np.array(kf_hist)
ekf_hist = np.array(ekf_hist)
ukf_hist = np.array(ukf_hist)

# Compute RMSE
def calc_rmse(est, gt):
    return np.sqrt(np.mean(np.sum((est - gt)**2, axis=1)))

print(f"KF  RMSE: {calc_rmse(kf_hist, true_state[1:]):.3f} m")
print(f"EKF RMSE: {calc_rmse(ekf_hist, true_state[1:]):.3f} m")
print(f"UKF RMSE: {calc_rmse(ukf_hist, true_state[1:]):.3f} m")

# Plot
plt.figure(figsize=(8, 6))
plt.plot(true_x, true_y, "k--", linewidth=2, label="Ground Truth")
plt.plot(kf_hist[:, 0], kf_hist[:, 1], linewidth=1.5, label="KF")
plt.plot(ekf_hist[:, 0], ekf_hist[:, 1], linewidth=1.5, label="EKF")
plt.plot(ukf_hist[:, 0], ukf_hist[:, 1], linewidth=1.5, label="UKF")
plt.scatter(true_x[1:], true_y[1:], c="gray", s=5, alpha=0.3, label="Measurements")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("KF vs EKF vs UKF Tracking Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("tracking_comparison.png", dpi=150)
plt.show()
print("Saved tracking_comparison.png")
