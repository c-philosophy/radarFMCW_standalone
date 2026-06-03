"""IMM 交互多模型跟踪演示：对比 IMM 与单 EKF 在机动目标场景下的表现。

场景说明：
  目标以 10 m/s 沿 x 轴匀速直线运动，在 frame 40 后逐渐转向，
  模拟车辆变道/转弯场景。

  对比项：
    - IMM（三分支：EKF-CV + EKF-CA + UKF-CTRA）
    - 单 EKF（仅 CV 模型）
    - 输出 RMSE 对比和轨迹图

运行:
  python examples/09_imm_tracking_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt

from fmcw.tracking.imm import InteractingMultipleModel
from fmcw.tracking.kalman import ExtendedKalmanFilter, UnscentedKalmanFilter
from fmcw.tracking.motion_model import CVModel, CAModel, CTRAModel
from fmcw.utils.metrics import rmse

# =========================================================================
# 1. 生成机动目标真值轨迹（笛卡尔坐标）
# =========================================================================
np.random.seed(42)
dt = 0.1
n_frames = 100

true_x = np.zeros(n_frames)
true_y = np.zeros(n_frames)
v = 10.0

# 目标从 (0, 50) 左前方开始靠近并转弯，确保角度始终良定义
true_x[0] = 0.0
true_y[0] = 50.0

for i in range(1, n_frames):
    if i < 40:
        # 向 x 正方向 + y 负方向运动（沿 y 轴靠近 + x 方向移动）
        true_x[i] = true_x[i-1] + v * 0.8 * dt
        true_y[i] = true_y[i-1] - v * 0.6 * dt
    else:
        # 逐渐转弯
        turn_rate = min((i - 40) * 1.0, 20.0)
        rad = np.deg2rad(turn_rate)
        dx = v * 0.8 * np.cos(rad) * dt - v * 0.6 * np.sin(rad) * dt
        dy = -v * 0.6 * np.cos(rad) * dt - v * 0.8 * np.sin(rad) * dt
        true_x[i] = true_x[i-1] + dx
        true_y[i] = true_y[i-1] + dy

# 生成含噪声的雷达极坐标测量值
true_range = np.sqrt(true_x**2 + true_y**2)
true_angle = np.rad2deg(np.arctan2(true_y, true_x))
meas_range = true_range + np.random.normal(0, 0.15, n_frames)
meas_angle = true_angle + np.random.normal(0, 0.3, n_frames)

# =========================================================================
# 2. 构建跟踪器
# =========================================================================

# 2a. IMM（三分支）
ekf_cv = ExtendedKalmanFilter(dt=dt, process_noise=0.1, model=CVModel())
ekf_ca = ExtendedKalmanFilter(dt=dt, process_noise=0.1, model=CAModel())
ukf_ctra = UnscentedKalmanFilter(dt=dt, process_noise=0.1, model=CTRAModel())

imm = InteractingMultipleModel(
    branches=[("cv", ekf_cv), ("ca", ekf_ca), ("ctra", ukf_ctra)],
    init_probs=np.array([0.8, 0.1, 0.1]),
)

# 2b. 单 EKF 基线（仅 CV）
ekf_baseline = ExtendedKalmanFilter(dt=dt, process_noise=0.1, model=CVModel())

# =========================================================================
# 3. 初始化
# =========================================================================
z0 = np.array([meas_range[0], meas_angle[0]])
imm.init(z0)
ekf_baseline.init(z0)

imm_hist = np.zeros((n_frames, 2))
ekf_hist = np.zeros((n_frames, 2))

# =========================================================================
# 4. 逐帧跟踪
# =========================================================================
for i in range(1, n_frames):
    z = np.array([meas_range[i], meas_angle[i]])

    imm.predict()
    imm.update(z)
    imm_hist[i] = imm.position

    ekf_baseline.predict()
    ekf_baseline.update(z)
    ekf_hist[i] = ekf_baseline.position

imm_hist[0] = imm.position
ekf_hist[0] = ekf_baseline.position

# =========================================================================
# 5. 计算 RMSE
# =========================================================================
true_xy = np.column_stack([true_x[1:], true_y[1:]])
imm_rmse = rmse(imm_hist[1:], true_xy)
ekf_rmse = rmse(ekf_hist[1:], true_xy)

print("=" * 55)
print(f"  IMM (CV+CA+CTRA) RMSE: {imm_rmse:.3f} m")
print(f"  单 EKF (CV) RMSE:     {ekf_rmse:.3f} m")
improvement = (1 - imm_rmse / ekf_rmse) * 100
print(f"  改善幅度:              {improvement:+.1f}%")
print("=" * 55)

# =========================================================================
# 6. 绘图（使用英文避免 CJK 字体缺失警告）
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 左图：轨迹对比
ax1 = axes[0]
ax1.plot(true_x, true_y, "k-", linewidth=2, label="Ground Truth")
ax1.plot(imm_hist[:, 0], imm_hist[:, 1], "b-", linewidth=1.5,
         label=f"IMM (RMSE={imm_rmse:.3f}m)")
ax1.plot(ekf_hist[:, 0], ekf_hist[:, 1], "r--", linewidth=1.5,
         label=f"EKF (RMSE={ekf_rmse:.3f}m)")
ax1.axvline(x=true_x[40], color="gray", linestyle=":", alpha=0.5)
ax1.annotate("Turn starts", xy=(true_x[40], true_y[40]),
             xytext=(true_x[40] + 2, 2), fontsize=9, color="gray")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_title("Trajectory: IMM vs Single EKF")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.axis("equal")

# 右图：逐帧位置误差
ax2 = axes[1]
pos_error_imm = np.sqrt((imm_hist[:, 0] - true_x)**2 + (imm_hist[:, 1] - true_y)**2)
pos_error_ekf = np.sqrt((ekf_hist[:, 0] - true_x)**2 + (ekf_hist[:, 1] - true_y)**2)
ax2.plot(range(n_frames), pos_error_ekf, "r-", alpha=0.7,
         label=f"EKF (mean={pos_error_ekf[1:].mean():.2f}m)")
ax2.plot(range(n_frames), pos_error_imm, "b-", alpha=0.7,
         label=f"IMM (mean={pos_error_imm[1:].mean():.2f}m)")
ax2.axvline(x=40, color="gray", linestyle=":", alpha=0.5)
ax2.set_xlabel("Frame")
ax2.set_ylabel("Position Error (m)")
ax2.set_title("Per-frame Position Error")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "..", "imm_tracking_demo.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")
print("Done.")
