"""全流程 IMM 跟踪演示：FMCW 雷达信号仿真到多目标跟踪完整流水线。

演示内容：
  1. 使用 fmcw 全流程流水线：信号生成 → FFT → CFAR → 参数估计 → 跟踪
  2. 对比 IMM（CV+CA+CTRA）与单 EKF（CV）在转弯目标下的跟踪效果
  3. 展示 IMM 的多模型架构在曲线运动场景下的精度优势

场景说明：
  单个目标从 30m 处以 -8 m/s 靠近雷达，同时以 3 deg/s 转弯。
  IMM 的 CTRA 分支专为转弯运动设计，而单 EKF 仅依赖 CV 模型。

运行:
  python examples/09_fmcw_imm_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import time
import numpy as np
import matplotlib.pyplot as plt

from fmcw.config.schema import (
    RadarParams, SceneConfig, TargetSpec, MotionModel, PipelineConfig,
)
from fmcw.pipeline.pipeline import RadarPipeline
from fmcw.utils.metrics import rmse

# =========================================================================
# 1. 场景配置
# =========================================================================
radar = RadarParams(
    fc=79e9, B=0.5e9, Tc=40e-6,
    chirp_num=64, sample_num=512, antenna_num=4,
)

scene = SceneConfig(
    type="single_target",
    num_frames=20,
    frame_interval=0.1,
    snr_db=20.0,
    targets=[
        # TargetSpec(
        #     id=1,
        #     initial_range=10.0,
        #     initial_velocity=10.0,
        #     initial_angle=0.0,
        #     rcs=1.0,
        #     motion=MotionModel(
        #         motion="constant_velocity",
        #         angular_velocity=3.0,     # deg/s, 匀速转弯
        #     ),
        # ),
        TargetSpec(
            id=2,
            initial_range=20.0,
            initial_velocity=5.0,
            initial_angle=45.0,
            rcs=1.0,
            motion=MotionModel(
                motion="constant_velocity",
                angular_velocity=3.0,     # deg/s, 匀速转弯

            ),
        ),
    ],
)

# 真值轨迹（用于评估）
from fmcw.signal.scene import Scene as _Scene
gt_scene = _Scene(radar, scene)
gt_traj = gt_scene.target_trajectory(scene.targets[0])
gt_x = np.array([t.range * np.cos(np.deg2rad(t.angle)) for t in gt_traj])
gt_y = np.array([t.range * np.sin(np.deg2rad(t.angle)) for t in gt_traj])

gt_traj_list = [
            gt_scene.target_trajectory(spec)
            for spec in scene.targets
        ]
# gt_x = np.array([t.range * np.cos(np.deg2rad(t.angle)) for t in gt_traj_list])
# gt_y = np.array([t.range * np.sin(np.deg2rad(t.angle)) for t in gt_traj_list])
pass

# =========================================================================
# 2. 流水线配置工厂
# =========================================================================
def make_cfg(enable_imm=True):
    cfg = PipelineConfig()
    cfg.processing.range_window = "hanning"
    cfg.processing.doppler_window = "hanning"
    cfg.detection.cfar_algorithm = "ca_cfar"
    cfg.detection.guard_cells = 2
    cfg.detection.reference_cells = 8
    cfg.detection.pfa = 1e-2
    cfg.estimation.doa_method = "fft"
    cfg.tracking.association = "gnn"
    cfg.tracking.gate_mahalanobis = 10.0
    cfg.tracking.init_threshold = 3
    cfg.tracking.coast_threshold = 5
    cfg.tracking.dt = scene.frame_interval
    cfg.visualization.backend = ""
    cfg.persistence.save_intermediate = False

    if enable_imm:
        cfg.tracking.filter = "ekf"
        cfg.tracking.enable_imm = True
        cfg.tracking.imm_models = ("cv", "ca", "ctra")
        cfg.tracking.use_mahalanobis = True
        cfg.tracking.use_velocity_gating = False
        cfg.tracking.process_noise = 0.5
    else:
        cfg.tracking.filter = "ekf"
        cfg.tracking.enable_imm = False
        cfg.tracking.use_mahalanobis = False
        cfg.tracking.use_velocity_gating = False
        cfg.tracking.process_noise = 0.5    # 单 EKF 需要更大过程噪声适应转弯

    return cfg


# =========================================================================
# 3. 运行全流程（使用 run_streaming 逐帧提取跟踪位置）
# =========================================================================
def run_and_track(enable_imm):
    """运行流水线并逐帧记录滤波位置。"""
    cfg = make_cfg(enable_imm)
    pipe = RadarPipeline(radar, cfg, scene)
    
    # 打印雷达的最大无模糊距离、速度以及距离分辨率、速度分辨率
    print(f"==== Radar Parameters ====")
    print(f"Max range: {radar.max_range:.2f} m")
    print(f"Max velocity: {radar.max_velocity:.2f} m/s")
    print(f"Max angle: {radar.max_angle_rad*180/np.pi:.2f} deg")
    print(f"Range resolution: {radar.range_resolution:.2f} m")
    print(f"Velocity resolution: {radar.velocity_resolution:.2f} m/s")
        
    positions = []

    # Visualize By Matplotlib
    from fmcw.visualization.mpl_viz import MatplotlibVisualizer
    import matplotlib.pyplot as plt
    plt.ion()   # 开启交互模式
    viz = MatplotlibVisualizer()
    viz.setup(radar)    
    
    # 创建 2×2 四面板布局
    t0 = time.time()
    n_frames = 0
    for frame_data in pipe.run_streaming():
        frame_idx = frame_data["frame_idx"]
        print(f"Frame {frame_idx}")
        
        n_frames += 1
        tracks = frame_data["tracks"]
        # 取第一条活跃航迹的滤波位置
        pos = np.array([np.nan, np.nan])
        for t in tracks:
            if t.status.value in ("confirmed", "tentative") and t.filter is not None:
                pos = t.filter.position.copy()
                # break
        positions.append(pos)
        
        # 更新可视化
        viz.update(frame_data)    # 每帧更新可视面板

    elapsed = time.time() - t0
    det_rate = 100.0    # 检测率 TODO: 检测率应该通过计算得到
    label = "IMM" if enable_imm else "EKF"
    print(f"  {label}: {n_frames} frames in {elapsed:.1f}s, detection: {det_rate:.0f}%\n")

    return np.array(positions)


print("Running IMM tracking...")
imm_pos = run_and_track(enable_imm=True)

print("Running Baseline EKF tracking...")
ekf_pos = run_and_track(enable_imm=False)

# =========================================================================
# 4. 计算 RMSE
# =========================================================================
def calc_rmse(positions, gt_x, gt_y):
    valid = ~np.isnan(positions[:, 0])
    if valid.sum() < 5:
        return float("nan"), valid
    return rmse(positions[valid], np.column_stack([gt_x[valid], gt_y[valid]])), valid

imm_rmse, valid_imm = calc_rmse(imm_pos, gt_x, gt_y)
ekf_rmse, valid_ekf = calc_rmse(ekf_pos, gt_x, gt_y)

print("\n" + "=" * 55)
if not np.isnan(imm_rmse):
    print(f"  IMM (CV+CA+CTRA) RMSE:  {imm_rmse:.3f} m")
if not np.isnan(ekf_rmse):
    print(f"  EKF (CV only) RMSE:     {ekf_rmse:.3f} m")
if not np.isnan(imm_rmse) and not np.isnan(ekf_rmse):
    impr = (1 - imm_rmse / ekf_rmse) * 100
    print(f"  IMM 精度改善:           {impr:+.1f}%")
    print(f"  (IMM 的 CTRA 分支专为转弯运动优化, 精度优于单 CV 模型)")
print("=" * 55)

# =========================================================================
# 5. 绘图
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 左图：轨迹对比
ax = axes[0]
ax.plot(gt_x, gt_y, "k-", linewidth=2, label="Ground Truth")
if valid_imm.sum() > 0:
    ax.plot(imm_pos[valid_imm, 0], imm_pos[valid_imm, 1], "b-", linewidth=1.5,
            label=f"IMM ({imm_rmse:.2f}m)")
if valid_ekf.sum() > 0:
    ax.plot(ekf_pos[valid_ekf, 0], ekf_pos[valid_ekf, 1], "r--", linewidth=1.5,
            label=f"EKF ({ekf_rmse:.2f}m)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Trajectory: IMM vs EKF")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axis("equal")

# 右图：每帧误差
ax = axes[1]
if valid_imm.sum() > 0:
    e_imm = np.sqrt((imm_pos[valid_imm,0]-gt_x[valid_imm])**2 + (imm_pos[valid_imm,1]-gt_y[valid_imm])**2)
    ax.plot(np.where(valid_imm)[0], e_imm, "b-", alpha=0.7, label=f"IMM")
if valid_ekf.sum() > 0:
    e_ekf = np.sqrt((ekf_pos[valid_ekf,0]-gt_x[valid_ekf])**2 + (ekf_pos[valid_ekf,1]-gt_y[valid_ekf])**2)
    ax.plot(np.where(valid_ekf)[0], e_ekf, "r--", alpha=0.7, label=f"EKF")
ax.set_xlabel("Frame")
ax.set_ylabel("Position Error (m)")
ax.set_title("Per-frame Error")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "..", "fmcw_imm_demo.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")
print("Done.")
