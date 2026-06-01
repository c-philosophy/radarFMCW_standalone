"""Multi-target detection demo with CFAR comparison."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from fmcw.config import load_config, RadarParams, SceneConfig, TargetSpec, MotionModel, PipelineConfig
from fmcw.pipeline.pipeline import RadarPipeline

# Build a multi-target scenario programmatically
radar = RadarParams(fc=79e9, B=0.5e9, Tc=40e-6,
                    chirp_num=64, sample_num=512, antenna_num=8)

scene = SceneConfig(
    type="multi_target",
    num_frames=100,
    frame_interval=0.1,
    snr_db=20.0,
    targets=[
        TargetSpec(id=1, initial_range=30, initial_velocity=1, initial_angle=20,
                   motion=MotionModel(motion="constant_velocity")),
        TargetSpec(id=2, initial_range=25, initial_velocity=0, initial_angle=-10,
                   motion=MotionModel(motion="constant_velocity")),
    ],
)

pipeline_cfg = PipelineConfig()
pipeline_cfg.detection.cfar_algorithm = "ca_cfar"  # Try CA-CFAR
# pipeline_cfg.detection.alpha = 6.0
pipeline_cfg.tracking.filter = "ekf"               # Use EKF for radar measurements
pipeline_cfg.estimation.doa_method = "fft"
 
print(f"CFAR: {pipeline_cfg.detection.cfar_algorithm}")
print(f"Tracker: {pipeline_cfg.tracking.filter}")
print(f"Targets: {len(scene.targets)} targets, {scene.num_frames} frames")
# 打印雷达的最大无模糊距离、速度以及距离分辨率、速度分辨率
print(f"Max range: {radar.max_range:.2f} m")
print(f"Max velocity: {radar.max_velocity:.2f} m/s")
print(f"Range resolution: {radar.range_resolution:.2f} m")
print(f"Velocity resolution: {radar.velocity_resolution:.2f} m/s")

pipe = RadarPipeline(radar, pipeline_cfg, scene)

# # 1.Run the pipeline
# results = pipe.run()

# # Summary
# for frame_idx, f_est in enumerate(results["estimates"]):
#     tracks = results["tracks"][frame_idx]
#     detections = results["estimates"][frame_idx]
#     confirmed = [t for t in tracks if t.status.value == "confirmed"]
#     print(f"Frame {frame_idx}: {len(f_est.targets)} detections, {len(confirmed)} confirmed tracks")
#     # print(f"Confirmed tracks: {[t.id for t in confirmed]}")
#     # print(f"Detections: {[(d.id, d.range, d.velocity, d.angle) for d in detections]}")

# 2.Visualize By Matplotlib
from fmcw.visualization.mpl_viz import MatplotlibVisualizer
import matplotlib.pyplot as plt
plt.ion()   # 开启交互模式

save_dir = os.path.join(os.path.dirname(__file__), "..", "data")
save_path = os.path.join(save_dir, "matplotlib_viz.gif")
viz = MatplotlibVisualizer()
viz.setup(radar)                         # 创建 2×2 四面板布局
for frame_data in pipe.run_streaming():
    frame_idx = frame_data["frame_idx"]
    print(f"Frame {frame_idx}")
    # 输出当前帧目标检测结果
    estimates = frame_data["estimates"]
    if estimates:
        detections = estimates.targets
        print(f"Detections number: {len(detections)}")
        if len(detections) > 0:
            print('        range velocity angle range_bin doppler_bin angle_bin snr_db')
        for i, detection in enumerate(detections):
            print(f"  Detection {i}: {detection.range, detection.velocity, detection.angle, detection.range_bin, detection.doppler_bin, detection.angle_bin, detection.snr_db}")
    
    viz.update(frame_data)               # 每帧更新 4 个面板
    # viz.save_frame(f"frame_{frame_idx:04d}.png") # 可选：保存单帧图片
viz.stop()                               # 可导出为 GIF 动画

print("Done.")