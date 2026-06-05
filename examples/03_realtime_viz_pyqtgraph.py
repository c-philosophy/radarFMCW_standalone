"""Real-time visualization demo using pyqtgraph backend.

Demonstrates the pyqtgraph real-time visualization backend with GPU-accelerated
rendering for live radar signal processing.

Requirements: pip install pyqtgraph PyQt5 (or PySide6)

Note: On headless systems or systems without display, this example will fail.
Use the matplotlib backend (example 04) for offline rendering.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fmcw.config.schema import RadarParams, SceneConfig, TargetSpec, MotionModel, PipelineConfig
from fmcw.pipeline.pipeline import RadarPipeline

print("=" * 60)
print("PyQtGraph Real-time Visualization Demo")
print("=" * 60)

# Configure radar parameters
radar = RadarParams(fc=79e9, B=0.5e9, Tc=40e-6, 
                    chirp_num=64, sample_num=512, antenna_num=8)

# Multi-target scenario
scene = SceneConfig(
    type="multi_target",
    num_frames=50,
    frame_interval=0.1,
    snr_db=20.0,
    targets=[
        TargetSpec(id=1, initial_range=15, initial_velocity=10, initial_angle=5,
                   motion=MotionModel(motion="constant_velocity")),
        # TargetSpec(id=2, initial_range=15, initial_velocity=10, initial_angle=-5,
        #            motion=MotionModel(motion="constant_velocity")),
        # TargetSpec(id=3, initial_range=20, initial_velocity=5, initial_angle=15,
        #            motion=MotionModel(motion="constant_velocity")),
        TargetSpec(id=4, initial_range=20, initial_velocity=10, initial_angle=-15,
                   motion=MotionModel(motion="constant_velocity")),
        TargetSpec(id=5, initial_range=20, initial_velocity=10, initial_angle=15,
                   motion=MotionModel(motion="constant_velocity")),
        # TargetSpec(id=6, initial_range=20, initial_velocity=-8, initial_angle=45,
        #            motion=MotionModel(motion="constant_velocity")),
        # TargetSpec(id=7, initial_range=20, initial_velocity=8, initial_angle=-45,
        #            motion=MotionModel(motion="constant_velocity")),
        # TargetSpec(id=8, initial_range=10, initial_velocity=6, initial_angle=90,
        #            motion=MotionModel(motion="constant_velocity")),
    ],
)

# Pipeline with pyqtgraph backend
pipeline_cfg = PipelineConfig()
pipeline_cfg.visualization.backend = "pyqtgraph" # "mpl" or "pyqtgraph"
pipeline_cfg.visualization.update_hz = 2
pipeline_cfg.visualization.panels = ["rd_map", "ra_map", "trajectory", "diagnostic"]
pipeline_cfg.tracking.filter = "kf"
pipeline_cfg.detection.cfar_algorithm = "ca_cfar"
pipeline_cfg.estimation.doa_method = "music"  # "music" / "esprit" / "mvdr"/ "fft"
enable_imm = False
if enable_imm:
    pipeline_cfg.tracking.filter = "ekf"
    pipeline_cfg.tracking.enable_imm = True
    pipeline_cfg.tracking.imm_models = ("cv", "ca", "ctra")
    pipeline_cfg.tracking.use_mahalanobis = True
    pipeline_cfg.tracking.use_velocity_gating = False
    pipeline_cfg.tracking.process_noise = 0.5

print(f"Backend: {pipeline_cfg.visualization.backend}")
print(f"Targets: {len(scene.targets)}, Frames: {scene.num_frames}")
print("Opening pyqtgraph window... (close the window to stop)")
# 打印雷达的最大无模糊距离、速度以及距离分辨率、速度分辨率
print(f"Max range: {radar.max_range:.2f} m")
print(f"Max velocity: {radar.max_velocity:.2f} m/s")
print(f"Range resolution: {radar.range_resolution:.2f} m")
print(f"Velocity resolution: {radar.velocity_resolution:.2f} m/s")

pipe = RadarPipeline(radar, pipeline_cfg, scene)
results = pipe.run(verbose=True)

print(f"Done. Processed {len(results['estimates'])} frames.")
