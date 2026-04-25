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
radar = RadarParams(fc=79e9, B=0.5e9, Tc=40e-6, chirp_num=128, sample_num=1024, antenna_num=8)

# Multi-target scenario
scene = SceneConfig(
    type="multi_target",
    num_frames=100,
    frame_interval=0.05,
    snr_db=25.0,
    targets=[
        TargetSpec(id=1, initial_range=50, initial_velocity=10, initial_angle=20,
                   motion=MotionModel(motion="constant_velocity")),
        TargetSpec(id=2, initial_range=35, initial_velocity=-5, initial_angle=-15,
                   motion=MotionModel(motion="constant_velocity")),
    ],
)

# Pipeline with pyqtgraph backend
pipeline_cfg = PipelineConfig()
pipeline_cfg.visualization.backend = "pyqtgraph"
pipeline_cfg.visualization.update_hz = 20
pipeline_cfg.visualization.panels = ["rd_map", "ra_map", "trajectory", "diagnostic"]
pipeline_cfg.tracking.filter = "ekf"
pipeline_cfg.detection.cfar_algorithm = "os_cfar"

print(f"Backend: {pipeline_cfg.visualization.backend}")
print(f"Targets: {len(scene.targets)}, Frames: {scene.num_frames}")
print("Opening pyqtgraph window... (close the window to stop)")

pipe = RadarPipeline(radar, pipeline_cfg, scene)
results = pipe.run()

print(f"Done. Processed {len(results['estimates'])} frames.")
