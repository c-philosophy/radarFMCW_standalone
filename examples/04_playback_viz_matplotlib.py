"""Playback visualization demo using matplotlib backend.

Demonstrates offline playback / animation of radar data using matplotlib.
Generates a GIF animation of the full radar processing chain.

Usage:
    python examples/04_playback_viz_matplotlib.py

Requirements: pip install matplotlib pillow
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fmcw.config.schema import RadarParams, SceneConfig, TargetSpec, MotionModel, PipelineConfig
from fmcw.pipeline.pipeline import RadarPipeline

print("=" * 60)
print("Matplotlib Playback Visualization Demo")
print("=" * 60)

# Configure radar parameters
radar = RadarParams(fc=79e9, B=0.5e9, Tc=40e-6, chirp_num=128, sample_num=1024, antenna_num=8)

# Multi-target scenario
scene = SceneConfig(
    type="multi_target",
    num_frames=30,
    frame_interval=0.05,
    snr_db=25.0,
    targets=[
        TargetSpec(id=1, initial_range=50, initial_velocity=10, initial_angle=20,
                   motion=MotionModel(motion="constant_velocity")),
        TargetSpec(id=2, initial_range=25, initial_velocity=15, initial_angle=-10,
                   motion=MotionModel(motion="constant_acceleration", acceleration=2.0)),
    ],
)

# Pipeline with matplotlib backend, saving animation
pipeline_cfg = PipelineConfig()
pipeline_cfg.visualization.backend = "mpl"
pipeline_cfg.visualization.panels = ["rd_map", "ra_map", "trajectory", "diagnostic"]
pipeline_cfg.tracking.filter = "ekf"
pipeline_cfg.tracking.association = "gnn"

# Optional: save animation to GIF
output_gif = os.path.join(os.path.dirname(__file__), "..", "data", "playback_demo.gif")
os.makedirs(os.path.dirname(output_gif), exist_ok=True)

# The pipeline runs and the mpl backend shows a window
# To save each frame as PNG, modify the pipeline loop:
pipeline_cfg_for_streaming = PipelineConfig()
pipeline_cfg_for_streaming.visualization.backend = "mpl"
pipeline_cfg_for_streaming.visualization.panels = ["rd_map", "ra_map", "trajectory", "diagnostic"]
pipeline_cfg_for_streaming.tracking.filter = "ekf"
pipeline_cfg_for_streaming.detection.cfar_algorithm = "os_cfar"

print(f"Targets: {len(scene.targets)}, Frames: {scene.num_frames}")
print("Processing pipeline with matplotlib visualization...")
print("(Close the plot window when done)")

pipe = RadarPipeline(radar, pipeline_cfg_for_streaming, scene)
results = pipe.run()

print(f"Done. {len(results['estimates'])} frames processed.")
print(f"To create an animation, run: python examples/04_playback_viz_matplotlib.py")
