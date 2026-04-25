"""Multi-target detection demo with CFAR comparison."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from fmcw.config import load_config, RadarParams, SceneConfig, TargetSpec, MotionModel, PipelineConfig
from fmcw.pipeline.pipeline import RadarPipeline

# Build a multi-target scenario programmatically
radar = RadarParams(fc=79e9, B=0.5e9, Tc=40e-6, chirp_num=128, sample_num=1024)

scene = SceneConfig(
    type="multi_target",
    num_frames=10,
    frame_interval=0.05,
    snr_db=25.0,
    targets=[
        TargetSpec(id=1, initial_range=50, initial_velocity=10, initial_angle=20,
                   motion=MotionModel(motion="constant_velocity")),
        TargetSpec(id=2, initial_range=25, initial_velocity=20, initial_angle=-10,
                   motion=MotionModel(motion="constant_velocity")),
    ],
)

pipeline_cfg = PipelineConfig()
pipeline_cfg.detection.cfar_algorithm = "os_cfar"  # Try OS-CFAR
pipeline_cfg.tracking.filter = "ekf"               # Use EKF for radar measurements

print(f"CFAR: {pipeline_cfg.detection.cfar_algorithm}")
print(f"Tracker: {pipeline_cfg.tracking.filter}")
print(f"Targets: {len(scene.targets)} targets, {scene.num_frames} frames")

pipe = RadarPipeline(radar, pipeline_cfg, scene)
results = pipe.run()

# Summary
for frame_idx, f_est in enumerate(results["estimates"]):
    tracks = results["tracks"][frame_idx]
    confirmed = [t for t in tracks if t.status.value == "confirmed"]
    print(f"Frame {frame_idx}: {len(f_est.targets)} detections, {len(confirmed)} confirmed tracks")
