"""Persistence demo: save and reload radar data.

Demonstrates:
    1. Saving raw signals (NPZ), detection CSV, tracking CSV, and JSON logs
    2. Reloading saved data for post-processing analysis
    3. Using the IOManager to coordinate persistence

Run this example, then inspect the 'data/' directory for outputs.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fmcw.config.schema import RadarParams, SceneConfig, TargetSpec, MotionModel, PipelineConfig
from fmcw.pipeline.pipeline import RadarPipeline

print("=" * 60)
print("Persistence Demo: Save & Reload Radar Data")
print("=" * 60)

radar = RadarParams(fc=79e9, B=0.5e9, Tc=40e-6, chirp_num=128, sample_num=1024, antenna_num=8)

scene = SceneConfig(
    type="multi_target",
    num_frames=3,
    frame_interval=0.05,
    snr_db=25.0,
    targets=[
        TargetSpec(id=1, initial_range=50, initial_velocity=10, initial_angle=20,
                   motion=MotionModel(motion="constant_velocity")),
    ],
)

# Pipeline with persistence enabled
pipeline_cfg = PipelineConfig()
pipeline_cfg.persistence.save_intermediate = True
pipeline_cfg.persistence.output_dir = "data"
pipeline_cfg.visualization.backend = ""  # no viz for this demo

print(f"Running pipeline with persistence...")
print(f"Output directory: {pipeline_cfg.persistence.output_dir}")

pipe = RadarPipeline(radar, pipeline_cfg, scene)
results = pipe.run()

# After run, check what was saved
import glob
output_base = f"{pipeline_cfg.persistence.output_dir}/{pipe._io.run_id}"
print(f"\nSaved outputs in: {output_base}/")
print("-" * 40)

# List saved files
import os as _os
if _os.path.exists(output_base):
    for f in sorted(_os.listdir(output_base)):
        fpath = _os.path.join(output_base, f)
        if _os.path.isfile(fpath):
            size = _os.path.getsize(fpath)
            print(f"  {f:30s} {size:>8d} bytes")

# Reload and analyze
io = pipe._io
print(f"\nReloading detection results...")
det_df = io.load_detections()
if det_df is not None:
    print(f"  Detection columns: {list(det_df.columns)}")
    print(f"  Rows: {len(det_df)}")
    print(f"  Detections (is_det=1): {(det_df['is_det'] == 1).sum()}")
    print()
    print(det_df.to_string(index=False))

print(f"\nReloading track results...")
trk_df = io.load_tracks()
if trk_df is not None:
    print(f"  Track columns: {list(trk_df.columns)}")
    print(f"  Rows: {len(trk_df)}")
    print()
    print(trk_df.to_string(index=False))

print(f"\nCheck JSON logs at: {output_base}/pipeline_logs.jsonl")
print("Demo complete!")
