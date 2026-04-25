"""Quickstart: single-target radar pipeline with visualization."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fmcw.config import load_config
from fmcw.pipeline.pipeline import RadarPipeline

# Load config
cfg = load_config("fmcw/config/defaults.yaml")
radar = cfg["radar"]
pipeline_cfg = cfg["pipeline"]
scene_cfg = cfg["scene"]

print(f"Radar: {radar.fc/1e9:.0f} GHz, B={radar.B/1e6:.0f} MHz")
print(f"Resolution: range={radar.range_resolution:.2f}m, vel={radar.velocity_resolution:.3f}m/s")
print(f"Scene: {scene_cfg.num_frames} frame(s), {len(scene_cfg.targets)} target(s)")

# Build and run pipeline
pipe = RadarPipeline(radar, pipeline_cfg, scene_cfg)
results = pipe.run()

# Print results
for f_est in results["estimates"]:
    print(f"\nFrame {f_est.frame_idx}:")
    for t in f_est.targets:
        print(f"  R={t.range:.1f}m V={t.velocity:.1f}m/s A={t.angle:.1f}deg SNR={t.snr_db}dB")

print(f"\nTracking: {len(results['tracks'][-1])} active track(s)")
print("Done!")
