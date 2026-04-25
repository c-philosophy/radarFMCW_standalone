"""CFAR algorithm comparison demo.

Compares 4 CFAR variants (CA, OS, GO, SO) on identical multi-target data
and reports detection statistics for each.

This demonstrates the plug-and-play CFAR algorithm swapping via config.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from fmcw.config.schema import RadarParams, SceneConfig, TargetSpec, MotionModel, PipelineConfig
from fmcw.pipeline.pipeline import RadarPipeline
from fmcw.core.registry import AlgorithmRegistry

print("=" * 60)
print("CFAR Algorithm Comparison")
print("=" * 60)

radar = RadarParams(fc=79e9, B=0.5e9, Tc=40e-6, chirp_num=128, sample_num=1024, antenna_num=8)

# Scene: 2 targets close together (challenging for CA-CFAR)
scene = SceneConfig(
    type="multi_target",
    num_frames=5,
    frame_interval=0.05,
    snr_db=20.0,  # lower SNR makes CFAR differences more visible
    targets=[
        TargetSpec(id=1, initial_range=30, initial_velocity=10, initial_angle=15,
                   motion=MotionModel(motion="constant_velocity")),
        TargetSpec(id=2, initial_range=35, initial_velocity=8, initial_angle=10,
                   motion=MotionModel(motion="constant_velocity")),
    ],
)

cfar_algorithms = ["ca_cfar", "os_cfar", "go_cfar", "so_cfar"]
results = {}

print(f"Scene: {len(scene.targets)} targets, {scene.num_frames} frames, SNR={scene.snr_db}dB\n")

for algo_name in cfar_algorithms:
    pipeline_cfg = PipelineConfig()
    pipeline_cfg.detection.cfar_algorithm = algo_name
    pipeline_cfg.visualization.backend = ""
    pipeline_cfg.persistence.save_intermediate = False

    pipe = RadarPipeline(radar, pipeline_cfg, scene)
    result = pipe.run()
    results[algo_name] = result

    # Collect stats across frames
    total_detections = sum(len(e.targets) for e in result["estimates"])
    confirmed_tracks_sum = 0
    for tracks in result["tracks"]:
        confirmed_tracks_sum += sum(1 for t in tracks if t.status.value == "confirmed")

    print(f"  {algo_name:>10}: {total_detections} total detections, "
          f"{confirmed_tracks_sum} confirmed track-frames"
          f"  (alpha={pipeline_cfg.detection.reference_cells}/{pipeline_cfg.detection.guard_cells})")

print("\nNote: OS-CFAR should show better detection of closely-spaced targets")
print("      GO-CFAR is better at clutter edges, SO-CFAR for close targets")
