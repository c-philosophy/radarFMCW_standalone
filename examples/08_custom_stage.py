"""Custom DAG pipeline stage demo.

Demonstrates how to create and insert a custom processing stage into the
DAGPipeline, showing the plug-and-play stage architecture.

This example creates a "mti_filter" stage (Moving Target Indicator) that
cancels stationary clutter and inserts it between the doppler_fft and
angle_fft stages. The DAG automatically orders stages based on their
declared input and output keys.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from fmcw.config.schema import RadarParams, SceneConfig, TargetSpec, PipelineConfig
from fmcw.core.stage import Stage
from fmcw.core.pipeline import DAGPipeline
from fmcw.core.context import PipelineContext


# Step 1: Define a custom stage
class MTIFilterStage(Stage):
    """Moving Target Indicator (MTI) filter stage.

    Cancels stationary clutter by subtracting the mean across chirps.

    Inputs:  s_rd  (range-doppler complex data)
    Outputs: s_rd_filtered  (clutter-suppressed range-doppler data)
    """

    def __init__(self):
        super().__init__(
            name="mti_filter",
            inputs=["s_rd"],
            outputs=["s_rd_filtered"],
        )

    def run(self, ctx: PipelineContext) -> None:
        s_rd = ctx["s_rd"]

        # Subtract the mean across chirps (removes zero-Doppler / stationary components)
        chirp_mean = np.mean(s_rd, axis=1, keepdims=True)
        s_rd_filtered = s_rd - chirp_mean

        ctx["s_rd_filtered"] = s_rd_filtered
        print(f"    [MTI Filter] Applied on shape {s_rd.shape}, "
              f"clutter power: {float(np.mean(np.abs(chirp_mean))):.4f}")


# Step 2: Demonstrate DAG pipeline construction with custom stage
def main():
    print("=" * 60)
    print("Custom DAG Pipeline Stage Demo")
    print("=" * 60)

    radar = RadarParams(fc=79e9, B=0.5e9, Tc=40e-6, chirp_num=128, sample_num=1024, antenna_num=8)

    scene = SceneConfig(
        type="single_target",
        num_frames=2,
        frame_interval=0.05,
        snr_db=30.0,
        targets=[
            TargetSpec(id=1, initial_range=50, initial_velocity=10, initial_angle=20),
        ],
    )

    # Step 3: Build a DAG pipeline manually with stages
    from fmcw.signal.scene import Scene
    from fmcw.processing.range_fft import range_fft as do_range_fft
    from fmcw.processing.doppler_fft import doppler_fft as do_doppler_fft
    from fmcw.processing.angle_fft import angle_fft as do_angle_fft

    radar_scene = Scene(radar, scene)
    print("\nGenerating signals...")

    pipe = DAGPipeline()

    # Add stages in any order — DAG resolver orders them by input/output keys
    # Stage 1: Range FFT
    pipe.add_stage(Stage(
        name="range_fft",
        inputs=["raw_signal"],
        outputs=["s_r"],
        run_fn=lambda ctx: ctx.set("s_r", do_range_fft(ctx["raw_signal"])),
    ))

    # Stage 2: Doppler FFT
    pipe.add_stage(Stage(
        name="doppler_fft",
        inputs=["s_r"],
        outputs=["s_rd"],
        run_fn=lambda ctx: ctx.set("s_rd", do_doppler_fft(ctx["s_r"])),
    ))

    # Stage 3: Custom MTI filter (consumes s_rd, produces s_rd_filtered)
    pipe.add_stage(MTIFilterStage())

    # Stage 4: Angle FFT (consumes s_rd_filtered from MTI filter)
    pipe.add_stage(Stage(
        name="angle_fft",
        inputs=["s_rd_filtered"],
        outputs=["s_rda"],
        run_fn=lambda ctx: ctx.set("s_rda", do_angle_fft(ctx["s_rd_filtered"], radar.angle_num)),
    ))

    print(f"DAG pipeline stages: {list(pipe._stages.keys())}")
    print(f"Execution order: {pipe.execution_order}")

    # Process one frame
    print("\nProcessing frame 0...")
    frame_idx, signal, targets = next(radar_scene.stream())
    ctx = PipelineContext(raw_signal=signal, frame_idx=frame_idx, targets=targets)
    result_ctx = pipe.run(ctx)

    print(f"\nPipeline context keys: {list(result_ctx._data.keys())}")
    print(f"s_rda shape: {result_ctx['s_rda'].shape}")
    print("\nCustom stage demo completed successfully!")


if __name__ == "__main__":
    # Ensure algorithm registrations are loaded
    import fmcw.detection.cfar      # noqa
    import fmcw.tracking.kalman     # noqa
    import fmcw.tracking.association  # noqa
    main()
