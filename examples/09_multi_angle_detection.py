"""同距离-速度、不同角度目标分辨能力验证示例。

验证方案 A1（角度多峰检测）对同一 Range-Doppler cell 中
多个角度目标的分离能力，对比 FFT 和 MUSIC 两种 DOA 方法。

测试场景：
    A) 单目标基线 — 确认正常检测不受影响
    B) 双目标同(R=50,V=10)不同角度(±30°) — 大角度分离, 核心验证
    C) 双目标同(R=50,V=10)不同角度(±15°) — 中等角度分离
    D) 双目标同(R=50,V=10)仅差 5°(±2.5°) — 分辨率极限测试

用法：
    python examples/09_multi_angle_detection.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from fmcw.config.schema import (
    RadarParams, SceneConfig, TargetSpec, MotionModel,
    PipelineConfig, EstimationConfig,
)
from fmcw.pipeline.pipeline import RadarPipeline


# ── 通用配置 ──────────────────────────────────────────────────
RADAR = RadarParams(
    fc=79e9, B=0.5e9, Tc=40e-6,
    chirp_num=64, sample_num=512, antenna_num=8,
)
FRAMES = 5
SNR = 25.0
COMMON_RANGE = 50.0
COMMON_VELOCITY = 10.0


def build_scene(name, targets):
    """构建场景配置。"""
    return SceneConfig(
        type="multi_target",
        num_frames=FRAMES,
        frame_interval=0.1,
        snr_db=SNR,
        targets=targets,
    )


def test_doa_method(name, scene, doa_method):
    """对指定场景和 DOA 方法运行流水线，返回逐帧检测数。"""
    cfg = PipelineConfig()
    cfg.visualization.backend = ""
    cfg.persistence.save_intermediate = False
    cfg.estimation = EstimationConfig(doa_method=doa_method)
    pipe = RadarPipeline(RADAR, cfg, scene)
    result = pipe.run()
    dets_per_frame = [e.targets for e in result["estimates"]]
    return dets_per_frame


# ── 场景定义 ──────────────────────────────────────────────────
SCENARIOS = {
    "A) 单目标基线": build_scene("baseline", [
        TargetSpec(
            id=1, initial_range=COMMON_RANGE,
            initial_velocity=COMMON_VELOCITY,
            initial_angle=20.0,
            motion=MotionModel(motion="constant_velocity"),
        ),
    ]),
    "B) 双目标 ±30° (同 R,V)": build_scene("wide_split", [
        TargetSpec(
            id=1, initial_range=COMMON_RANGE,
            initial_velocity=COMMON_VELOCITY,
            initial_angle=30.0,
            motion=MotionModel(motion="constant_velocity"),
        ),
        TargetSpec(
            id=2, initial_range=COMMON_RANGE,
            initial_velocity=COMMON_VELOCITY,
            initial_angle=-30.0,
            motion=MotionModel(motion="constant_velocity"),
        ),
    ]),
    "C) 双目标 ±15° (同 R,V)": build_scene("mid_split", [
        TargetSpec(
            id=1, initial_range=COMMON_RANGE,
            initial_velocity=COMMON_VELOCITY,
            initial_angle=15.0,
            motion=MotionModel(motion="constant_velocity"),
        ),
        TargetSpec(
            id=2, initial_range=COMMON_RANGE,
            initial_velocity=COMMON_VELOCITY,
            initial_angle=-15.0,
            motion=MotionModel(motion="constant_velocity"),
        ),
    ]),
    "D) 双目标 ±2.5° (同 R,V,极限)": build_scene("narrow_split", [
        TargetSpec(
            id=1, initial_range=COMMON_RANGE,
            initial_velocity=COMMON_VELOCITY,
            initial_angle=2.5,
            motion=MotionModel(motion="constant_velocity"),
        ),
        TargetSpec(
            id=2, initial_range=COMMON_RANGE,
            initial_velocity=COMMON_VELOCITY,
            initial_angle=-2.5,
            motion=MotionModel(motion="constant_velocity"),
        ),
    ]),
}


# ── 主程序 ────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("方案 A1 — 同 RD 不同角度多目标检测能力验证")
    print("=" * 65)
    print(f"雷达: {RADAR.fc / 1e9:.0f} GHz, "
          f"chirp={RADAR.chirp_num}, sample={RADAR.sample_num}, "
          f"天线={RADAR.antenna_num}")
    print(f"分辨率: range={RADAR.range_resolution:.2f} m, "
          f"velocity={RADAR.velocity_resolution:.2f} m/s")
    print()

    # ── 第一部分：FFT 模式 ──
    print("-" * 65)
    print("DOA 方法: FFT (角度多峰, 阈值=6dB, 最多3峰)")
    print("-" * 65)
    for scenario_name, scene in SCENARIOS.items():
        dets_per_frame = test_doa_method(scenario_name, scene, "fft")
        avg_det = np.mean([len(d) for d in dets_per_frame])
        # 收集第一帧的角度
        angles_f0 = [est.angle for est in dets_per_frame[0]]
        status = "OK" if len(angles_f0) == len(scene.targets) else "WARN"
        print(f"  {scenario_name}: ")
        print(f"    expected={len(scene.targets)}, "
              f"detected avg={avg_det:.1f}/frame, "
              f"frame0 angles={[f'{a:+.1f}deg' for a in angles_f0]} "
              f"[{status}]")
    print()

    # ── 第二部分：MUSIC 超分辨 ──
    print("-" * 65)
    print("DOA 方法: MUSIC (多源检测, n_sources=3, ang_res=0.5deg)")
    print("-" * 65)
    for scenario_name, scene in SCENARIOS.items():
        dets_per_frame = test_doa_method(scenario_name, scene, "music")
        avg_det = np.mean([len(d) for d in dets_per_frame])
        angles_f0 = [est.angle for est in dets_per_frame[0]]
        status = "OK" if len(angles_f0) == len(scene.targets) else "WARN"
        print(f"  {scenario_name}: ")
        print(f"    expected={len(scene.targets)}, "
              f"detected avg={avg_det:.1f}/frame, "
              f"frame0 angles={[f'{a:.1f}deg' for a in angles_f0]} "
              f"[{status}]")
    print()

    # ── 第三部分：详细打印第一帧 ──
    print("-" * 65)
    print("场景 B (±30°) 第一帧详细信息 (FFT 模式)")
    print("-" * 65)
    scene_b = SCENARIOS["B) 双目标 ±30° (同 R,V)"]
    dets_f0 = test_doa_method("B", scene_b, "fft")[0]
    print(f"  {'r(m)':>8s}  {'v(m/s)':>8s}  {'angle(deg)':>10s}  "
          f"{'r_bin':>6s}  {'d_bin':>6s}  {'a_bin':>6s}  {'SNR':>5s}")
    for est in dets_f0:
        print(f"  {est.range:8.1f}  {est.velocity:8.1f}  "
              f"{est.angle:10.1f}  {est.range_bin:6d}  "
              f"{est.doppler_bin:6d}  {est.angle_bin:6d}  "
              f"{est.snr_db:5.1f}")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
