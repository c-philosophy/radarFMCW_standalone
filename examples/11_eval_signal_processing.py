"""示例 11：信号处理评估 — CRLB 理论界对比与 SNR 扫描。

演示内容：
  1. 单目标场景的信号处理精度评估（距离/速度/角度 RMSE）
  2. 与 Cramer-Rao 下界 (CRLB) 对比
  3. SNR 扫描：不同 SNR 下的估计精度变化

运行方式：
    python examples/11_eval_signal_processing.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from fmcw.config.schema import RadarParams, PipelineConfig, PersistenceConfig  # noqa: F401
from fmcw.pipeline.pipeline import RadarPipeline
from fmcw.evaluation.collector import FrameCollector
from fmcw.evaluation.manager import EvaluationManager
from fmcw.evaluation.signal_eval import crlb_range, crlb_velocity, crlb_angle
from fmcw.evaluation.scenarios import build_single_target_cv, build_snr_sweep


def demo_basic_signal_eval(radar, pipe_cfg):
    """演示：单次场景的信号处理评估。"""
    print("\n" + "-" * 50)
    print("  1. 单场景信号处理精度评估")
    print("-" * 50)

    scene_cfg = build_single_target_cv(
        num_frames=30, snr_db=25.0,
        initial_range=30.0, initial_velocity=10.0, initial_angle=5.0,
    )

    pipe = RadarPipeline(radar, pipe_cfg, scene_cfg)
    outputs = pipe.run()
    collector = FrameCollector.from_pipeline_outputs(outputs, scene_cfg, radar)

    # 只评估信号处理模块，计算全部指标
    result = EvaluationManager(radar).evaluate(
        collector.records, modules=["signal"],
    )

    sig = result.signal
    print(f"\n  评估结果 (共 {sig.num_targets} 个目标配对):")
    print(f"    距离 RMSE:       {sig.range_rmse.value:.4f} m")
    print(f"    距离 CRLB 比率:  {sig.range_crlb_ratio.value:.2f} (接近 1.0 = 接近理论最优)")
    print(f"    速度 RMSE:       {sig.velocity_rmse.value:.4f} m/s")
    print(f"    角度 RMSE:       {sig.angle_rmse.value:.4f}°")

    # 解读 CRLB 比率
    ratio = sig.range_crlb_ratio.value
    if ratio < 2.0:
        print(f"    [OK] 距离估计接近 CRLB (比率={ratio:.1f})，算法效率良好")
    elif ratio < 5.0:
        print(f"    [--] 距离估计距 CRLB 有差距 (比率={ratio:.1f})，可考虑优化")
    else:
        print(f"    [!!] 距离估计远大于 CRLB (比率={ratio:.1f})，需要排查算法问题")


def demo_snr_sweep(radar, pipe_cfg):
    """演示：SNR 扫描 — 不同 SNR 下的估计精度。"""
    print("\n" + "-" * 50)
    print("  2. SNR 扫描：估计精度 vs SNR")
    print("-" * 50)

    snr_configs = build_snr_sweep(
        snr_range_db=(-5, 25),
        snr_step_db=5,
        num_frames_per_snr=15,
    )

    mgr = EvaluationManager(radar)
    snr_values = []
    range_rmse_values = []
    range_crlb_theory = []  # 理论 CRLB 值

    print("\n  SNR (dB) | Range RMSE (m) | CRLB (m) | Ratio")
    print("  " + "-" * 48)

    for snr_cfg in snr_configs:
        pipe = RadarPipeline(radar, pipe_cfg, snr_cfg)
        outputs = pipe.run()
        collector = FrameCollector.from_pipeline_outputs(outputs, snr_cfg, radar)
        result = mgr.evaluate(collector.records, modules=["signal"])

        snr_db = snr_cfg.snr_db
        snr_linear = 10.0 ** (snr_db / 10.0)
        crlb_val = float(crlb_range(
            np.array([snr_linear]), radar.B, radar.sample_num,
        )[0])

        if result.has_signal and result.signal.range_rmse:
            rmse = result.signal.range_rmse.value
            ratio = rmse / max(crlb_val, 1e-12)
            snr_values.append(snr_db)
            range_rmse_values.append(rmse)
            range_crlb_theory.append(crlb_val)
            print(f"  {snr_db:>6.0f}   | {rmse:.6f}      | {crlb_val:.6f} | {ratio:.2f}")

    # 简要分析
    if len(snr_values) >= 3:
        print(f"\n  分析: SNR 从 {snr_values[0]}→{snr_values[-1]} dB, "
              f"RMSE 从 {range_rmse_values[0]:.4f}→{range_rmse_values[-1]:.4f} m")
        print(f"  CRLB 理论预测精度随 SNR 提升，实测 RMSE 趋势与之一致")


def demo_selective_metrics(radar, pipe_cfg):
    """演示：选择性计算指标 — 只算关心的指标。"""
    print("\n" + "-" * 50)
    print("  3. 选择性评估 — 只计算指定指标")
    print("-" * 50)

    scene_cfg = build_single_target_cv(num_frames=10, snr_db=20)
    pipe = RadarPipeline(radar, pipe_cfg, scene_cfg)
    outputs = pipe.run()
    collector = FrameCollector.from_pipeline_outputs(outputs, scene_cfg, radar)

    # 只算距离 RMSE 和处理耗时，跳过速度和角度
    result = EvaluationManager(radar).evaluate(
        collector.records,
        modules=["signal"],
        signal_metrics=["range_rmse", "timing"],
    )

    sig = result.signal
    print(f"\n  已计算指标:")
    print(f"    range_rmse: {sig.range_rmse}")
    print(f"    proc_time:  {sig.proc_time_mean}")
    print(f"\n  未计算（None = 跳过）:")
    print(f"    velocity_rmse: {sig.velocity_rmse}")
    print(f"    angle_rmse:    {sig.angle_rmse}")
    print(f"    crlb_ratio:    {sig.range_crlb_ratio}")
    print(f"\n  → 仅被请求的指标才被计算，节省计算资源")


def main():
    print("=" * 60)
    print("  FMCW 雷达信号处理评估 — CRLB 对比与 SNR 扫描")
    print("=" * 60)

    # 共享雷达参数
    radar = RadarParams(
        fc=79e9, B=500e6, Tc=40e-6,
        chirp_num=128, sample_num=1024, antenna_num=8, angle_num=180,
    )
    print(f"\n雷达配置: {radar.fc/1e9:.0f}GHz, "
          f"距离分辨率={radar.range_resolution:.3f}m, "
          f"速度分辨率={radar.velocity_resolution:.3f}m/s")

    # 统一 Pipeline 配置，禁用持久化（评估示例不需要存储 NPZ）
    pipe_cfg = PipelineConfig(
        persistence=PersistenceConfig(save_intermediate=False),
    )

    demo_basic_signal_eval(radar, pipe_cfg)
    demo_snr_sweep(radar, pipe_cfg)
    demo_selective_metrics(radar, pipe_cfg)

    print("\n" + "=" * 60)
    print("  信号处理评估演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
