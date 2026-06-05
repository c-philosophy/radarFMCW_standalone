"""示例 12：CFAR 检测器对比评估 — 四种 CFAR 变体的 Pd/Pfa 对比。

演示内容：
  1. 四种 CFAR 变体 (CA/OS/GO/SO) 在单目标场景的检测性能
  2. 均匀噪声下 CA-CFAR 的理论最优性验证
  3. 多目标场景下 OS-CFAR 的抗遮蔽优势
  4. 低 SNR 场景的检测极限

运行方式：
    python examples/12_eval_cfar_comparison.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from fmcw.config.schema import (
    RadarParams, PipelineConfig, DetectionConfig, PersistenceConfig,
)
from fmcw.pipeline.pipeline import RadarPipeline
from fmcw.evaluation.collector import FrameCollector
from fmcw.evaluation.manager import EvaluationManager
from fmcw.evaluation.scenarios import (
    build_single_target_cv, build_two_targets_close,
)


def compare_cfar_variants(radar, scene_cfg, scene_label):
    """在给定场景下对比四种 CFAR 变体的检测性能。"""
    variants = ["ca_cfar", "os_cfar", "go_cfar", "so_cfar"]
    variant_labels = {
        "ca_cfar": "CA-CFAR (单元平均)",
        "os_cfar": "OS-CFAR (有序统计)",
        "go_cfar": "GO-CFAR (最大选择)",
        "so_cfar": "SO-CFAR (最小选择)",
    }

    mgr = EvaluationManager(radar)
    results = {}

    for variant in variants:
        pipe = RadarPipeline(
            radar,
            PipelineConfig(
                detection=DetectionConfig(cfar_algorithm=variant),
                persistence=PersistenceConfig(save_intermediate=False),
            ),
            scene_cfg,
        )
        outputs = pipe.run()
        collector = FrameCollector.from_pipeline_outputs(outputs, scene_cfg, radar)
        result = mgr.evaluate(
            collector.records,
            modules=["detection"],
            detection_metrics=["pd", "pfa", "peak_grouping"],
        )
        results[variant] = result.detection

    return results, variant_labels


def demo_uniform_noise_comparison(radar):
    """演示：均匀噪声下四种 CFAR 对比。"""
    print("\n" + "-" * 50)
    print("  1. 单目标均匀噪声 — CA-CFAR 理论最优")
    print("-" * 50)

    scene_cfg = build_single_target_cv(num_frames=20, snr_db=25)

    results, labels = compare_cfar_variants(radar, scene_cfg, "single")

    print(f"\n  {'变体':<22} {'Pd':>7} {'Pfa':>10} {'检测点/目标':>12}")
    print("  " + "-" * 54)
    for variant, det in results.items():
        pd_val = det.detection_probability.value if det.detection_probability else 0
        pfa_val = det.false_alarm_rate.value if det.false_alarm_rate else 0
        gp_val = det.peak_grouping_mean.value if det.peak_grouping_mean else 0
        print(f"  {labels[variant]:<22} {pd_val:>7.3f} {pfa_val:>10.2e} {gp_val:>12.2f}")

    # 分析
    print(f"\n  分析: 均匀噪声下 CA-CFAR 是理论最优检测器。")
    ca_pd = results["ca_cfar"].detection_probability.value
    print(f"  CA-CFAR Pd = {ca_pd:.3f}，可作为其他变体的基准对比")


def demo_two_close_targets(radar):
    """演示：近距双目标 — 检测 OS-CFAR 的抗目标遮蔽能力。"""
    print("\n" + "-" * 50)
    print("  2. 近距双目标 — OS-CFAR 抗遮蔽优势")
    print("-" * 50)

    # 两个目标在距离上仅相差 2×range_resolution (≈0.6m)
    scene_cfg = build_two_targets_close(
        num_frames=20, range1=30.0, range2=30.8,
        snr_db=25.0, angle1=0.0, angle2=5.0,
    )
    print(f"\n  场景: 2个目标 @ 30.0m & 30.8m (间距≈2×距离分辨率), SNR=25dB")

    results, labels = compare_cfar_variants(radar, scene_cfg, "two_close")

    print(f"\n  {'变体':<22} {'Pd':>7} {'检测点/目标':>12}")
    print("  " + "-" * 44)
    for variant, det in results.items():
        pd_val = det.detection_probability.value if det.detection_probability else 0
        gp_val = det.peak_grouping_mean.value if det.peak_grouping_mean else 0
        print(f"  {labels[variant]:<22} {pd_val:>7.3f} {gp_val:>12.2f}")

    # 分析：CA-CFAR 在紧邻目标时会产生目标遮蔽（弱目标被强目标拉高门限）
    ca_pd = results["ca_cfar"].detection_probability.value
    os_pd = results["os_cfar"].detection_probability.value
    print(f"\n  分析:")
    print(f"    CA-CFAR Pd={ca_pd:.3f} (可能受目标遮蔽影响)")
    print(f"    OS-CFAR Pd={os_pd:.3f} (抗遮蔽能力较强)")
    if os_pd >= ca_pd:
        print(f"    → OS-CFAR 在密集目标场景下表现更优 (Pd 高 {os_pd-ca_pd:.3f})")


def demo_low_snr_detection(radar):
    """演示：低 SNR 下 CFAR 检测极限。"""
    print("\n" + "-" * 50)
    print("  3. 低 SNR 检测极限 — 仅对比 CA-CFAR")
    print("-" * 50)

    mgr = EvaluationManager(radar)
    snr_list = [30, 25, 20, 15, 10, 5, 0]

    print(f"\n  {'SNR (dB)':>8} {'Pd':>7}")
    print("  " + "-" * 18)

    for snr in snr_list:
        scene_cfg = build_single_target_cv(num_frames=15, snr_db=snr)
        pipe = RadarPipeline(
            radar,
            PipelineConfig(persistence=PersistenceConfig(save_intermediate=False)),
            scene_cfg,
        )
        outputs = pipe.run()
        collector = FrameCollector.from_pipeline_outputs(outputs, scene_cfg, radar)
        result = mgr.evaluate(
            collector.records,
            modules=["detection"],
            detection_metrics=["pd"],
        )
        pd_val = result.detection.detection_probability.value if result.has_detection else 0
        bar = "█" * int(pd_val * 20) if pd_val > 0 else ""
        print(f"  {snr:>8} {pd_val:>7.3f}  {bar}")

    print(f"\n  分析: 高 SNR (≥20dB) 下 CA-CFAR 检测可靠 (Pd≈1.0)")
    print(f"        低 SNR (≤10dB) 下 Pd 显著下降，体现了 CFAR 的 SNR 依赖特性")


def main():
    print("=" * 60)
    print("  FMCW 雷达 CFAR 检测器对比评估")
    print("=" * 60)

    radar = RadarParams(
        fc=79e9, B=500e6, Tc=40e-6,
        chirp_num=64, sample_num=512, antenna_num=8, angle_num=180,
    )

    print(f"\n雷达: {radar.fc/1e9:.0f}GHz, B={radar.B/1e6:.0f}MHz")
    print(f"距离分辨率: {radar.range_resolution:.3f}m")

    demo_uniform_noise_comparison(radar)
    demo_two_close_targets(radar)
    demo_low_snr_detection(radar)

    print("\n" + "=" * 60)
    print("  CFAR 对比评估完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
