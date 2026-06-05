"""示例 14：完整基准测试 — 多场景 × 多算法 全流程自动对比。

演示内容：
  1. 使用 BenchmarkRunner 进行多场景多算法对比
  2. 自动生成 Markdown 对比报告
  3. CFAR 变体 × 跟踪器 的交叉评估
  4. Monte Carlo 多次试验的均值稳定性

运行方式：
    python examples/14_eval_benchmark.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fmcw.config.schema import RadarParams
from fmcw.evaluation.benchmark import BenchmarkRunner, AlgoVariant
from fmcw.evaluation.scenarios import (
    build_single_target_cv,
    build_single_target_ca,
    build_two_targets_close,
    build_multi_target_dense,
)


def demo_cfar_benchmark(radar):
    """演示：CFAR 变体基准对比。"""
    print("\n" + "-" * 50)
    print("  1. CFAR 检测器对比基准")
    print("-" * 50)

    scenarios = {
        "single_cv_25dB": build_single_target_cv(num_frames=20, snr_db=25),
        "two_close_25dB": build_two_targets_close(
            num_frames=20, range1=30.0, range2=30.8, snr_db=25,
        ),
    }

    algorithms = [
        AlgoVariant("CA-CFAR", {"detection": {"cfar_algorithm": "ca_cfar"}}),
        AlgoVariant("OS-CFAR", {"detection": {"cfar_algorithm": "os_cfar"}}),
        AlgoVariant("GO-CFAR", {"detection": {"cfar_algorithm": "go_cfar"}}),
        AlgoVariant("SO-CFAR", {"detection": {"cfar_algorithm": "so_cfar"}}),
    ]

    runner = BenchmarkRunner(radar, scenarios, algorithms, num_trials=1)
    report = runner.run()
    print(f"\n{report.summary_table}")

    # 最佳推荐
    best = _find_best_pd(report)
    if best:
        print(f"\n  推荐: {best[0]} 在 {best[1]} 场景下 Pd 最高 = {best[2]:.3f}")


def demo_tracker_benchmark(radar):
    """演示：跟踪器基准对比。"""
    print("\n" + "-" * 50)
    print("  2. 跟踪算法对比基准")
    print("-" * 50)

    scenarios = {
        "single_cv": build_single_target_cv(num_frames=30, snr_db=25),
        "single_ca": build_single_target_ca(num_frames=30, snr_db=25),
        "multi_dense": build_multi_target_dense(num_targets=4, num_frames=20, snr_db=25),
    }

    algorithms = [
        AlgoVariant("KF+GNN", {
            "tracking": {"filter": "kf", "association": "gnn"},
        }),
        AlgoVariant("EKF+GNN", {
            "tracking": {"filter": "ekf", "association": "gnn"},
        }),
        AlgoVariant("UKF+GNN", {
            "tracking": {"filter": "ukf", "association": "gnn"},
        }),
    ]

    runner = BenchmarkRunner(radar, scenarios, algorithms, num_trials=1)
    report = runner.run()
    print(f"\n{report.summary_table}")


def demo_cross_benchmark(radar):
    """演示：CFAR × 跟踪器 交叉对比。"""
    print("\n" + "-" * 50)
    print("  3. CFAR × 跟踪器 交叉对比")
    print("-" * 50)

    scene_cfg = build_single_target_cv(num_frames=20, snr_db=20)

    scenarios = {"single_cv_20dB": scene_cfg}

    algorithms = [
        AlgoVariant("CA-CFAR + KF", {
            "detection": {"cfar_algorithm": "ca_cfar"},
            "tracking": {"filter": "kf", "association": "gnn"},
        }),
        AlgoVariant("OS-CFAR + KF", {
            "detection": {"cfar_algorithm": "os_cfar"},
            "tracking": {"filter": "kf", "association": "gnn"},
        }),
        AlgoVariant("CA-CFAR + EKF", {
            "detection": {"cfar_algorithm": "ca_cfar"},
            "tracking": {"filter": "ekf", "association": "gnn"},
        }),
    ]

    runner = BenchmarkRunner(radar, scenarios, algorithms, num_trials=1)
    report = runner.run()
    print(f"\n{report.summary_table}")

    print(f"\n  分析: 交叉对比揭示 CFAR 检测质量如何影响下游跟踪性能")
    print(f"        CFAR Pd 低 → 漏检传入跟踪 → GOSPA 升高 / MOTA 降低")


def _find_best_pd(report):
    """在报告中找出 Pd 最高的 (algo, scenario, pd)。"""
    best = None
    for (scn, algo), r in report.results.items():
        if r.has_detection and r.detection.detection_probability:
            pd = r.detection.detection_probability.value
            if best is None or pd > best[2]:
                best = (algo, scn, pd)
    return best


def main():
    print("=" * 60)
    print("  FMCW 雷达全流程基准测试")
    print("=" * 60)

    radar = RadarParams(
        fc=79e9, B=500e6, Tc=40e-6,
        chirp_num=64, sample_num=512, antenna_num=8, angle_num=180,
    )
    print(f"\n雷达: {radar.fc/1e9:.0f}GHz, {radar.B/1e6:.0f}MHz")
    print(f"分辨率: 距离={radar.range_resolution:.3f}m, "
          f"速度={radar.velocity_resolution:.3f}m/s")

    demo_cfar_benchmark(radar)
    demo_tracker_benchmark(radar)
    demo_cross_benchmark(radar)

    print("\n" + "=" * 60)
    print("  全流程基准测试完成!")
    print("  Benchmark 的核心价值：")
    print("    1) 多场景覆盖 → 避免算法只在单一场景下有效")
    print("    2) 标准指标 → GOSPA/MOTA/Pd 同时衡量不同维度")
    print("    3) 可复现 → 固定种子 + 配置驱动")
    print("=" * 60)


if __name__ == "__main__":
    main()
