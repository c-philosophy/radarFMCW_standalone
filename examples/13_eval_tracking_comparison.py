"""示例 13：跟踪算法对比评估 — KF/EKF/UKF 的 GOSPA 和 MOTA 对比。

演示内容：
  1. 三种卡尔曼滤波器在单目标匀速场景的跟踪精度
  2. GOSPA/MOTA/MOTP/IDF1 全套跟踪指标
  3. 双目标交叉场景下的数据关联挑战
  4. 多目标密集场景的综合评估

运行方式：
    python examples/13_eval_tracking_comparison.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from fmcw.config.schema import (
    RadarParams, PipelineConfig, TrackingConfig, PersistenceConfig,
)
from fmcw.pipeline.pipeline import RadarPipeline
from fmcw.evaluation.collector import FrameCollector
from fmcw.evaluation.manager import EvaluationManager
from fmcw.evaluation.scenarios import (
    build_single_target_cv,
    build_single_target_ca,
    build_two_targets_crossing,
    build_multi_target_dense,
)


def compare_trackers(radar, scene_cfg, trackers, scene_label):
    """对比不同跟踪器在同一场景下的表现。"""
    tracker_labels = {
        "kf":  "KF  (线性卡尔曼)",
        "ekf": "EKF (扩展卡尔曼)",
        "ukf": "UKF (无迹卡尔曼)",
    }

    mgr = EvaluationManager(radar)
    results = {}

    for tracker in trackers:
        pipe = RadarPipeline(
            radar,
            PipelineConfig(
                tracking=TrackingConfig(filter=tracker, association="gnn"),
                persistence=PersistenceConfig(save_intermediate=False),
            ),
            scene_cfg,
        )
        outputs = pipe.run()
        collector = FrameCollector.from_pipeline_outputs(outputs, scene_cfg, radar)
        result = mgr.evaluate(
            collector.records,
            modules=["tracking"],
            tracking_metrics=["gospa", "mota", "motp", "idf1", "lifecycle"],
        )
        results[tracker] = result.tracking

    return results, tracker_labels


def demo_cv_tracking(radar):
    """演示：匀速运动下三者几乎等价。"""
    print("\n" + "-" * 50)
    print("  1. 单目标匀速 (CV) — KF/EKF/UKF 对比")
    print("-" * 50)

    scene_cfg = build_single_target_cv(
        num_frames=50, snr_db=25,
        initial_range=30.0, initial_velocity=10.0, initial_angle=5.0,
    )

    results, labels = compare_trackers(
        radar, scene_cfg, ["kf", "ekf", "ukf"], "cv",
    )

    print(f"\n  {'跟踪器':<20} {'GOSPA':>8} {'MOTA':>8} {'MOTP(m)':>8} {'IDF1':>8} {'确认延迟':>8}")
    print("  " + "-" * 65)
    best_gospa = min(r.gospa.value for r in results.values() if r.gospa)
    for tracker, trk in results.items():
        gs = f"{trk.gospa.value:.3f}" if trk.gospa else "N/A"
        mt = f"{trk.mota.value:.3f}" if trk.mota else "N/A"
        mp = f"{trk.motp.value:.3f}" if trk.motp else "N/A"
        i1 = f"{trk.idf1.value:.3f}" if trk.idf1 else "N/A"
        cd = f"{trk.track_lifecycle.get('confirmation_delay_frames', 0):.0f}" if trk.track_lifecycle else "N/A"
        star = " ★" if trk.gospa and trk.gospa.value <= best_gospa * 1.01 else ""
        print(f"  {labels.get(tracker, tracker):<20} {gs:>8} {mt:>8} {mp:>8} {i1:>8} {cd:>8}{star}")

    print(f"\n  分析: 匀速直线运动下，三者性能几乎等价（KF 略优，模型匹配）")
    print(f"        ★ 标记为最低 GOSPA（最佳综合性能）")


def demo_ca_tracking(radar):
    """演示：匀加速运动 — EKF/UKF 的优势。"""
    print("\n" + "-" * 50)
    print("  2. 单目标匀加速 (CA) — 机动跟踪考验")
    print("-" * 50)

    scene_cfg = build_single_target_ca(
        num_frames=50, snr_db=25,
        initial_range=50.0, initial_velocity=5.0, acceleration=2.0,
    )

    results, labels = compare_trackers(
        radar, scene_cfg, ["kf", "ekf", "ukf"], "ca",
    )

    print(f"\n  {'跟踪器':<20} {'GOSPA':>8} {'MOTA':>8} {'MOTP(m)':>8} {'IDF1':>8}")
    print("  " + "-" * 55)
    for tracker, trk in results.items():
        gs = f"{trk.gospa.value:.3f}" if trk.gospa else "N/A"
        mt = f"{trk.mota.value:.3f}" if trk.mota else "N/A"
        mp = f"{trk.motp.value:.3f}" if trk.motp else "N/A"
        i1 = f"{trk.idf1.value:.3f}" if trk.idf1 else "N/A"
        print(f"  {labels.get(tracker, tracker):<20} {gs:>8} {mt:>8} {mp:>8} {i1:>8}")

    print(f"\n  分析: 匀加速场景下 KF 模型失配 (CV model)，EKF/UKF 非线性处理")
    print(f"        加速运动使线性 KF 的残差增大，GOSPA 随之上升")


def demo_crossing_targets(radar):
    """演示：双目标交叉 — 数据关联考验。"""
    print("\n" + "-" * 50)
    print("  3. 双目标交叉轨迹 — 数据关联 + 身份保持")
    print("-" * 50)

    scene_cfg = build_two_targets_crossing(num_frames=60, snr_db=25)

    results, labels = compare_trackers(
        radar, scene_cfg, ["kf", "ekf", "ukf"], "crossing",
    )

    print(f"\n  {'跟踪器':<20} {'GOSPA':>8} {'MOTA':>8} {'IDF1':>8} {'IDSW':>6}")
    print("  " + "-" * 55)
    for tracker, trk in results.items():
        gs = f"{trk.gospa.value:.3f}" if trk.gospa else "N/A"
        mt = f"{trk.mota.value:.3f}" if trk.mota else "N/A"
        i1 = f"{trk.idf1.value:.3f}" if trk.idf1 else "N/A"
        sw = str(trk.identity_switches) if trk.identity_switches else "0"
        print(f"  {labels.get(tracker, tracker):<20} {gs:>8} {mt:>8} {i1:>8} {sw:>6}")

    print(f"\n  分析: IDF1 衡量身份保持能力，IDSW 统计身份切换次数")
    print(f"        交叉点附近最易发生身份切换，IDF1 下降")


def demo_multi_target_dense(radar):
    """演示：多目标密集场景 — 综合评估。"""
    print("\n" + "-" * 50)
    print("  4. 多目标密集场景 — 综合评估")
    print("-" * 50)

    scene_cfg = build_multi_target_dense(
        num_targets=4, num_frames=30, snr_db=25,
    )

    results, labels = compare_trackers(
        radar, scene_cfg, ["kf", "ekf", "ukf"], "dense",
    )

    print(f"\n  {'跟踪器':<20} {'GOSPA':>8} {'MOTA':>8} {'MOTP(m)':>8} {'航迹数':>6}")
    print("  " + "-" * 55)
    for tracker, trk in results.items():
        gs = f"{trk.gospa.value:.3f}" if trk.gospa else "N/A"
        mt = f"{trk.mota.value:.3f}" if trk.mota else "N/A"
        mp = f"{trk.motp.value:.3f}" if trk.motp else "N/A"
        nt = str(trk.track_lifecycle.get('total_tracks_created', 0)) if trk.track_lifecycle else "N/A"
        print(f"  {labels.get(tracker, tracker):<20} {gs:>8} {mt:>8} {mp:>8} {nt:>6}")

    print(f"\n  分析: 多目标场景下，GOSPA 综合衡量定位+基数误差")
    print(f"        GOSPA 越低→跟踪越接近真值（定位准+数量对）")


def main():
    print("=" * 60)
    print("  FMCW 雷达跟踪算法对比评估")
    print("=" * 60)

    radar = RadarParams(
        fc=79e9, B=500e6, Tc=40e-6,
        chirp_num=64, sample_num=512, antenna_num=8, angle_num=180,
    )
    print(f"\n雷达: {radar.fc/1e9:.0f}GHz, {radar.antenna_num}天线")
    print(f"距离分辨率: {radar.range_resolution:.3f}m")
    print(f"帧间隔: 50ms (20Hz)")

    demo_cv_tracking(radar)
    demo_ca_tracking(radar)
    demo_crossing_targets(radar)
    demo_multi_target_dense(radar)

    print("\n" + "=" * 60)
    print("  跟踪算法对比评估完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
