"""示例 10：评估模块快速入门 — 运行流水线后评估全部三个模块。

演示评估模块最基本的使用流程：
  1. 运行 RadarPipeline 获取数据
  2. 通过 FrameCollector 批量重建帧记录
  3. 通过 EvaluationManager 评估全部模块
  4. 打印评估报告

运行方式：
    python examples/10_eval_basic.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fmcw.config.schema import RadarParams, PipelineConfig, PersistenceConfig
from fmcw.pipeline.pipeline import RadarPipeline
from fmcw.evaluation.collector import FrameCollector
from fmcw.evaluation.manager import EvaluationManager
from fmcw.evaluation.scenarios import build_single_target_cv


def main():
    print("=" * 60)
    print("  FMCW 雷达评估模块 — 快速入门")
    print("=" * 60)

    # 1. 配置雷达参数和场景
    radar = RadarParams(
        fc=79e9, B=500e6, Tc=40e-6,
        chirp_num=128, sample_num=1024, antenna_num=8, angle_num=180,
    )
    scene_cfg = build_single_target_cv(
        num_frames=30,
        initial_range=30.0,
        initial_velocity=10.0,
        initial_angle=0.0,
        snr_db=25.0,
    )
    pipeline_cfg = PipelineConfig(
        persistence=PersistenceConfig(save_intermediate=False),
    )

    print(f"\n雷达参数: {radar.fc/1e9:.0f}GHz, B={radar.B/1e6:.0f}MHz, "
          f"{radar.antenna_num}天线, {radar.chirp_num}chirps")
    print(f"理论分辨率: 距离={radar.range_resolution:.2f}m, "
          f"速度={radar.velocity_resolution:.3f}m/s")
    print(f"场景: {scene_cfg.num_frames}帧, 单目标 CV, SNR={scene_cfg.snr_db}dB")

    # 2. 运行流水线
    print("\n[1/3] 运行 Radarpipeline ...")
    pipe = RadarPipeline(radar, pipeline_cfg, scene_cfg)
    outputs = pipe.run()
    print(f"      完成 {len(outputs['signals'])} 帧处理")

    # 3. 收集数据（批量模式，不修改 pipeline）
    print("[2/3] 收集数据 (FrameCollector) ...")
    collector = FrameCollector.from_pipeline_outputs(outputs, scene_cfg, radar)
    print(f"      收集 {len(collector)} 条帧记录")

    # 4. 评估全部三个模块
    print("[3/3] 执行评估 (EvaluationManager) ...")
    mgr = EvaluationManager(radar)
    result = mgr.evaluate(collector.records)

    # 5. 打印评估报告
    print("\n" + "=" * 60)
    print("  评估报告")
    print("=" * 60)

    # ── 信号处理层 ──
    if result.has_signal:
        sig = result.signal
        print(f"\n── 信号处理评估 (共 {sig.num_targets} 个目标配对) ──")
        if sig.range_rmse:
            print(f"  距离 RMSE:     {sig.range_rmse.value:.4f} m")
        if sig.range_mae:
            print(f"  距离 MAE:      {sig.range_mae.value:.4f} m")
        if sig.range_crlb_ratio:
            print(f"  距离 CRLB比率: {sig.range_crlb_ratio.value:.2f} (理想=1.0)")
        if sig.velocity_rmse:
            print(f"  速度 RMSE:     {sig.velocity_rmse.value:.4f} m/s")
        if sig.velocity_crlb_ratio:
            print(f"  速度 CRLB比率: {sig.velocity_crlb_ratio.value:.2f}")
        if sig.angle_rmse:
            print(f"  角度 RMSE:     {sig.angle_rmse.value:.4f}°")
        if sig.angle_crlb_ratio:
            print(f"  角度 CRLB比率: {sig.angle_crlb_ratio.value:.2f}")
        if sig.proc_time_mean:
            print(f"  处理耗时均值:  {sig.proc_time_mean.value:.1f} ms/帧")
        if sig.resolution:
            print(f"  理论分辨率:    距离={sig.resolution['range_resolution_m']:.2f}m, "
                  f"速度={sig.resolution['velocity_resolution_ms']:.3f}m/s, "
                  f"角度≈{sig.resolution['angle_resolution_deg']:.1f}°")

    # ── 检测层 ──
    if result.has_detection:
        det = result.detection
        print(f"\n── 目标检测评估 (共 {det.num_frames} 帧) ──")
        if det.detection_probability:
            print(f"  检测概率 Pd:   {det.detection_probability.value:.3f}")
        if det.false_alarm_rate:
            print(f"  虚警率 Pfa:    {det.false_alarm_rate.value:.2e}")
        if det.peak_grouping_mean:
            print(f"  每目标检测数:  {det.peak_grouping_mean.value:.2f} (理想=1.0)")
        if det.cfar_alpha_error_db:
            print(f"  Pfa 偏差:      {det.cfar_alpha_error_db.value:.1f} dB")

    # ── 跟踪层 ──
    if result.has_tracking:
        trk = result.tracking
        print(f"\n── 目标跟踪评估 (共 {trk.num_frames} 帧) ──")
        if trk.gospa:
            print(f"  GOSPA 距离:    {trk.gospa.value:.3f} (越小越好)")
        if trk.mota:
            print(f"  MOTA:          {trk.mota.value:.3f} (越大越好, 最大=1.0)")
        if trk.motp:
            print(f"  MOTP:          {trk.motp.value:.3f} m (越小越好)")
        if trk.idf1:
            print(f"  IDF1:          {trk.idf1.value:.3f} (越大越好)")
        if trk.identity_switches > 0:
            print(f"  身份切换次数:  {trk.identity_switches}")
        if trk.track_lifecycle:
            lc = trk.track_lifecycle
            print(f"  航迹确认延迟:  {lc.get('confirmation_delay_frames', 0):.1f} 帧")
            print(f"  创建航迹总数:  {lc.get('total_tracks_created', 0)}")

    print("\n" + "=" * 60)
    print("  评估完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
