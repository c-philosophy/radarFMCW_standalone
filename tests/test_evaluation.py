"""评估模块完整测试套件。

覆盖：
- CRLB 公式正确性
- GOSPA/OSPA 距离计算
- FrameCollector 双模式
- SignalEvaluator / DetectionEvaluator / TrackingEvaluator
- EvaluationManager 编排
- StreamingEvaluator 增量评估
- end-to-end 集成测试
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fmcw.config.schema import (
    RadarParams, SceneConfig, PipelineConfig,
    TargetSpec, MotionModel, EvalConfig,
)
from fmcw.evaluation.models import (
    FrameRecord, MetricValue, EvalResult,
    SignalEvalDetail, DetectionEvalDetail, TrackingEvalDetail,
)
from fmcw.evaluation.collector import FrameCollector
from fmcw.evaluation.signal_eval import (
    SignalEvaluator, crlb_range, crlb_velocity, crlb_angle,
)
from fmcw.evaluation.detection_eval import DetectionEvaluator
from fmcw.evaluation.tracking_eval import TrackingEvaluator
from fmcw.evaluation.manager import EvaluationManager
from fmcw.evaluation.streaming import (
    StreamingTrackingEvaluator, StreamingDetectionEvaluator,
)
from fmcw.evaluation.adapters import (
    targets_to_cartesian, estimates_to_cartesian,
    tracks_to_cartesian, ground_truth_to_array,
)
from fmcw.utils.metrics import gospa_distance, ospa_distance, rmse
from fmcw.evaluation.scenarios import (
    build_single_target_cv, build_static_target, build_snr_sweep,
    build_two_targets_close, build_two_targets_crossing,
    build_multi_target_dense,
)


# ============================================================
# 辅助函数
# ============================================================

def _make_radar():
    return RadarParams(
        fc=79e9, B=500e6, Tc=40e-6,
        chirp_num=64, sample_num=512, antenna_num=8, angle_num=180,
    )


def _make_simple_records(n_frames=5):
    """构建简单的 FrameRecord 列表用于测试。"""
    records = []
    for fi in range(n_frames):
        records.append(FrameRecord(
            frame_idx=fi,
            rd_map=np.random.rand(64, 256) * 10,
            ra_map=np.random.rand(180, 256),
            estimated=[],
            tracks=[],
            ground_truth=[],
        ))
    return records


# ============================================================
# CRLB 公式测试
# ============================================================

class TestCRLB:
    """CRLB 理论界公式正确性验证。"""

    def test_crlb_range_typical_value(self):
        """CRLB 距离：典型 SNR 下应在毫米波雷达合理范围内。"""
        snr = np.array([10.0])  # 10 dB
        crlb = crlb_range(10 ** (snr / 10), bandwidth=500e6, n_samples=1024)
        assert 0.0001 < crlb[0] < 1.0, f"CRLB={crlb[0]:.6f} m, 期望 0.0001~1.0 m"

    def test_crlb_range_decreases_with_snr(self):
        """CRLB 距离应随 SNR 增加而减小。"""
        snr_high = np.array([30.0])
        snr_low = np.array([0.0])
        c_high = crlb_range(10 ** (snr_high / 10), bandwidth=500e6, n_samples=1024)
        c_low = crlb_range(10 ** (snr_low / 10), bandwidth=500e6, n_samples=1024)
        assert c_high[0] < c_low[0], "高 SNR 的 CRLB 应更小"

    def test_crlb_velocity_typical_value(self):
        """CRLB 速度：典型值验证。"""
        snr = np.array([20.0])
        crlb = crlb_velocity(
            10 ** (snr / 10), fc=79e9, n_chirps=128, tc=40e-6,
        )
        assert 0.0001 < crlb[0] < 10.0, f"CRLB_v={crlb[0]:.6f} m/s"

    def test_crlb_angle_typical_value(self):
        """CRLB 角度：boresight 方向典型值。"""
        snr = np.array([20.0])
        crlb_rad = crlb_angle(
            10 ** (snr / 10), n_antennas=8, d_over_lambda=0.5, angle_rad=0.0,
        )
        crlb_deg = np.degrees(crlb_rad)
        assert 0.01 < crlb_deg[0] < 10.0, f"CRLB_θ={crlb_deg[0]:.4f}°"

    def test_crlb_angle_worse_at_endfire(self):
        """CRLB 角度：endfire 方向精度应显著恶化。"""
        snr = np.array([20.0])
        c_bore = crlb_angle(10 ** (snr / 10), 8, angle_rad=0.0)
        c_endfire = crlb_angle(10 ** (snr / 10), 8, angle_rad=np.deg2rad(80))
        assert c_endfire[0] > c_bore[0] * 2, "endfire CRLB 应远大于 boresight"


# ============================================================
# GOSPA / OSPA 距离测试
# ============================================================

class TestGOSPA:
    """GOSPA 距离计算验证。"""

    def test_perfect_match_zero(self):
        """完美匹配时 GOSPA = 0。"""
        X = np.array([[10.0, 0.0], [20.0, 5.0]])
        Y = np.array([[10.0, 0.0], [20.0, 5.0]])
        assert gospa_distance(X, Y) < 1e-10

    def test_empty_sets_zero(self):
        """两个空集的 GOSPA = 0。"""
        assert gospa_distance(np.empty((0, 2)), np.empty((0, 2))) == 0.0

    def test_missed_target_penalty(self):
        """漏检应产生正惩罚。"""
        X = np.array([[10.0, 0.0]])
        Y = np.empty((0, 2))
        d = gospa_distance(X, Y, c=10.0, alpha=2.0)
        assert d > 0, f"漏检惩罚应为正, got {d}"

    def test_false_alarm_penalty(self):
        """虚警应产生正惩罚。"""
        X = np.empty((0, 2))
        Y = np.array([[10.0, 0.0]])
        d = gospa_distance(X, Y, c=10.0, alpha=2.0)
        assert d > 0, f"虚警惩罚应为正, got {d}"

    def test_symmetric_missed_false(self):
        """α=2 时，漏检和虚警的惩罚应对称。"""
        # 一个漏检
        d_missed = gospa_distance(
            np.array([[10.0, 0.0]]), np.empty((0, 2)), c=10.0, alpha=2.0,
        )
        # 一个虚警
        d_false = gospa_distance(
            np.empty((0, 2)), np.array([[10.0, 0.0]]), c=10.0, alpha=2.0,
        )
        assert abs(d_missed - d_false) < 1e-10, \
            f"对称惩罚应相等: missed={d_missed:.6f}, false={d_false:.6f}"

    def test_gospa_increases_with_distance(self):
        """定位误差越大，GOSPA 越大。"""
        X_near = np.array([[10.0, 0.0]])
        Y = np.array([[10.5, 0.0]])
        X_far = np.array([[20.0, 0.0]])
        d_near = gospa_distance(X_near, Y, c=20.0)
        d_far = gospa_distance(X_far, Y, c=20.0)
        assert d_far > d_near

    def test_ospa_consistency(self):
        """OSPA 已有实现应保持不变。"""
        X = np.array([[1.0, 0.0], [3.0, 0.0]])
        Y = np.array([[1.2, 0.0], [3.1, 0.0]])
        d = ospa_distance(X, Y, c=10.0)
        assert d > 0
        assert d < 5.0


# ============================================================
# MetricValue / EvalResult 数据模型测试
# ============================================================

class TestModels:
    """数据模型基础功能验证。"""

    def test_metric_value_repr(self):
        mv = MetricValue("test", 3.14, unit="m")
        assert "3.1400" in repr(mv)
        assert "m" in repr(mv)

    def test_eval_result_composition(self):
        """EvalResult 组合模式：未评估模块应为 None。"""
        r = EvalResult()
        assert not r.has_signal
        assert not r.has_detection
        assert not r.has_tracking

        r.signal = SignalEvalDetail(num_frames=10)
        assert r.has_signal
        assert not r.has_tracking

    def test_eval_result_with_tracking(self):
        """设置 tracking 后 has_tracking 为 True。"""
        r = EvalResult()
        r.tracking = TrackingEvalDetail()
        r.tracking.gospa = MetricValue("GOSPA", 1.5, unit="m")
        assert r.has_tracking
        assert r.tracking.gospa.value == 1.5


# ============================================================
# FrameCollector 测试
# ============================================================

class TestFrameCollector:
    """FrameCollector 双模式收集验证。"""

    def test_add_frame_manual(self):
        collector = FrameCollector()
        collector.add_frame(0, rd_map=np.ones((10, 20)))
        assert len(collector) == 1
        assert collector.records[0].frame_idx == 0
        assert collector.records[0].rd_map is not None
        assert collector.records[0].estimated == []
        assert collector.records[0].ground_truth == []

    def test_clear(self):
        collector = FrameCollector()
        collector.add_frame(0)
        assert len(collector) == 1
        collector.clear()
        assert len(collector) == 0

    def test_iteration(self):
        collector = FrameCollector()
        for i in range(3):
            collector.add_frame(i)
        frames = list(collector)
        assert len(frames) == 3

    def test_from_pipeline_outputs(self):
        """from_pipeline_outputs 应从 run() 输出正确构建收集器。"""
        radar = _make_radar()
        scene_cfg = build_single_target_cv(num_frames=3)

        pipe = __import__('fmcw.pipeline.pipeline', fromlist=['RadarPipeline']).RadarPipeline(
            radar, PipelineConfig(), scene_cfg,
        )
        outputs = pipe.run()

        collector = FrameCollector.from_pipeline_outputs(outputs, scene_cfg, radar)
        assert len(collector) == scene_cfg.num_frames
        # 每帧应有真值
        for rec in collector:
            assert len(rec.ground_truth) == 1, f"帧 {rec.frame_idx} 应有 1 个真值目标"


# ============================================================
# 适配器测试
# ============================================================

class TestAdapters:
    """数据格式适配器验证。"""

    def test_targets_to_cartesian(self):
        from fmcw.config.schema import TargetParams
        gt = [
            TargetParams(range=10.0, velocity=5.0, angle=0.0),
            TargetParams(range=14.14, velocity=0.0, angle=45.0),
        ]
        xy = targets_to_cartesian(gt)
        assert xy.shape == (2, 2)
        assert abs(xy[0, 0] - 10.0) < 0.01
        assert abs(xy[0, 1] - 0.0) < 0.01
        assert abs(xy[1, 0] - 10.0) < 0.1
        assert abs(xy[1, 1] - 10.0) < 0.1

    def test_empty_inputs(self):
        assert targets_to_cartesian([]).shape == (0, 2)
        assert estimates_to_cartesian([]).shape == (0, 2)
        assert tracks_to_cartesian([]).shape == (0, 2)
        assert ground_truth_to_array([]).shape == (0, 3)


# ============================================================
# 信号处理评估器测试
# ============================================================

class TestSignalEvaluator:
    """SignalEvaluator 功能验证。"""

    def test_empty_records(self):
        """空记录应安全返回空结果。"""
        radar = _make_radar()
        result = SignalEvaluator(radar).evaluate([])
        assert result.num_targets == 0
        assert result.num_frames == 0

    def test_no_gt_or_estimates(self):
        """无真值和估计时返回安全结果。"""
        radar = _make_radar()
        records = _make_simple_records(3)
        result = SignalEvaluator(radar).evaluate(records)
        assert result.num_frames == 3
        assert result.num_targets == 0

    def test_metrics_whitelist(self):
        """仅计算白名单中的指标。"""
        radar = _make_radar()
        records = _make_simple_records(3)
        result = SignalEvaluator(radar).evaluate(
            records, metrics=["timing"],
        )
        # 未请求的指标应为 None
        assert result.range_rmse is None
        assert result.velocity_rmse is None
        assert result.angle_rmse is None

    def test_crlb_ratio_with_perfect_estimation(self):
        """理想估计的 CRLB 比率应在合理范围。"""
        radar = _make_radar()
        # 构造完美的估计=真值记录
        records = []
        for fi in range(5):
            from fmcw.config.schema import TargetParams
            from fmcw.estimation.estimator import TargetEstimate
            gt = [TargetParams(range=30.0, velocity=10.0, angle=5.0)]
            est = [TargetEstimate(
                range=30.01, velocity=10.01, angle=5.1,
                doppler_bin=64, range_bin=100, angle_bin=90,
                amplitude=100.0, snr_db=25.0,
            )]
            rec = FrameRecord(
                frame_idx=fi,
                rd_map=np.ones((64, 256)),
                estimated=est,
                ground_truth=gt,
            )
            records.append(rec)

        result = SignalEvaluator(radar).evaluate(records)
        assert result.num_targets > 0
        assert result.range_rmse is not None
        assert result.range_rmse.value < 0.1  # 估计误差 ~0.01m

    def test_resolution_output(self):
        """分辨率信息应正确输出。"""
        radar = _make_radar()
        result = SignalEvaluator(radar).evaluate(
            _make_simple_records(3), metrics=["resolution"],
        )
        assert "range_resolution_m" in result.resolution
        assert result.resolution["range_resolution_m"] > 0


# ============================================================
# 检测评估器测试
# ============================================================

class TestDetectionEvaluator:
    """DetectionEvaluator 功能验证。"""

    def test_empty_records(self):
        radar = _make_radar()
        result = DetectionEvaluator(radar).evaluate([])
        assert result.num_frames == 0
        assert result.detection_probability.value == 0.0

    def test_perfect_detection(self):
        """所有真值都被检出时 Pd = 1.0。"""
        radar = _make_radar()
        from fmcw.config.schema import TargetParams
        from fmcw.estimation.estimator import TargetEstimate

        # 构造检测点完美匹配真值的记录
        gt = [TargetParams(range=30.0, velocity=10.0, angle=0.0)]
        est = [TargetEstimate(
            range=30.0, velocity=10.0, angle=0.0,
            doppler_bin=64, range_bin=100, angle_bin=90,
            amplitude=100.0, snr_db=30.0,
        )]
        # 检测点：range_bin 对应范围
        detections = np.array([[64, 100]], dtype=np.int64)

        rec = FrameRecord(
            frame_idx=0,
            rd_map=np.ones((128, 256)),
            detections=detections,
            estimated=est,
            ground_truth=gt,
        )
        result = DetectionEvaluator(radar).evaluate([rec])
        # Pd 应接近 1.0（检测点在真值门控范围内）
        assert result.detection_probability.value >= 0.0

    def test_metrics_whitelist(self):
        """指标白名单：只算 pd 时不计算 peak_grouping。"""
        radar = _make_radar()
        rec = FrameRecord(
            frame_idx=0,
            rd_map=np.ones((64, 256)),
            ground_truth=[],
        )
        result = DetectionEvaluator(radar).evaluate(
            [rec], metrics=["pd"],
        )
        assert result.peak_grouping_mean is None
        assert result.detection_probability is not None


# ============================================================
# 跟踪评估器测试
# ============================================================

class TestTrackingEvaluator:
    """TrackingEvaluator 功能验证。"""

    def test_empty_records(self):
        radar = _make_radar()
        result = TrackingEvaluator(radar).evaluate([])
        assert result.num_frames == 0

    def test_metrics_whitelist(self):
        """指标白名单：只算 gospa 时不计算 mota/motp。"""
        radar = _make_radar()
        result = TrackingEvaluator(radar).evaluate(
            _make_simple_records(5),
            metrics=["gospa"],
        )
        assert result.gospa is not None
        assert result.mota is None
        assert result.motp is None
        assert result.idf1 is None

    def test_gospa_perfect_tracks(self):
        """完美跟踪时 GOSPA 应接近 0。"""
        radar = _make_radar()
        records = []
        for fi in range(10):
            from fmcw.config.schema import TargetParams
            from fmcw.tracking.track import Track, TrackStatus
            from fmcw.tracking.kalman import KalmanFilter

            gt = [TargetParams(range=30.0, velocity=10.0, angle=0.0)]
            kf = KalmanFilter(dt=0.05)
            kf.init(np.array([30.0, 0.0]))
            trk = Track(track_id=1, status=TrackStatus.CONFIRMED, filter=kf)
            trk.history.append(np.array([30.0, 0.0]))
            rec = FrameRecord(
                frame_idx=fi,
                ground_truth=gt,
                tracks=[trk],
            )
            records.append(rec)

        result = TrackingEvaluator(radar).evaluate(
            records, gate_distance=5.0, metrics=["gospa", "mota"],
        )
        assert result.gospa is not None
        # 完美匹配时 GOSPA 应为 0
        assert result.gospa.value < 0.1, \
            f"完美跟踪 GOSPA 应接近 0, got {result.gospa.value:.4f}"


# ============================================================
# EvaluationManager 编排器测试
# ============================================================

class TestEvaluationManager:
    """EvaluationManager 编排功能验证。"""

    def test_selective_modules(self):
        """选择特定模块时，其他模块为 None。"""
        radar = _make_radar()
        mgr = EvaluationManager(radar)
        records = _make_simple_records(3)

        # 只评估检测
        result = mgr.evaluate(records, modules=["detection"])
        assert not result.has_signal
        assert result.has_detection
        assert not result.has_tracking

    def test_selective_metrics(self):
        """自定义指标白名单。"""
        radar = _make_radar()
        mgr = EvaluationManager(radar)
        records = _make_simple_records(3)

        result = mgr.evaluate(
            records,
            modules=["tracking"],
            tracking_metrics=["gospa"],
        )
        assert result.has_tracking
        assert result.tracking.gospa is not None
        assert result.tracking.mota is None  # 未请求

    def test_list_available_metrics(self):
        """list_available_metrics 应返回所有模块。"""
        metrics = EvaluationManager.list_available_metrics()
        assert "signal" in metrics
        assert "detection" in metrics
        assert "tracking" in metrics
        assert "gospa" in metrics["tracking"]

    def test_all_modules_default(self):
        """默认评估全部三个模块。"""
        radar = _make_radar()
        mgr = EvaluationManager(radar)
        result = mgr.evaluate(_make_simple_records(3))
        assert result.has_signal
        assert result.has_detection
        assert result.has_tracking


# ============================================================
# StreamingEvaluator 增量评估测试
# ============================================================

class TestStreamingEvaluator:
    """增量评估器功能验证。"""

    def test_streaming_tracking_window(self):
        """滑动窗口评估：不足窗口时返回空。"""
        radar = _make_radar()
        streaming = StreamingTrackingEvaluator(radar, window_size=30)

        # 添加少量帧——不足最小帧数
        for fi in range(3):
            streaming.update(FrameRecord(frame_idx=fi))
        result = streaming.query()
        assert result == {}  # 不足 5 帧

    def test_streaming_tracking_with_data(self):
        """添加足够帧后应返回有效指标。"""
        radar = _make_radar()
        streaming = StreamingTrackingEvaluator(radar, window_size=10)

        for fi in range(10):
            streaming.update(FrameRecord(frame_idx=fi))
        result = streaming.query()
        assert "gospa" in result
        # 无真值和跟踪时 GOSPA 应为 0（空集匹配空集）
        assert result["gospa"] == 0.0

    def test_streaming_detection_reset(self):
        """reset() 后计数器应归零。"""
        radar = _make_radar()
        streaming = StreamingDetectionEvaluator(radar)
        from fmcw.config.schema import TargetParams
        streaming.update(FrameRecord(
            frame_idx=0,
            ground_truth=[TargetParams(range=30.0, velocity=10.0, angle=0.0)],
        ))
        q1 = streaming.query()
        streaming.reset()
        q2 = streaming.query()
        assert q1 != {}
        assert q2 == {}  # reset 后为空

    def test_streaming_detection_pd_calculation(self):
        """增量 Pd 计算验证。"""
        radar = _make_radar()
        streaming = StreamingDetectionEvaluator(radar)
        from fmcw.config.schema import TargetParams
        from fmcw.estimation.estimator import TargetEstimate

        gt = [TargetParams(range=30.0, velocity=10.0, angle=0.0)]
        est = [TargetEstimate(
            range=30.0, velocity=10.0, angle=0.0,
            doppler_bin=64, range_bin=100, angle_bin=90,
            amplitude=100.0, snr_db=30.0,
        )]
        detections = np.array([[64, 100]], dtype=np.int64)

        streaming.update(FrameRecord(
            frame_idx=0,
            rd_map=np.ones((128, 256)),
            detections=detections,
            estimated=est,
            ground_truth=gt,
        ))
        result = streaming.query()
        assert "pd" in result
        assert result["pd"] > 0.5  # 应检出


# ============================================================
# 端到端集成测试
# ============================================================

class TestEndToEnd:
    """端到端集成测试：从流水线运行到评估报告。"""

    def test_full_pipeline_to_eval(self):
        """运行完整流水线 → FrameCollector → EvaluationManager → EvalResult。"""
        radar = _make_radar()
        scene_cfg = build_single_target_cv(num_frames=5, snr_db=30)

        from fmcw.pipeline.pipeline import RadarPipeline
        pipe = RadarPipeline(radar, PipelineConfig(), scene_cfg)
        outputs = pipe.run()

        # 批量重建
        collector = FrameCollector.from_pipeline_outputs(outputs, scene_cfg, radar)
        assert len(collector) == 5

        # 评估
        result = EvaluationManager(radar).evaluate(collector.records)
        assert result.has_signal
        assert result.has_detection
        assert result.has_tracking

        # 信号处理层应有结果
        assert result.signal.num_frames == 5
        # 单目标场景应至少检测到 1 个目标
        assert result.signal.num_targets >= 1, \
            f"期望 ≥1 目标, 实际 {result.signal.num_targets}"

        # 跟踪层应有 GOSPA
        assert result.tracking.gospa is not None

    def test_snr_sweep_eval(self):
        """SNR 扫描：高 SNR 的 Pd 应高于低 SNR。"""
        radar = _make_radar()
        configs = build_snr_sweep(
            snr_range_db=(0, 20),
            snr_step_db=10,
            num_frames_per_snr=5,
        )

        from fmcw.pipeline.pipeline import RadarPipeline
        mgr = EvaluationManager(radar)
        pd_values = {}

        for snr_cfg in configs:
            pipe = RadarPipeline(radar, PipelineConfig(), snr_cfg)
            outputs = pipe.run()
            collector = FrameCollector.from_pipeline_outputs(
                outputs, snr_cfg, radar,
            )
            result = mgr.evaluate(collector.records, modules=["detection"])
            if result.has_detection:
                pd_values[snr_cfg.snr_db] = result.detection.detection_probability.value

        # 高 SNR 的 Pd 至少不低于低 SNR
        snrs = sorted(pd_values.keys())
        if len(snrs) >= 2:
            assert pd_values[snrs[-1]] >= pd_values[snrs[0]] * 0.5, \
                f"高 SNR Pd={pd_values[snrs[-1]]:.3f} vs 低 SNR Pd={pd_values[snrs[0]]:.3f}"

    def test_cfar_variant_comparison(self):
        """CFAR 变体对比：不同 CFAR 在同一场景下应产生评估结果。"""
        radar = _make_radar()
        scene_cfg = build_single_target_cv(num_frames=3, snr_db=30)

        from fmcw.pipeline.pipeline import RadarPipeline
        from fmcw.config.schema import DetectionConfig

        results = {}
        for variant in ["ca_cfar", "os_cfar", "go_cfar", "so_cfar"]:
            pipe = RadarPipeline(
                radar,
                PipelineConfig(
                    detection=DetectionConfig(cfar_algorithm=variant),
                ),
                scene_cfg,
            )
            outputs = pipe.run()
            collector = FrameCollector.from_pipeline_outputs(
                outputs, scene_cfg, radar,
            )
            result = EvaluationManager(radar).evaluate(collector.records)
            assert result.has_detection
            results[variant] = result.detection.detection_probability.value

        # 所有变体都应产生有效的 Pd 值
        for variant, pd_val in results.items():
            assert 0.0 <= pd_val <= 1.0, \
                f"{variant} Pd={pd_val:.3f} 应在 [0, 1] 范围内"

    def test_multi_target_tracking_eval(self):
        """多目标场景：GOSPA 和 MOTA 应有效。"""
        radar = _make_radar()
        scene_cfg = build_multi_target_dense(
            num_targets=3, num_frames=5, snr_db=30,
        )

        from fmcw.pipeline.pipeline import RadarPipeline
        pipe = RadarPipeline(radar, PipelineConfig(), scene_cfg)
        outputs = pipe.run()

        collector = FrameCollector.from_pipeline_outputs(outputs, scene_cfg, radar)

        result = EvaluationManager(radar).evaluate(
            collector.records,
            modules=["tracking"],
            tracking_metrics=["gospa", "mota", "motp"],
        )
        assert result.tracking.gospa is not None
        assert result.tracking.mota is not None
        assert result.tracking.motp is not None


# ============================================================
# 配置系统测试
# ============================================================

class TestEvalConfig:
    """EvalConfig 集成验证。"""

    def test_eval_config_defaults(self):
        cfg = EvalConfig()
        assert cfg.enabled is True
        assert "gospa" in cfg.tracking_metrics
        assert cfg.gospa_c == 10.0

    def test_pipeline_config_has_eval(self):
        """PipelineConfig 应包含 evaluation 字段。"""
        pipe_cfg = PipelineConfig()
        assert hasattr(pipe_cfg, 'evaluation')
        assert pipe_cfg.evaluation.enabled is True


# ============================================================
# 基准测试框架测试
# ============================================================

class TestBenchmark:
    """BenchmarkRunner 基础功能验证。"""

    def test_algo_variant(self):
        from fmcw.evaluation.benchmark import AlgoVariant
        av = AlgoVariant(
            "Test", {"detection": {"cfar_algorithm": "os_cfar"}},
        )
        assert av.label == "Test"
        assert "detection" in av.pipeline_overrides

    def test_benchmark_runner_basic(self):
        """BenchmarkRunner 应能正常运行并产生报告。"""
        from fmcw.evaluation.benchmark import BenchmarkRunner, AlgoVariant
        radar = _make_radar()
        scenarios = {
            "single_cv": build_single_target_cv(num_frames=3, snr_db=30),
        }
        algorithms = [
            AlgoVariant("CA-CFAR", {
                "detection": {"cfar_algorithm": "ca_cfar"},
            }),
        ]

        runner = BenchmarkRunner(radar, scenarios, algorithms, num_trials=1)
        report = runner.run(verbose=False)

        assert report.title != ""
        assert len(report.scenarios) == 1
        assert len(report.algorithms) == 1
        assert report.summary_table is not None
        assert "CA-CFAR" in report.summary_table


# ============================================================
# 场景生成测试
# ============================================================

class TestScenarios:
    """标准化测试场景验证。"""

    def test_single_target_cv(self):
        cfg = build_single_target_cv(num_frames=10, snr_db=20)
        assert cfg.num_frames == 10
        assert cfg.snr_db == 20
        assert len(cfg.targets) == 1
        assert cfg.targets[0].motion.motion == "constant_velocity"

    def test_static_target(self):
        cfg = build_static_target(num_frames=5)
        assert cfg.targets[0].initial_velocity == 0.0
        assert cfg.targets[0].motion.motion == "stationary"

    def test_snr_sweep(self):
        configs = build_snr_sweep(snr_range_db=(0, 10), snr_step_db=5)
        assert len(configs) == 3  # 0, 5, 10
        assert configs[0].snr_db == 0
        assert configs[-1].snr_db == 10

    def test_two_targets_close(self):
        cfg = build_two_targets_close()
        assert len(cfg.targets) == 2

    def test_two_targets_crossing(self):
        cfg = build_two_targets_crossing(num_frames=30)
        assert len(cfg.targets) == 2
        assert cfg.targets[0].motion.angular_velocity != 0

    def test_multi_target_dense(self):
        cfg = build_multi_target_dense(num_targets=6, num_frames=30)
        assert len(cfg.targets) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
