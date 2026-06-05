"""基准测试框架：多场景、多算法配置的对比评估执行器。

提供 AlgoVariant（算法变体描述）和 BenchmarkRunner（对比执行器）。
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
import time
import numpy as np

from fmcw.config.schema import (
    RadarParams, PipelineConfig, SceneConfig,
    DetectionConfig, TrackingConfig, PersistenceConfig,
)
from fmcw.pipeline.pipeline import RadarPipeline
from fmcw.evaluation.collector import FrameCollector
from fmcw.evaluation.manager import EvaluationManager
from fmcw.evaluation.models import EvalResult, BenchmarkReport


@dataclass
class AlgoVariant:
    """算法变体描述——一次评估中的一个算法配置。

    Usage:
        AlgoVariant("CA-CFAR + KF + GNN",
                     {"detection": {"cfar_algorithm": "ca_cfar"},
                      "tracking": {"filter": "kf", "association": "gnn"}})
    """
    label: str
    pipeline_overrides: dict = field(default_factory=dict)


class BenchmarkRunner:
    """多场景、多算法对比基准测试执行器。

    Usage:
        runner = BenchmarkRunner(radar, scenarios, algorithms, num_trials=3)
        report = runner.run()
        print(report.summary_table)
    """

    def __init__(
        self,
        radar: RadarParams,
        scenarios: Dict[str, SceneConfig],
        algorithms: List[AlgoVariant],
        num_trials: int = 1,
        seed: int = 42,
    ):
        self.radar = radar
        self.scenarios = scenarios
        self.algorithms = algorithms
        self.num_trials = num_trials
        self.rng = np.random.RandomState(seed)

    def run(self, verbose: bool = True) -> BenchmarkReport:
        """执行全部基准测试。

        Returns:
            BenchmarkReport 包含所有场景×算法×试验的结果。
        """
        report = BenchmarkReport(
            title="FMCW Radar Evaluation Benchmark",
            scenarios=list(self.scenarios.keys()),
            algorithms=[a.label for a in self.algorithms],
        )

        for scn_name, scene_cfg in self.scenarios.items():
            for algo in self.algorithms:
                if verbose:
                    print(f"  [{scn_name}] {algo.label} ...", end=" ", flush=True)
                t0 = time.perf_counter()

                trial_results = []
                for trial in range(self.num_trials):
                    seed = self.rng.randint(0, 2**31)
                    result = self._run_single(scene_cfg, algo, seed=seed)
                    trial_results.append(result)

                aggregated = self._aggregate_trials(trial_results)
                t_elapsed = time.perf_counter() - t0

                if verbose:
                    trk = aggregated.tracking
                    gs = (
                        f"GOSPA={trk.gospa.value:.2f}"
                        if (trk and trk.gospa) else ""
                    )
                    print(f"{t_elapsed:.1f}s {gs}")

                report.results[(scn_name, algo.label)] = aggregated

        report.summary_table = self._build_summary_table(report)
        return report

    def _run_single(
        self, scene_cfg: SceneConfig, algo: AlgoVariant, seed: int,
    ) -> EvalResult:
        """运行单次评估试验。"""
        pipeline_cfg = self._build_pipeline_config(algo)

        # 运行流水线
        pipe = RadarPipeline(self.radar, pipeline_cfg, scene_cfg)
        outputs = pipe.run()

        # 使用 FrameCollector 批量重建（真值通过 Scene 独立计算）
        collector = FrameCollector.from_pipeline_outputs(
            outputs, scene_cfg, self.radar,
        )

        # 通过 EvaluationManager 按需评估（默认全部三个模块）
        mgr = EvaluationManager(self.radar)
        result = mgr.evaluate(collector.records)
        result.scenario_name = scene_cfg.type
        result.algorithm_config = algo.pipeline_overrides
        return result

    def _build_pipeline_config(self, algo: AlgoVariant) -> PipelineConfig:
        """构建合并算法覆盖后的 PipelineConfig。"""
        det_cfg = DetectionConfig()
        trk_cfg = TrackingConfig()

        if "detection" in algo.pipeline_overrides:
            for k, v in algo.pipeline_overrides["detection"].items():
                setattr(det_cfg, k, v)
        if "tracking" in algo.pipeline_overrides:
            for k, v in algo.pipeline_overrides["tracking"].items():
                setattr(trk_cfg, k, v)

        return PipelineConfig(
            detection=det_cfg, tracking=trk_cfg,
            persistence=PersistenceConfig(save_intermediate=False),
        )

    def _aggregate_trials(self, trials: List[EvalResult]) -> EvalResult:
        """聚合多次 Monte Carlo 试验结果。

        当前为简化实现（返回首次结果）。完整实现应计算
        每个 MetricValue 的 mean/std/ci_95。
        """
        if not trials:
            return EvalResult()

        result = trials[0]
        if self.num_trials <= 1:
            return result

        # 收集所有试验的指标值并计算聚合
        # 注：完整实现应遍历 result 中所有 MetricValue 子字段。
        # 此处为基础框架，后续按需扩展。
        return result

    def _build_summary_table(self, report: BenchmarkReport) -> str:
        """生成 Markdown 对比表。"""
        lines = [
            "| Scenario | Algorithm | Range RMSE (m) | Angle RMSE (°) | Pd | GOSPA | MOTA |",
            "|----------|-----------|---------------|----------------|----|-------|------|",
        ]
        for (scn, algo), r in report.results.items():
            rr = (
                f"{r.signal.range_rmse.value:.3f}"
                if r.has_signal and r.signal.range_rmse else "N/A"
            )
            ar = (
                f"{r.signal.angle_rmse.value:.2f}"
                if r.has_signal and r.signal.angle_rmse else "N/A"
            )
            pd = (
                f"{r.detection.detection_probability.value:.3f}"
                if r.has_detection and r.detection.detection_probability else "N/A"
            )
            gs = (
                f"{r.tracking.gospa.value:.2f}"
                if r.has_tracking and r.tracking.gospa else "N/A"
            )
            mt = (
                f"{r.tracking.mota.value:.3f}"
                if r.has_tracking and r.tracking.mota else "N/A"
            )
            lines.append(
                f"| {scn} | {algo} | {rr} | {ar} | {pd} | {gs} | {mt} |"
            )
        return "\n".join(lines)
