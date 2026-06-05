"""评估编排器：按需选择模块+指标，统一执行入口。

解决三个核心需求：
1. 按需选择模块：modules=["signal", "tracking"] 只评估指定模块
2. 自定义指标白名单：tracking_metrics=["gospa", "mota"] 只计算指定指标
3. 统一参数传递：GOSPA 的 c、门控距离等通过 manager 统一传入
"""

from typing import List, Optional, Dict

from fmcw.evaluation.models import EvalResult, FrameRecord
from fmcw.evaluation.signal_eval import SignalEvaluator
from fmcw.evaluation.detection_eval import DetectionEvaluator
from fmcw.evaluation.tracking_eval import TrackingEvaluator


# 所有可用评估模块的注册表
AVAILABLE_MODULES: Dict[str, dict] = {
    "signal": {
        "evaluator_cls": "SignalEvaluator",
        "result_attr": "signal",
        "metrics": [
            "range_rmse", "velocity_rmse", "angle_rmse",
            "crlb_ratio", "resolution", "timing",
        ],
    },
    "detection": {
        "evaluator_cls": "DetectionEvaluator",
        "result_attr": "detection",
        "metrics": [
            "pd", "pfa", "roc", "peak_grouping", "cfar_alpha",
        ],
    },
    "tracking": {
        "evaluator_cls": "TrackingEvaluator",
        "result_attr": "tracking",
        "metrics": [
            "gospa", "mota", "motp", "idf1", "hota", "lifecycle",
        ],
    },
}


class EvaluationManager:
    """评估编排器——按需选择模块+指标，统一执行。

    这是用户使用评估模块的主要入口。内部持有各个 Evaluator
    的实例，根据用户指定的 modules 和 metrics 参数精确控制
    评估范围和计算量。

    Usage:
        mgr = EvaluationManager(radar)

        # 1. 只评估跟踪，仅计算 GOSPA 和 MOTA
        result = mgr.evaluate(
            records,
            modules=["tracking"],
            tracking_metrics=["gospa", "mota"],
        )

        # 2. 全模块评估，使用各模块默认指标集
        result = mgr.evaluate(records)

        # 3. 快速检测对比：只算 Pd/Pfa
        result = mgr.evaluate(
            records,
            modules=["detection"],
            detection_metrics=["pd", "pfa"],
        )
    """

    def __init__(self, radar):
        """初始化评估管理器。

        Args:
            radar: RadarParams 实例。
        """
        self.radar = radar

    def evaluate(
        self,
        records: List[FrameRecord],
        modules: Optional[List[str]] = None,
        # ── 各模块的自定义指标列表（None = 使用默认全部） ──
        signal_metrics: Optional[List[str]] = None,
        detection_metrics: Optional[List[str]] = None,
        tracking_metrics: Optional[List[str]] = None,
        # ── 检测参数 ──
        pfa_design: Optional[float] = None,
        detection_range_gate_m: Optional[float] = None,
        # ── 跟踪参数 ──
        tracking_gate_m: float = 5.0,
        gospa_p: float = 2.0,
        gospa_c: float = 10.0,
        gospa_alpha: float = 2.0,
        calc_hota: bool = False,
    ) -> EvalResult:
        """按需评估。

        Args:
            records: FrameRecord 列表（由 FrameCollector 生成）。
            modules: 要评估的模块列表。可选: "signal", "detection", "tracking"。
                     None = 全部三个模块。
            signal_metrics: 信号处理指标白名单。None = 默认全部。
            detection_metrics: 检测指标白名单。None = 默认全部。
            tracking_metrics: 跟踪指标白名单。None = 默认全部。
            pfa_design: 设计虚警率（用于 Pfa 偏差评估）。
            detection_range_gate_m: 检测-真值距离门控 (m)。
            tracking_gate_m: 跟踪-真值关联门控 (m)。
            gospa_p/c/alpha: GOSPA 参数。
            calc_hota: 是否计算 HOTA（计算量较大，默认关闭）。

        Returns:
            EvalResult，仅包含被请求模块的子结果。未请求的模块为 None。
        """
        modules = modules or list(AVAILABLE_MODULES.keys())
        result = EvalResult(num_frames=len(records))

        # ── 信号处理评估 ──
        if "signal" in modules:
            evaluator = SignalEvaluator(self.radar)
            result.signal = evaluator.evaluate(
                records,
                metrics=signal_metrics,
            )

        # ── 目标检测评估 ──
        if "detection" in modules:
            evaluator = DetectionEvaluator(self.radar)
            result.detection = evaluator.evaluate(
                records,
                metrics=detection_metrics,
                pfa_design=pfa_design,
                range_gate_m=detection_range_gate_m,
            )

        # ── 目标跟踪评估 ──
        if "tracking" in modules:
            evaluator = TrackingEvaluator(self.radar)
            result.tracking = evaluator.evaluate(
                records,
                metrics=tracking_metrics,
                gate_distance=tracking_gate_m,
                gospa_p=gospa_p,
                gospa_c=gospa_c,
                gospa_alpha=gospa_alpha,
                calc_hota=calc_hota,
            )

        return result

    @classmethod
    def list_available_metrics(cls) -> Dict[str, List[str]]:
        """列出所有可用评估模块及其指标。

        Returns:
            {module_name: [metric_names]} 字典。
        """
        return {k: list(v["metrics"]) for k, v in AVAILABLE_MODULES.items()}
