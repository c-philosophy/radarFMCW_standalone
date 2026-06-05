"""毫米波FMCW雷达全流程评估模块。

提供信号处理、目标检测、目标跟踪三大功能模块的
性能评估能力，支持离线批处理和在线增量两种评估模式。

主要入口:
    - EvaluationManager: 统一评估编排器（推荐）
    - FrameCollector: 数据收集器（流水线连接点）
    - SignalEvaluator / DetectionEvaluator / TrackingEvaluator: 独立评估器
    - BenchmarkRunner: 多场景多算法对比
    - StreamingEvaluator: 增量在线评估

版本: 1.0.0
"""

from fmcw.evaluation.models import (
    FrameRecord,
    MetricValue,
    EvalResult,
    BenchmarkReport,
    SignalEvalDetail,
    DetectionEvalDetail,
    TrackingEvalDetail,
)
from fmcw.evaluation.collector import FrameCollector
from fmcw.evaluation.manager import EvaluationManager
from fmcw.evaluation.signal_eval import (
    SignalEvaluator,
    crlb_range,
    crlb_velocity,
    crlb_angle,
)
from fmcw.evaluation.detection_eval import DetectionEvaluator
from fmcw.evaluation.tracking_eval import TrackingEvaluator
from fmcw.evaluation.streaming import (
    StreamingEvaluator,
    StreamingTrackingEvaluator,
    StreamingDetectionEvaluator,
)

__all__ = [
    # 数据模型
    "FrameRecord",
    "MetricValue",
    "EvalResult",
    "BenchmarkReport",
    "SignalEvalDetail",
    "DetectionEvalDetail",
    "TrackingEvalDetail",
    # 收集器
    "FrameCollector",
    # 编排器
    "EvaluationManager",
    # 评估器
    "SignalEvaluator",
    "DetectionEvaluator",
    "TrackingEvaluator",
    # CRLB 函数
    "crlb_range",
    "crlb_velocity",
    "crlb_angle",
    # 增量评估
    "StreamingEvaluator",
    "StreamingTrackingEvaluator",
    "StreamingDetectionEvaluator",
]
