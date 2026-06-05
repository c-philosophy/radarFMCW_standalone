"""评估数据模型：帧记录、指标值、评估结果、基准报告。

所有数据类均使用 dataclass 定义，保持与 fmcw/config/schema.py 风格一致。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


# ============================================================
# 单帧数据记录
# ============================================================

@dataclass
class FrameRecord:
    """单帧全量数据记录：流水线所有阶段的输入输出 + 真值。

    这是评估模块的核心数据结构。每帧的 FrameRecord 由
    FrameCollector 从流水线输出中提取，供后续评估器消费。
    """
    frame_idx: int
    timestamp: float = 0.0                     # 相对场景起始时间 (s)

    # ── 信号处理层 ──
    signal: Optional[np.ndarray] = None         # 原始 IF 信号 (A, D, S)
    rd_map: Optional[np.ndarray] = None         # RD 幅度图 (D, R)
    ra_map: Optional[np.ndarray] = None         # RA 幅度图 (A_angle, R)
    s_rda: Optional[np.ndarray] = None          # 3D FFT 立方体 (A_angle, D, R)
    proc_time_ms: float = 0.0                   # 本帧 FFT 处理耗时

    # ── 检测层 ──
    detections: Optional[np.ndarray] = None     # (M, 2) 检测点 [d_idx, r_idx]
    det_time_ms: float = 0.0                    # 本帧检测耗时

    # ── 估计层 ──
    estimated: List = field(default_factory=list)   # List[TargetEstimate]
    est_time_ms: float = 0.0

    # ── 跟踪层 ──
    tracks: List = field(default_factory=list)      # List[Track]
    trk_time_ms: float = 0.0

    # ── 真值 ──
    ground_truth: List = field(default_factory=list)  # List[TargetParams]


# ============================================================
# 指标值
# ============================================================

@dataclass
class MetricValue:
    """单个评估指标的值，支持置信区间。

    对于蒙特卡洛多次试验，存储均值和标准差；
    对于单次确定性计算，std_dev=None。
    """
    name: str                                   # 指标名称，如 "range_rmse_m"
    value: float                                # 均值（或单次值）
    std_dev: Optional[float] = None             # 标准差（多次试验）
    unit: str = ""                              # 单位，如 "m", "m/s", "deg"
    ci_95: Optional[tuple] = None               # 95% 置信区间 (lower, upper)
    direction: str = "lower_better"             # "lower_better" | "higher_better"
    reference_value: Optional[float] = None     # 参考/理论值（如 CRLB）

    def __repr__(self) -> str:
        std = f" ± {self.std_dev:.4f}" if self.std_dev is not None else ""
        return f"{self.name}={self.value:.4f}{std} {self.unit}".strip()


# ============================================================
# 子模块评估详情
# ============================================================

@dataclass
class SignalEvalDetail:
    """信号处理评估的详细结果。"""
    num_frames: int = 0
    num_targets: int = 0

    range_rmse: Optional[MetricValue] = None
    range_mae: Optional[MetricValue] = None
    range_crlb_ratio: Optional[MetricValue] = None

    velocity_rmse: Optional[MetricValue] = None
    velocity_crlb_ratio: Optional[MetricValue] = None

    angle_rmse: Optional[MetricValue] = None
    angle_crlb_ratio: Optional[MetricValue] = None

    resolution: dict = field(default_factory=dict)
    proc_time_mean: Optional[MetricValue] = None


@dataclass
class DetectionEvalDetail:
    """目标检测评估的详细结果。"""
    num_frames: int = 0

    detection_probability: Optional[MetricValue] = None     # Pd
    false_alarm_rate: Optional[MetricValue] = None          # Pfa actual
    cfar_alpha_error_db: Optional[MetricValue] = None       # Pfa 偏差 (dB)
    peak_grouping_mean: Optional[MetricValue] = None        # 每目标检测点数
    roc_data: dict = field(default_factory=dict)
    per_snr_pd: dict = field(default_factory=dict)          # {snr_bin: pd}


@dataclass
class TrackingEvalDetail:
    """目标跟踪评估的详细结果。"""
    num_frames: int = 0

    gospa: Optional[MetricValue] = None
    mota: Optional[MetricValue] = None
    motp: Optional[MetricValue] = None
    idf1: Optional[MetricValue] = None
    hota: Optional[MetricValue] = None

    identity_switches: int = 0
    track_fragmentation: int = 0
    track_lifecycle: dict = field(default_factory=dict)


# ============================================================
# 评估结果（组合模式）
# ============================================================

@dataclass
class EvalResult:
    """单次评估运行的完整结果集合——组合模式。

    不预设所有模块的指标字段，而是通过三个可选子对象按需组合。
    哪个模块被评估，对应的子对象就非 None；未评估的模块为 None。

    Usage:
        # 只评估跟踪
        result = manager.evaluate(records, modules=["tracking"])
        if result.has_tracking:
            print(result.tracking.gospa)
    """
    # ── 元数据 ──
    scenario_name: str = ""
    num_frames: int = 0
    algorithm_config: dict = field(default_factory=dict)
    total_runtime_ms: float = 0.0

    # ── 按需组合的子结果（None = 该模块未被评估） ──
    signal: Optional[SignalEvalDetail] = None
    detection: Optional[DetectionEvalDetail] = None
    tracking: Optional[TrackingEvalDetail] = None

    # ── 便捷属性 ──
    @property
    def has_signal(self) -> bool:
        return self.signal is not None

    @property
    def has_detection(self) -> bool:
        return self.detection is not None

    @property
    def has_tracking(self) -> bool:
        return self.tracking is not None

    # ── 逐帧详情（可选，调试用） ──
    per_frame_metrics: List[dict] = field(default_factory=list)


# ============================================================
# 基准测试报告
# ============================================================

@dataclass
class BenchmarkReport:
    """多场景、多算法配置的对比评估报告。"""
    title: str = ""
    scenarios: List[str] = field(default_factory=list)
    algorithms: List[str] = field(default_factory=list)
    results: dict = field(default_factory=dict)  # (scenario, algo) → EvalResult
    summary_table: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)


# ============================================================
# 增量评估状态
# ============================================================

@dataclass
class StreamingEvalState:
    """增量评估器内部累积状态容器。"""
    total_frames: int = 0
    metrics_history: List[dict] = field(default_factory=list)
    latest_snapshot: dict = field(default_factory=dict)
