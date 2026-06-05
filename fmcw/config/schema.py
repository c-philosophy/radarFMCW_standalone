"""配置数据类：雷达参数、目标和流水线配置。"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from fmcw.utils.constants import C


@dataclass
class RadarParams:
    """FMCW 雷达系统参数。

    字段命名遵循雷达信号处理文献惯例：
        - fc: 中心频率（Hz）- RF 工程标准符号
        - B: 带宽（Hz）- 带宽通用符号
        - Tc: 调频周期（s）- 'c' 表示 chirp
        - Fs: 采样频率（Hz）- 标准符号

    所有派生参数（S, wl, antenna_spacing, Fs, 分辨率, 最大范围）在
    __post_init__ 中自动计算。
    """

    # --- 调频波形参数 ---
    fc: float = 79.0e9                # Center frequency (Hz)
    B: float = 0.5e9                  # Bandwidth (Hz)
    Tc: float = 40.0e-6               # Chirp duration (s)

    # --- 采样参数 ---
    chirp_num: int = 128              # Chirps per frame
    sample_num: int = 1024            # ADC samples per chirp
    antenna_num: int = 8              # RX antenna count
    angle_num: int = 180              # Angle FFT output bins (zero-padded)

    # --- RF 链参数 ---
    noise_figure_db: float = 10.0     # Receiver noise figure (dB)
    rx_gain_db: float = 20.0          # Receiver gain (dB)

    # --- 派生参数（自动计算）---
    S: float = field(init=False)               # Chirp slope (Hz/s)
    wl: float = field(init=False)              # Wavelength (m)
    antenna_spacing: float = field(init=False)  # Antenna element spacing (m): 阵元间距
    Fs: float = field(init=False)              # ADC sampling rate (Hz)
    range_resolution: float = field(init=False)
    velocity_resolution: float = field(init=False)
    max_range: float = field(init=False)
    max_velocity: float = field(init=False)
    max_angle_rad: float = field(init=False)

    def __post_init__(self):
        self.S = self.B / self.Tc
        self.wl = C / self.fc
        self.antenna_spacing = self.wl / 2.0
        self.Fs = self.sample_num / self.Tc
        self.range_resolution = C / (2.0 * self.B)
        self.velocity_resolution = self.wl / (2.0 * self.Tc * self.chirp_num)
        self.max_range = self.sample_num * C / (4.0 * self.B)
        self.max_velocity = self.wl / (4.0 * self.Tc)
        self.max_angle_rad = np.arcsin(min(self.wl / (2.0 * self.antenna_spacing), 1.0))


@dataclass
class TargetParams:
    """Parameters for a single point target in one frame."""

    range: float                      # Range (m)
    velocity: float                   # Radial velocity (m/s, positive = approaching)
    angle: float                      # Azimuth angle (degrees, 0 = boresight)
    rcs: float = 1.0                  # Relative RCS / amplitude scaling factor


@dataclass
class MotionModel:
    """Motion model for a target across frames."""

    motion: str = "constant_velocity"  # constant_velocity, constant_acceleration, stationary
    acceleration: float = 0.0          # m/s² (for constant_acceleration)
    angular_velocity: float = 0.0      # deg/s (angle change rate, for all motion types)


@dataclass
class TargetSpec:
    """Full specification of a target including motion across frames."""

    id: int
    initial_range: float
    initial_velocity: float
    initial_angle: float
    rcs: float = 1.0
    motion: MotionModel = field(default_factory=MotionModel)


@dataclass
class SceneConfig:
    """Multi-frame scenario definition."""

    type: str = "single_target"
    num_frames: int = 1
    frame_interval: float = 0.05      # seconds between frames
    targets: List[TargetSpec] = field(default_factory=list)
    snr_db: float = 25.0              # Target SNR
    swerling_model: int = 0           # Swerling 0-4

@dataclass
class ProcessingConfig:
    """FFT processing stage configuration.

    Controls the 3-stage FFT processing chain parameters.
    """

    range_window: str = "hanning"     # Range FFT 窗函数
    doppler_window: str = "hanning"   # Doppler FFT 窗函数
    angle_window: str = "hanning"     # Angle FFT 窗函数
    downsample: int = 1               # Range FFT 降采样步长


@dataclass
class DetectionConfig:
    """Detection stage configuration."""

    cfar_algorithm: str = "ca_cfar"   # ca_cfar, os_cfar, go_cfar, so_cfar
    guard_cells: int = 2
    reference_cells: int = 8
    pfa: float = 1e-4
    os_rank_ratio: float = 0.75       # For OS-CFAR
    alpha: Optional[float] = None     # None = 从 pfa 自动计算（否则使用显式值）


@dataclass
class EstimationConfig:
    """Estimation stage configuration.

    Controls the DOA (Direction of Arrival) angle estimation method.
    """

    doa_method: str = "fft"            # fft, esprit, music, mvdr
    n_sources: int = 1                 # Number of signal sources (for MUSIC/ESPRIT)
    angle_resolution: float = 0.1      # Angular grid spacing in deg (for MUSIC/MVDR)
    scan_range: Tuple[float, float] = (-80.0, 80.0)  # Scan range (for MUSIC/MVDR), exclude endfire
    doa_multi_peak: bool = True        # 启用角度多峰检测（同 RD 不同角度目标分辨）
    doa_peak_threshold_db: float = 6.0 # 次峰相对主峰的门限 (dB)
    doa_max_peaks: int = 3             # 每 RD cell 最多检测的角度峰数


@dataclass
class TrackingConfig:
    """Tracking stage configuration."""

    filter: str = "kf"                # kf, ekf, ukf
    association: str = "gnn"          # nn, gnn, jpda
    init_threshold: int = 3           # M-of-N hits to confirm
    coast_threshold: int = 5          # consecutive misses to delete
    dt: float = 0.05                  # Time step (s)
    process_noise: float = 0.01
    measurement_noise: float = 0.1

    # --- IMM 配置 ---
    enable_imm: bool = False
    imm_models: tuple = ("cv", "ca", "ctra")

    # --- 关联优化 ---
    use_mahalanobis: bool = True
    gate_mahalanobis: float = 6.0
    use_velocity_gating: bool = True
    gate_velocity: float = 3.0


@dataclass
class VisualizationConfig:
    """Visualization stage configuration."""

    backend: str = "mpl"              # mpl or pyqtgraph
    update_hz: int = 20
    panels: List[str] = field(default_factory=lambda: ["rd_map", "ra_map", "trajectory", "diagnostic"])


@dataclass
class PersistenceConfig:
    """Persistence stage configuration."""

    format: str = "npz"               # hdf5 or npz for raw signals
    save_intermediate: bool = True
    output_dir: str = "data"


@dataclass
class PipelineConfig:
    """Full pipeline configuration — which algorithms to use at each stage."""

    stages: List[str] = field(default_factory=lambda: [
        "signal_generation",
        "range_fft",
        "doppler_fft",
        "angle_fft",
        "peak_detection",
        "cfar",
        "estimation",
        "tracking",
    ])
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    estimation: EstimationConfig = field(default_factory=EstimationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    evaluation: "EvalConfig" = field(default_factory=lambda: EvalConfig())


# ============================================================
# 评估配置（新增）
# ============================================================

@dataclass
class EvalConfig:
    """评估模块配置。"""
    # ── 通用 ──
    enabled: bool = True                        # 是否启用评估
    output_dir: str = "eval_results"            # 评估报告输出目录
    save_report: bool = True                    # 是否保存评估报告

    # ── 信号处理评估 ──
    signal_metrics: List[str] = field(default_factory=lambda: [
        "range_rmse", "velocity_rmse", "angle_rmse",
        "crlb_ratio", "resolution", "timing",
    ])

    # ── 检测评估 ──
    detection_metrics: List[str] = field(default_factory=lambda: [
        "pd", "pfa", "roc", "peak_grouping",
    ])
    detection_range_gate_m: float = 2.0         # 检测-真值距离门控 (m)
    pfa_design: Optional[float] = None          # 设计 Pfa

    # ── 跟踪评估 ──
    tracking_metrics: List[str] = field(default_factory=lambda: [
        "gospa", "mota", "motp", "idf1", "hota", "lifecycle",
    ])
    tracking_gate_m: float = 5.0                # 跟踪-真值关联门控 (m)
    gospa_p: float = 2.0
    gospa_c: float = 10.0
    gospa_alpha: float = 2.0

    # ── 基准测试 ──
    benchmark_scenarios: List[str] = field(default_factory=list)
    num_monte_carlo: int = 1
