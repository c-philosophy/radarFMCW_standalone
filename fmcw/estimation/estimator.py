"""参数估计器：将检测峰值转换为物理量（距离、速度、角度）。

角度估计支持多种 DOA 方法：
    - fft:    FFT 波束形成（Bartlett），默认，零额外依赖
    - esprit: TLS-ESPRIT 子空间方法，无需扫描网格，解析求解
    - music:  MUSIC 伪谱法，噪声子空间投影，高分辨
    - mvdr:   MVDR/Capon 波束形成，最小方差无畸变响应

DOA 方法通过 EstimationConfig.doa_method 配置。
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from fmcw.config.schema import RadarParams, EstimationConfig
from fmcw.utils.constants import C


@dataclass
class TargetEstimate:
    """Single target estimate for one frame."""

    range: float                     # Range (m)
    velocity: float                  # Radial velocity (m/s), + = approaching
    angle: float                     # Azimuth angle (degrees), 0 = boresight
    doppler_bin: int                 # Doppler FFT bin (fftshift 后，DC 在中心)
    range_bin: int                   # Range FFT bin
    angle_bin: int                   # Angle FFT bin (fftshift 后，0° 在中心),
                                     # -1 for super-res methods
    amplitude: float                 # Signal magnitude at peak
    snr_db: float = 0.0              # Estimated SNR


@dataclass
class FrameEstimates:
    """All target estimates for one processing frame."""

    frame_idx: int
    targets: List[TargetEstimate]
    timestamp: float = 0.0           # Relative to start of scenario (s)


class ParameterEstimator:
    """Convert detection peaks to physical parameters.

    Uses standard FMCW formulas:
        R = (r_peak / N_sample * Fs) * c / (2 * S)
        v = (d_peak / N_chirp * 2*pi) * lambda / (4 * pi * Tc)
        theta = arcsin(lambda * (a_peak / N_angle * 2*pi) / (2 * pi * d))   
    Supports configurable DOA estimation methods via EstimationConfig.

    Usage:
        # Default: FFT beamforming
        est = ParameterEstimator(radar)

        # Super-resolution: ESPRIT
        cfg = EstimationConfig(doa_method="esprit")
        est = ParameterEstimator(radar, config=cfg)
    """

    def __init__(
        self,
        radar: RadarParams,
        config: Optional[EstimationConfig] = None,
    ):
        self.radar = radar
        self.config = config or EstimationConfig()

        # Lazy-initialized DOA engines (avoid overhead when not used)
        self._esprit_engine = None
        self._music_engine = None
        self._mvdr_engine = None

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def estimate(
        self,
        s_rda: np.ndarray,
        filtered_peaks: np.ndarray,
        frame_idx: int = 0,
        timestamp: float = 0.0,
        ground_truth: Optional[List] = None,
        s_rd: Optional[np.ndarray] = None,
    ) -> FrameEstimates:
        """Convert peak indices to physical parameters.

        Args:
            s_rda: 3D complex FFT cube (angle_bins, D_bins, R_bins).
                   Used for FFT beamforming and amplitude extraction.
            filtered_peaks: (N, 2) [doppler_idx, range_idx].
            frame_idx: Frame index.
            timestamp: Frame timestamp (s).
            ground_truth: Optional list of TargetParams for SNR calculation.
            s_rd: 3D complex cube (n_antennas, D_bins, R_bins) before
                  angle FFT. Required for super-resolution DOA methods
                  (ESPRIT/MUSIC/MVDR) which need raw array snapshots.
                  If None and a super-res method is configured, falls
                  back to using s_rda (less accurate).

        Returns:
            FrameEstimates with all estimated targets.
        """
        targets = []
        for d_peak, r_peak in filtered_peaks:
            d_idx = int(d_peak)
            r_idx = int(r_peak)

            # Range
            r_est = self._range_from_bin(r_idx)

            # Velocity (fftshift 后 d_idx 直接对应带符号速度)
            v_est = self._velocity_from_bin(d_idx)

            # Angle (dispatched by config; super-res uses s_rd)
            a_bin, a_est = self._estimate_angle(
                s_rda, d_idx, r_idx, s_rd=s_rd,
            )

            # Amplitude: for FFT methods use angle_bin; for super-res use max
            if a_bin >= 0:
                amp = float(np.abs(s_rda[a_bin, d_idx, r_idx]))
            else:
                amp = float(np.max(np.abs(s_rda[:, d_idx, r_idx])))

            # SNR estimate
            rd_slice = np.abs(s_rda[:, d_idx, r_idx])
            noise_floor = np.median(rd_slice)
            snr = 20.0 * np.log10(max(amp / (noise_floor + 1e-30), 1.0))

            targets.append(TargetEstimate(
                range=r_est,
                velocity=v_est,
                angle=a_est,
                doppler_bin=d_idx,
                range_bin=r_idx,
                angle_bin=a_bin,
                amplitude=amp,
                snr_db=round(snr, 1),
            ))

        return FrameEstimates(
            frame_idx=frame_idx,
            targets=targets,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    # 角度估计分发
    # ------------------------------------------------------------------

    def _estimate_angle(
        self,
        s_rda: np.ndarray,
        d_peak: int,
        r_peak: int,
        s_rd: Optional[np.ndarray] = None,
    ) -> Tuple[int, float]:
        """根据配置分发到对应的 DOA 方法。

        Args:
            s_rda: 角度 FFT 数据 (n_angle, D, R)，FFT 方法使用。
            d_peak: Doppler bin。
            r_peak: Range bin。
            s_rd: 原始天线数据 (n_antennas, D, R)，超分辨方法使用。

        Returns:
            (angle_bin, angle_deg): angle_bin 对超分辨方法返回 -1。
        """
        method = self.config.doa_method
        if method == "fft":
            return self._angle_fft(s_rda, d_peak, r_peak)
        elif method == "esprit":
            return self._angle_esprit(s_rd, d_peak, r_peak)
        elif method == "music":
            return self._angle_music(s_rd, d_peak, r_peak)
        elif method == "mvdr":
            return self._angle_mvdr(s_rd, d_peak, r_peak)
        else:
            raise ValueError(
                f"Unknown DOA method: {method}. "
                f"Valid options: fft, esprit, music, mvdr"
            )

    # ------------------------------------------------------------------
    # FFT 波束形成（现有逻辑，重命名为 _angle_fft）
    # ------------------------------------------------------------------

    def _angle_fft(
        self,
        s_rda: np.ndarray,
        d_peak: int,
        r_peak: int,
    ) -> Tuple[int, float]:
        """FFT 波束形成（Bartlett）角度估计。

        s_rda 已做 fftshift，boresight (0°) 在 bin N/2。
        antenna_spacing = λ/2 时，sin(θ) = 2*(a_bin - N/2) / N。
        """
        angle_spectrum = np.abs(s_rda[:, d_peak, r_peak])
        a_peak = int(np.argmax(angle_spectrum))

        # fftshift 后直接计算带符号角度
        a_centered = a_peak - self.radar.angle_num // 2
        # sin(θ) = 2 * a_centered / N  (antenna_spacing = λ/2 时成立)
        sin_theta = 2.0 * a_centered / self.radar.angle_num
        sin_theta = np.clip(sin_theta, -1.0, 1.0)
        return a_peak, float(np.degrees(np.arcsin(sin_theta)))

    # ------------------------------------------------------------------
    # 超分辨 DOA 方法
    # ------------------------------------------------------------------

    def _collect_snapshots(
        self,
        s_antenna: np.ndarray,
        d_peak: int,
        r_peak: int,
    ) -> np.ndarray:
        """从 RD 邻域收集多快照阵列矩阵。

        单个 RD cell 仅提供 1 个快照（n_antennas 维向量），不足以估计
        协方差矩阵。从 ±1 邻域收集最多 9 个伪快照。

        Args:
            s_antenna: 原始天线域数据 (n_antennas, D, R)，即 s_rd。
            d_peak: Doppler bin。
            r_peak: Range bin。

        Returns:
            (n_antennas, n_snapshots) 复数阵列快照矩阵。
        """
        snapshots = []
        for dd in (-1, 0, 1):
            for dr in (-1, 0, 1):
                di = d_peak + dd
                ri = r_peak + dr
                if 0 <= di < s_antenna.shape[1] and 0 <= ri < s_antenna.shape[2]:
                    snapshots.append(s_antenna[:, di, ri])
        return np.column_stack(snapshots)

    # --- ESPRIT ---

    def _get_esprit_engine(self):
        """懒初始化 TLS-ESPRIT 引擎。"""
        if self._esprit_engine is None:
            from fmcw.processing.super_resolution import TLSESPRIT
            self._esprit_engine = TLSESPRIT(
                self.radar,
                n_sources=self.config.n_sources,
            )
        return self._esprit_engine

    def _angle_esprit(
        self,
        s_antenna: np.ndarray,
        d_peak: int,
        r_peak: int,
    ) -> Tuple[int, float]:
        """TLS-ESPRIT 超分辨角度估计。

        Args:
            s_antenna: 原始天线域数据 (n_antennas, D, R)，即 s_rd。
        """
        X = self._collect_snapshots(s_antenna, d_peak, r_peak)
        angles = self._get_esprit_engine().estimate(X)
        if len(angles) == 0:
            return -1, 0.0
        best = float(angles[np.argmin(np.abs(angles))])
        return -1, best

    # --- MUSIC ---

    def _get_music_engine(self):
        """懒初始化 MUSIC 引擎。"""
        if self._music_engine is None:
            from fmcw.estimation.doa import MUSIC
            self._music_engine = MUSIC(
                self.radar,
                n_sources=self.config.n_sources,
                angle_resolution=self.config.angle_resolution,
                scan_range_deg=self.config.scan_range,
            )
        return self._music_engine

    def _angle_music(
        self,
        s_antenna: np.ndarray,
        d_peak: int,
        r_peak: int,
    ) -> Tuple[int, float]:
        """MUSIC 伪谱超分辨角度估计。

        Args:
            s_antenna: 原始天线域数据 (n_antennas, D, R)，即 s_rd。
        """
        X = self._collect_snapshots(s_antenna, d_peak, r_peak)
        angles = self._get_music_engine().estimate(X, n_peaks=1)
        if len(angles) == 0:
            return -1, 0.0
        return -1, float(angles[0])

    # --- MVDR ---

    def _get_mvdr_engine(self):
        """懒初始化 MVDR 引擎。"""
        if self._mvdr_engine is None:
            from fmcw.estimation.doa import MVDR
            self._mvdr_engine = MVDR(
                self.radar,
                angle_resolution=self.config.angle_resolution,
                scan_range_deg=self.config.scan_range,
            )
        return self._mvdr_engine

    def _angle_mvdr(
        self,
        s_antenna: np.ndarray,
        d_peak: int,
        r_peak: int,
    ) -> Tuple[int, float]:
        """MVDR/Capon 超分辨角度估计。

        Args:
            s_antenna: 原始天线域数据 (n_antennas, D, R)，即 s_rd。
        """
        X = self._collect_snapshots(s_antenna, d_peak, r_peak)
        angles = self._get_mvdr_engine().estimate(X, n_peaks=1)
        if len(angles) == 0:
            return -1, 0.0
        return -1, float(angles[0])

    # ------------------------------------------------------------------
    # 距离/速度转换（不变）
    # ------------------------------------------------------------------

    def _range_from_bin(self, r_bin: int) -> float:
        """Convert range FFT bin to range in meters."""
        delta_f = r_bin / self.radar.sample_num * self.radar.Fs
        return delta_f * C / (2.0 * self.radar.S)

    def _velocity_from_bin(self, d_bin: int) -> float:
        """Convert Doppler bin to velocity (m/s).

        s_rd 已做 fftshift，DC 在 bin chirp_num/2。
        带符号速度 = (d_bin - D/2) * velocity_resolution。
        """
        d_centered = d_bin - self.radar.chirp_num // 2
        return d_centered * self.radar.velocity_resolution
