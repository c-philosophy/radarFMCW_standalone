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


def _estimate_n_sources_eigratio(
    eigvals: np.ndarray,
) -> int:
    """特征值比值法自动估计信源数。

    信号特征值远大于噪声特征值时，比值 λ_k / λ_{k+1}
    在信号-噪声边界处产生最大跳变。

    信源数 = argmax(λ_k / λ_{k+1}), k ∈ {1, ..., M-1}

    Args:
        eigvals: 降序排列的协方差矩阵特征值。

    Returns:
        估计信源数 (int, 1 ≤ k ≤ M-1)。
    """
    M = len(eigvals)
    if M <= 2:
        return 1
    ratios = np.array([eigvals[i] / (eigvals[i + 1] + 1e-30) for i in range(M - 1)])
    return int(np.argmax(ratios) + 1)


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
            # 返回列表: [(a_bin, a_deg), ...] — 多峰检测支持同 RD 多目标
            angle_results = self._estimate_angle(
                s_rda, d_idx, r_idx, s_rd=s_rd,
            )
            # 每个角度峰生成独立的 TargetEstimate
            for a_bin, a_est in angle_results:
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
    ) -> list:
        """根据配置分发到对应的 DOA 方法，返回角度列表。

        方案 A1（Multi-Object Beamforming 等效）：
        每个 RD cell 检测角度谱中的多个峰值，支持分辨同距离/同速度
        但不同角度的目标。超分辨方法（MUSIC/MVDR/ESPRIT）天然支持多源。

        Returns:
            List of (angle_bin, angle_deg) tuples.
            angle_bin = -1 for super-resolution methods.
        """
        method = self.config.doa_method
        if method == "fft":
            if self.config.doa_multi_peak:
                return self._angle_fft_multipeak(s_rda, d_peak, r_peak)
            else:
                return [self._angle_fft_single(s_rda, d_peak, r_peak)]
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

    def _angle_fft_single(
        self,
        s_rda: np.ndarray,
        d_peak: int,
        r_peak: int,
    ) -> Tuple[int, float]:
        """FFT 波束形成（Bartlett）角度估计 — 仅取最大值（legacy 模式）。"""
        return self._angle_fft_multipeak(s_rda, d_peak, r_peak)[0]

    def _angle_fft_multipeak(
        self,
        s_rda: np.ndarray,
        d_peak: int,
        r_peak: int,
    ) -> list:
        """FFT 波束形成角度多峰检测。

        在角度谱中查找所有超过相对门限的局部最大值。
        相对门限 = 主峰幅度 - doa_peak_threshold_db。

        s_rda 已做 fftshift，boresight (0°) 在 bin N/2。
        antenna_spacing = λ/2 时，sin(θ) = 2*(a_bin - N/2) / N。

        Returns:
            List of (angle_bin, angle_deg), sorted by amplitude descending.
        """
        spectrum = np.abs(s_rda[:, d_peak, r_peak])
        max_val = np.max(spectrum)
        if max_val <= 0:
            return [(-1, 0.0)]

        abs_threshold = max_val * 10.0 ** (-self.config.doa_peak_threshold_db / 20.0)
        N = self.radar.angle_num

        peaks = []
        for i in range(1, len(spectrum) - 1):
            if spectrum[i] <= spectrum[i - 1] or spectrum[i] <= spectrum[i + 1]:
                continue
            if spectrum[i] < abs_threshold:
                continue
            a_centered = i - N // 2
            sin_theta = 2.0 * a_centered / N
            sin_theta = np.clip(sin_theta, -1.0, 1.0)
            peaks.append((i, spectrum[i], float(np.degrees(np.arcsin(sin_theta)))))

        # 按幅度降序排列，取前 doa_max_peaks 个
        peaks.sort(key=lambda x: x[1], reverse=True)
        return [(p[0], p[2]) for p in peaks[:self.config.doa_max_peaks]]

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

    # --- 空间平滑 ---

    def _spatial_smoothing_covariance(
        self,
        x: np.ndarray,
    ) -> tuple:
        """前向-后向空间平滑协方差估计 (Shan, Wax & Kailath, 1985)。

        将 N 元均匀线阵拆为 L 个 M 元重叠子阵，对各子阵协方差
        取平均，恢复因信号相干导致的秩亏。

        Args:
            x: 单快照阵列数据，shape (n_antennas,)。

        Returns:
            (R_fb, M): 平滑后的 (M×M) 协方差矩阵 和 子阵大小 M。
        """
        N = len(x)
        # 子阵大小 M：默认保证 L ≥ 3 个子阵
        M = max(N - 2, N // 2 + 1)  # N=8 → M=6, L=3
        L = N - M + 1

        x = np.asarray(x, dtype=np.complex128).ravel()

        # --- 前向平滑 ---
        R_f = np.zeros((M, M), dtype=np.complex128)
        for li in range(L):
            xl = x[li:li + M]
            R_f += np.outer(xl, xl.conj())
        R_f /= L

        # --- 后向平滑 ---
        x_b = np.flipud(x.conj())
        R_b = np.zeros((M, M), dtype=np.complex128)
        for li in range(L):
            xl = x_b[li:li + M]
            R_b += np.outer(xl, xl.conj())
        R_b /= L

        R_fb = (R_f + R_b) / 2.0
        return R_fb, M

    @staticmethod
    def _sub_steering_matrix(
        n_ant: int,
        antenna_spacing: float,
        wl: float,
        scan_angles_rad: np.ndarray,
    ) -> np.ndarray:
        """Compute steering vectors for a given array size."""
        n = np.arange(n_ant)
        phase = (-2.0 * np.pi * (antenna_spacing / wl)
                 * np.sin(scan_angles_rad[:, np.newaxis]) * n[np.newaxis, :])
        return np.exp(1j * phase)

    # --- ESPRIT (仍用邻域快照，空间平滑 + ESPRIT 需额外适配) ---

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
    ) -> list:
        """TLS-ESPRIT 超分辨多源角度估计（邻域快照模式）。

        注意：ESPRIT 需要原始快照矩阵，空间平滑产生的协方差矩阵
        不直接兼容。对同 RD 多目标场景建议使用 MUSIC 方法。
        """
        X = self._collect_snapshots(s_antenna, d_peak, r_peak)
        angles = self._get_esprit_engine().estimate(X)
        if len(angles) == 0:
            return [(-1, 0.0)]
        return [(-1, float(a)) for a in angles]

    # --- MUSIC (空间平滑) ---

    def _angle_music(
        self,
        s_antenna: np.ndarray,
        d_peak: int,
        r_peak: int,
    ) -> list:
        """MUSIC 超分辨多源角度估计（空间平滑 + MDL 自动信源数估计）。

        1. 前向-后向空间平滑恢复协方差秩 (Shan et al., 1985)
        2. MDL 信息论准则自动估计信源数 (Wax & Kailath, 1985)
        3. MUSIC 伪谱多峰查找
        """
        from scipy import linalg

        x = s_antenna[:, d_peak, r_peak]
        n_pks = self.config.doa_max_peaks if self.config.doa_multi_peak else 1

        # 空间平滑协方差估计
        R_ss, M = self._spatial_smoothing_covariance(x)

        # 特征分解
        eigvals, eigvecs = linalg.eigh(R_ss)
        idx = np.argsort(eigvals)[::-1]
        eigvals_sorted = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # 特征值比值法自动估计信源数
        n_sources = _estimate_n_sources_eigratio(eigvals_sorted)
        # 安全约束：不超过用户配置的最大峰数，且保留噪声子空间
        n_sources = max(1, min(n_sources, n_pks, M - 2))

        noise_subspace = eigvecs[:, n_sources:]

        # 扫描网格
        scan_angles_deg = np.arange(
            self.config.scan_range[0], self.config.scan_range[1],
            self.config.angle_resolution,
        )
        scan_rad = np.deg2rad(scan_angles_deg)

        # 子阵导向矢量
        A = self._sub_steering_matrix(
            M, self.radar.antenna_spacing, self.radar.wl, scan_rad,
        )

        # MUSIC 伪谱
        proj = A @ noise_subspace
        denom = np.sum(np.abs(proj) ** 2, axis=1)
        spectrum_db = 10.0 * np.log10(1.0 / (denom + 1e-30))

        # 峰值查找（先找所有局部最大值，再按幅度门限过滤）
        from fmcw.estimation.doa import _find_spectrum_peaks
        # 先找足够多的候选峰
        candidate_angles = _find_spectrum_peaks(
            spectrum_db, scan_angles_deg, self.config.angle_resolution,
            min(n_pks * 2, len(scan_angles_deg)),  # 收集更多候选
        )
        if len(candidate_angles) == 0:
            return [(-1, 0.0)]

        # 幅度门限过滤：次峰需在主峰的 doa_peak_threshold_db 范围内
        peak_vals = np.interp(candidate_angles, scan_angles_deg, spectrum_db)
        max_val = np.max(peak_vals)
        threshold = max_val - self.config.doa_peak_threshold_db
        filtered = [a for a, v in zip(candidate_angles, peak_vals) if v >= threshold]
        angles = filtered[:n_pks]
        if len(angles) == 0:
            return [(-1, float(candidate_angles[0]))]
        return [(-1, float(a)) for a in angles]

    # --- MVDR (空间平滑) ---

    def _angle_mvdr(
        self,
        s_antenna: np.ndarray,
        d_peak: int,
        r_peak: int,
    ) -> list:
        """MVDR/Capon 超分辨多源角度估计（前向-后向空间平滑）。

        使用空间平滑替代邻域伪快照。
        """
        from scipy import linalg

        x = s_antenna[:, d_peak, r_peak]
        n_pks = self.config.doa_max_peaks if self.config.doa_multi_peak else 1

        R_ss, M = self._spatial_smoothing_covariance(x)
        R_inv = linalg.pinvh(R_ss)

        scan_angles_deg = np.arange(
            self.config.scan_range[0], self.config.scan_range[1],
            self.config.angle_resolution,
        )
        scan_rad = np.deg2rad(scan_angles_deg)
        A = self._sub_steering_matrix(
            M, self.radar.antenna_spacing, self.radar.wl, scan_rad,
        )

        power = np.einsum("ij,jk,ik->i", A.conj(), R_inv, A, optimize=True)
        spectrum_db = 10.0 * np.log10(1.0 / np.real(power + 1e-30))

        from fmcw.estimation.doa import _find_spectrum_peaks
        angles = _find_spectrum_peaks(
            spectrum_db, scan_angles_deg, self.config.angle_resolution, n_pks,
        )
        if len(angles) == 0:
            return [(-1, 0.0)]
        return [(-1, float(a)) for a in angles]

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
