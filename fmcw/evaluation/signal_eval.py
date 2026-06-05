"""信号处理评估器：距离/速度/角度估计精度与 CRLB 对比。

提供：
- CRLB 理论界计算（距离/速度/角度）
- 估计-真值配对与 RMSE/MAE 评估
- CRLB 比率指标（RMSE/√CRLB）
- 分辨率验证

参考文献：
    [1] Kay, S.M. "Fundamentals of Statistical Signal Processing:
        Estimation Theory." Prentice-Hall, 1993.
    [2] Van Trees, H.L. "Detection, Estimation, and Modulation Theory,
        Part IV: Optimum Array Processing." Wiley, 2002.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from fmcw.config.schema import RadarParams
from fmcw.evaluation.models import (
    FrameRecord, MetricValue, SignalEvalDetail,
)


# ============================================================
# CRLB 理论界计算
# ============================================================

def crlb_range(
    snr_linear: np.ndarray,
    bandwidth: float,
    n_samples: int,
) -> np.ndarray:
    """距离估计 CRLB（AWGN 信道）。

    σ_R² ≥ 3c² / (8π² · SNR · B² · N)

    参考文献: Kay (1993), Eq. (3.31) for frequency estimation,
              mapped to range via R = c·f_beat / (2·S).

    Args:
        snr_linear: 线性 SNR（非 dB），shape (K,)。
        bandwidth: 有效带宽 (Hz)，即雷达的调频带宽 B。
        n_samples: ADC 采样点数。

    Returns:
        距离 CRLB 标准差 (m)，shape 与 snr_linear 相同。
    """
    C = 3.0e8
    sigma_sq = (3.0 * C**2) / (
        8.0 * np.pi**2 * snr_linear * bandwidth**2 * n_samples
    )
    return np.sqrt(np.maximum(sigma_sq, 0.0))


def crlb_velocity(
    snr_linear: np.ndarray,
    fc: float,
    n_chirps: int,
    tc: float,
) -> np.ndarray:
    """速度估计 CRLB。

    σ_v² ≥ 3λ² / (8π² · SNR · (N_c·T_c)² · N_c)

    Args:
        snr_linear: 线性 SNR。
        fc: 载频 (Hz)。
        n_chirps: 每帧 chirp 数。
        tc: chirp 周期 (s)。

    Returns:
        速度 CRLB 标准差 (m/s)。
    """
    C = 3.0e8
    wl = C / fc
    T_total = n_chirps * tc
    sigma_sq = (3.0 * wl**2) / (
        8.0 * np.pi**2 * snr_linear * T_total**2 * n_chirps
    )
    return np.sqrt(np.maximum(sigma_sq, 0.0))


def crlb_angle(
    snr_linear: np.ndarray,
    n_antennas: int,
    d_over_lambda: float = 0.5,
    angle_rad: float = 0.0,
) -> np.ndarray:
    """角度估计 CRLB（ULA 阵列）。

    σ_θ² ≥ 6 / (π² · SNR · N·(N²-1) · (d/λ)² · cos²θ)

    参考文献: Van Trees (2002), Eq. (8.91).

    Args:
        snr_linear: 线性 SNR。
        n_antennas: 接收天线数。
        d_over_lambda: 阵元间距与波长之比（默认 λ/2 = 0.5）。
        angle_rad: 目标方位角 (rad)，boresight=0。

    Returns:
        角度 CRLB 标准差 (rad)。
    """
    N = n_antennas
    cos_factor = np.cos(angle_rad)
    if abs(cos_factor) < 1e-10:
        cos_factor = 1e-10  # 避免除零（endfire 方向）
    sigma_sq = 6.0 / (
        np.pi**2 * snr_linear * N * (N**2 - 1)
        * d_over_lambda**2 * cos_factor**2
    )
    return np.sqrt(np.maximum(sigma_sq, 0.0))


# ============================================================
# 信号处理评估器
# ============================================================

class SignalEvaluator:
    """信号处理层评估器。

    评估距离、速度、角度估计的精度，与 CRLB 对比，
    验证分辨率是否达到理论值。

    Usage:
        evaluator = SignalEvaluator(radar)
        result = evaluator.evaluate(records, metrics=["range_rmse", "angle_rmse"])
    """

    def __init__(self, radar: RadarParams):
        self.radar = radar

    def evaluate(
        self,
        records: List[FrameRecord],
        metrics: Optional[List[str]] = None,
    ) -> SignalEvalDetail:
        """对收集的帧数据进行信号处理评估。

        预估-真值配对策略：
        - 使用匈牙利算法将每帧的估计值匹配到最近的真值
        - 门控阈值 = max_range × 0.1（避免跨目标误匹配）
        - 配对仅在需要计算 RMSE/CRLB 指标时执行（按需计算）

        Args:
            records: FrameRecord 列表。
            metrics: 要计算的指标名称列表。None = 全部计算。
                     可选: range_rmse, velocity_rmse, angle_rmse,
                           crlb_ratio, resolution, timing。

        Returns:
            SignalEvalDetail 包含被请求的所有指标。
        """
        metrics = metrics or [
            "range_rmse", "velocity_rmse", "angle_rmse",
            "crlb_ratio", "resolution", "timing",
        ]

        result = SignalEvalDetail()
        result.num_frames = len(records)

        # ── 按需收集估计-真值配对（仅当需要时） ──
        needs_pairs = any(m in metrics for m in [
            "range_rmse", "velocity_rmse", "angle_rmse", "crlb_ratio",
        ])
        pairs = None
        if needs_pairs:
            pairs = self._collect_estimation_pairs(records)

        if pairs is None or pairs["range_gt"].size == 0:
            result.num_targets = 0
            # 即使无目标，也可计算 timing 和 resolution（仅依赖雷达参数）
            if "timing" in metrics:
                pt = np.array([r.proc_time_ms for r in records if r.proc_time_ms > 0])
                if len(pt) > 0:
                    result.proc_time_mean = MetricValue(
                        "proc_time_ms", float(np.mean(pt)), unit="ms",
                    )
            if "resolution" in metrics:
                result.resolution = self._build_resolution_info()
            return result

        result.num_targets = len(pairs["range_gt"])

        # ── 距离 RMSE ──
        if "range_rmse" in metrics or "crlb_ratio" in metrics:
            re = pairs["range_est"] - pairs["range_gt"]
            if "range_rmse" in metrics:
                rmse_val = float(np.sqrt(np.mean(re**2)))
                mae_val = float(np.mean(np.abs(re)))
                result.range_rmse = MetricValue("range_rmse", rmse_val, unit="m")
                result.range_mae = MetricValue("range_mae", mae_val, unit="m")
            if "crlb_ratio" in metrics:
                crlb_r = crlb_range(
                    pairs["snr_linear"], self.radar.B, self.radar.sample_num,
                )
                crlb_mean = float(np.mean(crlb_r))
                rmse_for_crlb = float(np.sqrt(np.mean(re**2)))
                result.range_crlb_ratio = MetricValue(
                    "range_crlb_ratio",
                    rmse_for_crlb / max(crlb_mean, 1e-12),
                    direction="lower_better", reference_value=1.0,
                )

        # ── 速度 RMSE ──
        if "velocity_rmse" in metrics or "crlb_ratio" in metrics:
            ve = pairs["velocity_est"] - pairs["velocity_gt"]
            if "velocity_rmse" in metrics:
                rmse_v = float(np.sqrt(np.mean(ve**2)))
                result.velocity_rmse = MetricValue(
                    "velocity_rmse", rmse_v, unit="m/s",
                )
            if "crlb_ratio" in metrics:
                crlb_v = crlb_velocity(
                    pairs["snr_linear"], self.radar.fc,
                    self.radar.chirp_num, self.radar.Tc,
                )
                crlb_v_mean = float(np.mean(crlb_v))
                rmse_v2 = float(np.sqrt(np.mean(ve**2)))
                result.velocity_crlb_ratio = MetricValue(
                    "velocity_crlb_ratio",
                    rmse_v2 / max(crlb_v_mean, 1e-12),
                    direction="lower_better", reference_value=1.0,
                )

        # ── 角度 RMSE ──
        if "angle_rmse" in metrics or "crlb_ratio" in metrics:
            ae = pairs["angle_est"] - pairs["angle_gt"]
            if "angle_rmse" in metrics:
                rmse_a = float(np.sqrt(np.mean(ae**2)))
                result.angle_rmse = MetricValue("angle_rmse", rmse_a, unit="deg")
            if "crlb_ratio" in metrics:
                crlb_a_rad = crlb_angle(
                    pairs["snr_linear"], self.radar.antenna_num,
                )
                crlb_a_deg = np.degrees(crlb_a_rad)
                crlb_a_mean = float(np.mean(crlb_a_deg))
                rmse_a2 = float(np.sqrt(np.mean(ae**2)))
                result.angle_crlb_ratio = MetricValue(
                    "angle_crlb_ratio",
                    rmse_a2 / max(crlb_a_mean, 1e-12),
                    direction="lower_better", reference_value=1.0,
                )

        # ── 分辨率验证 ──
        if "resolution" in metrics:
            result.resolution = self._build_resolution_info()

        # ── 处理耗时 ──
        if "timing" in metrics:
            pt = np.array([r.proc_time_ms for r in records if r.proc_time_ms > 0])
            if len(pt) > 0:
                result.proc_time_mean = MetricValue(
                    "proc_time_ms", float(np.mean(pt)), unit="ms",
                )

        return result

    def _build_resolution_info(self) -> dict:
        """构建理论分辨率信息（仅依赖雷达参数，不依赖数据）。"""
        return {
            "range_resolution_m": self.radar.range_resolution,
            "velocity_resolution_ms": self.radar.velocity_resolution,
            "angle_resolution_deg": float(
                np.degrees(np.arcsin(
                    min(2.0 / self.radar.antenna_num, 1.0)
                ))
            ),
        }

    # ── 内部：估计-真值配对 ──

    def _collect_estimation_pairs(self, records: List[FrameRecord]) -> dict:
        """从所有帧中收集估计值-真值配对。

        配对策略：使用匈牙利算法将每帧的估计值匹配到最近的真值。
        距离门控阈值 = max_range × 0.1（避免跨目标误匹配）。

        Returns:
            dict with keys: range_gt, range_est, velocity_gt, velocity_est,
                           angle_gt, angle_est, snr_linear
        """
        gate = self.radar.max_range * 0.1
        all_range_gt, all_range_est = [], []
        all_vel_gt, all_vel_est = [], []
        all_angle_gt, all_angle_est = [], []
        all_snr = []

        for rec in records:
            gt = rec.ground_truth
            est_list = rec.estimated
            if not gt or not est_list:
                continue

            # 真值数组: (N, 3) [range, velocity, angle]
            gt_arr = np.array([[g.range, g.velocity, g.angle] for g in gt])
            est_arr = np.array([[e.range, e.velocity, e.angle] for e in est_list])

            # 距离矩阵 + 匈牙利匹配
            dist = np.zeros((len(est_arr), len(gt_arr)))
            for i in range(len(est_arr)):
                for j in range(len(gt_arr)):
                    dist[i, j] = abs(est_arr[i, 0] - gt_arr[j, 0])

            row, col = linear_sum_assignment(dist)
            for i, j in zip(row, col):
                if dist[i, j] <= gate:
                    all_range_gt.append(gt_arr[j, 0])
                    all_range_est.append(est_arr[i, 0])
                    all_vel_gt.append(gt_arr[j, 1])
                    all_vel_est.append(est_arr[i, 1])
                    all_angle_gt.append(gt_arr[j, 2])
                    all_angle_est.append(est_arr[i, 2])
                    all_snr.append(est_list[i].snr_db)

        snr_linear = (
            10.0 ** (np.array(all_snr) / 10.0) if all_snr
            else np.array([])
        )
        return {
            "range_gt": np.array(all_range_gt),
            "range_est": np.array(all_range_est),
            "velocity_gt": np.array(all_vel_gt),
            "velocity_est": np.array(all_vel_est),
            "angle_gt": np.array(all_angle_gt),
            "angle_est": np.array(all_angle_est),
            "snr_linear": snr_linear,
        }
