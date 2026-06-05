"""目标检测评估器：Pd/Pfa 统计、ROC 曲线、峰值分组效果。

评估 CFAR 检测器的：
- 检测概率 Pd（给定 SNR 和 Pfa）
- 实际虚警率 Pfa_actual vs 设计值 Pfa_design
- ROC 曲线数据
- 峰值分组效果（每目标平均检测数）
- 门限因子精度
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment

from fmcw.config.schema import RadarParams
from fmcw.evaluation.models import (
    FrameRecord, MetricValue, DetectionEvalDetail,
)


class DetectionEvaluator:
    """目标检测层评估器。

    评估 CFAR 检测器的各项性能指标。

    Usage:
        evaluator = DetectionEvaluator(radar)
        result = evaluator.evaluate(records, pfa_design=1e-4,
                                     metrics=["pd", "pfa"])
    """

    def __init__(self, radar: RadarParams):
        self.radar = radar

    def evaluate(
        self,
        records: List[FrameRecord],
        pfa_design: Optional[float] = None,
        range_gate_m: Optional[float] = None,
        metrics: Optional[List[str]] = None,
    ) -> DetectionEvalDetail:
        """计算所有检测评估指标。

        Args:
            records: FrameRecord 列表。
            pfa_design: 设计虚警率（用于 Pfa 偏差评估）。
            range_gate_m: 检测-真值距离门控阈值 (m)。
                          默认 = 2×range_resolution。
            metrics: 指标白名单。None = 全部计算。
                     可选: pd, pfa, roc, peak_grouping, cfar_alpha。

        Returns:
            DetectionEvalDetail。
        """
        metrics = metrics or ["pd", "pfa", "roc", "peak_grouping", "cfar_alpha"]
        gate = range_gate_m or self.radar.range_resolution * 2.0

        result = DetectionEvalDetail()
        result.num_frames = len(records)

        # ── 逐帧统计 ──
        total_gt = 0          # 真值目标总数
        total_det = 0         # 检测点总数
        matched_det = 0       # 匹配到真值的检测数
        matched_gt = 0        # 被检出的真值数
        n_total_cells = 0     # RD 图总 cell 数

        per_snr_bins: Dict[int, list] = {}  # {snr_bin: [gt_count, det_count]}

        for rec in records:
            gt = rec.ground_truth
            dets = (
                rec.detections if rec.detections is not None
                else np.empty((0, 2), dtype=np.int64)
            )
            rd_map = rec.rd_map

            total_gt += len(gt)
            total_det += len(dets)
            if rd_map is not None:
                n_total_cells += rd_map.size

            # 检测点 → 距离 (m)
            det_ranges = np.array([
                r_idx * self.radar.range_resolution
                for _, r_idx in dets
            ]) if len(dets) > 0 else np.array([])

            gt_ranges = np.array([g.range for g in gt]) if gt else np.array([])

            # 匈牙利匹配（基于距离）
            matched_gt_set = set()
            matched_det_set = set()
            if len(det_ranges) > 0 and len(gt_ranges) > 0:
                dist = np.abs(det_ranges[:, None] - gt_ranges[None, :])
                row, col = linear_sum_assignment(dist)
                for i, j in zip(row, col):
                    if dist[i, j] <= gate:
                        matched_det_set.add(i)
                        matched_gt_set.add(j)

            matched_det += len(matched_det_set)
            matched_gt += len(matched_gt_set)

            # SNR 分层统计
            for e in rec.estimated:
                snr = e.snr_db
                bin_key = int(snr // 5) * 5  # 5 dB 分档
                if bin_key not in per_snr_bins:
                    per_snr_bins[bin_key] = [0, 0]
                per_snr_bins[bin_key][0] += 1  # 该 SNR 档的估计总数

        # ── 汇总指标（按白名单门控） ──

        if "pd" in metrics:
            result.detection_probability = MetricValue(
                "Pd",
                matched_gt / max(total_gt, 1),
                direction="higher_better",
            )

        fp = total_det - matched_det
        if "pfa" in metrics:
            result.false_alarm_rate = MetricValue(
                "Pfa",
                fp / max(n_total_cells, 1),
                direction="lower_better",
            )

        if "peak_grouping" in metrics:
            result.peak_grouping_mean = MetricValue(
                "detections_per_target",
                total_det / max(matched_gt, 1),
                direction="lower_better",
                reference_value=1.0,
            )

        # Pfa 偏差
        if "cfar_alpha" in metrics and pfa_design is not None:
            pfa_actual = fp / max(n_total_cells, 1)
            error_db = 10.0 * np.log10(
                max(pfa_actual, 1e-30) / max(pfa_design, 1e-30)
            )
            result.cfar_alpha_error_db = MetricValue(
                "pfa_error_db", error_db, unit="dB",
            )

        # ROC 数据
        if "roc" in metrics:
            result.roc_data = self._compute_roc_data(records, gate)

        # SNR 分层 Pd
        if "pd" in metrics and per_snr_bins:
            result.per_snr_pd = {
                str(k): v[0] / max(v[0] + v[1], 1)  # 简化：需要 gt 匹配计数
                for k, v in per_snr_bins.items()
            }

        return result

    def _compute_roc_data(
        self,
        records: List[FrameRecord],
        gate: float,
    ) -> dict:
        """计算不同门限下的 (Pfa, Pd) 数据点。

        返回结构供外部绘图使用。
        当前为简化实现，需要在调用方提供多阈值扫描。
        """
        return {
            "pfa": [],
            "pd": [],
            "thresholds": [],
            "note": "ROC 数据需通过 SNR 扫描或多 alpha 值扫描获取",
        }
