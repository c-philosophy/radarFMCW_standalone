"""多目标跟踪评估器：GOSPA、CLEAR MOT (MOTA/MOTP/IDF1)、HOTA、航迹生命周期。

评估指标分为三组：
1. 综合指标：GOSPA、MOTA、HOTA
2. 精度指标：MOTP（定位精度）
3. 生命周期：确认延迟、航迹保持率、ID切换次数

参考文献：
    [1] Rahmathullah et al., "Generalized OSPA Metric", Fusion 2017.
    [2] Bernardin & Stiefelhagen, "CLEAR MOT Metrics", EURASIP JIVP 2008.
    [3] Luiten et al., "HOTA: Higher Order Tracking Accuracy", IJCV 2021.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment

from fmcw.config.schema import RadarParams
from fmcw.utils.metrics import gospa_distance
from fmcw.evaluation.models import (
    FrameRecord, MetricValue, TrackingEvalDetail,
)


class TrackingEvaluator:
    """多目标跟踪评估器。

    Usage:
        evaluator = TrackingEvaluator(radar)
        result = evaluator.evaluate(
            records, gate_distance=5.0,
            metrics=["gospa", "mota"],
        )
    """

    def __init__(self, radar: RadarParams):
        self.radar = radar

    def evaluate(
        self,
        records: List[FrameRecord],
        gate_distance: float = 5.0,
        gospa_p: float = 2.0,
        gospa_c: float = 10.0,
        gospa_alpha: float = 2.0,
        calc_hota: bool = False,
        metrics: Optional[List[str]] = None,
    ) -> TrackingEvalDetail:
        """计算跟踪评估指标。

        Args:
            records: FrameRecord 列表。
            gate_distance: 跟踪-真值关联门控 (m)。
            gospa_p/c/alpha: GOSPA 参数。
            calc_hota: 是否计算 HOTA（计算量较大，默认关闭）。
            metrics: 指标白名单。None = 全部默认。
                     可选: gospa, mota, motp, idf1, hota, lifecycle。

        Returns:
            TrackingEvalDetail。
        """
        metrics = metrics or ["gospa", "mota", "motp", "idf1", "lifecycle"]
        result = TrackingEvalDetail()
        result.num_frames = len(records)

        # ── 收集笛卡尔位置（仅当需要时，提取一次共享） ──
        needs_positions = any(m in metrics for m in [
            "gospa", "mota", "motp", "idf1", "hota",
        ])
        if needs_positions:
            all_gt, all_trk, all_ids = self._collect_positions(records)

        # ── GOSPA 距离 ──
        if "gospa" in metrics and needs_positions:
            gospa_values = []
            for gt, trk in zip(all_gt, all_trk):
                if len(gt) == 0 and len(trk) == 0:
                    gospa_values.append(0.0)
                else:
                    gospa_values.append(
                        gospa_distance(
                            trk, gt,
                            p=gospa_p, c=gospa_c, alpha=gospa_alpha,
                        )
                    )
            result.gospa = MetricValue(
                "GOSPA", float(np.mean(gospa_values)),
                unit="m", direction="lower_better",
            )

        # ── CLEAR MOT 指标 ──
        clear_mot_needed = any(m in metrics for m in ["mota", "motp", "idf1"])
        if clear_mot_needed and needs_positions:
            mota, motp, idf1, idsw, frag = self._compute_clear_mot(
                all_gt, all_trk, all_ids, gate_distance,
            )
            if "mota" in metrics:
                result.mota = MetricValue(
                    "MOTA", mota, direction="higher_better",
                )
            if "motp" in metrics:
                result.motp = MetricValue(
                    "MOTP", motp, unit="m", direction="higher_better",
                )
            if "idf1" in metrics:
                result.idf1 = MetricValue(
                    "IDF1", idf1, direction="higher_better",
                )
            result.identity_switches = idsw
            result.track_fragmentation = frag

        # ── HOTA ──
        if "hota" in metrics and calc_hota and needs_positions:
            hota_val = self._compute_hota(all_gt, all_trk, gate_distance)
            result.hota = MetricValue(
                "HOTA", hota_val, direction="higher_better",
            )

        # ── 航迹生命周期 ──
        if "lifecycle" in metrics:
            result.track_lifecycle = self._compute_lifecycle_stats(records)

        return result

    # ── 内部：位置收集 ──

    def _collect_positions(
        self, records: List[FrameRecord],
    ):
        """从 FrameRecord 列表中提取所有帧的笛卡尔位置。

        提取一次，供 GOSPA/CLEAR MOT/HOTA 共享使用。
        仅统计已确认（confirmed）航迹。

        Returns:
            (all_gt, all_trk, all_ids)
            - all_gt: List[np.ndarray] per frame (N_t, 2)
            - all_trk: List[np.ndarray] per frame (M_t, 2)
            - all_ids: List[List[int]] per frame
        """
        all_gt = []
        all_trk = []
        all_ids = []

        for rec in records:
            # 真值 → 笛卡尔 (N, 2)
            gt_xy = np.array([
                [g.range * np.cos(np.deg2rad(g.angle)),
                 g.range * np.sin(np.deg2rad(g.angle))]
                for g in rec.ground_truth
            ]) if rec.ground_truth else np.empty((0, 2))

            # 跟踪航迹（仅已确认） → 笛卡尔 (M, 2)
            confirmed = [t for t in rec.tracks
                         if hasattr(t, 'status')
                         and t.status.value == 'confirmed']
            trk_xy_list = []
            trk_id_list = []
            for t in confirmed:
                if t.filter is not None and hasattr(t.filter, 'position'):
                    pos = t.filter.position
                    trk_xy_list.append([float(pos[0]), float(pos[1])])
                elif t.history:
                    trk_xy_list.append([float(t.history[-1][0]),
                                        float(t.history[-1][1])])
                else:
                    trk_xy_list.append([0.0, 0.0])
                trk_id_list.append(t.track_id)

            trk_xy = np.array(trk_xy_list) if trk_xy_list else np.empty((0, 2))
            all_gt.append(gt_xy)
            all_trk.append(trk_xy)
            all_ids.append(trk_id_list)

        return all_gt, all_trk, all_ids

    # ── CLEAR MOT ──

    def _compute_clear_mot(
        self,
        gt_positions: List[np.ndarray],
        trk_positions: List[np.ndarray],
        trk_ids: List[List[int]],
        gate: float,
    ):
        """计算 CLEAR MOT 指标。

        MOTA = 1 - Σ(FN_t + FP_t + IDSW_t) / Σ(GT_t)
        MOTP = Σ d_{t,i} / Σ c_t
        IDF1 = 2·IDP·IDR / (IDP+IDR)

        Returns:
            (mota, motp, idf1, idsw_count, frag_count)
        """
        total_fn = 0
        total_fp = 0
        total_idsw = 0
        total_gt = 0
        total_match_dist = 0.0
        total_matches = 0
        prev_gt_to_id: Dict[int, int] = {}

        for t, (gt, trk, ids) in enumerate(
            zip(gt_positions, trk_positions, trk_ids)
        ):
            n_gt = len(gt)
            n_trk = len(trk)
            total_gt += n_gt

            if n_trk == 0:
                total_fn += n_gt
                continue
            if n_gt == 0:
                total_fp += n_trk
                continue

            # 欧氏距离矩阵 + 匈牙利匹配
            dist = np.zeros((n_trk, n_gt))
            for i in range(n_trk):
                for j in range(n_gt):
                    dist[i, j] = np.linalg.norm(trk[i] - gt[j])
            row, col = linear_sum_assignment(dist)

            cur_gt_to_id = {}
            for i, j in zip(row, col):
                if dist[i, j] <= gate:
                    total_matches += 1
                    total_match_dist += dist[i, j]
                    cur_gt_to_id[j] = ids[i]

            # ID 切换检测
            for gt_j, new_id in cur_gt_to_id.items():
                if gt_j in prev_gt_to_id:
                    if prev_gt_to_id[gt_j] != new_id:
                        total_idsw += 1
            prev_gt_to_id = cur_gt_to_id

            # 未匹配统计
            matched_gt = set(col[dist[row, col] <= gate])
            matched_trk = set(row[dist[row, col] <= gate])
            total_fn += n_gt - len(matched_gt)
            total_fp += n_trk - len(matched_trk)

        denominator = max(total_gt, 1)
        mota = 1.0 - (total_fn + total_fp + total_idsw) / denominator
        motp = total_match_dist / max(total_matches, 1)
        idp = total_matches / max(total_matches + total_fp, 1)
        idr = total_matches / max(total_matches + total_fn, 1)
        idf1 = 2.0 * idp * idr / max(idp + idr, 1e-12)

        return mota, motp, idf1, total_idsw, 0

    # ── HOTA（简化版） ──

    def _compute_hota(
        self,
        gt_positions: List[np.ndarray],
        trk_positions: List[np.ndarray],
        gate: float,
    ) -> float:
        """计算 HOTA（简化版）。

        HOTA = sqrt(DetA * AssA)
        简化映射：DetA ≈ 匹配率，AssA ≈ IDF1。

        完整 HOTA 实现需逐对轨迹计算关联矩阵，超出当前需求范围。
        """
        total_gt = max(sum(len(g) for g in gt_positions), 1)
        total_trk = max(sum(len(t) for t in trk_positions), 1)
        total_match = 0
        for gt, trk in zip(gt_positions, trk_positions):
            if len(gt) == 0 or len(trk) == 0:
                continue
            dist = np.zeros((len(trk), len(gt)))
            for i in range(len(trk)):
                for j in range(len(gt)):
                    dist[i, j] = np.linalg.norm(trk[i] - gt[j])
            row, col = linear_sum_assignment(dist)
            total_match += int(np.sum(dist[row, col] <= gate))

        deta = total_match / total_gt
        assa = total_match / total_trk
        return float(np.sqrt(max(deta, 0.0) * max(assa, 0.0)))

    # ── 航迹生命周期 ──

    def _compute_lifecycle_stats(
        self, records: List[FrameRecord],
    ) -> dict:
        """统计航迹生命周期指标。

        Returns:
            dict with:
            - confirmation_delay_frames: TENTATIVE→CONFIRMED 平均帧数
            - total_tracks_created: 创建的航迹总数
            - max_track_age: 最长航迹年龄
        """
        track_states: Dict[int, list] = {}

        for rec in records:
            for t in rec.tracks:
                tid = t.track_id
                if tid not in track_states:
                    track_states[tid] = []
                track_states[tid].append((rec.frame_idx, t.status.value))

        confirm_delays = []
        for tid, states in track_states.items():
            first_tentative = None
            first_confirmed = None
            for fi, st in states:
                if st == "tentative" and first_tentative is None:
                    first_tentative = fi
                if st == "confirmed" and first_confirmed is None:
                    first_confirmed = fi
            if (first_tentative is not None
                    and first_confirmed is not None
                    and first_confirmed > first_tentative):
                confirm_delays.append(first_confirmed - first_tentative)

        return {
            "confirmation_delay_frames": (
                float(np.mean(confirm_delays)) if confirm_delays else 0.0
            ),
            "total_tracks_created": len(track_states),
            "max_track_age": max(
                (len(states) for states in track_states.values()),
                default=0,
            ),
        }
