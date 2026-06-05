"""增量评估器：支持逐帧更新 + 随时查询当前指标的在线评估模式。

与离线评估器的区别：
- 离线：collect_all() → evaluate_all() → 返回聚合结果
- 增量：update(frame) → update(frame) → ... → query() → 返回当前结果
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from fmcw.evaluation.models import FrameRecord
from fmcw.evaluation.tracking_eval import TrackingEvaluator


class StreamingEvaluator(ABC):
    """增量评估器抽象基类。

    子类需实现 update() 和 query()。
    """

    @abstractmethod
    def update(self, record: FrameRecord) -> None:
        """摄入一帧数据，增量更新内部累积状态。

        时间复杂度要求：O(1) 或 O(window_size)。
        """
        ...

    @abstractmethod
    def query(self) -> dict:
        """查询当前累积的指标值。

        时间复杂度要求：O(window_size)，不阻塞。
        返回空的 dict 表示数据不足无法计算。

        Returns:
            {metric_name: float_value} 字典。
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """重置所有累积状态，用于开始新一轮评估。"""
        ...


class StreamingTrackingEvaluator(StreamingEvaluator):
    """增量跟踪评估器——滑动窗口内计算 GOSPA 和 MOTA。

    适用场景：pyqtgraph 实时可视化中，每 N 帧查询一次跟踪质量，
    在诊断面板上显示滚动的 GOSPA/MOTA 曲线。

    实现策略：
    - 维护固定大小的双端队列（deque），存储最近 window_size 帧
    - query() 时对窗口内数据执行完整的 TrackingEvaluator.evaluate()
    - 窗口大小建议 20-50 帧（1-2.5 秒 @ 20Hz）
    """

    def __init__(self, radar, window_size: int = 30):
        self._radar = radar
        self._window: deque = deque(maxlen=window_size)
        self._evaluator = TrackingEvaluator(radar)

    def update(self, record: FrameRecord) -> None:
        """将新帧加入滑动窗口。"""
        self._window.append(record)

    def query(self) -> dict:
        """计算滑动窗口内的 GOSPA/MOTA/MOTP。"""
        if len(self._window) < 5:
            return {}
        result = self._evaluator.evaluate(
            list(self._window),
            metrics=["gospa", "mota", "motp"],
        )
        return {
            "gospa": result.gospa.value if result.gospa else None,
            "mota": result.mota.value if result.mota else None,
            "motp": result.motp.value if result.motp else None,
        }

    def reset(self) -> None:
        self._window.clear()


class StreamingDetectionEvaluator(StreamingEvaluator):
    """增量检测评估器——累积 Pd/Pfa 统计。

    实现策略：
    - 维护累计计数器（total_gt, total_det, matched_gt, matched_det）
    - update() 仅做 O(M×N) 的计数器增量（M/N 通常很小）
    - query() 直接基于累计值计算 Pd 和 Pfa
    """

    def __init__(self, radar, range_gate_m: Optional[float] = None):
        self._radar = radar
        self._range_gate = range_gate_m or radar.range_resolution * 2.0
        self.reset()

    def update(self, record: FrameRecord) -> None:
        """增量更新检测统计计数器。"""
        gt = record.ground_truth
        dets = (
            record.detections if record.detections is not None
            else np.empty((0, 2), dtype=np.int64)
        )
        rd_map = record.rd_map

        self._total_gt += len(gt)
        self._total_det += len(dets)
        if rd_map is not None:
            self._total_cells += rd_map.size

        # 匈牙利匹配检测点→真值
        if len(gt) > 0 and len(dets) > 0:
            det_ranges = np.array([
                r_idx * self._radar.range_resolution
                for _, r_idx in dets
            ])
            gt_ranges = np.array([g.range for g in gt])
            dist = np.abs(det_ranges[:, None] - gt_ranges[None, :])
            row, col = linear_sum_assignment(dist)
            n_matched = int(np.sum(dist[row, col] <= self._range_gate))
            self._matched_det += n_matched
            self._matched_gt += n_matched

    def query(self) -> dict:
        """基于累计计数器计算 Pd 和 Pfa。"""
        if self._total_gt == 0:
            return {}
        return {
            "pd": self._matched_gt / max(self._total_gt, 1),
            "pfa": (
                (self._total_det - self._matched_det)
                / max(self._total_cells, 1)
            ),
            "total_frames_processed": self._total_frames,
        }

    def reset(self) -> None:
        self._total_gt = 0
        self._total_det = 0
        self._matched_gt = 0
        self._matched_det = 0
        self._total_cells = 0
        self._total_frames = 0
