"""帧数据收集器：评估模块与 Radarpipeline 之间的唯一连接点。

支持两种收集模式：
- 模式 A（流式）：从 run_streaming() 逐帧收集
- 模式 B（批量）：从 run() 返回值 + Scene 独立重建真值
"""

from typing import List, Optional
import numpy as np

from fmcw.evaluation.models import FrameRecord
from fmcw.config.schema import SceneConfig, RadarParams


class FrameCollector:
    """从 RadarPipeline 输出逐帧或批量收集数据。

    评估模块与流水线之间的唯一连接点。支持两种收集模式。

    Usage:
        # 模式 A: run_streaming() 逐帧收集
        collector = FrameCollector()
        for frame_data in pipe.run_streaming():
            collector.collect_from_stream(frame_data)
        records = collector.records

        # 模式 B: run() 后批量重建（不修改 pipeline）
        outputs = pipe.run()
        collector = FrameCollector.from_pipeline_outputs(
            outputs, scene_cfg, radar,
        )
        records = collector.records
    """

    def __init__(self):
        self._records: List[FrameRecord] = []

    # ── 属性 ──

    @property
    def records(self) -> List[FrameRecord]:
        return self._records

    # ── 逐帧手动添加 ──

    def add_frame(
        self,
        frame_idx: int,
        timestamp: float = 0.0,
        signal: Optional[np.ndarray] = None,
        rd_map: Optional[np.ndarray] = None,
        ra_map: Optional[np.ndarray] = None,
        s_rda: Optional[np.ndarray] = None,
        detections: Optional[np.ndarray] = None,
        estimated: Optional[list] = None,
        tracks: Optional[list] = None,
        ground_truth: Optional[list] = None,
        proc_time_ms: float = 0.0,
        det_time_ms: float = 0.0,
        est_time_ms: float = 0.0,
        trk_time_ms: float = 0.0,
    ):
        """手动添加单帧数据。"""
        self._records.append(FrameRecord(
            frame_idx=frame_idx, timestamp=timestamp,
            signal=signal, rd_map=rd_map, ra_map=ra_map, s_rda=s_rda,
            detections=detections,
            estimated=estimated or [],
            tracks=tracks or [],
            ground_truth=ground_truth or [],
            proc_time_ms=proc_time_ms, det_time_ms=det_time_ms,
            est_time_ms=est_time_ms, trk_time_ms=trk_time_ms,
        ))

    # ── 模式 A：从 run_streaming() 收集 ──

    def collect_from_stream(self, frame_data: dict):
        """从 run_streaming() 的 yield 结果添加一帧。

        frame_data 需包含: frame_idx, signal, rd_map, ra_map,
                        detections, estimates, tracks, ground_truth
        """
        estimates = frame_data.get("estimates")
        targets = estimates.targets if estimates else []
        self.add_frame(
            frame_idx=frame_data["frame_idx"],
            signal=frame_data.get("signal"),
            rd_map=frame_data.get("rd_map"),
            ra_map=frame_data.get("ra_map"),
            detections=frame_data.get("detections"),
            estimated=targets,
            tracks=frame_data.get("tracks", []),
            ground_truth=frame_data.get("ground_truth", []),
        )

    # ── 模式 B：从 run() 批量重建（不需要 pipeline 修改） ──

    @classmethod
    def from_pipeline_outputs(
        cls,
        outputs: dict,
        scene_config: SceneConfig,
        radar: Optional[RadarParams] = None,
    ) -> "FrameCollector":
        """从 RadarPipeline.run() 的返回值批量构建 FrameCollector。

        核心设计：真值通过 Scene.all_trajectories() 独立重建，
        完全不依赖 RadarPipeline 内部修改。

        Args:
            outputs: RadarPipeline.run() 的返回值
                     (keys: signals, results, estimates, tracks)。
            scene_config: 原始场景配置（用于重建真值）。
            radar: 雷达系统参数（可选，scene 构建时需要）。

        Returns:
            填充完整的 FrameCollector 实例。
        """
        from fmcw.signal.scene import Scene

        # 使用默认雷达参数（如果未提供）
        if radar is None:
            from fmcw.config.schema import RadarParams
            radar = RadarParams()

        # 独立重建所有帧的真值——不依赖 pipeline 内部
        scene = Scene(radar, scene_config)
        all_gt = scene.all_trajectories()  # List[List[TargetParams]]

        collector = cls()
        n_frames = len(outputs.get("signals", []))

        detections_list = outputs.get("detections", [])
        for fi in range(n_frames):
            result = outputs["results"][fi] if fi < len(outputs.get("results", [])) else None
            est = outputs["estimates"][fi] if fi < len(outputs.get("estimates", [])) else None
            trk = outputs["tracks"][fi] if fi < len(outputs.get("tracks", [])) else []
            det = detections_list[fi] if fi < len(detections_list) else None
            gt = all_gt[fi] if fi < len(all_gt) else []

            collector.add_frame(
                frame_idx=fi,
                signal=outputs["signals"][fi] if fi < len(outputs.get("signals", [])) else None,
                rd_map=result.rd_map if result else None,
                ra_map=result.ra_map if result else None,
                s_rda=result.s_rda if result else None,
                detections=det,
                estimated=(est.targets if est else []),
                tracks=trk,
                ground_truth=gt,
            )

        return collector

    # ── 辅助 ──

    def clear(self):
        """清空所有已收集的帧记录。"""
        self._records.clear()

    def __len__(self):
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def __getitem__(self, idx):
        return self._records[idx]
