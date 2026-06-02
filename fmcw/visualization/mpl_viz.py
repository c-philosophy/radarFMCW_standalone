"""Matplotlib 可视化：离线回放和出版质量绘图。"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np

# 先切换到安全的 matplotlib 后端，避免 PySide6/Qt 绑定缺失导致的 KeyError
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .base import BaseVisualizer
from .panels import (
    draw_rd_map, draw_ra_map, draw_trajectory, draw_diagnostic,
    draw_detection_table, draw_tracking_table, MAX_VISIBLE_ROWS,
)

_logger = logging.getLogger(__name__)


def _position_slider(slider, slider_ax, table_ax, val_max: int):
    """更新滑动条范围/可见性，并将位置紧贴到表格右侧（在 draw 之后调用）。"""
    if slider is None or slider_ax is None or table_ax is None:
        return
    if val_max <= 0:
        slider_ax.set_visible(False)
        return

    slider_ax.set_visible(True)
    slider.valmax = val_max
    slider.ax.set_ylim(0, val_max)
    if slider.val > val_max:
        slider.set_val(val_max)

    # 滑动条位置 = 表格 axes 右边缘 + 间隙
    pos = table_ax.get_position()
    slider_ax.set_position([pos.x1 + 0.008, pos.y0, 0.015, pos.height])


class MatplotlibVisualizer(BaseVisualizer):
    """Matplotlib backend: 4-panel layout for offline / playback visualization.

    Supports both:
        - Frame-by-frame manual update (for integration with pipeline)
        - FuncAnimation-based playback (from saved data)

    Usage:
        viz = MatplotlibVisualizer()
        viz.setup(radar)
        for frame in pipeline.run_streaming():
            viz.update(frame)
            viz.save_frame(f"frame_{frame['frame_idx']:04d}.png")
        viz.stop()
    """

    def __init__(
        self,
        panels: List[str] = None,
        figsize: tuple = (14, 12),  # GridSpec 高度比 2.5:2.5:1
        save_path: Optional[str] = None,
        save_dpi: int = 100,
    ):
        super().__init__(panels)
        self.figsize = figsize
        self.save_path = save_path
        self.save_dpi = save_dpi
        self.fig = None
        self.axes = {}
        self._frames_data = []
        self._anim = None
        # 滑动条（setup 中创建）
        self._det_slider = None
        self._trk_slider = None
        self._det_slider_ax = None
        self._trk_slider_ax = None

    def setup(self, radar_params, scene_config=None):
        """Create the 6-panel figure (3×2 layout，带滑动条)。"""
        from matplotlib.widgets import Slider

        # GridSpec: 图表行 2.5 倍于表格行高度
        self.fig = plt.figure(figsize=self.figsize, constrained_layout=True)
        gs = self.fig.add_gridspec(3, 2, height_ratios=[2.5, 2.5, 1.0])
        self.radar = radar_params

        panel_map = {
            "rd_map": (0, 0),
            "ra_map": (0, 1),
            "trajectory": (1, 0),
            "diagnostic": (1, 1),
        }

        self.axes = {}
        for panel in self.panels:
            if panel in panel_map:
                row, col = panel_map[panel]
                self.axes[panel] = self.fig.add_subplot(gs[row, col])

        # 信息表格在 Row 2
        self.axes["det_table"] = self.fig.add_subplot(gs[2, 0])
        self.axes["trk_table"] = self.fig.add_subplot(gs[2, 1])

        # 创建两个垂直滑动条（紧贴表格右侧，初始范围 >0 避免 ylim 警告）
        self._det_slider_ax = self.fig.add_axes([0.435, 0.06, 0.02, 0.10])
        self._det_slider = Slider(
            self._det_slider_ax, "", valmin=0, valmax=1, valinit=0, valstep=1,
            orientation="vertical",
        )
        self._det_slider_ax.set_visible(False)
        self._trk_slider_ax = self.fig.add_axes([0.905, 0.06, 0.02, 0.10])
        self._trk_slider = Slider(
            self._trk_slider_ax, "", valmin=0, valmax=1, valinit=0, valstep=1,
            orientation="vertical",
        )
        self._trk_slider_ax.set_visible(False)

    def update(self, frame_data: Dict[str, Any]):
        """Render one frame."""
        if self.fig is None:
            raise RuntimeError("Call setup() before update()")

        rd_map = frame_data.get("rd_map")
        ra_map = frame_data.get("ra_map")
        detections = frame_data.get("detections")
        estimates = frame_data.get("estimates")
        tracks = frame_data.get("tracks", [])
        ground_truth = frame_data.get("ground_truth")

        # RD Map
        if "rd_map" in self.axes and rd_map is not None:
            draw_rd_map(
                self.axes["rd_map"], rd_map,
                detections=detections,
                radar=self.radar,
            )

        # RA Map
        if "ra_map" in self.axes and ra_map is not None:
            draw_ra_map(
                self.axes["ra_map"], ra_map,
                estimates=estimates.targets if estimates else None,
                radar=self.radar,
            )

        # Trajectory
        if "trajectory" in self.axes:
            gt_positions = None
            if ground_truth:
                gt_positions = []
                for gt in ground_truth:
                    r = gt.range if hasattr(gt, 'range') else gt[0]
                    a = gt.angle if hasattr(gt, 'angle') else gt[2]
                    x = r * np.cos(np.deg2rad(a))
                    y = r * np.sin(np.deg2rad(a))
                    gt_positions.append([x, y])

            draw_trajectory(
                self.axes["trajectory"],
                tracks=tracks,
                ground_truth=gt_positions,
            )

        # Diagnostic
        if "diagnostic" in self.axes:
            metrics = {}
            n_tracks_key = "n_tracks"
            n_dets_key = "n_detections"
            if not hasattr(self, "_metrics_history"):
                self._metrics_history = {}
            if n_tracks_key not in self._metrics_history:
                self._metrics_history[n_tracks_key] = []
            if n_dets_key not in self._metrics_history:
                self._metrics_history[n_dets_key] = []
            self._metrics_history[n_tracks_key].append(len(tracks))
            self._metrics_history[n_dets_key].append(
                len(estimates.targets) if estimates else 0
            )
            draw_diagnostic(self.axes["diagnostic"], self._metrics_history)

        # Detection / Tracking 信息表格 + 滑动条
        n_targets = len(estimates.targets) if (
            estimates and hasattr(estimates, "targets")
        ) else 0
        n_tracks = len(tracks)

        det_max = max(0, n_targets - MAX_VISIBLE_ROWS)
        trk_max = max(0, n_tracks - MAX_VISIBLE_ROWS)

        # 获取滑动条当前偏移（跨帧保持）
        if det_max > 0 and self._det_slider is not None:
            det_offset = min(int(self._det_slider.val), det_max)
        else:
            det_offset = 0
        if trk_max > 0 and self._trk_slider is not None:
            trk_offset = min(int(self._trk_slider.val), trk_max)
        else:
            trk_offset = 0

        # 绘制表格（单次，不再双重绘制）
        if "det_table" in self.axes:
            draw_detection_table(
                self.axes["det_table"], estimates,
                scroll_offset=det_offset, max_visible=MAX_VISIBLE_ROWS,
            )
        if "trk_table" in self.axes:
            draw_tracking_table(
                self.axes["trk_table"], tracks,
                scroll_offset=trk_offset, max_visible=MAX_VISIBLE_ROWS,
            )

        # 先 draw 让 constrained_layout 计算所有 axes 位置
        self.fig.canvas.draw()

        # 在 draw 之后更新滑动条范围和位置（此时 axes 位置已确定）
        _position_slider(self._det_slider, self._det_slider_ax,
                         self.axes.get("det_table"), det_max)
        _position_slider(self._trk_slider, self._trk_slider_ax,
                         self.axes.get("trk_table"), trk_max)

        self.fig.canvas.flush_events()

    def run(self):
        """Show the figure (non-blocking for interactive use)."""
        if self.fig is not None:
            plt.show(block=False)

    def stop(self):
        """Close the figure."""
        if self.save_path and self._frames_data:
            self._save_animation()
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

    def save_frame(self, path: str):
        """Save current frame as image."""
        if self.fig is not None:
            self.fig.savefig(path, dpi=self.save_dpi, bbox_inches="tight")

    def _save_animation(self):
        """Save all recorded frames as an animation."""
        if not self._frames_data:
            return
        try:
            from matplotlib.animation import FuncAnimation
            anim = FuncAnimation(
                self.fig,
                lambda i: self.update(self._frames_data[i]),
                frames=len(self._frames_data),
                interval=200,
            )
            anim.save(self.save_path, dpi=self.save_dpi, writer="pillow")
            _logger.info("Animation saved to %s", self.save_path)
        except Exception as e:
            _logger.error("Failed to save animation: %s", e)
