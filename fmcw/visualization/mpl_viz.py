"""Matplotlib 可视化：离线回放和出版质量绘图。"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .base import BaseVisualizer
from .panels import draw_rd_map, draw_ra_map, draw_trajectory, draw_diagnostic

_logger = logging.getLogger(__name__)


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
        figsize: tuple = (14, 12),
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

    def setup(self, radar_params, scene_config=None):
        """Create the 4-panel figure."""
        self.fig = plt.figure(figsize=self.figsize)
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
                self.axes[panel] = self.fig.add_subplot(2, 2, panel_map[panel][0] * 2 + panel_map[panel][1] + 1)

        plt.tight_layout()

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
                # Convert polar ground truth to Cartesian
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

        self.fig.canvas.draw()
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
