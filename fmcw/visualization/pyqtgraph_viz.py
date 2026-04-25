"""PyQtGraph-based real-time visualization backend.

Provides GPU-accelerated (via OpenGL) real-time radar visualization at 20+ FPS
with four configurable panels: RD map, RA map, XY trajectory, and diagnostics.

Requires: pyqtgraph (with PyQt5/PySide6)

Usage:
    viz = PyQtGraphVisualizer(panels=["rd_map", "ra_map", "trajectory", "diagnostic"])
    viz.setup(radar_params)
    for frame_data in pipeline.run_streaming():
        viz.update(frame_data)
    viz.run()
    viz.stop()
"""

from typing import Dict, Any, List, Optional
import numpy as np

from .base import BaseVisualizer


class PyQtGraphVisualizer(BaseVisualizer):
    """Real-time pyqtgraph visualization with 20+ FPS target.

    Four panels:
        1. RD Map: Range-Doppler heatmap with detection overlays
        2. RA Map: Range-Angle heatmap
        3. Trajectory: XY Cartesian track plot
        4. Diagnostic: metrics over time (track count, detections, etc.)
    """

    def __init__(
        self,
        panels: List[str] = None,
        update_hz: int = 20,
        window_title: str = "FMCW Radar — Real-time Visualization",
        **kwargs,
    ):
        super().__init__(panels)
        self.update_hz = update_hz
        self.window_title = window_title
        self._win = None
        self._update_timer = None
        self._pending_data = None
        self._frame_count = 0

        # Diagnostic history
        self._n_tracks_history: List[int] = []
        self._n_dets_history: List[int] = []
        self._frame_history: List[int] = []

    def setup(self, radar_params, scene_config=None):
        """Initialize the pyqtgraph window with 4 panels."""
        try:
            import pyqtgraph as pg
            from pyqtgraph.Qt import QtCore, QtWidgets
        except ImportError:
            raise ImportError(
                "pyqtgraph is required. Install with: pip install pyqtgraph"
            )

        # Enable OpenGL for GPU-accelerated rendering if available
        pg.setConfigOptions(useOpenGL=True, antialias=True, imageAxisOrder="row-major")

        self._radar = radar_params
        self._app = QtWidgets.QApplication.instance()
        if self._app is None:
            self._app = QtWidgets.QApplication([])

        # Main window
        self._win = pg.GraphicsLayoutWidget(title=self.window_title, show=True)
        self._win.resize(1400, 900)

        # Panel dictionary
        self._panels = {}

        # Layout: 2x2 grid
        for i, panel in enumerate(self.panels):
            row = i // 2
            col = i % 2
            if panel == "rd_map":
                self._setup_rd_panel(row, col)
            elif panel == "ra_map":
                self._setup_ra_panel(row, col)
            elif panel == "trajectory":
                self._setup_trajectory_panel(row, col)
            elif panel == "diagnostic":
                self._setup_diagnostic_panel(row, col)

        # Update timer
        self._update_timer = QtCore.QTimer()
        self._update_timer.timeout.connect(self._on_timer)
        interval_ms = max(1, int(1000.0 / self.update_hz))
        self._update_timer.start(interval_ms)

    def _setup_rd_panel(self, row: int, col: int):
        """Setup Range-Doppler map panel."""
        import pyqtgraph as pg

        plot = self._win.addPlot(row=row, col=col)
        plot.setTitle("Range-Doppler Map")
        plot.setLabel("bottom", "Range bin")
        plot.setLabel("left", "Doppler bin")
        plot.setAspectLocked(False)

        # Image item for the RD heatmap
        img = pg.ImageItem()
        plot.addItem(img)

        # Scatter for detected peaks
        scatter = pg.ScatterPlotItem(pen=pg.mkPen("w", width=2), brush=None, symbol="x", size=10)
        plot.addItem(scatter)

        self._panels["rd_map"] = {"plot": plot, "image": img, "scatter": scatter}

    def _setup_ra_panel(self, row: int, col: int):
        """Setup Range-Angle map panel."""
        import pyqtgraph as pg

        plot = self._win.addPlot(row=row, col=col)
        plot.setTitle("Range-Angle Map")
        plot.setLabel("bottom", "Range bin")
        plot.setLabel("left", "Angle bin")
        plot.setAspectLocked(False)

        img = pg.ImageItem()
        plot.addItem(img)

        self._panels["ra_map"] = {"plot": plot, "image": img}

    def _setup_trajectory_panel(self, row: int, col: int):
        """Setup XY trajectory panel."""
        import pyqtgraph as pg

        plot = self._win.addPlot(row=row, col=col)
        plot.setTitle("Tracking Trajectories")
        plot.setLabel("bottom", "X (m)")
        plot.setLabel("left", "Y (m)")
        plot.setAspectLocked(True)
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.setXRange(-10, 80)
        plot.setYRange(-30, 30)

        # Empty dict of plot items, keyed by track_id
        self._panels["trajectory"] = {
            "plot": plot,
            "lines": {},      # track_id -> PlotDataItem
            "gt_line": None,  # ground truth line
        }

    def _setup_diagnostic_panel(self, row: int, col: int):
        """Setup diagnostic metrics panel."""
        import pyqtgraph as pg

        plot = self._win.addPlot(row=row, col=col)
        plot.setTitle("Diagnostics")
        plot.setLabel("bottom", "Frame")
        plot.setLabel("left", "Count")
        plot.showGrid(x=True, y=True, alpha=0.3)

        # Lines for different metrics
        track_line = plot.plot([], [], pen="c", name="Tracks")
        det_line = plot.plot([], [], pen="y", name="Detections")

        self._panels["diagnostic"] = {
            "plot": plot,
            "track_line": track_line,
            "det_line": det_line,
        }

    def update(self, frame_data: Dict[str, Any]):
        """Queue incoming frame data (called from pipeline thread)."""
        self._pending_data = frame_data

    def _on_timer(self):
        """Timer callback: render the latest frame data."""
        if self._pending_data is None or self._win is None:
            return

        data = self._pending_data
        self._pending_data = None
        self._frame_count += 1

        rd_map = data.get("rd_map")
        ra_map = data.get("ra_map")
        detections = data.get("detections")
        tracks = data.get("tracks", [])
        estimates = data.get("estimates")

        # RD Map
        if "rd_map" in self._panels and rd_map is not None:
            p = self._panels["rd_map"]
            p["image"].setImage(rd_map.T, autoLevels=False)
            # Only auto-level on first frame
            if self._frame_count == 1:
                p["image"].autoLevels()

            # Update detection scatter
            if detections is not None and len(detections) > 0:
                spots = [{"pos": (d[1], d[0])} for d in detections]
                p["scatter"].setData(spots)
            else:
                p["scatter"].setData([])

        # RA Map
        if "ra_map" in self._panels and ra_map is not None:
            p = self._panels["ra_map"]
            p["image"].setImage(ra_map.T, autoLevels=False)
            if self._frame_count == 1:
                p["image"].autoLevels()

        # Trajectory
        if "trajectory" in self._panels:
            self._update_trajectory(tracks, data.get("ground_truth"))

        # Diagnostic
        if "diagnostic" in self._panels:
            self._update_diagnostic(tracks, estimates)

    def _update_trajectory(self, tracks, ground_truth):
        """Update the trajectory panel with current track positions."""
        p = self._panels["trajectory"]
        plot = p["plot"]
        lines = p["lines"]

        # Track IDs currently displayed
        displayed = set(lines.keys())
        current_ids = set()

        for track in tracks:
            tid = track.track_id
            current_ids.add(tid)

            # Build position history array
            positions = []
            if hasattr(track, "history") and track.history:
                positions = track.history  # list of [x, y]

            if tid not in lines and len(positions) > 0:
                # Create new line for this track
                color = self._track_color(tid)
                line = plot.plot([], [], pen=color, symbol="o", symbolSize=5,
                                 name=f"T{tid}", symbolBrush=color)
                lines[tid] = line

            # Update existing line
            if tid in lines and len(positions) > 0:
                pts = np.array(positions)
                lines[tid].setData(pts[:, 0], pts[:, 1])

            # If only 1 position, just show the point
            if tid in lines and len(positions) == 0:
                lines[tid].setData([], [])

        # Remove tracks no longer present
        for tid in list(displayed - current_ids):
            plot.removeItem(lines[tid])
            del lines[tid]

    def _update_diagnostic(self, tracks, estimates):
        """Update diagnostic panel with metrics history."""
        p = self._panels["diagnostic"]

        n_tracks = len(tracks)
        n_dets = len(estimates.targets) if estimates and hasattr(estimates, "targets") else 0

        self._n_tracks_history.append(n_tracks)
        self._n_dets_history.append(n_dets)
        self._frame_history.append(self._frame_count)

        frames = self._frame_history
        p["track_line"].setData(frames, self._n_tracks_history)
        p["det_line"].setData(frames, self._n_dets_history)

    def run(self):
        """Start the Qt event loop."""
        if self._app is None or self._win is None:
            return
        import pyqtgraph as pg
        pg.exec()

    def stop(self):
        """Stop the visualization and close the window."""
        if self._update_timer is not None:
            self._update_timer.stop()
            self._update_timer = None
        if self._win is not None:
            self._win.close()
            self._win = None

    @staticmethod
    def _track_color(track_id: int):
        """Generate a consistent color for a given track ID."""
        import pyqtgraph as pg
        colors = [
            (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
            (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
            (188, 189, 34), (23, 190, 207),
        ]
        c = colors[track_id % len(colors)]
        return pg.mkPen(c, width=2)
