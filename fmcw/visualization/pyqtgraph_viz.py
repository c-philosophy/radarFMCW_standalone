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


# ---------------------------------------------------------------------------
# Jet colormap（与 matplotlib 'jet' 一致），用于 ImageItem LUT
# ---------------------------------------------------------------------------
def _jet_lut(n_pts: int = 256):
    """构建 jet colormap 查找表 (N, 4) RGBA uint8。"""
    import pyqtgraph as pg

    pos = np.array([0.0, 0.125, 0.375, 0.625, 0.875, 1.0])
    colors = np.array([
        [0, 0, 143, 255],
        [0, 0, 255, 255],
        [0, 255, 255, 255],
        [255, 255, 0, 255],
        [255, 0, 0, 255],
        [128, 0, 0, 255],
    ], dtype=np.ubyte)
    cmap = pg.ColorMap(pos, colors)
    return cmap.getLookupTable(nPts=n_pts, alpha=False)


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

        # Info panel labels (created in setup)
        self._det_label = None
        self._trk_label = None

    def setup(self, radar_params, scene_config=None):
        """Initialize the pyqtgraph window with 4 panels."""
        try:
            import pyqtgraph as pg
            from pyqtgraph.Qt import QtCore, QtWidgets
        except ImportError:
            raise ImportError(
                "pyqtgraph is required. Install with: pip install pyqtgraph"
            )

        # 白色背景 + 黑色前景，提升热力图对比度和面板边界可见性
        pg.setConfigOptions(background='w', foreground='k',
                            useOpenGL=True, antialias=True, imageAxisOrder="row-major")

        self._radar = radar_params
        self._app = QtWidgets.QApplication.instance()
        if self._app is None:
            self._app = QtWidgets.QApplication([])

        # Main window
        self._win = pg.GraphicsLayoutWidget(title=self.window_title, show=True)
        self._win.resize(1400, 1100)  # 3×2 布局，纵向留信息面板空间

        # Panel dictionary
        self._panels = {}

        # Layout: 3×2 grid
        for i, panel in enumerate(self.panels):
            if panel == "rd_map":
                self._setup_rd_panel(row=0, col=0)
            elif panel == "ra_map":
                self._setup_ra_panel(row=0, col=1)
            elif panel == "trajectory":
                self._setup_trajectory_panel(row=1, col=0)
            elif panel == "diagnostic":
                self._setup_diagnostic_panel(row=1, col=1)

        # 信息面板始终显示（固定在 Row 2）
        self._setup_detection_info_panel(row=2, col=0)
        self._setup_tracking_info_panel(row=2, col=1)

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
        plot.setLabel("bottom", "Range (m)")
        plot.setLabel("left", "Velocity (m/s)")
        plot.setAspectLocked(False)
        # 禁止自动缩放，确保图像与坐标轴边界贴齐
        plot.disableAutoRange()
        plot.setXRange(0, self._radar.max_range, padding=0)
        plot.setYRange(-self._radar.max_velocity, self._radar.max_velocity, padding=0)

        # Image item with jet colormap
        img = pg.ImageItem()
        img.setLookupTable(_jet_lut())
        plot.addItem(img)

        # Scatter for detected peaks
        scatter = pg.ScatterPlotItem(
            pen=pg.mkPen("k", width=1.5), brush="w", symbol="x", size=10,
            name="Detections",
        )
        plot.addItem(scatter)

        self._panels["rd_map"] = {"plot": plot, "image": img, "scatter": scatter}

    def _setup_ra_panel(self, row: int, col: int):
        """Setup Range-Angle map panel.

        Y 轴内部用 sin(θ) 坐标（ImageItem 要求线性像素间距），
        通过自定义刻度标签显示实际角度度数。
        """
        import pyqtgraph as pg

        plot = self._win.addPlot(row=row, col=col)
        plot.setTitle("Range-Angle Map")
        plot.setLabel("bottom", "Range (m)")
        plot.setLabel("left", "Angle (deg)")
        plot.setAspectLocked(False)
        # 禁止自动缩放，确保图像与坐标轴边界贴齐
        plot.disableAutoRange()
        plot.setXRange(0, self._radar.max_range, padding=0)
        plot.setYRange(-1.0, 1.0, padding=0)

        # 自定义 Y 轴刻度：sin 位置 → 角度度数标签
        sin_ticks = [
            (-1.0, "-90°"), (-0.866, "-60°"), (-0.5, "-30°"),
            (0.0, "0°"),
            (0.5, "30°"), (0.866, "60°"), (1.0, "90°"),
        ]
        plot.getAxis("left").setTicks([sin_ticks])

        # Image item with jet colormap
        img = pg.ImageItem()
        img.setLookupTable(_jet_lut())
        plot.addItem(img)

        # Scatter for estimated target positions
        scatter = pg.ScatterPlotItem(
            pen=pg.mkPen("k", width=1.5), brush="w", symbol="x", size=12,
            name="Estimates",
        )
        plot.addItem(scatter)

        self._panels["ra_map"] = {"plot": plot, "image": img, "scatter": scatter}

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

        # 图例：小字体 + 灰色半透明背景框
        legend = plot.addLegend(offset=(-10, 10), labelTextSize="8pt")
        legend.setPen(pg.mkPen("gray", width=1))
        legend.setBrush(pg.mkBrush(245, 245, 245, 220))

        # Empty dict of plot items, keyed by track_id / gt_index
        self._panels["trajectory"] = {
            "plot": plot,
            "lines": {},       # track_id -> PlotDataItem
            "gt_lines": {},    # gt_index -> PlotDataItem (每个目标独立GT轨迹)
        }
        # 真值历史：{gt_index: [[x,y], ...]}
        self._gt_history: Dict[int, list] = {}

    def _setup_diagnostic_panel(self, row: int, col: int):
        """Setup diagnostic metrics panel."""
        import pyqtgraph as pg

        plot = self._win.addPlot(row=row, col=col)
        plot.setTitle("Diagnostics")
        plot.setLabel("bottom", "Frame")
        plot.setLabel("left", "Count")
        plot.showGrid(x=True, y=True, alpha=0.3)

        # 图例：小字体 + 灰色半透明背景框
        legend = plot.addLegend(offset=(-10, 10), labelTextSize="8pt")
        legend.setPen(pg.mkPen("gray", width=1))
        legend.setBrush(pg.mkBrush(245, 245, 245, 220))

        # 深色线条在白色背景下清晰可见
        track_line = plot.plot([], [], pen=pg.mkPen((31, 119, 180), width=2),
                               name="Tracks")
        det_line = plot.plot([], [], pen=pg.mkPen((214, 39, 40), width=2),
                             name="Detections")

        self._panels["diagnostic"] = {
            "plot": plot,
            "track_line": track_line,
            "det_line": det_line,
        }

    # ------------------------------------------------------------------
    # 信息显示面板（HTML 富文本表格）
    # ------------------------------------------------------------------

    def _setup_detection_info_panel(self, row: int, col: int):
        """检测信息面板：显示每帧目标估计参数。"""
        from pyqtgraph.Qt import QtWidgets, QtCore

        self._det_label = QtWidgets.QLabel()
        self._det_label.setTextFormat(QtCore.Qt.RichText)
        self._det_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        # 左边留白给标题缩进，右边不留白让表格贴齐边框
        self._det_label.setStyleSheet(
            "padding: 8px 2px 8px 8px; background: #FAFBFC;"
            "border: 2px solid #C8CCD4; border-left: 4px solid #4472C4;"
            "border-radius: 4px;"
        )
        self._det_label.setMinimumWidth(500)
        self._det_label.setMinimumHeight(120)
        self._det_label.setWordWrap(True)
        self._det_label.setText(
            "<i style='color:gray'>No detections</i>"
        )

        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(self._det_label)
        self._win.addItem(proxy, row=row, col=col)

    def _setup_tracking_info_panel(self, row: int, col: int):
        """跟踪信息面板：显示航迹状态参数。"""
        from pyqtgraph.Qt import QtWidgets, QtCore

        self._trk_label = QtWidgets.QLabel()
        self._trk_label.setTextFormat(QtCore.Qt.RichText)
        self._trk_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self._trk_label.setStyleSheet(
            "padding: 8px 2px 8px 8px; background: #FAFBFC;"
            "border: 2px solid #C8CCD4; border-left: 4px solid #2B579A;"
            "border-radius: 4px;"
        )
        self._trk_label.setMinimumWidth(500)
        self._trk_label.setMinimumHeight(120)
        self._trk_label.setWordWrap(True)
        self._trk_label.setText(
            "<i style='color:gray'>No tracks</i>"
        )

        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(self._trk_label)
        self._win.addItem(proxy, row=row, col=col)

    @staticmethod
    def _build_detection_html(estimates) -> str:
        """从 TargetEstimate 列表构建检测信息 HTML 表格。"""
        n_targets = len(estimates.targets) if (
            estimates and hasattr(estimates, "targets") and estimates.targets
        ) else 0

        if n_targets == 0:
            return (
                "<div style='font-size:13px; font-weight:bold; color:#4472C4;"
                " border-bottom:1px solid #C8CCD4; padding-bottom:3px;"
                " margin-bottom:3px; padding-left:2px'>Target Detections</div>"
                "<i style='color:gray; font-size:11px; padding-left:2px'>No detections</i>"
            )

        rows = []
        for i, e in enumerate(estimates.targets, 1):
            x = e.range * np.cos(np.deg2rad(e.angle))
            y = e.range * np.sin(np.deg2rad(e.angle))
            bg = "#FFFFFF" if i % 2 == 1 else "#F0F3F7"
            rows.append(
                f"<tr style='background:{bg}'>"
                f"<td style='text-align:center; width:6%; font-size:11px'>{i}</td>"
                f"<td style='text-align:right; width:15%; font-size:11px'>{e.range:.2f}</td>"
                f"<td style='text-align:right; width:15%; font-size:11px'>{e.velocity:+.2f}</td>"
                f"<td style='text-align:right; width:15%; font-size:11px'>{e.angle:+.1f}</td>"
                f"<td style='text-align:right; width:15%; font-size:11px'>{x:.2f}</td>"
                f"<td style='text-align:right; width:15%; font-size:11px'>{y:.2f}</td>"
                f"<td style='text-align:right; width:19%; font-size:11px'>{e.snr_db:.1f}</td>"
                f"</tr>"
            )

        header = (
            "<tr style='background:#4472C4; color:white; font-weight:bold;"
            " font-size:11px'>"
            "<th style='width:6%'>#</th>"
            "<th style='width:15%'>Range<br>(m)</th>"
            "<th style='width:15%'>Vel<br>(m/s)</th>"
            "<th style='width:15%'>Angle<br>(&deg;)</th>"
            "<th style='width:15%'>X<br>(m)</th>"
            "<th style='width:15%'>Y<br>(m)</th>"
            "<th style='width:19%'>SNR<br>(dB)</th></tr>"
        )

        title = (
            f"<div style='font-size:13px; font-weight:bold; color:#4472C4;"
            f" border-bottom:1px solid #C8CCD4; padding-bottom:3px;"
            f" margin-bottom:3px'>"
            f"Target Detections"
            f"<span style='color:gray; font-weight:normal; font-size:11px'>"
            f" &mdash; {n_targets} target(s)</span></div>"
        )

        return (
            f"{title}"
            f"<table border='0' cellpadding='2' cellspacing='0'"
            f" style='font-size:11px; width:100%'>"
            f"{header}{''.join(rows)}</table>"
        )

    @staticmethod
    def _build_tracking_html(tracks) -> str:
        """从 Track 列表构建跟踪信息 HTML 表格（含状态颜色编码）。"""
        n_tracks = len(tracks) if tracks else 0

        if n_tracks == 0:
            return (
                "<div style='font-size:13px; font-weight:bold; color:#2B579A;"
                " border-bottom:1px solid #C8CCD4; padding-bottom:3px;"
                " margin-bottom:3px; padding-left:2px'>Track Status</div>"
                "<i style='color:gray; font-size:11px; padding-left:2px'>No tracks</i>"
            )

        # 状态颜色
        status_colors = {
            "confirmed": "#228B22",
            "tentative": "#FF8C00",
            "coasting": "#808080",
        }

        rows = []
        for i, t in enumerate(tracks):
            status = t.status.value if hasattr(t.status, "value") else str(t.status)
            color = status_colors.get(status, "#000000")

            # 位置：优先从滤波器获取
            if t.filter is not None and hasattr(t.filter, "position"):
                pos = t.filter.position
                x, y = pos[0], pos[1]
            elif t.history:
                x, y = t.history[-1][0], t.history[-1][1]
            else:
                x, y = float("nan"), float("nan")

            # 速度：从滤波器状态向量提取
            if t.filter is not None and hasattr(t.filter, "state"):
                st = t.filter.state
                vx, vy = st[2], st[3] if len(st) >= 4 else (float("nan"), float("nan"))
            else:
                vx, vy = float("nan"), float("nan")

            bg = "#FFFFFF" if i % 2 == 1 else "#F0F3F7"
            x_str = f"{x:.2f}" if not np.isnan(x) else "--"
            y_str = f"{y:.2f}" if not np.isnan(y) else "--"
            vx_str = f"{vx:+.2f}" if not np.isnan(vx) else "--"
            vy_str = f"{vy:+.2f}" if not np.isnan(vy) else "--"

            rows.append(
                f"<tr style='background:{bg}'>"
                f"<td style='text-align:center; width:8%; font-size:11px'>{t.track_id}</td>"
                f"<td style='text-align:center; width:16%; color:{color};"
                f" font-weight:bold; font-size:11px'>{status}</td>"
                f"<td style='text-align:right; width:15%; font-size:11px'>{x_str}</td>"
                f"<td style='text-align:right; width:15%; font-size:11px'>{y_str}</td>"
                f"<td style='text-align:right; width:15%; font-size:11px'>{vx_str}</td>"
                f"<td style='text-align:right; width:15%; font-size:11px'>{vy_str}</td>"
                f"<td style='text-align:center; width:16%; font-size:11px'>{t.age}</td>"
                f"</tr>"
            )

        header = (
            "<tr style='background:#2B579A; color:white; font-weight:bold;"
            " font-size:11px'>"
            "<th style='width:8%'>ID</th>"
            "<th style='width:16%'>Status</th>"
            "<th style='width:15%'>X<br>(m)</th>"
            "<th style='width:15%'>Y<br>(m)</th>"
            "<th style='width:15%'>Vx<br>(m/s)</th>"
            "<th style='width:15%'>Vy<br>(m/s)</th>"
            "<th style='width:16%'>Age</th></tr>"
        )

        n_confirmed = sum(
            1 for t in tracks
            if (t.status.value if hasattr(t.status, "value") else str(t.status)) == "confirmed"
        )
        n_tentative = n_tracks - n_confirmed

        title = (
            f"<div style='font-size:13px; font-weight:bold; color:#2B579A;"
            f" border-bottom:1px solid #C8CCD4; padding-bottom:3px;"
            f" margin-bottom:3px'>"
            f"Track Status"
            f"<span style='color:gray; font-weight:normal; font-size:11px'>"
            f" &mdash; {n_tracks} track(s)"
        )

        if n_confirmed > 0:
            title += (
                f"<span style='color:#228B22; font-weight:bold; font-size:11px'>"
                f" ({n_confirmed} confirmed</span>"
                f"<span style='color:gray; font-weight:normal; font-size:11px'>"
                f", {n_tentative} tentative)</span>"
            )
        else:
            title += (
                f"<span style='color:gray; font-weight:normal; font-size:11px'>"
                f", {n_tentative} tentative</span>"
            )

        title += "</span></div>"

        return (
            f"{title}"
            f"<table border='0' cellpadding='2' cellspacing='0'"
            f" style='font-size:11px; width:100%'>"
            f"{header}{''.join(rows)}</table>"
        )

    # ------------------------------------------------------------------

    def update(self, frame_data: Dict[str, Any]):
        """Queue incoming frame data (called from pipeline thread)."""
        self._pending_data = frame_data

    def _on_timer(self):
        """Timer callback: render the latest frame data."""
        if self._pending_data is None or self._win is None:
            return

        from pyqtgraph.Qt import QtCore

        data = self._pending_data
        self._pending_data = None
        self._frame_count += 1

        rd_map = data.get("rd_map")
        ra_map = data.get("ra_map")
        detections = data.get("detections")
        tracks = data.get("tracks", [])
        estimates = data.get("estimates")

        # RD Map —— 去掉 .T 转置，使用物理坐标 rect
        if "rd_map" in self._panels and rd_map is not None:
            p = self._panels["rd_map"]
            rmax = self._radar.max_range
            vmax = self._radar.max_velocity
            # 负高度：ImageItem top-left origin 下
            #   np.flipud 后 row 0 = +vmax → 映射到 y=vmax (顶部)
            #   row D-1 = -vmax → 映射到 y=-vmax (底部)
            rd_rect = QtCore.QRectF(0, vmax, rmax, -2 * vmax)
            p["image"].setImage(
                np.flipud(rd_map), autoLevels=(self._frame_count == 1), rect=rd_rect,
            )

            # detection scatter → 真实物理坐标
            if detections is not None and len(detections) > 0:
                D = rd_map.shape[0]
                spots = [{"pos": (
                    d[1] * self._radar.range_resolution,
                    (d[0] - D // 2) * self._radar.velocity_resolution,
                )} for d in detections]
                p["scatter"].setData(spots)
            else:
                p["scatter"].setData([])

        # RA Map —— 去掉 .T 转置，Y 轴用 sin 坐标
        if "ra_map" in self._panels and ra_map is not None:
            p = self._panels["ra_map"]
            rmax = self._radar.max_range
            A = ra_map.shape[0]
            # 负高度：np.flipud 后 row 0 = sin(+1) → 顶部 y=1.0
            #   row A-1 = sin(-1) → 底部 y=-1.0
            ra_rect = QtCore.QRectF(0, 1.0, rmax, -2.0)
            p["image"].setImage(
                np.flipud(ra_map), autoLevels=(self._frame_count == 1), rect=ra_rect,
            )

            # estimate scatter → sin 坐标
            if estimates is not None and hasattr(estimates, "targets") and estimates.targets:
                spots = [{"pos": (
                    e.range,
                    2.0 * (e.angle_bin - A // 2) / A,
                )} for e in estimates.targets]
                p["scatter"].setData(spots)
            else:
                p["scatter"].setData([])

        # Trajectory
        if "trajectory" in self._panels:
            self._update_trajectory(tracks, data.get("ground_truth"))

        # Diagnostic
        if "diagnostic" in self._panels:
            self._update_diagnostic(tracks, estimates)

        # Detection / Tracking 信息面板
        if self._det_label is not None:
            self._det_label.setText(self._build_detection_html(estimates))
        if self._trk_label is not None:
            self._trk_label.setText(self._build_tracking_html(tracks))

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
                import pyqtgraph as pg
                color = self._track_color(tid)
                line = plot.plot([], [], pen=pg.mkPen(color, width=2), symbol="o", symbolSize=5,
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

        # 更新真值轨迹（每个目标独立累积历史 → 各自独立的虚线）
        gt_lines = p["gt_lines"]
        if ground_truth:
            current_gt_ids = set()
            for i, gt in enumerate(ground_truth):
                r = gt.range if hasattr(gt, 'range') else gt[0]
                a = gt.angle if hasattr(gt, 'angle') else gt[2]
                x = r * np.cos(np.deg2rad(a))
                y = r * np.sin(np.deg2rad(a))

                current_gt_ids.add(i)
                if i not in self._gt_history:
                    self._gt_history[i] = []
                self._gt_history[i].append([x, y])

                # 创建或更新 GT 线
                if i not in gt_lines:
                    import pyqtgraph as pg
                    from pyqtgraph.Qt import QtCore
                    gt_lines[i] = plot.plot(
                        [], [], pen=pg.mkPen("k", width=1.5, style=QtCore.Qt.DashLine),
                        name=f"GT-{i}",
                    )
                hist = np.array(self._gt_history[i])
                gt_lines[i].setData(hist[:, 0], hist[:, 1])

            # 清理已消失的目标
            for gt_id in list(gt_lines.keys()):
                if gt_id not in current_gt_ids:
                    plot.removeItem(gt_lines[gt_id])
                    del gt_lines[gt_id]
                    if gt_id in self._gt_history:
                        del self._gt_history[gt_id]
        else:
            # 无真值时清理所有 GT 线
            for gt_id in list(gt_lines.keys()):
                plot.removeItem(gt_lines[gt_id])
            gt_lines.clear()
            self._gt_history.clear()

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

    def refresh(self):
        """Process Qt events so the timer callback renders the latest frame."""
        if self._app is None:
            return
        from pyqtgraph.Qt import QtCore
        QtCore.QCoreApplication.processEvents()
        # QtCore.QCoreApplication.flush()

    @staticmethod
    def _track_color(track_id: int):
        """Generate a consistent color for a given track ID."""
        import pyqtgraph as pg
        colors = [
            (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
            (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
            (188, 189, 34), (23, 190, 207),
        ]
        return colors[track_id % len(colors)]
