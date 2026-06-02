"""Panel drawing functions shared between visualization backends.

Each function draws one panel onto a given matplotlib Axes.
"""

import numpy as np
from typing import List, Optional

from fmcw.estimation.estimator import FrameEstimates


def draw_rd_map(
    ax,
    rd_map: np.ndarray,
    detections: Optional[np.ndarray] = None,
    ground_truth_bins: Optional[np.ndarray] = None,
    radar=None,
):
    """Draw Range-Doppler heatmap with detection overlay.

    Args:
        ax: matplotlib Axes.
        rd_map: (D, R) magnitude map (fftshift'd, DC at Doppler center).
        detections: (N, 2) [doppler_idx, range_idx] of detected peaks (fftshift'd).
        ground_truth_bins: (M, 2) ground truth peak locations.
        radar: RadarParams for axis labels and unit conversion.
    """
    ax.clear()

    # 使用真实物理单位设置坐标范围
    if radar is not None:
        r_extent = [0, radar.max_range]
        v_extent = [-radar.max_velocity, radar.max_velocity]
        x_label = "Range (m)"
        y_label = "Velocity (m/s)"
    else:
        r_extent = [0, rd_map.shape[1]]
        v_extent = [0, rd_map.shape[0]]
        x_label = "Range bin"
        y_label = "Doppler bin"

    ax.imshow(
        rd_map, origin="lower", aspect="auto", cmap="jet",
        extent=[r_extent[0], r_extent[1], v_extent[0], v_extent[1]],
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Range-Doppler Map", fontsize=10)

    if detections is not None and len(detections) > 0:
        # 将 bin 索引转换为真实物理坐标
        if radar is not None:
            r_vals = detections[:, 1] * radar.range_resolution
            v_vals = (detections[:, 0] - radar.chirp_num // 2) * radar.velocity_resolution
        else:
            r_vals = detections[:, 1]
            v_vals = detections[:, 0]
        ax.scatter(
            r_vals, v_vals,
            c="white", marker="x", s=40, linewidths=2, label="Detections",
        )
    if ground_truth_bins is not None and len(ground_truth_bins) > 0:
        if radar is not None:
            gt_r = ground_truth_bins[:, 1] * radar.range_resolution
            gt_v = (ground_truth_bins[:, 0] - radar.chirp_num // 2) * radar.velocity_resolution
        else:
            gt_r = ground_truth_bins[:, 1]
            gt_v = ground_truth_bins[:, 0]
        ax.scatter(
            gt_r, gt_v,
            c="lime", marker="o", s=50, facecolors="none", linewidths=2, label="GT",
        )
    if detections is not None or ground_truth_bins is not None:
        ax.legend(loc="upper right", fontsize=7)


def draw_ra_map(
    ax,
    ra_map: np.ndarray,
    estimates: Optional[List] = None,
    radar=None,
):
    """Draw Range-Angle heatmap.

    使用 pcolormesh 绘制，Y 轴直接以角度（度）为单位。
    FFT bins 在 sin(θ) 空间均匀分布，通过 arcsin 转换为实际角度
    作为网格边界，确保像素位置与物理角度精确对应。

    Args:
        ax: matplotlib Axes.
        ra_map: (A, R) magnitude map (fftshift'd, boresight at angle center).
        estimates: List of TargetEstimate.
        radar: RadarParams for axis labels and unit conversion.
    """
    ax.clear()
    A, R = ra_map.shape

    # X 轴（Range）网格边界
    if radar is not None:
        x_edges = np.linspace(0, radar.max_range, R + 1)
        x_label = "Range (m)"
    else:
        x_edges = np.arange(R + 1)
        x_label = "Range bin"

    # Y 轴（Angle）网格边界
    # FFT bins 在 sin 空间等间距，转为实际角度后不等间距
    # 构建 A+1 个边界：sin 从 -1 到 +1 均匀采样，再 arcsin 转角度
    sin_edges = np.linspace(-1.0, 1.0, A + 1)
    y_edges = np.degrees(np.arcsin(sin_edges))

    ax.pcolormesh(
        x_edges, y_edges, ra_map,
        cmap="jet", shading="flat", rasterized=True,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Range-Angle Map", fontsize=10)

    if estimates:
        if radar is not None:
            r_vals = [e.range for e in estimates]
        else:
            r_vals = [e.range_bin for e in estimates]
        # Y 轴直接用估计的物理角度值，与 pcolormesh 网格坐标系一致
        a_vals = [e.angle for e in estimates]
        ax.scatter(r_vals, a_vals, c="white", marker="x", s=80, linewidths=2)


def draw_trajectory(
    ax,
    tracks: List = None,
    ground_truth: Optional[List] = None,
    xlim: tuple = (-10, 80),
    ylim: tuple = (-30, 30),
):
    """Draw XY Cartesian trajectory plot.

    Args:
        ax: matplotlib Axes.
        tracks: List of Track objects with history.
        ground_truth: List of ground truth positions.
    """
    ax.clear()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Tracking Trajectories", fontsize=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axhline(0, color="gray", alpha=0.3)
    ax.axvline(0, color="gray", alpha=0.3)
    ax.grid(True, alpha=0.3)

    colors = plt_color_cycle()

    if tracks:
        for track in tracks:
            c = next(colors)
            history = np.array(track.history)
            if len(history) >= 2:
                ax.plot(history[:, 0], history[:, 1], "-o", color=c,
                        markersize=3, label=f"T{track.track_id}")
            elif len(history) == 1:
                ax.plot(history[0, 0], history[0, 1], "o", color=c,
                        markersize=6, label=f"T{track.track_id}")

    if ground_truth:
        gt = np.array(ground_truth)
        if gt.ndim == 2 and gt.shape[1] >= 2:
            ax.plot(gt[:, 0], gt[:, 1], "k--", linewidth=1.5, label="Ground Truth")

    if tracks or ground_truth:
        ax.legend(loc="upper right", fontsize=7)


def draw_diagnostic(
    ax,
    metrics_history: dict = None,
):
    """Draw diagnostic panel: errors and statistics over time.

    Args:
        ax: matplotlib Axes.
        metrics_history: dict with keys like 'rmse', 'n_tracks', 'n_detections'.
    """
    ax.clear()
    ax.set_xlabel("Frame")
    ax.set_ylabel("Value")
    ax.set_title("Diagnostics", fontsize=10)
    ax.grid(True, alpha=0.3)

    if metrics_history:
        for name, values in metrics_history.items():
            if values:
                ax.plot(range(len(values)), values, marker=".", label=name)
        ax.legend(fontsize=7)


def plt_color_cycle():
    """Generator of distinct colors."""
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    i = 0
    while True:
        yield colors[i % len(colors)]
        i += 1


# ------------------------------------------------------------------
# 信息表格绘制（matplotlib ax.table）
# ------------------------------------------------------------------

# 每个表格最多同时显示的行数
MAX_VISIBLE_ROWS = 6

# 表格字体大小
TABLE_FONTSIZE = 9


def draw_detection_table(ax, estimates, scroll_offset=0, max_visible=MAX_VISIBLE_ROWS):
    """在 ax 上渲染目标检测信息表格。

    Args:
        ax: matplotlib Axes（隐藏刻度，显示带框面板）。
        estimates: FrameEstimates 或 None。
        scroll_offset: 滑动偏移量（跳过的行数）。
        max_visible: 最大可见行数。
    """
    ax.clear()

    # 隐藏刻度，用 spines 直接作为面板边框（与 axes 边界精确对齐）
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#FAFBFC")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#B0B8C4")
        spine.set_linewidth(1.8)
        # 调整 spine 高度，确保与表格内容对齐---不正确
        # ax.spines["bottom"].set_bounds(0, max_visible)

    n_targets = len(estimates.targets) if (
        estimates and hasattr(estimates, "targets") and estimates.targets
    ) else 0

    if n_targets == 0:
        ax.text(0.5, 0.5, "No detections", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray",
                style="italic")
        ax.set_title("Target Detections", fontsize=12, fontweight="bold",
                     color="#4472C4", loc="left", pad=12)
        return

    col_labels = ["#", "Range\n(m)", "Vel\n(m/s)", "Angle\n(°)",
                  "X\n(m)", "Y\n(m)", "SNR\n(dB)"]

    all_rows = []
    for i, e in enumerate(estimates.targets, 1):
        x = e.range * np.cos(np.deg2rad(e.angle))
        y = e.range * np.sin(np.deg2rad(e.angle))
        all_rows.append([
            str(i),
            f"{e.range:.2f}",
            f"{e.velocity:+.2f}",
            f"{e.angle:+.1f}",
            f"{x:.2f}",
            f"{y:.2f}",
            f"{e.snr_db:.1f}",
        ])

    # 滑动窗口切片后，用空行填充到固定 max_visible 行
    end_idx = min(scroll_offset + max_visible, n_targets)
    cell_text = all_rows[scroll_offset:end_idx]
    visible_rows = len(cell_text)
    # 填充空行到固定行数，确保表格高度恒定
    while len(cell_text) < max_visible:
        cell_text.append([""] * len(col_labels))

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_FONTSIZE)
    # 纵向拉伸 1.7 倍确保多行表头不被裁切
    table.scale(1.0, 1.7)

    _style_table(table, max_visible, col_header_color="#4472C4")

    if n_targets > max_visible:
        title = (f"Target Detections — {n_targets} targets "
                 f"(showing {scroll_offset + 1}–{end_idx})")
    else:
        title = f"Target Detections — {n_targets} target(s)"

    ax.set_title(title, fontsize=12, fontweight="bold", color="#4472C4",
                 loc="left", pad=12)


def draw_tracking_table(ax, tracks, scroll_offset=0, max_visible=MAX_VISIBLE_ROWS):
    """在 ax 上渲染目标跟踪信息表格（含状态颜色编码）。

    Args:
        ax: matplotlib Axes（隐藏刻度，显示带框面板）。
        tracks: List[Track] 或 None。
        scroll_offset: 滑动偏移量（跳过的行数）。
        max_visible: 最大可见行数。
    """
    ax.clear()

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#FAFBFC")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#B0B8C4")
        spine.set_linewidth(1.8)

    if not tracks:
        ax.text(0.5, 0.5, "No tracks", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray",
                style="italic")
        ax.set_title("Track Status", fontsize=12, fontweight="bold",
                     color="#2B579A", loc="left", pad=12)
        return

    status_colors = {
        "confirmed": "#228B22",
        "tentative": "#FF8C00",
        "coasting": "#808080",
    }

    col_labels = ["ID", "Status", "X\n(m)", "Y\n(m)",
                  "Vx\n(m/s)", "Vy\n(m/s)", "Age"]

    all_rows = []
    all_status_colors = []
    for t in tracks:
        status = t.status.value if hasattr(t.status, "value") else str(t.status)

        if t.filter is not None and hasattr(t.filter, "position"):
            px, py = t.filter.position[0], t.filter.position[1]
        elif t.history:
            px, py = t.history[-1][0], t.history[-1][1]
        else:
            px, py = float("nan"), float("nan")

        if t.filter is not None and hasattr(t.filter, "state") and len(t.filter.state) >= 4:
            vx, vy = t.filter.state[2], t.filter.state[3]
        else:
            vx, vy = float("nan"), float("nan")

        x_str = f"{px:.2f}" if not np.isnan(px) else "--"
        y_str = f"{py:.2f}" if not np.isnan(py) else "--"
        vx_str = f"{vx:+.2f}" if not np.isnan(vx) else "--"
        vy_str = f"{vy:+.2f}" if not np.isnan(vy) else "--"

        all_rows.append([
            str(t.track_id), status, x_str, y_str, vx_str, vy_str, str(t.age),
        ])
        all_status_colors.append(status_colors.get(status, "#000000"))

    n_tracks = len(tracks)
    end_idx = min(scroll_offset + max_visible, n_tracks)
    cell_text = all_rows[scroll_offset:end_idx]
    row_colors = all_status_colors[scroll_offset:end_idx]
    # 填充空行到固定行数，确保表格高度恒定
    while len(cell_text) < max_visible:
        cell_text.append([""] * len(col_labels))

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_FONTSIZE)
    table.scale(1.0, 1.7)

    _style_table(table, max_visible, col_header_color="#2B579A")

    for i, color in enumerate(row_colors):
        cell = table[i + 1, 1]
        cell.get_text().set_color(color)
        cell.get_text().set_fontweight("bold")

    n_confirmed = sum(
        1 for t in tracks
        if (t.status.value if hasattr(t.status, "value") else str(t.status)) == "confirmed"
    )
    if n_tracks > max_visible:
        title = (f"Track Status — {n_tracks} tracks "
                 f"(showing {scroll_offset + 1}–{end_idx})")
    else:
        title = f"Track Status — {n_tracks} track(s)"
    if n_confirmed > 0:
        title += f" ({n_confirmed} confirmed)"

    ax.set_title(title, fontsize=12, fontweight="bold", color="#2B579A",
                 loc="left", pad=12)


def _style_table(table, n_rows: int, col_header_color: str = "#4472C4"):
    """给 matplotlib table 应用统一样式：表头着色 + 隔行交替 + 细边框。"""
    n_cols = max(k[1] for k in table.get_celld() if k[0] == 0) + 1

    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor(col_header_color)
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")
        cell.get_text().set_fontsize(TABLE_FONTSIZE)

    for i in range(n_rows):
        bg = "#F0F3F7" if i % 2 == 1 else "#FFFFFF"
        for j in range(n_cols):
            cell = table[i + 1, j]
            cell.set_facecolor(bg)
            cell.get_text().set_fontsize(TABLE_FONTSIZE)

    for cell in table.get_celld().values():
        cell.set_edgecolor("#C8CCD4")
        cell.set_linewidth(0.5)
