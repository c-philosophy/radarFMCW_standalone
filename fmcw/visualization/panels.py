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
        rd_map: (D, R) magnitude map.
        detections: (N, 2) [doppler_idx, range_idx] of detected peaks.
        ground_truth_bins: (M, 2) ground truth peak locations.
        radar: RadarParams for axis labels.
    """
    ax.clear()
    ax.imshow(
        rd_map, origin="lower", aspect="auto", cmap="jet",
        extent=[0, rd_map.shape[1], 0, rd_map.shape[0]],
    )
    ax.set_xlabel("Range bin")
    ax.set_ylabel("Doppler bin")
    ax.set_title("Range-Doppler Map", fontsize=10)

    if detections is not None and len(detections) > 0:
        ax.scatter(
            detections[:, 1], detections[:, 0],
            c="white", marker="x", s=40, linewidths=2, label="Detections",
        )
    if ground_truth_bins is not None and len(ground_truth_bins) > 0:
        ax.scatter(
            ground_truth_bins[:, 1], ground_truth_bins[:, 0],
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

    Args:
        ax: matplotlib Axes.
        ra_map: (A, R) magnitude map.
        estimates: List of TargetEstimate.
        radar: RadarParams for axis labels.
    """
    ax.clear()
    angle_num = ra_map.shape[0]
    ax.imshow(
        ra_map, origin="lower", aspect="auto", cmap="jet",
        extent=[0, ra_map.shape[1], -angle_num // 2, angle_num // 2],
    )
    ax.set_xlabel("Range bin")
    ax.set_ylabel("Angle bin")
    ax.set_title("Range-Angle Map", fontsize=10)

    if estimates:
        r_bins = [e.range_bin for e in estimates]
        a_bins = [e.angle_bin for e in estimates]  # fftshift 后已居中
        ax.scatter(r_bins, a_bins, c="white", marker="x", s=80, linewidths=2)


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
