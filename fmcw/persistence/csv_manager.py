"""CSV writer/reader for detection and tracking results.

Writes comma-separated CSV with fixed-width column alignment.
This ensures both machine readability (standard CSV, readable by Excel/pandas)
and human readability (columns visually aligned when viewed in monospace text).

Detection CSV columns:
    frame, peak_idx, range_bin, doppler_bin, angle_bin,
    range_m, velocity_ms, angle_deg, amplitude, is_det,
    gt_id, gt_range_m, gt_velocity_ms, gt_angle_deg

Tracking CSV columns:
    frame, track_id, status, x, y, vx, vy, P_xx, P_yy,
    gt_id, gt_x, gt_y, gt_vx, gt_vy, association
"""

import csv
from pathlib import Path
from typing import List, Optional
import numpy as np


# Column headers
DETECTION_HEADERS = [
    "frame", "peak_idx", "range_bin", "doppler_bin", "angle_bin",
    "range_m", "velocity_ms", "angle_deg", "amplitude", "is_det",
    "gt_id", "gt_range_m", "gt_velocity_ms", "gt_angle_deg",
]

# Column widths for visual alignment
DETECTION_WIDTHS = [7, 10, 11, 13, 11, 10, 13, 11, 11, 7, 7, 12, 16, 13]

TRACK_HEADERS = [
    "frame", "track_id", "status", "x", "y", "vx", "vy",
    "P_xx", "P_yy", "gt_id", "gt_x", "gt_y", "gt_vx", "gt_vy", "association",
]

TRACK_WIDTHS = [7, 10, 12, 9, 9, 9, 9, 9, 9, 7, 9, 9, 9, 9, 13]


def _fmt(val, default="-") -> str:
    """Format a single value for CSV."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    if isinstance(val, str):
        return val
    if hasattr(val, "item"):
        val = val.item()
    return str(val)


def _pad(s: str, width: int, align: str = "right") -> str:
    """Pad string to given width; truncate from left if too long."""
    if len(s) >= width:
        return s[-width:]
    if align == "left":
        return s.ljust(width)
    return s.rjust(width)


def _write_detection_row(writer, row: list):
    """Write one detection row with column-aligned comma-separated values."""
    cleaned = []
    for i, val in enumerate(row):
        width = DETECTION_WIDTHS[i] if i < len(DETECTION_WIDTHS) else 12
        if isinstance(val, float):
            cleaned.append(_pad(f"{val:.2f}", width))
        elif isinstance(val, int) and not isinstance(val, bool):
            cleaned.append(_pad(str(val), width))
        else:
            cleaned.append(_pad(_fmt(val), width))
    writer.writerow(cleaned)


def _write_track_row(writer, row: list):
    """Write one track row with column-aligned comma-separated values."""
    cleaned = []
    for i, val in enumerate(row):
        width = TRACK_WIDTHS[i] if i < len(TRACK_WIDTHS) else 10
        if isinstance(val, float):
            cleaned.append(_pad(f"{val:.3f}", width))
        elif isinstance(val, int) and not isinstance(val, bool):
            cleaned.append(_pad(str(val), width))
        else:
            cleaned.append(_pad(_fmt(val), width))
    writer.writerow(cleaned)


class DetectionCSVWriter:
    """Write detection results to CSV with ground truth annotations."""

    def __init__(self, path: str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    def _ensure_header(self):
        if not self._initialized:
            with open(self._path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                padded = [_pad(h, DETECTION_WIDTHS[i])
                          for i, h in enumerate(DETECTION_HEADERS)]
                writer.writerow(padded)
            self._initialized = True

    def write_frame(
        self,
        frame: int,
        estimates,           # FrameEstimates
        filter_results,      # filtered peaks (N, 2)
        all_candidates,      # all candidate peaks before CFAR (K, 2)
        ground_truth=None,   # List[TargetParams] or None
    ):
        """Write one frame of detection results.

        Args:
            frame: Frame index.
            estimates: FrameEstimates with target estimates.
            filter_results: (M, 2) peaks that passed CFAR.
            all_candidates: (K, 2) all candidate peaks.
            ground_truth: Optional ground truth target list.
        """
        self._ensure_header()
        with open(self._path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            matched_gt = set()

            # Write CFAR-passed detections first
            for i, est in enumerate(estimates.targets):
                gt_id = "-"
                gt_r = "-"
                gt_v = "-"
                gt_a = "-"
                if ground_truth:
                    best_dist = float("inf")
                    for gi, gt in enumerate(ground_truth):
                        if gi in matched_gt:
                            continue
                        dist = abs(est.range - gt.range)
                        if dist < 2.0 and dist < best_dist:
                            best_dist = dist
                            gt_id = str(getattr(gt, "id", gi))
                            gt_r = gt.range
                            gt_v = gt.velocity
                            gt_a = gt.angle
                    if best_dist < float("inf"):
                        matched_gt.add(gt_id)

                _write_detection_row(writer, [
                    frame, i, est.range_bin, est.doppler_bin, est.angle_bin,
                    est.range, est.velocity, est.angle, est.amplitude,
                    1, gt_id, gt_r, gt_v, gt_a,
                ])

            # Write CFAR-rejected candidates
            filtered_set = set()
            for fp in filter_results:
                filtered_set.add((int(fp[0]), int(fp[1])))

            peak_idx_offset = len(estimates.targets)
            for j, peak in enumerate(all_candidates):
                p = (int(peak[0]), int(peak[1]))
                if p in filtered_set:
                    continue
                _write_detection_row(writer, [
                    frame, peak_idx_offset + j, int(peak[1]), int(peak[0]), -1,
                    -1.0, -1.0, -1.0, -1.0,
                    0, "-", "-", "-", "-",
                ])


class TrackingCSVWriter:
    """Write tracking results to CSV with ground truth annotations."""

    def __init__(self, path: str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    def _ensure_header(self):
        if not self._initialized:
            with open(self._path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                padded = [_pad(h, TRACK_WIDTHS[i])
                          for i, h in enumerate(TRACK_HEADERS)]
                writer.writerow(padded)
            self._initialized = True

    def write_frame(
        self,
        frame: int,
        tracks: List,         # List[Track]
        assignments: dict = None,  # track_id -> association label
        ground_truth=None,    # List of ground truth states
    ):
        """Write one frame of tracking results.

        Args:
            frame: Frame index.
            tracks: List of active Track objects.
            assignments: Optional dict mapping track_id to association label.
            ground_truth: Optional ground truth states.
        """
        self._ensure_header()
        with open(self._path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for track in tracks:
                if track.filter is not None:
                    x = track.filter.state[0]
                    y = track.filter.state[1]
                    vx = track.filter.state[2]
                    vy = track.filter.state[3]
                    p_diag = np.diag(track.filter.P) if hasattr(track.filter, 'P') else [0, 0, 0, 0]
                    P_xx = p_diag[0]
                    P_yy = p_diag[1]
                else:
                    x, y, vx, vy, P_xx, P_yy = 0, 0, 0, 0, 0, 0

                gt_id = "-"
                gt_x = "-"
                gt_y = "-"
                gt_vx = "-"
                gt_vy = "-"
                association = assignments.get(track.track_id, "-") if assignments else "-"

                _write_track_row(writer, [
                    frame, track.track_id, track.status.value,
                    x, y, vx, vy, P_xx, P_yy,
                    gt_id, gt_x, gt_y, gt_vx, gt_vy, association,
                ])
