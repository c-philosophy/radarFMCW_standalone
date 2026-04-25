"""统一 IO 管理器：协调日志、CSV 和信号存储。"""

import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from .log_manager import PipelineLogger
from .csv_manager import DetectionCSVWriter, TrackingCSVWriter
from .signal_store import NPZSignalStore

_logger = logging.getLogger(__name__)


class IOManager:
    """Unified IO interface for all persistence needs.

    Usage:
        io = IOManager("data", "run_001")
        io.logger.log_stage(0, "cfar", ...)
        io.save_detections(frame, estimates, filtered, candidates)
        io.save_tracks(frame, tracks)
        io.save_signal(signal, frame_idx=0)

    Output structure:
        data/run_001/
            pipeline_logs.jsonl
            detections.csv
            tracks.csv
            signal_frame_0000.npz
    """

    def __init__(self, output_dir: str = "data", run_id: str = None):
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id
        self.output_dir = Path(output_dir) / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = PipelineLogger(str(self.output_dir))
        self.detection_writer = DetectionCSVWriter(
            str(self.output_dir / "detections.csv")
        )
        self.track_writer = TrackingCSVWriter(
            str(self.output_dir / "tracks.csv")
        )
        self.signal_store = NPZSignalStore(str(self.output_dir))

        # Log run start
        self.logger.log_pipeline_start({"run_id": self.run_id})

    def save_detections(
        self,
        frame: int,
        estimates,              # FrameEstimates
        filtered_peaks,         # (M, 2) ndarray
        candidate_peaks,        # (K, 2) ndarray
        ground_truth=None,      # List[TargetParams]
    ):
        """Write detection results to CSV."""
        self.detection_writer.write_frame(
            frame=frame,
            estimates=estimates,
            filter_results=filtered_peaks,
            all_candidates=candidate_peaks,
            ground_truth=ground_truth,
        )

    def save_tracks(
        self,
        frame: int,
        tracks: List,           # List[Track]
        assignments: dict = None,
        ground_truth=None,
    ):
        """Write tracking results to CSV."""
        self.track_writer.write_frame(
            frame=frame,
            tracks=tracks,
            assignments=assignments,
            ground_truth=ground_truth,
        )

    def save_signal(
        self,
        signal,
        frame_idx: int = 0,
        radar_params=None,
        targets=None,
    ):
        """Save raw IF signal in NPZ format."""
        self.signal_store.save_frame(
            signal=signal,
            frame_idx=frame_idx,
            radar_params=radar_params,
            targets=targets,
        )

    def load_detections(self):
        """Load detection CSV as a pandas DataFrame.

        Returns:
            pd.DataFrame with detection results, or None if pandas unavailable.
        """
        try:
            import pandas as pd
            path = self.output_dir / "detections.csv"
            if path.exists():
                return pd.read_csv(path, skipinitialspace=True)
            return None
        except ImportError:
            _logger.warning("pandas required for load_tracks(). Install: pip install pandas")
            return None

    def load_tracks(self):
        """Load tracking CSV as a pandas DataFrame.

        Returns:
            pd.DataFrame with track results, or None if unavailable.
        """
        try:
            import pandas as pd
            path = self.output_dir / "tracks.csv"
            if path.exists():
                return pd.read_csv(path, skipinitialspace=True)
            return None
        except ImportError:
            print("pandas required for load_tracks(). Install: pip install pandas")
            return None

    def finish(self, total_frames: int, total_duration_s: float):
        """Mark pipeline as finished."""
        self.logger.log_pipeline_end(total_frames, total_duration_s)
