"""持久化模块测试（日志、CSV、信号存储）。"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
import shutil
import numpy as np

from fmcw.persistence.log_manager import PipelineLogger
from fmcw.persistence.csv_manager import DetectionCSVWriter, TrackingCSVWriter
from fmcw.persistence.signal_store import NPZSignalStore
from fmcw.persistence.io_manager import IOManager
from fmcw.estimation.estimator import FrameEstimates, TargetEstimate


def test_pipeline_logger_creation():
    """PipelineLogger should create valid JSON-lines output."""
    tmpdir = tempfile.mkdtemp()
    try:
        logger = PipelineLogger(tmpdir)
        logger.log_stage(0, "range_fft", duration_ms=10.5,
                         params={"window": "hanning"},
                         stats={"peak_bin": 67})

        # Check the log file
        log_path = logger.log_path
        assert log_path.exists()
        content = log_path.read_text()
        assert '"frame": 0' in content
        assert '"stage": "range_fft"' in content
        assert '"duration_ms": 10.5' in content or '"duration_ms": 10.5' in content.replace(" ", "")
    finally:
        shutil.rmtree(tmpdir)


def test_pipeline_logger_start_end():
    """PipelineLogger should log start and end events."""
    tmpdir = tempfile.mkdtemp()
    try:
        logger = PipelineLogger(tmpdir)
        logger.log_pipeline_start({"param": "value"})
        logger.log_pipeline_end(10, 1.5)

        content = logger.log_path.read_text()
        assert "pipeline_start" in content
        assert "pipeline_end" in content
        assert '"total_frames": 10' in content
    finally:
        shutil.rmtree(tmpdir)


def test_detection_csv_writer():
    """DetectionCSVWriter should produce valid CSV with correct columns."""
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "detections.csv")
        writer = DetectionCSVWriter(path)

        frame = 0
        estimates = FrameEstimates(
            frame_idx=0,
            targets=[
                TargetEstimate(range=50.1, velocity=10.0, angle=20.1,
                               doppler_bin=20, range_bin=67, angle_bin=30,
                               amplitude=1500.0, snr_db=25.0),
            ],
            timestamp=0.0,
        )
        filtered = np.array([[20, 67]])
        candidates = np.array([[20, 67], [30, 90], [5, 10]])

        writer.write_frame(frame, estimates, filtered, candidates)
        content = open(path).read()

        assert "frame" in content
        assert "range_m" in content
        assert "gt_id" in content
        assert "50.1" in content
        assert len(content.split("\n")) >= 3  # header + data row + empty
    finally:
        shutil.rmtree(tmpdir)


def test_tracking_csv_writer():
    """TrackingCSVWriter should produce valid CSV with correct columns."""
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "tracks.csv")
        writer = TrackingCSVWriter(path)

        # Create a mock track
        from fmcw.tracking.track import Track, TrackStatus, TrackManager
        tracker_factory = lambda: None  # no filter

        frame = 0
        track = Track(track_id=0, status=TrackStatus.CONFIRMED)
        track.history.append(np.array([20.0, 5.0]))

        writer.write_frame(frame, [track], assignments={0: "matched"})
        content = open(path).read()

        assert "frame" in content
        assert "track_id" in content
        assert "association" in content
    finally:
        shutil.rmtree(tmpdir)


def test_signal_store():
    """NPZSignalStore should save and reload signals correctly."""
    tmpdir = tempfile.mkdtemp()
    try:
        store = NPZSignalStore(tmpdir)
        signal = np.random.randn(8, 128, 256).astype(np.float32)

        store.save_frame(signal, frame_idx=0)
        data = store.load_frame(0)

        assert "signal" in data
        assert data["signal"].shape == signal.shape
        assert np.allclose(data["signal"], signal)
    finally:
        shutil.rmtree(tmpdir)


def test_io_manager():
    """IOManager should create the output directory structure."""
    tmpdir = tempfile.mkdtemp()
    try:
        io = IOManager(tmpdir, run_id="test_run")

        # Check directories and files
        output_path = os.path.join(tmpdir, "test_run")
        assert os.path.exists(output_path)
        assert os.path.exists(os.path.join(output_path, "pipeline_logs.jsonl"))
    finally:
        shutil.rmtree(tmpdir)


def test_io_manager_save_detections():
    """IOManager.save_detections should create detection CSV."""
    tmpdir = tempfile.mkdtemp()
    try:
        io = IOManager(tmpdir, run_id="test_det")

        estimates = FrameEstimates(
            frame_idx=0,
            targets=[
                TargetEstimate(range=50.1, velocity=10.0, angle=20.1,
                               doppler_bin=20, range_bin=67, angle_bin=30,
                               amplitude=1500.0, snr_db=25.0),
            ],
            timestamp=0.0,
        )
        io.save_detections(0, estimates,
                           filtered_peaks=np.array([[20, 67]]),
                           candidate_peaks=np.array([[20, 67]]))

        csv_path = os.path.join(tmpdir, "test_det", "detections.csv")
        assert os.path.exists(csv_path), f"Expected {csv_path} to exist"

        # Test load
        df = io.load_detections()
        assert df is not None
        assert len(df) >= 1
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
