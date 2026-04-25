"""Structured logging for pipeline intermediate results (JSON-lines format).

Writes one JSON object per line — easy to grep, parse, or load into pandas.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any


class PipelineLogger:
    """Structured logger for pipeline stages.

    Outputs to:
        1. Console (via Python logging)
        2. JSON-lines file (one JSON object per line)

    Usage:
        logger = PipelineLogger("data/run_001/")
        logger.log_stage(0, "cfar", duration_ms=3.2,
                         params={"algorithm":"ca_cfar"},
                         stats={"n_detections": 2})
    """

    def __init__(self, output_dir: str, run_name: str = "pipeline"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.output_dir / f"{run_name}_logs.jsonl"
        self._json_logger = logging.getLogger(f"pipeline.jsonl.{run_name}")
        self._json_logger.setLevel(logging.DEBUG)
        self._start_time = time.perf_counter()

    def log_stage(
        self,
        frame: int,
        stage: str,
        duration_ms: float = 0.0,
        params: Optional[Dict[str, Any]] = None,
        stats: Optional[Dict[str, Any]] = None,
        level: str = "INFO",
    ):
        """Log one pipeline stage execution.

        Args:
            frame: Frame index.
            stage: Stage name (e.g., 'range_fft', 'cfar', 'tracking').
            duration_ms: Elapsed wall-clock time in milliseconds.
            params: Stage parameters (algorithm name, thresholds, etc.).
            stats: Stage output statistics (counts, rates, etc.).
            level: Log level string.
        """
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "frame": frame,
            "stage": stage,
            "duration_ms": round(duration_ms, 3),
            "elapsed_s": round(time.perf_counter() - self._start_time, 3),
        }
        if params:
            entry["params"] = {k: self._serialize_value(v) for k, v in params.items()}
        if stats:
            entry["stats"] = {k: self._serialize_value(v) for k, v in stats.items()}

        # Write to JSON-lines file
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Also emit to Python logging
        msg = f"[Frame {frame}] {stage}: {duration_ms:.1f}ms"
        if stats:
            msg += " | " + " ".join(f"{k}={v}" for k, v in stats.items())
        getattr(logging, level.lower())(msg)

    def log_pipeline_start(self, config_summary: Dict[str, Any]):
        """Log pipeline configuration at start of run."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "pipeline_start",
            "config": config_summary,
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logging.info("Pipeline started")

    def log_pipeline_end(self, total_frames: int, total_duration_s: float):
        """Log pipeline completion statistics."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "pipeline_end",
            "total_frames": total_frames,
            "total_duration_s": round(total_duration_s, 3),
            "fps": round(total_frames / max(total_duration_s, 0.001), 1),
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logging.info(f"Pipeline done: {total_frames} frames in {total_duration_s:.1f}s")

    @staticmethod
    def _serialize_value(v: Any) -> Any:
        """Convert numpy values to Python-native types for JSON serialization."""
        if hasattr(v, "item"):           # numpy scalar
            return v.item()
        if isinstance(v, (list, tuple)):
            return [PipelineLogger._serialize_value(x) for x in v]
        return v

    @property
    def log_path(self) -> Path:
        return self._log_path
