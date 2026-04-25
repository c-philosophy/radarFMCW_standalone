"""Full radar processing pipeline: wires all stages together.

Provides the main entry point for end-to-end radar signal processing.
"""

from typing import List, Optional, Dict, Any
import time
import numpy as np

from fmcw.config.schema import RadarParams, SceneConfig, PipelineConfig
from fmcw.signal.scene import Scene
from fmcw.processing.processor import FMCWProcessor
from fmcw.detection.detector import DetectionPipeline
from fmcw.estimation.estimator import ParameterEstimator
from fmcw.tracking.multi_tracker import MultiTargetTracker
from fmcw.visualization.factory import create_visualizer
from fmcw.persistence.io_manager import IOManager


class RadarPipeline:
    """End-to-end FMCW radar processing pipeline.

    Usage:
        pipe = RadarPipeline(radar, pipeline_cfg, scene_cfg)
        results = pipe.run()

        # Or frame-by-frame:
        for frame_data in pipe.run_streaming():
            viz.update(frame_data)
    """

    def __init__(
        self,
        radar: RadarParams,
        pipeline_config: PipelineConfig,
        scene_config: SceneConfig,
    ):
        self.radar = radar
        self.pipeline_config = pipeline_config
        self.scene_config = scene_config

        # Lazy-built components
        self._scene: Optional[Scene] = None
        self._processor: Optional[FMCWProcessor] = None
        self._detector: Optional[DetectionPipeline] = None
        self._estimator: Optional[ParameterEstimator] = None
        self._tracker: Optional[MultiTargetTracker] = None
        self._viz = None
        self._io: Optional[IOManager] = None

    def _build(self):
        """Instantiate all processing components."""
        # Ensure all algorithm registrations are loaded
        import fmcw.detection.cfar      # noqa: registers CA/OS/GO/SO-CFAR
        import fmcw.tracking.kalman     # noqa: registers KF/EKF/UKF
        import fmcw.tracking.association  # noqa: registers NN/GNN/JPDA

        self._scene = Scene(self.radar, self.scene_config)

        self._processor = FMCWProcessor(
            self.radar,
            downsample=1,
            range_window=None,
            doppler_window=None,
            angle_window=None,
        )

        det_cfg = self.pipeline_config.detection
        self._detector = DetectionPipeline(
            cfar_algorithm=det_cfg.cfar_algorithm,
        )

        self._estimator = ParameterEstimator(self.radar)

        self._trk_cfg = self.pipeline_config.tracking
        self._tracker = MultiTargetTracker(
            tracker_type=self._trk_cfg.filter,
            associator_type=self._trk_cfg.association,
            dt=self._trk_cfg.dt,
            confirm_threshold=self._trk_cfg.init_threshold,
            delete_threshold=self._trk_cfg.coast_threshold,
        )

        viz_cfg = self.pipeline_config.visualization
        if viz_cfg.backend:
            self._viz = create_visualizer(
                backend=viz_cfg.backend,
                panels=viz_cfg.panels,
            )

        pers_cfg = self.pipeline_config.persistence
        if pers_cfg.save_intermediate:
            self._io = IOManager(
                output_dir=pers_cfg.output_dir,
            )
            self._io.logger.log_pipeline_start({
                "radar": {
                    "fc_hz": self.radar.fc,
                    "B_hz": self.radar.B,
                    "chirp_num": self.radar.chirp_num,
                    "sample_num": self.radar.sample_num,
                    "antenna_num": self.radar.antenna_num,
                },
                "scene": {
                    "num_frames": self.scene_config.num_frames,
                    "num_targets": len(self.scene_config.targets),
                },
            })

    def run(self) -> Dict[str, Any]:
        """Run the full pipeline for all frames.

        Returns:
            Dict with keys: 'signals', 'results', 'estimates', 'tracks'.
        """
        self._build()

        t_start = time.perf_counter()
        all_results = []
        all_estimates = []
        all_tracks = []
        all_signals = []

        # Setup visualization
        if self._viz:
            self._viz.setup(self.radar, self.scene_config)

        # Process frame-by-frame
        for frame_idx, signal, targets in self._scene.stream():
            t_frame = time.perf_counter()
            all_signals.append(signal)

            # Stage 1: Signal processing
            t0 = time.perf_counter()
            result = self._processor.process(signal)
            dt_proc = (time.perf_counter() - t0) * 1000

            if self._io:
                self._io.logger.log_stage(
                    frame_idx, "processing", dt_proc,
                    stats={"rd_max": float(result.rd_map.max())},
                )

            # Stage 2: Detection
            t0 = time.perf_counter()
            det_cfg = self.pipeline_config.detection
            candidates = self._detector.peak_finder.detect(result.rd_map)
            filtered = self._detector.run(
                result.rd_map,
                guard_cells=det_cfg.guard_cells,
                reference_cells=det_cfg.reference_cells,
                alpha=20.0,
                pfa=det_cfg.pfa,
            )
            dt_det = (time.perf_counter() - t0) * 1000

            if self._io:
                self._io.logger.log_stage(
                    frame_idx, "detection", dt_det,
                    params={"cfar": det_cfg.cfar_algorithm},
                    stats={"n_candidates": len(candidates), "n_detections": len(filtered)},
                )

            # Stage 3: Estimation
            t0 = time.perf_counter()
            estimates = self._estimator.estimate(
                result.s_rda, filtered,
                frame_idx=frame_idx,
                timestamp=frame_idx * self.scene_config.frame_interval,
                ground_truth=targets,
            )
            dt_est = (time.perf_counter() - t0) * 1000

            if self._io:
                self._io.logger.log_stage(
                    frame_idx, "estimation", dt_est,
                    stats={"n_estimates": len(estimates.targets)},
                )

            # Stage 4: Tracking
            t0 = time.perf_counter()
            tracks, new_tracks, _ = self._tracker.process_frame(estimates)
            dt_trk = (time.perf_counter() - t0) * 1000

            if self._io:
                self._io.logger.log_stage(
                    frame_idx, "tracking", dt_trk,
                    params={"filter": self._trk_cfg.filter, "association": self._trk_cfg.association},
                    stats={
                        "n_confirmed": sum(1 for t in tracks if t.status.value == "confirmed"),
                        "n_tentative": sum(1 for t in tracks if t.status.value == "tentative"),
                    },
                )

            # Persistence
            if self._io:
                self._io.save_signal(signal, frame_idx, self.radar, targets)
                self._io.save_detections(frame_idx, estimates, filtered, candidates, targets)
                self._io.save_tracks(frame_idx, tracks)

            # Visualization
            if self._viz:
                frame_data = {
                    "frame_idx": frame_idx,
                    "rd_map": result.rd_map,
                    "ra_map": result.ra_map,
                    "detections": filtered,
                    "estimates": estimates,
                    "tracks": tracks,
                    "ground_truth": targets,
                }
                self._viz.update(frame_data)

            # Collect results
            all_results.append(result)
            all_estimates.append(estimates)
            all_tracks.append(tracks)

        # Finalize
        t_total = time.perf_counter() - t_start
        if self._io:
            self._io.finish(self.scene_config.num_frames, t_total)

        if self._viz:
            self._viz.run()
            self._viz.stop()

        return {
            "signals": all_signals,
            "results": all_results,
            "estimates": all_estimates,
            "tracks": all_tracks,
        }

    def run_streaming(self):
        """Generator: yields frame data dicts for custom handling."""
        self._build()
        for frame_idx, signal, targets in self._scene.stream():
            result = self._processor.process(signal)
            filtered = self._detector.run(
                result.rd_map,
                guard_cells=self.pipeline_config.detection.guard_cells,
                reference_cells=self.pipeline_config.detection.reference_cells,
                alpha=20.0,
            )
            estimates = self._estimator.estimate(result.s_rda, filtered, frame_idx)
            tracks, _, _ = self._tracker.process_frame(estimates)

            yield {
                "frame_idx": frame_idx,
                "signal": signal,
                "rd_map": result.rd_map,
                "ra_map": result.ra_map,
                "detections": filtered,
                "estimates": estimates,
                "tracks": tracks,
                "ground_truth": targets,
            }
