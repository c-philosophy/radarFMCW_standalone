"""Full radar processing pipeline: wires all stages together.

Provides the main entry point for end-to-end radar signal processing.
"""

from typing import List, Optional, Dict, Any
import time
import numpy as np

from fmcw.config.schema import RadarParams, SceneConfig, PipelineConfig
from fmcw.signal.scene import Scene
from fmcw.processing.processor import FMCWProcessor
from fmcw.detection.detector import DetectionPipeline, _group_estimates_3d
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

        proc_cfg = self.pipeline_config.processing
        self._processor = FMCWProcessor(
            self.radar,
            downsample=proc_cfg.downsample,
            range_window=proc_cfg.range_window,
            doppler_window=proc_cfg.doppler_window,
            angle_window=proc_cfg.angle_window,
        )

        det_cfg = self.pipeline_config.detection
        self._detector = DetectionPipeline(
            cfar_algorithm=det_cfg.cfar_algorithm,
        )

        est_cfg = self.pipeline_config.estimation
        self._estimator = ParameterEstimator(self.radar, config=est_cfg)

        self._trk_cfg = self.pipeline_config.tracking
        self._tracker = MultiTargetTracker(
            tracker_type=self._trk_cfg.filter,
            associator_type=self._trk_cfg.association,
            dt=self._trk_cfg.dt,
            confirm_threshold=self._trk_cfg.init_threshold,
            delete_threshold=self._trk_cfg.coast_threshold,
            enable_imm=self._trk_cfg.enable_imm,
            imm_models=self._trk_cfg.imm_models,
            use_mahalanobis=self._trk_cfg.use_mahalanobis,
            gate_mahalanobis=self._trk_cfg.gate_mahalanobis,
            use_velocity_gating=self._trk_cfg.use_velocity_gating,
            gate_velocity=self._trk_cfg.gate_velocity,
            gate=self._trk_cfg.gate_mahalanobis,
            process_noise=self._trk_cfg.process_noise,
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

    def run(self, verbose: bool = False) -> Dict[str, Any]:
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
            self._viz.refresh()  # 让窗口先显示出来

        # Process frame-by-frame
        # 将计时起点重置到循环开始前，用于实时帧率控制
        for frame_idx, signal, targets in self._scene.stream():
            if verbose:
                print(f"Processing frame {frame_idx}...")

            # 实时帧率控制：按 frame_interval 节奏推进
            target_elapsed = frame_idx * self.scene_config.frame_interval
            actual_elapsed = time.perf_counter() - t_start
            if actual_elapsed < target_elapsed:
                time.sleep(target_elapsed - actual_elapsed)

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
            filtered = self._detector.run(
                result.rd_map,
                guard_cells=det_cfg.guard_cells,
                reference_cells=det_cfg.reference_cells,
                alpha=det_cfg.alpha,
                pfa=det_cfg.pfa,
            )
            dt_det = (time.perf_counter() - t0) * 1000

            if self._io:
                self._io.logger.log_stage(
                    frame_idx, "detection", dt_det,
                    params={"cfar": det_cfg.cfar_algorithm},
                    stats={"n_detections": len(filtered)},
                )

            # Stage 3: Estimation
            t0 = time.perf_counter()
            estimates = self._estimator.estimate(
                result.s_rda, filtered,
                frame_idx=frame_idx,
                timestamp=frame_idx * self.scene_config.frame_interval,
                ground_truth=targets,
                s_rd=result.s_rd,
            )
            dt_est = (time.perf_counter() - t0) * 1000

            # 3D 峰值分组（方案 A1）：合并 FFT 旁瓣产生的相邻检测，
            # 但保留同一 RD cell 不同角度的目标（角度多峰检测结果）
            raw_count = len(estimates.targets)
            if raw_count > 1:
                grouped = _group_estimates_3d(
                    estimates.targets, result.rd_map,
                    angle_num=self.radar.angle_num,
                )
                estimates.targets = grouped

            if verbose:
                print(f"Estimation: {raw_count} raw -> {len(estimates.targets)} grouped")
                print(f"index | range (m) | vel (m/s) | angle (deg) | doppler bin | range bin | angle bin| amplitude | SNR")
                for i, est in enumerate(estimates.targets):
                    print(f"{i} | {est.range:.2f} | {est.velocity:.2f} | {est.angle:.2f} | {est.doppler_bin} | {est.range_bin} | {est.angle_bin} | {est.amplitude:.2f} | {est.snr_db:.2f}")

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
                self._io.save_detections(
                    frame_idx, estimates, filtered,
                    np.empty((0, 2), dtype=np.int64), targets,
                )
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
                self._viz.refresh()  # 若使用pyqtgraph，处理 Qt 事件，触发定时器渲染当前帧

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
            # 1. Signal processing
            result = self._processor.process(signal)
            # 2. Detection
            filtered = self._detector.run(
                result.rd_map,
                guard_cells=self.pipeline_config.detection.guard_cells,
                reference_cells=self.pipeline_config.detection.reference_cells,
                alpha=self.pipeline_config.detection.alpha,
                pfa=self.pipeline_config.detection.pfa,
            )
            # 3. Estimation
            estimates = self._estimator.estimate(result.s_rda, filtered, frame_idx, s_rd=result.s_rd)
            # 4. Tracking
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
