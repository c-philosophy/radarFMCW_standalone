"""Multi-target tracker: orchestrates association + filtering per frame."""

from typing import List, Optional, Tuple
import numpy as np

from fmcw.estimation.estimator import FrameEstimates, TargetEstimate
from fmcw.core.registry import AlgorithmRegistry
from .track import Track, TrackManager
from .kalman import create_tracker


class MultiTargetTracker:
    """Orchestrates multi-target tracking frame-by-frame.

    Data flow per frame:
        1. Predict all existing tracks forward.
        2. Extract TrackState (position + covariance + velocity) for each track.
        3. Convert estimates from polar to Cartesian for association.
        4. Associate measurements to tracks (Mahalanobis distance + velocity gating).
        5. Update matched tracks, coast unmatched, initiate new tracks.
        6. Prune expired tracks.

    Supports IMM (Interacting Multiple Model) and single-filter modes.
    """

    def __init__(
        self,
        tracker_type: str = "kf",
        associator_type: str = "gnn",
        dt: float = 0.05,
        confirm_threshold: int = 3,
        delete_threshold: int = 5,
        gate: float = 3.0,
        # IMM 配置
        enable_imm: bool = False,
        imm_models: tuple = ("cv", "ca", "ctra"),
        # 关联优化
        use_mahalanobis: bool = True,
        gate_mahalanobis: float = 6.0,
        use_velocity_gating: bool = True,
        gate_velocity: float = 3.0,
        **tracker_kwargs,
    ):
        self.dt = dt
        self.associator_type = associator_type
        self._tracker_type = tracker_type
        self.enable_imm = enable_imm
        self.imm_models = imm_models
        self._tracker_kwargs = tracker_kwargs

        def make_tracker():
            if self.enable_imm:
                return self._build_imm()
            cls = AlgorithmRegistry.get("tracker", tracker_type)
            return cls(dt=dt, **tracker_kwargs)

        self.track_manager = TrackManager(
            confirm_threshold=confirm_threshold,
            delete_threshold=delete_threshold,
            tracker_factory=make_tracker,
        )

        assoc_cls = AlgorithmRegistry.get("associator", associator_type)
        self.associator = assoc_cls(
            gate=gate_mahalanobis if use_mahalanobis else gate,
            use_mahalanobis=use_mahalanobis,
            use_velocity_gating=use_velocity_gating,
            gate_velocity=gate_velocity,
        )

    def _build_imm(self):
        """构造 IMM 实例。"""
        from fmcw.tracking.imm import InteractingMultipleModel
        from fmcw.tracking.kalman import ExtendedKalmanFilter, UnscentedKalmanFilter
        from fmcw.tracking.motion_model import CVModel, CAModel, CTRAModel

        model_map = {
            "cv":   CVModel,
            "ca":   CAModel,
            "ctra": CTRAModel,
        }
        filter_map = {
            "cv":   ExtendedKalmanFilter,
            "ca":   ExtendedKalmanFilter,
            "ctra": UnscentedKalmanFilter,
        }

        branches = []
        for m in self.imm_models:
            model_cls = model_map[m]
            filt_cls = filter_map[m]
            filt = filt_cls(dt=self.dt, model=model_cls(), **self._tracker_kwargs)
            branches.append((m, filt))

        return InteractingMultipleModel(branches=branches)

    # ── 每帧处理 ─────────────────────────────────────────────────────────

    def process_frame(
        self,
        estimates: FrameEstimates,
    ) -> Tuple[List[Track], List[Track], List[Track]]:
        """Process one frame of detection estimates.

        Args:
            estimates: FrameEstimates from the parameter estimator.

        Returns:
            (updated_tracks, new_tracks, deleted_tracks).
        """
        # Step 1: Predict all tracks
        self.track_manager.predict_all()

        # Step 2: Extract TrackState for each track
        track_states = self._get_track_states()

        # Step 3: Convert estimates to Cartesian + extract velocities
        measurements, velocities = self._extract_measurements(estimates)

        # Step 4: Association
        result = self.associator.associate(track_states, measurements, velocities)

        assignments = result[0]
        unassigned_tracks = result[1]
        unassigned_meas = result[2]

        # Step 5: Update associated tracks
        active_tracks = self.track_manager.get_active()
        for i, track in enumerate(active_tracks):
            if assignments[i] >= 0:
                meas = measurements[assignments[i]]
                if self._tracker_type in ("ekf", "ukf") and not self.enable_imm:
                    # 单 EKF/UKF 模式：极坐标观测
                    r = np.sqrt(meas[0]**2 + meas[1]**2)
                    th = np.rad2deg(np.arctan2(meas[1], meas[0]))
                    self.track_manager.update(track, np.array([r, th]))
                else:
                    # KF 或 IMM 模式：直接使用笛卡尔坐标
                    self.track_manager.update(track, meas)

        # Coast unassigned tracks
        for idx in unassigned_tracks:
            if idx < len(active_tracks):
                self.track_manager.coast(active_tracks[idx])

        # Initiate new tracks
        new_tracks = []
        for idx in unassigned_meas:
            meas = measurements[idx]
            track = self.track_manager.initiate(meas)
            new_tracks.append(track)

        # Prune deleted tracks
        self.track_manager.prune()

        return (
            self.track_manager.get_active(),
            new_tracks,
            [],
        )

    # ── 辅助方法 ─────────────────────────────────────────────────────────

    def _get_track_states(self) -> List:
        """从活跃航迹提取 TrackState 列表。"""
        from fmcw.tracking.association import TrackState
        states = []
        for track in self.track_manager.get_active():
            if track.filter is not None:
                pos = track.filter.position
                # 位置协方差子矩阵
                P = track.filter.P[:2, :2] if hasattr(track.filter, 'P') else np.eye(2)

                # 径向速度预测
                x, y = pos[0], pos[1]
                r = np.sqrt(x**2 + y**2)
                if r > 1e-6 and hasattr(track.filter, 'x') and len(track.filter.x) >= 4:
                    vx, vy = track.filter.x[2], track.filter.x[3]
                    v_pred = (x * vx + y * vy) / r
                    # 从 P 矩阵提取速度方差
                    p_v = (track.filter.P[2, 2] + track.filter.P[3, 3]) / 2 if hasattr(track.filter, 'P') else 1.0
                else:
                    v_pred = 0.0
                    p_v = 1.0

                states.append(TrackState(
                    position=pos.copy(),
                    covariance=P.copy(),
                    velocity=v_pred,
                    p_velocity=p_v,
                ))
            else:
                states.append(TrackState(
                    position=track.history[-1][:2] if track.history else np.zeros(2),
                    covariance=np.eye(2) * 10.0,
                ))
        return states

    def _extract_measurements(self, estimates: FrameEstimates):
        """从 FrameEstimates 提取位置和速度。"""
        n = len(estimates.targets)
        measurements = np.empty((n, 2))
        velocities = np.empty(n)
        for i, t in enumerate(estimates.targets):
            th = np.deg2rad(t.angle)
            measurements[i] = [t.range * np.cos(th), t.range * np.sin(th)]
            velocities[i] = t.velocity
        return measurements, velocities

    @property
    def tracks(self) -> List[Track]:
        return self.track_manager.get_active()

    @property
    def confirmed_tracks(self) -> List[Track]:
        return self.track_manager.get_confirmed()
