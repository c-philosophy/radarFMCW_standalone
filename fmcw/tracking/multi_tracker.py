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
        2. Convert estimates from polar to Cartesian for association.
        3. Associate measurements to tracks.
        4. Update matched tracks, coast unmatched, initiate new tracks.
        5. Prune expired tracks.
    """

    def __init__(
        self,
        tracker_type: str = "kf",
        associator_type: str = "gnn",
        dt: float = 0.05,
        confirm_threshold: int = 3,
        delete_threshold: int = 5,
        gate: float = 3.0,
        **tracker_kwargs,
    ):
        self.dt = dt
        self.associator_type = associator_type
        self._tracker_type = tracker_type

        def make_tracker():
            # Import inside to avoid circular import
            from .kalman import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
            cls = AlgorithmRegistry.get("tracker", tracker_type)
            return cls(dt=dt, **tracker_kwargs)

        self.track_manager = TrackManager(
            confirm_threshold=confirm_threshold,
            delete_threshold=delete_threshold,
            tracker_factory=make_tracker,
        )

        assoc_cls = AlgorithmRegistry.get("associator", associator_type)
        self.associator = assoc_cls(gate=gate)

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

        # Step 2: Convert estimates to Cartesian for association
        measurements = []
        for t in estimates.targets:
            r = t.range
            th = np.deg2rad(t.angle)
            x = r * np.cos(th)
            y = r * np.sin(th)
            measurements.append([x, y])
        measurements = np.array(measurements) if measurements else np.empty((0, 2))

        # Step 3: Association
        track_positions = self.track_manager.get_positions()
        result = self.associator.associate(track_positions, measurements)

        assignments = result[0]
        unassigned_tracks = result[1]
        unassigned_meas = result[2]
        beta = result[3] if len(result) > 3 else None  # JPDA betas

        # Step 4: Update associated tracks
        active_tracks = self.track_manager.get_active()
        for i, track in enumerate(active_tracks):
            if assignments[i] >= 0:
                meas = measurements[assignments[i]]
                # For EKF/UKF, the measurement is in polar coordinates
                if self._tracker_type in ("ekf", "ukf"):
                    # Convert Cartesian → polar for EKF/UKF
                    r = np.sqrt(meas[0]**2 + meas[1]**2)
                    th = np.rad2deg(np.arctan2(meas[1], meas[0]))
                    polar_meas = np.array([r, th])
                    self.track_manager.update(track, polar_meas)
                else:
                    self.track_manager.update(track, meas)

        # Coast unassigned tracks
        for idx in unassigned_tracks:
            if idx < len(active_tracks):
                self.track_manager.coast(active_tracks[idx])

        # Initiate new tracks for unassigned measurements
        new_tracks = []
        for idx in unassigned_meas:
            meas = measurements[idx]
            track = self.track_manager.initiate(meas)
            new_tracks.append(track)

        # Prune deleted tracks
        deleted = self.track_manager.prune()

        return (
            self.track_manager.get_active(),
            new_tracks,
            [],  # deleted tracks (already pruned)
        )

    @property
    def tracks(self) -> List[Track]:
        return self.track_manager.get_active()

    @property
    def confirmed_tracks(self) -> List[Track]:
        return self.track_manager.get_confirmed()
