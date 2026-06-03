"""Track data structure and TrackManager for multi-target tracking.

Manages the lifecycle: Tentative → Confirmed → Coasting → Deleted.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import numpy as np


class TrackStatus(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"
    DELETED = "deleted"


@dataclass
class Track:
    """A single track with state, history, and status."""

    track_id: int
    status: TrackStatus = TrackStatus.TENTATIVE
    filter: Optional[object] = None       # KalmanFilter (or other)
    meas_history: List[np.ndarray] = field(default_factory=list) # [(range, angle, velocity), ...]
    history: List[np.ndarray] = field(default_factory=list)      # [(x, y), ...]
    hits: int = 0
    misses: int = 0
    age: int = 0
    total_hits: int = 0                   # Since creation


class TrackManager:
    """Manages the lifecycle of multiple tracks.

    Lifecycle:
        New measurement → Tentative track
        N hits in M frames → Confirmed
        No measurement → Coasting
        K consecutive misses → Deleted
    """

    def __init__(
        self,
        confirm_threshold: int = 3,
        delete_threshold: int = 5,
        tracker_factory: Optional[Callable[[], object]] = None,
    ):
        """
        Args:
            confirm_threshold: Number of hits to confirm a track.
            delete_threshold: Consecutive misses before deletion.
            tracker_factory: Callable that returns a new KalmanFilter.
        """
        self.confirm_threshold = confirm_threshold
        self.delete_threshold = delete_threshold
        self.tracker_factory = tracker_factory

        self.tracks: Dict[int, Track] = {}
        self._next_id = 0

    def initiate(self, measurement: np.ndarray, velocity: Optional[np.ndarray] = None) -> Track:
        """Create a new tentative track from an unassociated measurement.

        Args:
            measurement: (2,) or (4,) array [x, y] or [range, angle, vx, vy, ax, ay].

        Returns:
            The new Track.
        """
        track = Track(track_id=self._next_id)
        self._next_id += 1

        if self.tracker_factory is not None:
            kf = self.tracker_factory()
            if hasattr(kf, 'init'):
                kf.init(measurement, velocity)
            track.filter = kf
        # Add measurement to history
        track.meas_history.append(measurement.copy()) 
        cur_state = track.filter.state.copy() # (x, y, vx, vy, ax, ay)
        track.history.append(cur_state[0:2])
        
        self.tracks[track.track_id] = track
        return track

    def predict_all(self):
        """Run predict step for all active tracks."""
        for track in self.tracks.values():
            if track.status != TrackStatus.DELETED and track.filter is not None:
                track.filter.predict()

    def update(self, track: Track, measurement: np.ndarray):
        """Update track with associated measurement."""
        if track.filter is not None:
            track.filter.update(measurement)
            
        # Add measurement to history
        track.meas_history.append(measurement.copy()) 
        cur_state = track.filter.state.copy() # (x, y, vx, vy, ax, ay)
        track.history.append(cur_state[0:2])

        track.hits += 1
        track.total_hits += 1
        track.misses = 0
        track.age += 1

        if track.status == TrackStatus.TENTATIVE and track.hits >= self.confirm_threshold:
            track.status = TrackStatus.CONFIRMED
        elif track.status == TrackStatus.COASTING:
            track.status = TrackStatus.CONFIRMED

    def coast(self, track: Track):
        """Mark track as coasting (no measurement this frame)."""
        track.misses += 1
        track.age += 1
        if track.status == TrackStatus.CONFIRMED:
            track.status = TrackStatus.COASTING
        elif track.status == TrackStatus.COASTING and track.misses >= self.delete_threshold:
            track.status = TrackStatus.DELETED
        elif track.status == TrackStatus.TENTATIVE and track.misses >= 2:
            track.status = TrackStatus.DELETED

    def prune(self) -> List[int]:
        """Remove deleted tracks. Returns list of removed IDs."""
        deleted = [tid for tid, t in self.tracks.items()
                   if t.status == TrackStatus.DELETED]
        for tid in deleted:
            del self.tracks[tid]
        return deleted

    def get_active(self) -> List[Track]:
        """Get all non-deleted tracks."""
        return [t for t in self.tracks.values()
                if t.status != TrackStatus.DELETED]

    def get_confirmed(self) -> List[Track]:
        """Get only confirmed tracks."""
        return [t for t in self.tracks.values()
                if t.status == TrackStatus.CONFIRMED]

    def get_positions(self) -> List[np.ndarray]:
        """Get predicted positions [x, y] for all active tracks."""
        positions = []
        for t in self.get_active():
            if t.filter is not None:
                positions.append(t.filter.position.copy())
            elif t.history:
                positions.append(t.history[-1][:2])
            else:
                positions.append(np.zeros(2))
        return positions
