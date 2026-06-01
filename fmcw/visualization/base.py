"""Abstract base for visualization backends."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseVisualizer(ABC):
    """Unified interface for real-time and playback visualization.

    Subclasses implement `setup`, `update`, `run`, `stop` for specific backends.
    """

    def __init__(self, panels: List[str] = None):
        self.panels = panels or ["rd_map", "ra_map", "trajectory", "diagnostic"]

    @abstractmethod
    def setup(self, radar_params, scene_config=None):
        """Initialize figure/window and axes."""

    @abstractmethod
    def update(self, frame_data: Dict[str, Any]):
        """Render one frame of data.

        Expected keys in frame_data:
            'frame_idx': int
            'rd_map': ndarray (D, R)
            'ra_map': ndarray (A, R)
            'detections': ndarray (N, 2) [d_idx, r_idx]
            'estimates': FrameEstimates
            'tracks': List[Track]
            'ground_truth': List[TargetParams] (optional)
        """

    @abstractmethod
    def run(self):
        """Start the visualization loop (blocking for real-time backends)."""

    @abstractmethod
    def stop(self):
        """Stop / close the visualization."""

    def refresh(self):
        """Process pending GUI events (for real-time backends).

        Called after each frame update to allow the GUI to render.
        Default is no-op; real-time backends override to process events.
        """

    def save_frame(self, path: str):
        """Save current frame to file (for matplotlib backend)."""
        raise NotImplementedError
