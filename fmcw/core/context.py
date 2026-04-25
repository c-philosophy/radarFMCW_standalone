"""PipelineContext: shared state container passed between pipeline stages."""

from typing import Any, Dict, Optional
import numpy as np


class PipelineContext:
    """Mutable key-value store shared across all pipeline stages.

    Stages read from and write to this context via dict-like access.
    Common keys include:
        'signal' — raw IF signal ndarray
        's_r', 's_rd', 's_rda' — FFT results
        'rd_map', 'ra_map' — magnitude maps
        'candidate_peaks', 'filtered_peaks' — detection results
        'estimates' — FrameEstimates
        'tracks' — List[Track]
    """

    def __init__(self, **kwargs):
        self._data: Dict[str, Any] = dict(kwargs)
        self._frame_idx: int = 0

    # --- dict-like access ---

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any):
        self._data[key] = value
        return self  # chainable

    # --- frame tracking ---

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    def advance_frame(self):
        self._frame_idx += 1

    # --- convenience ---

    @property
    def signal(self) -> Optional[np.ndarray]:
        return self._data.get("signal")

    @property
    def rd_map(self) -> Optional[np.ndarray]:
        return self._data.get("rd_map")

    @property
    def ra_map(self) -> Optional[np.ndarray]:
        return self._data.get("ra_map")

    @property
    def tracks(self) -> list:
        return self._data.get("tracks", [])

    def __repr__(self) -> str:
        keys = list(self._data.keys())
        return f"PipelineContext(frame={self._frame_idx}, keys={keys})"
