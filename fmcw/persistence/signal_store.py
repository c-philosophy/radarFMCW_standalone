"""Raw signal storage: HDF5 and NPZ backends."""

from pathlib import Path
from typing import List, Optional
import numpy as np


class NPZSignalStore:
    """Store/reload raw IF signals using numpy .npz format.

    Usage:
        store = NPZSignalStore("data/run_001/")
        store.save_frame(signal, frame_idx=0, radar=radar, targets=targets)
        data = store.load_frame(0)
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_frame(
        self,
        signal: np.ndarray,
        frame_idx: int,
        radar_params=None,
        targets=None,
    ):
        """Save one frame of IF signal with metadata."""
        path = self.output_dir / f"signal_frame_{frame_idx:04d}.npz"
        save_dict = {"signal": signal.astype(np.float32)}
        if radar_params is not None:
            save_dict["fc"] = radar_params.fc
            save_dict["B"] = radar_params.B
            save_dict["Tc"] = radar_params.Tc
            save_dict["chirp_num"] = radar_params.chirp_num
            save_dict["sample_num"] = radar_params.sample_num
            save_dict["antenna_num"] = radar_params.antenna_num
        np.savez_compressed(path, **save_dict)

    def load_frame(self, frame_idx: int) -> dict:
        """Load one frame. Returns dict with 'signal' and metadata keys."""
        path = self.output_dir / f"signal_frame_{frame_idx:04d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Signal file not found: {path}")
        data = np.load(path)
        result = {k: data[k] for k in data.files}
        data.close()
        return result

    def save_all_frames(
        self,
        signals: List[np.ndarray],
        radar_params=None,
        targets=None,
    ):
        """Save all frames to a single NPZ file."""
        path = self.output_dir / "signals_all.npz"
        save_dict = {}
        for i, sig in enumerate(signals):
            save_dict[f"frame_{i:04d}"] = sig.astype(np.float32)
        if radar_params is not None:
            save_dict["fc"] = radar_params.fc
        np.savez_compressed(path, **save_dict)
