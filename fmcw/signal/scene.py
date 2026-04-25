"""Multi-frame scene generation: target trajectories and frame-by-frame targets."""

from typing import List, Optional
import numpy as np

from fmcw.config.schema import TargetSpec, TargetParams, SceneConfig, RadarParams
from fmcw.signal.generator import IFSignalGenerator
from fmcw.signal.target_fluctuation import TargetFluctuation


class Scene:
    """Generates multi-frame radar scenarios from a SceneConfig.

    Handles target motion (CV, CA models) and produces a list of
    TargetParams per frame, plus the IF signals.

    Usage:
        scene = Scene(radar_params, scene_config)
        signals = scene.generate_signals()
        # Or iterate frame-by-frame:
        for frame_idx, (signal, targets) in enumerate(scene.stream()):
            process(signal, targets)
    """

    def __init__(self, radar: RadarParams, scene_config: SceneConfig):
        self.radar = radar
        self.config = scene_config
        self._generator = IFSignalGenerator(radar)
        self._fluctuation = TargetFluctuation(
            model=scene_config.swerling_model,
        )

    @property
    def num_frames(self) -> int:
        return self.config.num_frames

    @property
    def dt(self) -> float:
        return self.config.frame_interval

    def target_trajectory(self, spec: TargetSpec) -> List[TargetParams]:
        """Compute the full trajectory for a single target spec.

        Motion models:
            constant_velocity: range += v * t, velocity constant
            constant_acceleration: velocity += a * t
            stationary: no change
        Angle changes with angular_velocity (deg/s) for all motion types.
        """
        motion = spec.motion
        result = []
        for frame in range(self.config.num_frames):
            t = frame * self.config.frame_interval
            if motion.motion == "constant_velocity":
                r = spec.initial_range + spec.initial_velocity * t
                v = spec.initial_velocity
            elif motion.motion == "constant_acceleration":
                a = motion.acceleration
                v = spec.initial_velocity + a * t
                r = spec.initial_range + spec.initial_velocity * t + 0.5 * a * t**2
            else:  # stationary
                r = spec.initial_range
                v = 0.0

            # Update angle: angle changes at constant angular velocity
            angle = spec.initial_angle + motion.angular_velocity * t

            result.append(
                TargetParams(
                    range=r,
                    velocity=v,
                    angle=angle,
                    rcs=spec.rcs,
                )
            )
        return result

    def all_trajectories(self) -> List[List[TargetParams]]:
        """Compute trajectories for all targets.

        Returns:
            List of shape (num_frames, num_targets) of TargetParams.
        """
        traj_list = [
            self.target_trajectory(spec)
            for spec in self.config.targets
        ]
        # Transpose: (num_targets, num_frames) → (num_frames, num_targets)
        return [
            [traj_list[t][f] for t in range(len(self.config.targets))]
            for f in range(self.num_frames)
        ]

    def generate_signals(self) -> List[np.ndarray]:
        """Generate IF signals for all frames.

        Returns:
            List of signal arrays, one per frame.
        """
        targets_per_frame = self.all_trajectories()
        return self._generator.generate_frames(
            targets_list=targets_per_frame,
            dt=self.dt,
            snr_db=self.config.snr_db,
            fluctuation_model=self._fluctuation,
        )

    def stream(self):
        """Generator yielding (frame_idx, signal, targets) for each frame.

        Yields:
            Tuple of (frame_idx, signal_array, List[TargetParams]).
        """
        targets_per_frame = self.all_trajectories()
        for frame_idx, targets in enumerate(targets_per_frame):
            fluc = self._fluctuation.generate(len(targets))
            signal = self._generator.generate_frame(
                targets=targets,
                frame_idx=frame_idx,
                dt=self.dt,
                snr_db=self.config.snr_db,
                fluctuation_factors=fluc,
            )
            yield frame_idx, signal, targets
