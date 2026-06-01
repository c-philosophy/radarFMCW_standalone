"""Target fluctuation models: Swerling 0-4."""

from enum import IntEnum
from typing import Optional
import numpy as np


class SwerlingModel(IntEnum):
    SWERLING_0 = 0  # Non-fluctuating
    SWERLING_1 = 1  # Slow fluctuation (scan-to-scan, chi-square 2 DOF)
    SWERLING_2 = 2  # Fast fluctuation (pulse-to-pulse, chi-square 2 DOF)
    SWERLING_3 = 3  # Slow fluctuation (scan-to-scan, chi-square 4 DOF)
    SWERLING_4 = 4  # Fast fluctuation (pulse-to-pulse, chi-square 4 DOF)


class TargetFluctuation:
    """Generate fluctuating RCS values per target per frame.

    Swerling 0: constant RCS (returns 1.0 always). 
    
    Swerling 1/2: chi-square with 2 DOF → exponential distribution. 
    
    Swerling 3/4: chi-square with 4 DOF → gamma distribution (k=2).

    The mean RCS is normalized to the target's rcs parameter.
    """

    def __init__(
        self,
        model: int = 0,
        seed: Optional[int] = None,
    ):
        self.model = SwerlingModel(model)
        self._rng = np.random.default_rng(seed)

    def generate(self, num_targets: int, num_pulses: int = 1) -> np.ndarray:
        """Generate fluctuation factors.

        Args:
            num_targets: Number of targets.
            num_pulses: Number of pulses/chirps (only matters for Swerling 2/4).

        Returns:
            Array of shape (num_targets,) for Swerling 0/1/3,
            or (num_targets, num_pulses) for Swerling 2/4.
            Values centered around 1.0.
        """
        if self.model == SwerlingModel.SWERLING_0:
            return np.ones(num_targets)

        elif self.model == SwerlingModel.SWERLING_1:
            # Slow: same value for all pulses in a frame
            # Exponential(1) → chi-square 2 DOF mean=2 → divide by 2
            return self._rng.exponential(1.0, size=num_targets) / 2.0

        elif self.model == SwerlingModel.SWERLING_2:
            # Fast: independent per pulse
            return self._rng.exponential(1.0, size=(num_targets, num_pulses)) / 2.0

        elif self.model == SwerlingModel.SWERLING_3:
            # Slow, chi-square 4 DOF = sum of 2 exponentials / 2
            e1 = self._rng.exponential(1.0, size=num_targets)
            e2 = self._rng.exponential(1.0, size=num_targets)
            return (e1 + e2) / 4.0

        elif self.model == SwerlingModel.SWERLING_4:
            # Fast, chi-square 4 DOF, independent per pulse
            e1 = self._rng.exponential(1.0, size=(num_targets, num_pulses))
            e2 = self._rng.exponential(1.0, size=(num_targets, num_pulses))
            return (e1 + e2) / 4.0

        else:
            raise ValueError(f"Unknown Swerling model: {self.model}")
