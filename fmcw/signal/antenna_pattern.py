"""Antenna array patterns: ULA and URA."""

import numpy as np


class UniformLinearArray:
    """Uniform Linear Array (ULA) with configurable element count.

    Elements spaced at lambda/2 by default.
    """

    def __init__(self, n_elements: int, d_over_lambda: float = 0.5):
        self.n_elements = n_elements
        self.d_over_lambda = d_over_lambda

    def steering_vector(self, angle_deg: float) -> np.ndarray:
        """Complex steering vector for a given azimuth angle.

        a(theta) = exp(-j * 2*pi * d/lambda * sin(theta) * [0, 1, ..., N-1])

        Args:
            angle_deg: Azimuth angle in degrees (0 = boresight).

        Returns:
            Complex array of shape (n_elements,).
        """
        theta = np.deg2rad(angle_deg)
        n = np.arange(self.n_elements)
        phase = -2.0 * np.pi * self.d_over_lambda * np.sin(theta) * n
        return np.exp(1j * phase)

    def pattern(self, angles_deg: np.ndarray) -> np.ndarray:
        """Array factor magnitude for a set of angles.

        Returns:
            Real array of shape (len(angles_deg),), normalized to max=1.
        """
        theta = np.deg2rad(np.asarray(angles_deg))
        psi = np.pi * np.sin(theta)  # d = lambda/2
        # sin(N*psi/2) / sin(psi/2)
        n = self.n_elements
        with np.errstate(divide="ignore", invalid="ignore"):
            af = np.sin(n * psi / 2.0) / (n * np.sin(psi / 2.0 + 1e-15))
            af = np.where(np.abs(np.sin(psi / 2.0)) < 1e-10, 1.0, af)
        return np.abs(af)

    def element_gain(self, angle_deg: float) -> float:
        """Simple cosine element factor (approximate patch antenna pattern)."""
        theta = np.deg2rad(angle_deg)
        return max(np.cos(theta), 0.01)


class UniformRectangularArray:
    """Uniform Rectangular Array (URA) — for future 4D radar elevation support."""

    def __init__(
        self,
        n_azimuth: int,
        n_elevation: int,
        d_over_lambda: float = 0.5,
    ):
        self.n_az = n_azimuth
        self.n_el = n_elevation
        self.d_over_lambda = d_over_lambda
        self.n_elements = n_azimuth * n_elevation

    def steering_vector(self, az_deg: float, el_deg: float) -> np.ndarray:
        """2D steering vector."""
        az_rad = np.deg2rad(az_deg)
        el_rad = np.deg2rad(el_deg)
        m = np.arange(self.n_az)
        n = np.arange(self.n_el)
        mm, nn = np.meshgrid(m, n)
        phase = -2j * np.pi * self.d_over_lambda * (
            np.sin(az_rad) * np.cos(el_rad) * mm.flatten()
            + np.sin(el_rad) * nn.flatten()
        )
        return np.exp(phase)
