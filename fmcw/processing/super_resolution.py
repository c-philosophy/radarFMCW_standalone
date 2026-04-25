"""Super-resolution angle estimation: ESPRIT and MUSIC wrappers.

Provides enhanced angular resolution beyond the FFT-based beamforming limit.
These algorithms exploit the structure of the array covariance matrix to
resolve targets within the same FFT beamwidth.

Algorithms:
    - TLS-ESPRIT: Total Least Squares ESPRIT using shift invariance
    - Beamspace MUSIC: convenience wrapper for per-range-Doppler cell use

References:
    - Roy, R. and Kailath, T., "ESPRIT - Estimation of Signal Parameters
      via Rotational Invariance Techniques," IEEE Trans. ASSP, 1989.
    - Stoica, P. and Moses, R., "Spectral Analysis of Signals," 2005.
"""

from typing import Optional, Tuple
import numpy as np
from scipy import linalg

from fmcw.config.schema import RadarParams
from fmcw.utils.constants import RAD2DEG


class TLSESPRIT:
    """TLS-ESPRIT (Total Least Squares ESPRIT) angle estimator.

    Exploits the rotational invariance of a uniform linear array:
    the steering vectors for sub-arrays shifted by one element are related
    by a phase rotation exp(-j * 2*pi * d/lambda * sin(theta)).

    Does not require a scanning grid — angles are computed analytically from
    eigenvalues of the shift-invariance equation, making it computationally
    efficient for real-time applications.

    Usage:
        esprit = TLSESPRIT(radar, n_sources=2)
        angles = esprit.estimate(X)

    Notes:
        - Requires at least n_sources + 1 antennas.
        - Works with a single snapshot IF the effective rank can be estimated.
    """

    def __init__(
        self,
        radar: RadarParams,
        n_sources: int = 1,
    ):
        """
        Args:
            radar: Radar system parameters.
            n_sources: Number of signal sources (targets).
        """
        self.radar = radar
        self.n_sources = n_sources

    def estimate(self, X: np.ndarray) -> np.ndarray:
        """Estimate DOA angles using TLS-ESPRIT.

        Args:
            X: Array snapshots (n_antennas, n_snapshots).

        Returns:
            Array of estimated angles in degrees, sorted ascending.
        """
        X = np.atleast_2d(X)
        if X.shape[0] != self.radar.antenna_num:
            X = X.T

        n_ant, n_snap = X.shape

        if n_ant < self.n_sources + 1:
            raise ValueError(
                f"Need at least n_sources + 1 antennas ({self.n_sources + 1}), "
                f"got {n_ant}"
            )

        # Step 1: Compute array covariance matrix
        R = X @ X.conj().T / n_snap

        # Step 2: Eigendecomposition
        eigvals, eigvecs = linalg.eigh(R)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]

        # Signal subspace: eigenvectors corresponding to the n_sources largest eigenvalues
        Es = eigvecs[:, :self.n_sources]  # (n_ant, n_sources)

        # Step 3: Split into two overlapping sub-arrays
        # Sub-array 1: first n_ant - 1 elements
        # Sub-array 2: last n_ant - 1 elements
        E1 = Es[:-1, :]  # (n_ant-1, n_sources)
        E2 = Es[1:, :]   # (n_ant-1, n_sources)

        # Step 4: Solve the shift invariance equation E1 * Psi = E2
        # Total Least Squares solution:
        #   [E1, E2] = U @ Sigma @ V^H
        #   V is partitioned into (n_sources x n_sources) blocks
        #   Psi = -V12 @ inv(V22)
        C = np.hstack([E1, E2])  # (n_ant-1, 2*n_sources)
        _, _, Vh = linalg.svd(C, full_matrices=False)
        V = Vh.conj().T  # (2*n_sources, 2*n_sources)

        V12 = V[:self.n_sources, self.n_sources:]
        V22 = V[self.n_sources:, self.n_sources:]

        # Psi = -V12 @ V22^{-1}
        Psi = -V12 @ linalg.inv(V22)

        # Step 5: Eigenvalues of Psi contain the angle information
        phi = linalg.eigvals(Psi)

        # Step 6: Convert eigenvalues to angles
        # phi_k = exp(-j * 2*pi * d/lambda * sin(theta_k))
        # theta_k = arcsin(-angle(phi_k) * lambda / (2*pi*d))
        angles_rad = np.arcsin(
            np.clip(
                -np.angle(phi) * self.radar.wl / (2.0 * np.pi * self.radar.antenna_spacing),
                -1.0, 1.0,
            )
        )

        angles = np.degrees(angles_rad)
        return np.sort(angles)


class BeamspaceMUSIC:
    """MUSIC applied per range-Doppler cell in the beam space.

    Convenience wrapper that extracts snapshots from the range-Doppler-angle
    data cube and applies MUSIC for high-resolution angle estimation at
    specific range-Doppler coordinates.

    Usage:
        bmusic = BeamspaceMUSIC(radar, n_sources=2)
        angles = bmusic.estimate_at_cell(s_rda, doppler_idx, range_idx)
    """

    def __init__(
        self,
        radar: RadarParams,
        n_sources: int = 1,
        angle_resolution: float = 0.1,
    ):
        """
        Args:
            radar: Radar system parameters.
            n_sources: Number of signal sources.
            angle_resolution: Angular grid resolution in degrees.
        """
        self.radar = radar
        self.n_sources = n_sources

        # Internal MUSIC instance
        from .doa import MUSIC as _MUSIC
        self._music = _MUSIC(
            radar=radar,
            n_sources=n_sources,
            angle_resolution=angle_resolution,
        )

    def estimate_at_cell(
        self,
        s_rda: np.ndarray,
        doppler_idx: int,
        range_idx: int,
    ) -> np.ndarray:
        """Estimate angles at a specific range-Doppler cell.

        The snapshot across antennas at (doppler_idx, range_idx) is used
        as the array observation. For a single snapshot, a spatial smoothing
        or averaging over adjacent cells can be applied.

        Args:
            s_rda: Angle-Doppler-Range complex cube (n_angle, n_doppler, n_range).
            doppler_idx: Doppler FFT bin index.
            range_idx: Range FFT bin index.

        Returns:
            Estimated angles in degrees.
        """
        # Extract the antenna snapshot at the given RD cell
        # s_rda shape is (angle, doppler, range) = (antenna, chirp, sample_half)
        snapshot = s_rda[:, doppler_idx, range_idx]  # (n_antennas,)

        # Use adjacent cells as additional snapshots for better covariance estimation
        n_ant = self.radar.antenna_num
        snapshots = [snapshot]

        # Add neighbors within 1 bin for more snapshots
        for dd in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                if dd == 0 and dr == 0:
                    continue
                d_idx = doppler_idx + dd
                r_idx = range_idx + dr
                if 0 <= d_idx < s_rda.shape[1] and 0 <= r_idx < s_rda.shape[2]:
                    snapshots.append(s_rda[:, d_idx, r_idx])

        X = np.column_stack(snapshots)  # (n_ant, n_snapshots)
        return self._music.estimate(X, n_peaks=self.n_sources)
