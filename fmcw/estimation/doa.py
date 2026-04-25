"""DOA（到达方向）估计：MUSIC 和 MVDR。

两种算法都对从距离-多普勒单元提取的多快拍阵列协方差矩阵进行操作，
提供比 FFT 波束成形更高的角度分辨率，尤其适用于密集目标。

参考文献：
    - Schmidt, R. O., "Multiple Emitter Location and Signal Parameter Estimation,"
      IEEE Trans. AP, 1986. (MUSIC)
    - Capon, J., "High-Resolution Frequency-Wavenumber Spectrum Analysis,"
      Proceedings of the IEEE, 1969. (MVDR/Capon)
"""

from typing import Tuple, Optional
import numpy as np
from scipy import linalg

from fmcw.utils.constants import C, RAD2DEG
from fmcw.config.schema import RadarParams


def _find_spectrum_peaks(
    spectrum_db: np.ndarray,
    scan_angles: np.ndarray,
    angle_resolution: float,
    n_peaks: int,
) -> np.ndarray:
    """在频谱中查找局部最大值，并进行二次插值精化。

    Args:
        spectrum_db: 频谱值（dB）。
        scan_angles: 对应的角度值（度）。
        angle_resolution: 角度步长（度）。
        n_peaks: 需返回的峰值数量。

    Returns:
        精化后的峰值角度数组（度）。
    """
    peaks = []
    for i in range(1, len(spectrum_db) - 1):
        if spectrum_db[i] > spectrum_db[i - 1] and spectrum_db[i] > spectrum_db[i + 1]:
            peaks.append((i, spectrum_db[i]))

    # 按幅度降序排序，取前 n_peaks 个
    peaks.sort(key=lambda x: x[1], reverse=True)
    peak_indices = [p[0] for p in peaks[:n_peaks]]

    # 二次插值精化
    refined_angles = []
    for idx in peak_indices:
        if 0 < idx < len(scan_angles) - 1:
            y0, y1, y2 = spectrum_db[idx - 1], spectrum_db[idx], spectrum_db[idx + 1]
            delta = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-30)
            refined_angle = scan_angles[idx] + delta * angle_resolution
        else:
            refined_angle = scan_angles[idx]
        refined_angles.append(refined_angle)

    return np.array(sorted(refined_angles))


class MUSIC:
    """MUSIC (MUltiple SIgnal Classification) DOA estimator.

    Uses eigendecomposition of the array covariance matrix to separate the
    signal and noise subspaces. The pseudospectrum is peaked where the steering
    vector is orthogonal to the noise subspace.

    Usage:
        music = MUSIC(radar, n_sources=2)
        spectrum = music.spectrum(X)              # full spectrum
        angles = music.estimate(X)                 # peak angles only
    """

    def __init__(
        self,
        radar: RadarParams,
        n_sources: int = 1,
        angle_resolution: float = 0.1,
        scan_range_deg: Tuple[float, float] = (-90.0, 90.0),
    ):
        """
        Args:
            radar: Radar system parameters.
            n_sources: Number of signal sources (targets) to resolve.
            angle_resolution: Angular grid spacing in degrees.
            scan_range_deg: (min_angle, max_angle) scanning range.
        """
        self.radar = radar
        self.n_sources = n_sources
        self.angle_resolution = angle_resolution
        self.scan_range = scan_range_deg

        # Pre-compute the angle scanning grid
        self._scan_angles = np.arange(scan_range_deg[0], scan_range_deg[1], angle_resolution)
        self._scan_rad = np.deg2rad(self._scan_angles)

    def _steering_vector(self, theta_rad: np.ndarray) -> np.ndarray:
        """Compute steering vectors for angles in radians.

        Args:
            theta_rad: Angle(s) in radians, scalar or 1D array.

        Returns:
            Steering vectors of shape (n_angles, n_antennas) or (n_antennas,).
        """
        n_ant = self.radar.antenna_num
        d = self.radar.antenna_spacing  # 元间距
        wl = self.radar.wl

        theta_arr = np.atleast_1d(np.asarray(theta_rad, dtype=np.float64))
        n = np.arange(n_ant)
        # a(theta) = exp(-j * 2*pi * d/lambda * sin(theta) * n)
        phase = -2.0 * np.pi * (d / wl) * np.sin(theta_arr[:, np.newaxis]) * n[np.newaxis, :]
        steering = np.exp(1j * phase)

        if np.ndim(theta_rad) == 0:
            return steering[0]
        return steering

    def _array_covariance(self, X: np.ndarray) -> np.ndarray:
        """Compute the sample array covariance matrix.

        Args:
            X: Snapshots matrix (n_antennas, n_snapshots) or (n_antennas,).

        Returns:
            Covariance matrix (n_antennas, n_antennas).
        """
        X = np.atleast_2d(X)
        if X.shape[0] != self.radar.antenna_num:
            X = X.T
        # Ensure we have the right shape
        if X.ndim == 1:
            X = X[:, np.newaxis]
        n_snapshots = X.shape[1]
        return X @ X.conj().T / n_snapshots

    def spectrum(self, X: np.ndarray) -> np.ndarray:
        """Compute the MUSIC pseudospectrum over the scanning grid.

        Args:
            X: Array snapshots (n_antennas, n_snapshots) from a range-Doppler cell.

        Returns:
            Pseudospectrum array (len(scan_angles),) in dB.
        """
        R = self._array_covariance(X)
        n_ant = R.shape[0]

        # Eigendecomposition
        eigvals, eigvecs = linalg.eigh(R)
        # Sort eigvenvalues in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]

        # Noise subspace: eigenvectors corresponding to the smallest (n_ant - n_sources) eigenvalues
        noise_subspace = eigvecs[:, self.n_sources:]

        # Compute pseudospectrum over scanning grid
        A = self._steering_vector(self._scan_rad)  # (n_angles, n_ant)
        # P(theta) = 1 / (a(theta)^H * En * En^H * a(theta))
        # where En is the noise subspace
        projection = A @ noise_subspace  # (n_angles, n_ant - n_sources)
        denominator = np.sum(np.abs(projection) ** 2, axis=1)

        pseudospectrum = 10.0 * np.log10(1.0 / (denominator + 1e-30))
        return pseudospectrum

    def estimate(self, X: np.ndarray, n_peaks: Optional[int] = None) -> np.ndarray:
        """通过 MUSIC 频谱峰值估计 DOA 角度。

        Args:
            X: 阵列快拍（n_antennas, n_snapshots）。
            n_peaks: 需返回的峰值数量（默认: n_sources）。

        Returns:
            估计角度数组（度）。
        """
        if n_peaks is None:
            n_peaks = self.n_sources

        spectrum_db = self.spectrum(X)
        return _find_spectrum_peaks(spectrum_db, self._scan_angles, self.angle_resolution, n_peaks)


class MVDR:
    """MVDR (Minimum Variance Distortionless Response) / Capon beamformer.

    Also known as the Capon beamformer. Minimizes output power while maintaining
    unity gain in the look direction, resulting in narrower beams and better
    interference rejection compared to FFT beamforming.

    Usage:
        mvdr = MVDR(radar)
        spectrum = mvdr.spectrum(X)
        angles = mvdr.estimate(X)
    """

    def __init__(
        self,
        radar: RadarParams,
        angle_resolution: float = 0.1,
        scan_range_deg: Tuple[float, float] = (-90.0, 90.0),
        diagonal_loading: float = 0.0,
    ):
        """
        Args:
            radar: Radar system parameters.
            angle_resolution: Angular grid spacing in degrees.
            scan_range_deg: (min_angle, max_angle) scanning range.
            diagonal_loading: Diagonal loading factor (fraction of trace/N)
                              to improve numerical stability. 0 = none.
        """
        self.radar = radar
        self.angle_resolution = angle_resolution
        self.scan_range = scan_range_deg
        self.diagonal_loading = diagonal_loading

        self._scan_angles = np.arange(scan_range_deg[0], scan_range_deg[1], angle_resolution)
        self._scan_rad = np.deg2rad(self._scan_angles)

    def _steering_vector(self, theta_rad: np.ndarray) -> np.ndarray:
        """与 MUSIC 相同的导向向量约定。"""
        n_ant = self.radar.antenna_num
        d = self.radar.antenna_spacing  # 元间距
        wl = self.radar.wl

        theta_arr = np.atleast_1d(np.asarray(theta_rad, dtype=np.float64))
        n = np.arange(n_ant)
        phase = -2.0 * np.pi * (d / wl) * np.sin(theta_arr[:, np.newaxis]) * n[np.newaxis, :]
        steering = np.exp(1j * phase)

        if np.ndim(theta_rad) == 0:
            return steering[0]
        return steering

    def _array_covariance(self, X: np.ndarray) -> np.ndarray:
        """Compute array covariance matrix with optional diagonal loading."""
        X = np.atleast_2d(X)
        if X.shape[0] != self.radar.antenna_num:
            X = X.T
        if X.ndim == 1:
            X = X[:, np.newaxis]
        n_snapshots = X.shape[1]
        R = X @ X.conj().T / n_snapshots

        if self.diagonal_loading > 0.0:
            loading = self.diagonal_loading * np.trace(R) / R.shape[0]
            R += loading * np.eye(R.shape[0], dtype=R.dtype)

        return R

    def spectrum(self, X: np.ndarray) -> np.ndarray:
        """Compute the MVDR/Capon spectrum.

        P(theta) = 1 / (a(theta)^H * R^{-1} * a(theta))

        Args:
            X: Array snapshots (n_antennas, n_snapshots).

        Returns:
            MVDR spectrum array in dB.
        """
        R = self._array_covariance(X)
        R_inv = linalg.pinvh(R)  # pseudo-inverse for numerical stability

        A = self._steering_vector(self._scan_rad)  # (n_angles, n_ant)

        # Capon spectrum: 1 / (a^H * R^{-1} * a)
        # For each angle: a_i^H @ R_inv @ a_i
        power = np.einsum("ij,jk,ik->i", A.conj(), R_inv, A, optimize=True)
        spectrum = 10.0 * np.log10(1.0 / np.real(power + 1e-30))

        return spectrum

    def estimate(self, X: np.ndarray, n_peaks: int = 1) -> np.ndarray:
        """通过 MVDR 频谱峰值估计 DOA 角度。

        Args:
            X: 阵列快拍（n_antennas, n_snapshots）。
            n_peaks: 需查找的峰值数量。

        Returns:
            估计角度数组（度）。
        """
        spectrum_db = self.spectrum(X)
        return _find_spectrum_peaks(spectrum_db, self._scan_angles, self.angle_resolution, n_peaks)
