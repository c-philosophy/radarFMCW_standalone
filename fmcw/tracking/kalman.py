"""Pure NumPy Kalman filter implementations: KF, EKF, UKF.

Zero external dependencies — all matrix operations using NumPy.

Registered in AlgorithmRegistry under category "tracker": kf, ekf, ukf.
"""

from typing import Optional, Callable, Tuple
import numpy as np
from scipy.linalg import cholesky

from fmcw.core.registry import register_algorithm


class KalmanFilter:
    """Linear Kalman Filter with constant-velocity model.

    State: [x, y, vx, vy]
    Observation: [x, y]

    Usage:
        kf = KalmanFilter(dt=0.1)
        kf.init(measurement)
        for z in measurements:
            kf.predict()
            kf.update(z)
    """

    def __init__(
        self,
        dt: float = 0.1,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
    ):
        self.dt = dt
        dim_x = 4
        dim_z = 2

        # State transition: constant velocity
        self.F = np.eye(dim_x)
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        # Observation: direct position
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # Covariance matrices
        self.Q = np.eye(dim_x) * process_noise
        self.R = np.eye(dim_z) * measurement_noise**2

        # State
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x) * 100.0

    def init(self, z: np.ndarray):
        """Initialize filter with first measurement."""
        self.x[0] = z[0]
        self.x[1] = z[1]
        self.x[2] = 0.0
        self.x[3] = 0.0

    def predict(self) -> np.ndarray:
        """Predict step: x = F @ x, P = F @ P @ F.T + Q."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z: np.ndarray) -> np.ndarray:
        """Update step with measurement z."""
        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + self.R      # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)      # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
        return self.x

    @property
    def state(self) -> np.ndarray:
        return self.x

    @property
    def position(self) -> np.ndarray:
        return self.x[:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:]


class ExtendedKalmanFilter:
    """Extended Kalman Filter with nonlinear observation model.

    Uses [range, azimuth] as observation (typical for radar).
    State: [x, y, vx, vy] in Cartesian.

    The user provides h(x) and its Jacobian H_jac(x).
    """

    def __init__(
        self,
        dt: float = 0.1,
        process_noise: float = 0.01,
        measurement_noise_range: float = 0.1,
        measurement_noise_angle: float = 0.02,
    ):
        self.dt = dt
        dim_x = 4
        dim_z = 2

        self.F = np.eye(dim_x)
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        self.Q = np.eye(dim_x) * process_noise
        self.R = np.diag([measurement_noise_range**2, measurement_noise_angle**2])

        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x) * 100.0

    def init(self, z: np.ndarray):
        """Initialize from [range, angle] measurement."""
        r, th = z[0], np.deg2rad(z[1])
        self.x[0] = r * np.cos(th)
        self.x[1] = r * np.sin(th)
        self.x[2] = 0.0
        self.x[3] = 0.0

    def hx(self, x: np.ndarray) -> np.ndarray:
        """Observation function: state → [range, angle]."""
        r = np.sqrt(x[0]**2 + x[1]**2)
        th = np.arctan2(x[1], x[0])
        return np.array([r, np.rad2deg(th)])

    def H_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of hx at state x."""
        r2 = x[0]**2 + x[1]**2
        r = np.sqrt(r2)
        if r < 1e-6:
            return np.zeros((2, 4))
        return np.array([
            [x[0] / r, x[1] / r, 0.0, 0.0],
            [-x[1] / r2, x[0] / r2, 0.0, 0.0],
        ])

    def predict(self) -> np.ndarray:
        """Linear predict (CV model)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z: np.ndarray) -> np.ndarray:
        """EKF update with nonlinear h(x) and analytic Jacobian."""
        H = self.H_jacobian(self.x)
        z_pred = self.hx(self.x)
        y = z - z_pred
        # Wrap angle innovation to [-180, 180]
        while y[1] > 180:
            y[1] -= 360
        while y[1] < -180:
            y[1] += 360

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        return self.x

    @property
    def state(self) -> np.ndarray:
        return self.x

    @property
    def position(self) -> np.ndarray:
        return self.x[:2]


class UnscentedKalmanFilter:
    """Unscented Kalman Filter using Merwe scaled sigma points.

    Handles nonlinear process model f(x) and observation model h(x)
    through statistical linearization via sigma points.
    """

    def __init__(
        self,
        dt: float = 0.1,
        process_noise: float = 0.01,
        measurement_noise_range: float = 0.1,
        measurement_noise_angle: float = 0.02,
        alpha: float = 0.1,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.dt = dt
        self.n = 4                         # state dim
        self.m = 2                         # measurement dim

        self.F = np.eye(4)
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        self.Q = np.eye(self.n) * process_noise
        self.R = np.diag([measurement_noise_range**2, measurement_noise_angle**2])

        # Merwe sigma point parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lam = alpha**2 * (self.n + kappa) - self.n

        # Weights
        self.Wm = np.full(2 * self.n + 1, 0.5 / (self.n + self.lam))
        self.Wc = np.full(2 * self.n + 1, 0.5 / (self.n + self.lam))
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.lam / (self.n + self.lam) + (1 - alpha**2 + beta)

        self.x = np.zeros(self.n)
        self.P = np.eye(self.n) * 100.0

    def init(self, z: np.ndarray):
        """Initialize from [range, angle] measurement."""
        r, th = z[0], np.deg2rad(z[1])
        self.x[0] = r * np.cos(th)
        self.x[1] = r * np.sin(th)

    def sigma_points(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate 2n+1 sigma points."""
        n = self.n
        L = cholesky((n + self.lam) * self.P, lower=True)

        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = self.x
        for i in range(n):
            sigmas[i + 1] = self.x + L[:, i]
            sigmas[n + i + 1] = self.x - L[:, i]

        return sigmas, self.Wm, self.Wc

    def hx(self, x: np.ndarray) -> np.ndarray:
        """Observation: Cartesian → [range, azimuth_deg]."""
        r = np.sqrt(x[0]**2 + x[1]**2)
        th = np.arctan2(x[1], x[0])
        return np.array([r, np.rad2deg(th)])

    def predict(self) -> np.ndarray:
        """UKF predict (linear CV model)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z: np.ndarray) -> np.ndarray:
        """UKF update."""
        n = self.n
        sigmas, Wm, Wc = self.sigma_points()

        # Propagate sigma points through observation function
        Z = np.array([self.hx(s) for s in sigmas])

        # Predicted measurement mean
        z_mean = Z[0] * Wm[0]
        for i in range(1, 2 * n + 1):
            z_mean = z_mean + Wm[i] * Z[i]

        # Innovation covariance
        y = Z - z_mean
        # Angle wrapping
        y[:, 1] = (y[:, 1] + 180) % 360 - 180

        S = self.R.copy()
        for i in range(2 * n + 1):
            S = S + Wc[i] * np.outer(y[i], y[i])

        # Cross covariance
        X_diff = sigmas - self.x
        Pxz = np.zeros((n, self.m))
        for i in range(2 * n + 1):
            Pxz = Pxz + Wc[i] * np.outer(X_diff[i], y[i])

        # Kalman gain
        K = Pxz @ np.linalg.inv(S)

        # Innovation
        innovation = z - z_mean
        while innovation[1] > 180:
            innovation[1] -= 360
        while innovation[1] < -180:
            innovation[1] += 360

        self.x = self.x + K @ innovation
        self.P = self.P - K @ S @ K.T
        return self.x

    @property
    def state(self) -> np.ndarray:
        return self.x

    @property
    def position(self) -> np.ndarray:
        return self.x[:2]


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_tracker(
    tracker_type: str,
    dt: float = 0.1,
    **kwargs,
):
    """Create a tracker instance by name. Supported: 'kf', 'ekf', 'ukf'."""
    from fmcw.core.registry import AlgorithmRegistry
    cls = AlgorithmRegistry.get("tracker", tracker_type)
    return cls(dt=dt, **kwargs)
