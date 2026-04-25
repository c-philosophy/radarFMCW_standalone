"""Coordinate transforms and unit conversions for radar data."""

import numpy as np
from .constants import DEG2RAD, RAD2DEG


def polar_to_cartesian(range_m: np.ndarray, angle_deg: np.ndarray) -> tuple:
    """Convert polar (range, angle) to Cartesian (x, y).

    Args:
        range_m: Range in meters, shape (N,) or scalar.
        angle_deg: Azimuth angle in degrees, shape (N,) or scalar.

    Returns:
        (x, y) tuple, each same shape as input.
    """
    angle_rad = np.asarray(angle_deg) * DEG2RAD
    x = np.asarray(range_m) * np.cos(angle_rad)
    y = np.asarray(range_m) * np.sin(angle_rad)
    return x, y


def cartesian_to_polar(x: np.ndarray, y: np.ndarray) -> tuple:
    """Convert Cartesian (x, y) to polar (range, angle).

    Returns:
        (range_m, angle_deg) tuple.
    """
    r = np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2)
    a = np.arctan2(np.asarray(y), np.asarray(x)) * RAD2DEG
    return r, a


def db(x: np.ndarray) -> np.ndarray:
    """Convert linear amplitude to dB (20*log10)."""
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def linear_from_db(db_val: float) -> float:
    """Convert dB to linear amplitude."""
    return 10.0 ** (db_val / 20.0)


def power_db(x: np.ndarray) -> np.ndarray:
    """Convert linear power to dB (10*log10)."""
    return 10.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def wrap_angle_deg(angle: float) -> float:
    """Wrap angle to [-180, 180] degrees."""
    return (angle + 180.0) % 360.0 - 180.0
