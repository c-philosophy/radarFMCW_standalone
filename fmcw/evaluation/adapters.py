"""数据格式适配器：将现有项目数据结构转换为评估模块可用的标准格式。

所有函数为纯函数，无副作用，零依赖（仅依赖 numpy）。
"""

from typing import List
import numpy as np


def targets_to_cartesian(ground_truth: list) -> np.ndarray:
    """将 TargetParams 列表转换为笛卡尔坐标矩阵。

    Args:
        ground_truth: List[TargetParams]，每个元素含 range/velocity/angle。

    Returns:
        (N, 2) 数组，列 [x, y] (m)。
    """
    if not ground_truth:
        return np.empty((0, 2))
    result = np.empty((len(ground_truth), 2))
    for i, g in enumerate(ground_truth):
        th = np.deg2rad(g.angle)
        result[i, 0] = g.range * np.cos(th)
        result[i, 1] = g.range * np.sin(th)
    return result


def estimates_to_cartesian(estimates: list) -> np.ndarray:
    """将 TargetEstimate 列表转换为笛卡尔坐标矩阵。

    Args:
        estimates: List[TargetEstimate]。

    Returns:
        (M, 2) 数组，列 [x, y] (m)。
    """
    if not estimates:
        return np.empty((0, 2))
    result = np.empty((len(estimates), 2))
    for i, e in enumerate(estimates):
        th = np.deg2rad(e.angle)
        result[i, 0] = e.range * np.cos(th)
        result[i, 1] = e.range * np.sin(th)
    return result


def tracks_to_cartesian(tracks: list, confirmed_only: bool = True) -> np.ndarray:
    """将 Track 列表转换为笛卡尔坐标矩阵。

    Args:
        tracks: List[Track]。
        confirmed_only: True 时仅提取 CONFIRMED 航迹。

    Returns:
        (K, 2) 数组，列 [x, y] (m)。
    """
    if confirmed_only:
        tracks = [t for t in tracks
                  if hasattr(t, 'status') and t.status.value == 'confirmed']

    if not tracks:
        return np.empty((0, 2))

    result = np.empty((len(tracks), 2))
    for i, t in enumerate(tracks):
        if t.filter is not None and hasattr(t.filter, 'position'):
            pos = t.filter.position
            result[i, 0] = float(pos[0])
            result[i, 1] = float(pos[1])
        elif t.history:
            result[i, 0] = float(t.history[-1][0])
            result[i, 1] = float(t.history[-1][1])
        else:
            result[i, 0] = 0.0
            result[i, 1] = 0.0
    return result


def tracks_to_ids(tracks: list, confirmed_only: bool = True) -> List[int]:
    """从 Track 列表提取 track_id 列表。

    Args:
        tracks: List[Track]。
        confirmed_only: True 时仅提取 CONFIRMED 航迹。

    Returns:
        List[int] track_id 列表。
    """
    if confirmed_only:
        tracks = [t for t in tracks
                  if hasattr(t, 'status') and t.status.value == 'confirmed']
    return [t.track_id for t in tracks]


def ground_truth_to_array(ground_truth: list) -> np.ndarray:
    """将 TargetParams 列表转换为 (N, 3) 矩阵 [range, velocity, angle]。

    Args:
        ground_truth: List[TargetParams]。

    Returns:
        (N, 3) 数组。
    """
    if not ground_truth:
        return np.empty((0, 3))
    result = np.empty((len(ground_truth), 3))
    for i, g in enumerate(ground_truth):
        result[i, 0] = g.range
        result[i, 1] = g.velocity
        result[i, 2] = g.angle
    return result


def estimates_to_array(estimates: list) -> np.ndarray:
    """将 TargetEstimate 列表转换为 (M, 3) 矩阵 [range, velocity, angle]。

    Args:
        estimates: List[TargetEstimate]。

    Returns:
        (M, 3) 数组。
    """
    if not estimates:
        return np.empty((0, 3))
    result = np.empty((len(estimates), 3))
    for i, e in enumerate(estimates):
        result[i, 0] = e.range
        result[i, 1] = e.velocity
        result[i, 2] = e.angle
    return result
