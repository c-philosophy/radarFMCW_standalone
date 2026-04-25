"""跟踪模块：卡尔曼滤波器、数据关联、航迹管理。"""

from fmcw.core.registry import AlgorithmRegistry

# 注册卡尔曼滤波器变体（导入时注册）
from .kalman import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter, create_tracker
AlgorithmRegistry.register("tracker", "kf", KalmanFilter)
AlgorithmRegistry.register("tracker", "ekf", ExtendedKalmanFilter)
AlgorithmRegistry.register("tracker", "ukf", UnscentedKalmanFilter)

# 注册关联器（导入时注册）
from .association import NearestNeighbor, GNN, JPDA
from .track import Track, TrackManager, TrackStatus
from .multi_tracker import MultiTargetTracker

__all__ = [
    "KalmanFilter", "ExtendedKalmanFilter", "UnscentedKalmanFilter", "create_tracker",
    "NearestNeighbor", "GNN", "JPDA",
    "Track", "TrackManager", "TrackStatus", "MultiTargetTracker",
]
