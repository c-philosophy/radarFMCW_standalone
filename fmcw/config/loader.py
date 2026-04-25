"""YAML/JSON 配置加载器，带验证功能。"""

import yaml
import json
from pathlib import Path
from typing import Union, Dict, Any

from .schema import (
    RadarParams, TargetParams, TargetSpec, MotionModel, SceneConfig,
    DetectionConfig, TrackingConfig, VisualizationConfig, PersistenceConfig,
    PipelineConfig,
)


def _dict_to_radar_params(d: dict) -> RadarParams:
    """将字典转换为 RadarParams 数据类，忽略未知键。

    Args:
        d: 包含雷达参数的字典。

    Returns:
        RadarParams 实例，仅包含已知字段。
    """
    known = {f.name for f in RadarParams.__dataclass_fields__.values() if f.init}
    return RadarParams(**{k: v for k, v in d.items() if k in known})


def _dict_to_target_params(d: dict) -> TargetParams:
    """将字典转换为 TargetParams 数据类。

    Args:
        d: 包含目标参数的字典。预期键：
            - 'range': 目标距离（米），必需
            - 'velocity': 径向速度（米/秒），默认 0.0
            - 'angle': 方位角（度），默认 0.0
            - 'rcs': RCS 缩放因子，默认 1.0

    Returns:
        TargetParams 实例。
    """
    return TargetParams(
        range=d["range"],
        velocity=d.get("velocity", 0.0),
        angle=d.get("angle", 0.0),
        rcs=d.get("rcs", 1.0),
    )


def _dict_to_target_spec(d: dict) -> TargetSpec:
    """将字典转换为 TargetSpec 数据类。

    Args:
        d: 包含目标规格的字典。预期键：
            - 'id': 目标标识符，必需
            - 'initial_range': 初始距离（米），必需
            - 'initial_velocity': 初始速度（米/秒），默认 0.0
            - 'initial_angle': 初始角度（度），默认 0.0
            - 'rcs': RCS 缩放因子，默认 1.0
            - 'motion': 运动模型配置，可选

    Returns:
        TargetSpec 实例。
    """
    motion_raw = d.get("motion", {})
    if isinstance(motion_raw, str):
        motion = MotionModel(motion=motion_raw)
    elif isinstance(motion_raw, dict):
        motion = MotionModel(**motion_raw)
    else:
        motion = MotionModel()

    return TargetSpec(
        id=d["id"],
        initial_range=d["initial_range"],
        initial_velocity=d.get("initial_velocity", 0.0),
        initial_angle=d.get("initial_angle", 0.0),
        rcs=d.get("rcs", 1.0),
        motion=motion,
    )


def _dict_to_scene_config(d: dict) -> SceneConfig:
    """将字典转换为 SceneConfig 数据类。

    Args:
        d: 包含场景配置的字典。预期键：
            - 'type': 场景类型，默认 'single_target'
            - 'num_frames': 帧数，默认 1
            - 'frame_interval': 帧间隔（秒），默认 0.05
            - 'targets': 目标规格列表，默认空列表
            - 'snr_db': 目标 SNR（dB），默认 25.0
            - 'swerling_model': Swerling 模型编号（0-4），默认 0

    Returns:
        SceneConfig 实例。
    """
    return SceneConfig(
        type=d.get("type", "single_target"),
        num_frames=d.get("num_frames", 1),
        frame_interval=d.get("frame_interval", 0.05),
        targets=[_dict_to_target_spec(t) for t in d.get("targets", [])],
        snr_db=d.get("snr_db", 25.0),
        swerling_model=d.get("swerling_model", 0),
    )


def _dict_to_detection_config(d: dict) -> DetectionConfig:
    """将字典转换为 DetectionConfig 数据类。

    Args:
        d: 包含检测配置的字典。

    Returns:
        DetectionConfig 实例，仅包含已知字段。
    """
    known = {f.name for f in DetectionConfig.__dataclass_fields__.values() if f.init}
    return DetectionConfig(**{k: v for k, v in d.items() if k in known})


def _dict_to_tracking_config(d: dict) -> TrackingConfig:
    """将字典转换为 TrackingConfig 数据类。

    Args:
        d: 包含跟踪配置的字典。

    Returns:
        TrackingConfig 实例，仅包含已知字段。
    """
    known = {f.name for f in TrackingConfig.__dataclass_fields__.values() if f.init}
    return TrackingConfig(**{k: v for k, v in d.items() if k in known})


def _dict_to_viz_config(d: dict) -> VisualizationConfig:
    """将字典转换为 VisualizationConfig 数据类。

    Args:
        d: 包含可视化配置的字典。

    Returns:
        VisualizationConfig 实例，仅包含已知字段。
    """
    known = {f.name for f in VisualizationConfig.__dataclass_fields__.values() if f.init}
    return VisualizationConfig(**{k: v for k, v in d.items() if k in known})


def _dict_to_persistence_config(d: dict) -> PersistenceConfig:
    """将字典转换为 PersistenceConfig 数据类。

    Args:
        d: 包含持久化配置的字典。

    Returns:
        PersistenceConfig 实例，仅包含已知字段。
    """
    known = {f.name for f in PersistenceConfig.__dataclass_fields__.values() if f.init}
    return PersistenceConfig(**{k: v for k, v in d.items() if k in known})


def _dict_to_pipeline_config(d: dict) -> PipelineConfig:
    """将字典转换为 PipelineConfig 数据类。

    Args:
        d: 包含流水线配置的字典。预期键：
            - 'stages': 流水线阶段列表，可选
            - 'detection': 检测配置字典，可选
            - 'tracking': 跟踪配置字典，可选
            - 'visualization': 可视化配置字典，可选
            - 'persistence': 持久化配置字典，可选

    Returns:
        PipelineConfig 实例。
    """
    return PipelineConfig(
        stages=d.get("stages", PipelineConfig().stages),
        detection=_dict_to_detection_config(d.get("detection", {})),
        tracking=_dict_to_tracking_config(d.get("tracking", {})),
        visualization=_dict_to_viz_config(d.get("visualization", {})),
        persistence=_dict_to_persistence_config(d.get("persistence", {})),
    )


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML or JSON configuration file and return structured config objects.

    Returns a dict with keys: 'radar', 'pipeline', 'scene'.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix in (".yaml", ".yml"):
            raw = yaml.safe_load(f)
        elif path.suffix == ".json":
            raw = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    return {
        "radar": _dict_to_radar_params(raw.get("radar", {})),
        "pipeline": _dict_to_pipeline_config(raw.get("pipeline", {})),
        "scene": _dict_to_scene_config(raw.get("scenario", {})),
    }
