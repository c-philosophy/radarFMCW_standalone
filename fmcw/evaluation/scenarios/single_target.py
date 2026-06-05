"""单目标测试场景：匀速、匀加速、静止、SNR 扫描。"""

from typing import List
from fmcw.config.schema import (
    SceneConfig, TargetSpec, MotionModel,
)


def build_single_target_cv(
    num_frames: int = 50,
    initial_range: float = 30.0,
    initial_velocity: float = 10.0,
    initial_angle: float = 0.0,
    snr_db: float = 25.0,
    swerling: int = 0,
    frame_interval: float = 0.05,
) -> SceneConfig:
    """单目标匀速运动场景。

    Args:
        num_frames: 总帧数。
        initial_range: 初始距离 (m)。
        initial_velocity: 径向速度 (m/s)，正值=接近雷达。
        initial_angle: 方位角 (deg)，0=boresight。
        snr_db: 目标 SNR (dB)。
        swerling: Swerling 起伏模型 (0-4)。
        frame_interval: 帧间隔 (s)。

    Returns:
        SceneConfig 可直接用于 RadarPipeline。
    """
    return SceneConfig(
        type="single_target_cv",
        num_frames=num_frames,
        frame_interval=frame_interval,
        targets=[
            TargetSpec(
                id=1,
                initial_range=initial_range,
                initial_velocity=initial_velocity,
                initial_angle=initial_angle,
                rcs=1.0,
                motion=MotionModel(motion="constant_velocity"),
            )
        ],
        snr_db=snr_db,
        swerling_model=swerling,
    )


def build_single_target_ca(
    num_frames: int = 50,
    initial_range: float = 50.0,
    initial_velocity: float = 5.0,
    acceleration: float = 2.0,
    initial_angle: float = 10.0,
    snr_db: float = 25.0,
) -> SceneConfig:
    """单目标匀加速运动场景——测试机动跟踪。

    Args:
        num_frames: 总帧数。
        initial_range: 初始距离 (m)。
        initial_velocity: 初始径向速度 (m/s)。
        acceleration: 加速度 (m/s²)。
        initial_angle: 初始方位角 (deg)。
        snr_db: 目标 SNR (dB)。
    """
    return SceneConfig(
        type="single_target_ca",
        num_frames=num_frames,
        frame_interval=0.05,
        targets=[
            TargetSpec(
                id=1,
                initial_range=initial_range,
                initial_velocity=initial_velocity,
                initial_angle=initial_angle,
                rcs=1.0,
                motion=MotionModel(
                    motion="constant_acceleration",
                    acceleration=acceleration,
                ),
            )
        ],
        snr_db=snr_db,
        swerling_model=0,
    )


def build_static_target(
    num_frames: int = 20,
    initial_range: float = 25.0,
    initial_angle: float = 0.0,
    snr_db: float = 30.0,
) -> SceneConfig:
    """静止目标场景——距离精度基准，零速检测验证。

    Args:
        num_frames: 总帧数。
        initial_range: 距离 (m)。
        initial_angle: 方位角 (deg)。
        snr_db: SNR (dB)。
    """
    return SceneConfig(
        type="static_target",
        num_frames=num_frames,
        frame_interval=0.05,
        targets=[
            TargetSpec(
                id=1,
                initial_range=initial_range,
                initial_velocity=0.0,
                initial_angle=initial_angle,
                rcs=1.0,
                motion=MotionModel(motion="stationary"),
            )
        ],
        snr_db=snr_db,
        swerling_model=0,
    )


def build_snr_sweep(
    snr_range_db: tuple = (-5, 30),
    snr_step_db: float = 5.0,
    num_frames_per_snr: int = 20,
    initial_range: float = 30.0,
    initial_velocity: float = 10.0,
) -> List[SceneConfig]:
    """SNR 扫描场景：生成多个 SceneConfig，覆盖不同 SNR 值。

    用于绘制 Pd-vs-SNR 曲线和 CRLB 比率曲线。

    Args:
        snr_range_db: (min_snr, max_snr) 范围 (dB)。
        snr_step_db: SNR 步长 (dB)。
        num_frames_per_snr: 每个 SNR 值的帧数。
        initial_range: 初始距离 (m)。
        initial_velocity: 初始速度 (m/s)。

    Returns:
        List[SceneConfig] 列表，每个元素对应一个 SNR 值。
    """
    configs = []
    snr_min, snr_max = snr_range_db
    snr = snr_min
    while snr <= snr_max:
        configs.append(build_single_target_cv(
            num_frames=num_frames_per_snr,
            snr_db=snr,
            initial_range=initial_range,
            initial_velocity=initial_velocity,
        ))
        snr += snr_step_db
    return configs
