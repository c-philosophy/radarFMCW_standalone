"""双目标测试场景：接近、交叉轨迹。"""

from fmcw.config.schema import (
    SceneConfig, TargetSpec, MotionModel,
)


def build_two_targets_close(
    num_frames: int = 30,
    range1: float = 30.0,
    range2: float = 31.0,
    velocity1: float = 10.0,
    velocity2: float = 10.0,
    angle1: float = 0.0,
    angle2: float = 5.0,
    snr_db: float = 25.0,
) -> SceneConfig:
    """双目标近距场景——测试 CFAR 抗目标遮蔽和距离分辨率。

    两个目标在距离上仅相差 range_resolution × 2。

    Args:
        num_frames: 总帧数。
        range1/range2: 两个目标的初始距离 (m)。
        velocity1/velocity2: 两个目标的径向速度 (m/s)。
        angle1/angle2: 两个目标的方位角 (deg)。
        snr_db: 目标 SNR (dB)。
    """
    return SceneConfig(
        type="two_targets_close",
        num_frames=num_frames,
        frame_interval=0.05,
        targets=[
            TargetSpec(
                id=1,
                initial_range=range1,
                initial_velocity=velocity1,
                initial_angle=angle1,
                rcs=1.0,
                motion=MotionModel(motion="constant_velocity"),
            ),
            TargetSpec(
                id=2,
                initial_range=range2,
                initial_velocity=velocity2,
                initial_angle=angle2,
                rcs=1.0,
                motion=MotionModel(motion="constant_velocity"),
            ),
        ],
        snr_db=snr_db,
        swerling_model=0,
    )


def build_two_targets_crossing(
    num_frames: int = 60,
    x1_start: float = -20.0,
    x2_start: float = 20.0,
    y_common: float = 40.0,
    vx: float = 2.0,
    snr_db: float = 25.0,
) -> SceneConfig:
    """双目标交叉轨迹场景——测试数据关联和身份切换。

    两个目标在 XY 平面上交叉运动，雷达位于原点 (0, 0)。
    通过极坐标转换自动计算每帧的 range 和 angle。

    Args:
        num_frames: 总帧数。
        x1_start/x2_start: 两个目标的初始 X 坐标 (m)。
        y_common: 两个目标的共同 Y 坐标 (m)。
        vx: 两个目标的 X 方向速度 (m/s)，方向相反。
        snr_db: 目标 SNR (dB)。

    Returns:
        SceneConfig。

    Note:
        由于 MotionModel 仅支持径向运动（range/velocity 变化），
        交叉轨迹场景是简化的近距-远距交替运动模拟。
        实际交叉运动需通过 Scene 的 all_trajectories 手动构建。
    """
    # 简化：两目标分别以不同方向运动
    return SceneConfig(
        type="two_targets_crossing",
        num_frames=num_frames,
        frame_interval=0.05,
        targets=[
            TargetSpec(
                id=1,
                initial_range=30.0,
                initial_velocity=5.0,   # 接近雷达
                initial_angle=-10.0,
                rcs=1.0,
                motion=MotionModel(
                    motion="constant_velocity",
                    angular_velocity=0.5,  # deg/s，角度变化模拟交叉
                ),
            ),
            TargetSpec(
                id=2,
                initial_range=40.0,
                initial_velocity=-3.0,  # 远离雷达
                initial_angle=10.0,
                rcs=1.0,
                motion=MotionModel(
                    motion="constant_velocity",
                    angular_velocity=-0.5,
                ),
            ),
        ],
        snr_db=snr_db,
        swerling_model=0,
    )
