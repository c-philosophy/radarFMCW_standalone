"""杂波边缘测试场景：测试 GO-CFAR/SO-CFAR 在杂波跃变下的表现。"""

from fmcw.config.schema import (
    SceneConfig, TargetSpec, MotionModel,
)


def build_clutter_edge(
    num_frames: int = 30,
    target_range: float = 35.0,
    target_velocity: float = 8.0,
    target_angle: float = 0.0,
    snr_db: float = 20.0,
) -> SceneConfig:
    """杂波边缘场景——在噪声功率跃变边界放置目标。

    当前实现为简化版本：使用单目标 + 低 SNR 模拟杂波环境。
    完整杂波边缘模拟需在信号生成层增加空间变噪声功率，
    超出当前 Scene 模型能力。

    评估建议：在实际杂波数据上额外测试 GO-CFAR 的虚警控制。

    Args:
        num_frames: 总帧数。
        target_range: 目标距离 (m)。
        target_velocity: 目标径向速度 (m/s)。
        target_angle: 目标方位角 (deg)。
        snr_db: 目标 SNR (dB) —— 使用较低值模拟杂波环境。
    """
    return SceneConfig(
        type="clutter_edge",
        num_frames=num_frames,
        frame_interval=0.05,
        targets=[
            TargetSpec(
                id=1,
                initial_range=target_range,
                initial_velocity=target_velocity,
                initial_angle=target_angle,
                rcs=0.3,  # 低 RCS 模拟弱目标
                motion=MotionModel(motion="constant_velocity"),
            )
        ],
        snr_db=snr_db,
        swerling_model=2,  # 快起伏模拟杂波变化
    )
