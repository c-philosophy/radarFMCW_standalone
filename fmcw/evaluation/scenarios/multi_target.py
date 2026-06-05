"""多目标测试场景：4-8个目标密集分布。"""

import numpy as np
from fmcw.config.schema import (
    SceneConfig, TargetSpec, MotionModel,
)


def build_multi_target_dense(
    num_targets: int = 4,
    num_frames: int = 50,
    snr_db: float = 25.0,
    seed: int = 42,
) -> SceneConfig:
    """多目标密集分布场景——测试 MOTA/GOSPA 和数据关联。

    目标在距离 15-60 m、角度 -30°~+30°、速度 -15~+15 m/s
    范围内随机分布。

    Args:
        num_targets: 目标数量（建议 4-8）。
        num_frames: 总帧数。
        snr_db: 目标 SNR (dB)。
        seed: 随机种子（可复现）。

    Returns:
        SceneConfig。
    """
    rng = np.random.RandomState(seed)
    targets = []
    for i in range(num_targets):
        targets.append(TargetSpec(
            id=i + 1,
            initial_range=float(rng.uniform(15, 60)),
            initial_velocity=float(rng.uniform(-15, 15)),
            initial_angle=float(rng.uniform(-30, 30)),
            rcs=float(rng.uniform(0.5, 2.0)),
            motion=MotionModel(motion="constant_velocity"),
        ))

    return SceneConfig(
        type="multi_target_dense",
        num_frames=num_frames,
        frame_interval=0.05,
        targets=targets,
        snr_db=snr_db,
        swerling_model=0,
    )
