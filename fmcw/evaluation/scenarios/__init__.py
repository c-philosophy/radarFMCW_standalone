"""标准化测试场景：预定义雷达评估场景集合。

每个场景返回 SceneConfig，可直接注入 RadarPipeline。
"""

from fmcw.evaluation.scenarios.single_target import (
    build_single_target_cv,
    build_single_target_ca,
    build_static_target,
    build_snr_sweep,
)
from fmcw.evaluation.scenarios.two_targets import (
    build_two_targets_close,
    build_two_targets_crossing,
)
from fmcw.evaluation.scenarios.multi_target import (
    build_multi_target_dense,
)
from fmcw.evaluation.scenarios.clutter_edge import (
    build_clutter_edge,
)

__all__ = [
    "build_single_target_cv",
    "build_single_target_ca",
    "build_static_target",
    "build_snr_sweep",
    "build_two_targets_close",
    "build_two_targets_crossing",
    "build_multi_target_dense",
    "build_clutter_edge",
]
