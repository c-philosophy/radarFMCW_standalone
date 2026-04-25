"""检测模块：峰值检测和 CFAR 检测器。"""

# 确保 CFAR 算法在导入时注册
from . import cfar
from .peak_finder import PeakFinder
from .detector import DetectionPipeline

__all__ = ["PeakFinder", "DetectionPipeline"]
