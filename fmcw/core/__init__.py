"""Core framework: AlgorithmRegistry, Stage, PipelineContext, DAGPipeline."""

from .stage import Stage
from .context import PipelineContext
from .pipeline import DAGPipeline
from .registry import AlgorithmRegistry, register_algorithm

__all__ = [
    "Stage",
    "PipelineContext",
    "DAGPipeline",
    "AlgorithmRegistry",
    "register_algorithm",
]
