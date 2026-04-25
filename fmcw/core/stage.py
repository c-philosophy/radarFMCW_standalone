"""Stage: a single processing step in the pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional, Callable
from .context import PipelineContext


@dataclass
class Stage:
    """A single processing stage in the DAG pipeline.

    Each stage declares its input and output context keys so the pipeline
    can validate the DAG and track data provenance.

    Minimal usage (function-based):
        Stage(name="my_stage", run_fn=my_function, inputs=["rd_map"], outputs=["filtered_peaks"])

    Class-based usage (inherit and override run()):
        class MyStage(Stage):
            def run(self, ctx): ...
    """

    name: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    run_fn: Optional[Callable] = None  # Convenience: function-based stage

    def run(self, ctx: PipelineContext) -> None:
        """Execute this stage. Override in subclasses or provide run_fn."""
        if self.run_fn is not None:
            self.run_fn(ctx)
        else:
            raise NotImplementedError(
                f"Stage '{self.name}' has no run() implementation. "
                f"Override run() or provide run_fn."
            )

    def validate_inputs(self, ctx: PipelineContext) -> List[str]:
        """Check that all required inputs are present. Returns list of missing keys."""
        return [k for k in self.inputs if k not in ctx]

    def __repr__(self) -> str:
        return f"Stage('{self.name}', in={self.inputs}, out={self.outputs})"
