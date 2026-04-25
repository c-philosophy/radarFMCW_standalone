"""DAGPipeline: Directed Acyclic Graph pipeline engine.

Stages can be added in any order; the pipeline topologically sorts them
based on their declared inputs and outputs.
"""

from typing import List, Dict, Optional, Iterator, Tuple
from collections import deque
import time
import logging

from .stage import Stage
from .context import PipelineContext

logger = logging.getLogger(__name__)


class DAGPipeline:
    """A DAG-based processing pipeline.

    Stages are nodes; edges are implied by input/output key dependencies.
    add_stage() wires a stage after its dependency stages automatically.

    Usage:
        pipe = DAGPipeline()
        pipe.add_stage(Stage("gen", outputs=["signal"]))
        pipe.add_stage(Stage("fft", inputs=["signal"], outputs=["rd_map"]))
        pipe.add_stage(Stage("cfar", inputs=["rd_map"], outputs=["peaks"]))
        ctx = PipelineContext()
        pipe.run(ctx)
    """

    def __init__(self, name: str = "pipeline"):
        self.name = name
        self._stages: Dict[str, Stage] = {}       # name -> Stage
        self._order: List[str] = []               # topological order (cached)

    # --- build ---

    def add_stage(self, stage: Stage) -> "DAGPipeline":
        """Add a stage. Returns self for fluent chaining."""
        if stage.name in self._stages:
            raise ValueError(f"Duplicate stage name: '{stage.name}'")
        self._stages[stage.name] = stage
        self._order = []  # invalidate cache
        return self

    def remove_stage(self, name: str) -> "DAGPipeline":
        """Remove a stage by name."""
        if name not in self._stages:
            raise KeyError(f"Stage not found: '{name}'")
        del self._stages[name]
        self._order = []
        return self

    def insert_before(self, target: str, stage: Stage) -> "DAGPipeline":
        """Insert a stage so it executes before the target stage."""
        if target not in self._stages:
            raise KeyError(f"Target stage not found: '{target}'")
        if stage.name in self._stages:
            raise ValueError(f"Duplicate stage name: '{stage.name}'")
        self._stages[stage.name] = stage
        self._order = []
        return self

    # --- topology ---

    def _topological_sort(self) -> List[str]:
        """Compute topological order based on input/output key dependencies."""
        # Build producer map: key -> stage that produces it
        producer: Dict[str, str] = {}
        for stage in self._stages.values():
            for out_key in stage.outputs:
                if out_key in producer:
                    logger.warning(
                        f"Key '{out_key}' produced by both '{producer[out_key]}' "
                        f"and '{stage.name}'; using the latter."
                    )
                producer[out_key] = stage.name

        # Build adjacency and in-degree
        in_degree: Dict[str, int] = {name: 0 for name in self._stages}
        adj: Dict[str, List[str]] = {name: [] for name in self._stages}

        for stage in self._stages.values():
            for in_key in stage.inputs:
                src = producer.get(in_key)
                if src is not None and src != stage.name:
                    if stage.name not in adj[src]:
                        adj[src].append(stage.name)
                        in_degree[stage.name] += 1

        # Kahn's algorithm
        queue = deque([name for name, deg in in_degree.items() if deg == 0])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self._stages):
            remaining = set(self._stages) - set(order)
            raise RuntimeError(
                f"Cycle detected in pipeline DAG. "
                f"Unresolved stages: {remaining}"
            )

        return order

    @property
    def execution_order(self) -> List[str]:
        """Return the topological execution order (cached)."""
        if not self._order:
            self._order = self._topological_sort()
        return list(self._order)

    # --- run ---

    def run(self, ctx: Optional[PipelineContext] = None) -> PipelineContext:
        """Execute all stages in topological order.

        Args:
            ctx: Existing context (creates new one if None).

        Returns:
            The PipelineContext after all stages have run.
        """
        if ctx is None:
            ctx = PipelineContext()

        order = self.execution_order
        logger.info(f"Pipeline '{self.name}': executing {len(order)} stages")

        for name in order:
            stage = self._stages[name]
            missing = stage.validate_inputs(ctx)
            if missing:
                raise RuntimeError(
                    f"Stage '{name}' missing inputs: {missing}. "
                    f"Context keys: {list(ctx._data.keys())}"
                )
            t0 = time.perf_counter()
            stage.run(ctx)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.debug(f"  [{name}] {elapsed_ms:.1f} ms")

        return ctx

    def run_streaming(self, ctx_factory, num_frames: int) -> Iterator[PipelineContext]:
        """Run the pipeline frame-by-frame, yielding context after each frame.

        Args:
            ctx_factory: Callable[[int], PipelineContext] that creates a
                         fresh context for each frame index.
            num_frames: Total frames to process.

        Yields:
            PipelineContext after each frame is processed.
        """
        for frame_idx in range(num_frames):
            ctx = ctx_factory(frame_idx)
            yield self.run(ctx)

    # --- info ---

    def summary(self) -> str:
        """Return a text summary of the pipeline structure."""
        order = self.execution_order
        lines = [f"Pipeline '{self.name}' ({len(order)} stages):"]
        for i, name in enumerate(order):
            stage = self._stages[name]
            lines.append(
                f"  {i + 1}. {stage.name}  "
                f"in={stage.inputs}  out={stage.outputs}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"DAGPipeline('{self.name}', {len(self._stages)} stages)"
