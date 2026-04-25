"""Visualizer factory: create_visualizer(backend_type)."""

from typing import Optional, List


def create_visualizer(
    backend: str = "mpl",
    panels: Optional[List[str]] = None,
    **kwargs,
):
    """Create a visualizer instance by backend name.

    Args:
        backend: 'mpl' for matplotlib, 'pyqtgraph' for real-time.
        panels: List of panel names.
        **kwargs: Passed to the visualizer constructor.

    Returns:
        BaseVisualizer instance.
    """
    if backend == "mpl":
        from .mpl_viz import MatplotlibVisualizer
        return MatplotlibVisualizer(panels=panels, **kwargs)
    elif backend == "pyqtgraph":
        try:
            from .pyqtgraph_viz import PyQtGraphVisualizer
            return PyQtGraphVisualizer(panels=panels, **kwargs)
        except ImportError as e:
            raise ImportError(
                f"pyqtgraph backend requested but import failed: {e}. "
                f"Install with: pip install pyqtgraph"
            )
    else:
        raise ValueError(f"Unknown visualization backend: '{backend}'. Use 'mpl' or 'pyqtgraph'.")
