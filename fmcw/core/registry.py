"""AlgorithmRegistry: global registry for plug-and-play algorithm swapping."""

from typing import Dict, Type, Any


class AlgorithmRegistry:
    """Global registry mapping algorithm category -> name -> class.

    Enables plug-and-play: swap CA-CFAR for OS-CFAR by changing a config string.

    Usage:
        @register_algorithm("cfar", "ca_cfar")
        class CACFAR: ...

        cls = AlgorithmRegistry.get("cfar", "ca_cfar")
        cfar = cls()
    """

    _registry: Dict[str, Dict[str, Type]] = {}

    @classmethod
    def register(cls, category: str, name: str, algorithm_cls: Type) -> None:
        """Register an algorithm class under a category."""
        if category not in cls._registry:
            cls._registry[category] = {}
        cls._registry[category][name] = algorithm_cls

    @classmethod
    def get(cls, category: str, name: str) -> Type:
        """Retrieve a registered algorithm class.

        Raises:
            KeyError: If category or name is unknown.
        """
        if category not in cls._registry:
            raise KeyError(
                f"Unknown algorithm category: '{category}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        if name not in cls._registry[category]:
            available = list(cls._registry[category].keys())
            raise KeyError(
                f"Unknown algorithm '{name}' in category '{category}'. "
                f"Available: {available}"
            )
        return cls._registry[category][name]

    @classmethod
    def list_algorithms(cls, category: str) -> list:
        """List all registered algorithms in a category."""
        return list(cls._registry.get(category, {}).keys())

    @classmethod
    def list_categories(cls) -> list:
        """List all algorithm categories."""
        return list(cls._registry.keys())

    @classmethod
    def create(cls, category: str, name: str, **kwargs) -> Any:
        """Create an instance of a registered algorithm."""
        alg_cls = cls.get(category, name)
        return alg_cls(**kwargs)


def register_algorithm(category: str, name: str):
    """Class decorator that registers the class in the global registry.

    Usage:
        @register_algorithm("cfar", "ca_cfar")
        class CACFAR(BaseCFAR):
            ...
    """
    def decorator(cls):
        AlgorithmRegistry.register(category, name, cls)
        return cls
    return decorator
