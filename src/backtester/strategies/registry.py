"""Strategy registry: discover strategies by name."""

from backtester.strategies.base import Strategy

_REGISTRY: dict[str, type[Strategy]] = {}


def register_strategy(name: str):
    """Decorator to register a strategy class by name.

    Usage:
        @register_strategy("sma_crossover")
        class SmaCrossover(Strategy): ...
    """
    def decorator(cls: type[Strategy]) -> type[Strategy]:
        if name in _REGISTRY:
            raise ValueError(f"Strategy '{name}' already registered")
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_strategy(name: str) -> Strategy:
    """Instantiate a registered strategy by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    return _REGISTRY[name]()


def list_strategies() -> list[str]:
    """Return sorted list of registered strategy names."""
    return sorted(_REGISTRY.keys())
