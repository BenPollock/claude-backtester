"""Strategy registry: discover strategies by name."""

import importlib
import logging
import pkgutil

from backtester.strategies.base import Strategy

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[Strategy]] = {}

# Modules that are part of the strategies package but don't contain
# strategy implementations (no @register_strategy usage).
_INFRASTRUCTURE_MODULES = {"base", "registry", "indicators", "__init__"}

_discovered = False


def register_strategy(name: str):
    """Decorator to register a strategy class by name.

    Usage:
        @register_strategy("sma_crossover")
        class SmaCrossover(Strategy): ...
    """
    def decorator(cls: type[Strategy]) -> type[Strategy]:
        if name in _REGISTRY:
            existing = _REGISTRY[name]
            if existing is cls:
                return cls  # same class re-imported; safe to skip
            raise ValueError(f"Strategy '{name}' already registered")
        _REGISTRY[name] = cls
        return cls
    return decorator


def discover_strategies() -> None:
    """Scan the strategies package and import all strategy modules.

    Imports every module in ``backtester.strategies`` except known
    infrastructure modules (base, registry, indicators, __init__).
    Each module's ``@register_strategy`` decorators execute on import,
    populating the registry.

    Safe to call multiple times -- subsequent calls are no-ops.
    """
    global _discovered
    if _discovered:
        return

    import backtester.strategies as pkg

    for finder, module_name, is_pkg in pkgutil.iter_modules(pkg.__path__):
        if module_name in _INFRASTRUCTURE_MODULES:
            continue
        fqn = f"backtester.strategies.{module_name}"
        try:
            importlib.import_module(fqn)
        except Exception:
            logger.warning("Failed to import strategy module %s", fqn, exc_info=True)

    _discovered = True


def get_strategy(name: str) -> Strategy:
    """Instantiate a registered strategy by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    return _REGISTRY[name]()


def list_strategies() -> list[str]:
    """Return sorted list of registered strategy names."""
    return sorted(_REGISTRY.keys())
