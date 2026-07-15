"""Registry for the analytic multibasin systems."""

from typing import Dict, List, Type
from skae.dynamics.analytic.base import CatalogSystem

ANALYTIC_REGISTRY: Dict[str, Type[CatalogSystem]] = {}
# Historical public name retained for checkpoint/import compatibility.
CATALOG_REGISTRY = ANALYTIC_REGISTRY


def register(cls: Type[CatalogSystem]) -> Type[CatalogSystem]:
    """Decorator to register a system class."""
    ANALYTIC_REGISTRY[cls.name] = cls
    return cls


def get_system(name: str, **kwargs) -> CatalogSystem:
    """Instantiate a system by name."""
    if name not in ANALYTIC_REGISTRY:
        raise ValueError(
            f"Unknown system '{name}'. Available: {list(ANALYTIC_REGISTRY.keys())}"
        )
    return ANALYTIC_REGISTRY[name](**kwargs)


def list_systems() -> List[str]:
    """List all registered system names."""
    return sorted(ANALYTIC_REGISTRY.keys())
