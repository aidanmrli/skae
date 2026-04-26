"""Registry for all catalog systems."""

from typing import Dict, List, Type
from skae.claude_catalog.base import CatalogSystem

CATALOG_REGISTRY: Dict[str, Type[CatalogSystem]] = {}


def register(cls: Type[CatalogSystem]) -> Type[CatalogSystem]:
    """Decorator to register a system class."""
    CATALOG_REGISTRY[cls.name] = cls
    return cls


def get_system(name: str, **kwargs) -> CatalogSystem:
    """Instantiate a system by name."""
    if name not in CATALOG_REGISTRY:
        raise ValueError(
            f"Unknown system '{name}'. Available: {list(CATALOG_REGISTRY.keys())}"
        )
    return CATALOG_REGISTRY[name](**kwargs)


def list_systems() -> List[str]:
    """List all registered system names."""
    return sorted(CATALOG_REGISTRY.keys())
