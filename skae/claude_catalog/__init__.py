"""Registry for the analytic systems retained by the controlled paper roster."""

from __future__ import annotations

import importlib

from skae.claude_catalog.base import CatalogSystem, rk4_step
from skae.claude_catalog.registry import CATALOG_REGISTRY, get_system, list_systems

_CATALOG_MODULES = (
    "skae.claude_catalog.paper_systems",
)


def ensure_catalog_registered() -> None:
    """Import the retained paper systems so the registry is populated."""
    for module_name in _CATALOG_MODULES:
        importlib.import_module(module_name)
