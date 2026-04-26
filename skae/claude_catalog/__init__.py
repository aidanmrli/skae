"""Transition-rich dynamical system catalog for basin partitioning benchmark.

Generated on branch: claude-transition-rich-systems-gen
"""

from __future__ import annotations

import importlib

from skae.claude_catalog.base import CatalogSystem, rk4_step
from skae.claude_catalog.registry import CATALOG_REGISTRY, get_system, list_systems

_CATALOG_MODULES = (
    "skae.claude_catalog.systems_gradient",
    "skae.claude_catalog.systems_bio_physical",
    "skae.claude_catalog.systems_creative",
    "skae.claude_catalog.systems_novel",
    "skae.claude_catalog.systems_tuned",
    "skae.claude_catalog.systems_variants",
    "skae.claude_catalog.systems_hybrid",
    "skae.claude_catalog.systems_flagship",
)


def ensure_catalog_registered() -> None:
    """Import all catalog modules so the registry is fully populated."""
    for module_name in _CATALOG_MODULES:
        importlib.import_module(module_name)
