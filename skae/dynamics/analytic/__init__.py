"""Descriptively named registry for the paper's analytic multibasin systems."""

from __future__ import annotations

import importlib

from skae.dynamics.analytic.base import CatalogSystem, rk4_step
from skae.dynamics.analytic.registry import (
    ANALYTIC_REGISTRY,
    CATALOG_REGISTRY,
    get_system,
    list_systems,
)

_SYSTEM_MODULES = ("skae.dynamics.analytic.systems",)


def ensure_catalog_registered() -> None:
    """Import the retained analytic systems so the registry is populated."""

    for module_name in _SYSTEM_MODULES:
        importlib.import_module(module_name)


__all__ = [
    "CatalogSystem",
    "rk4_step",
    "ANALYTIC_REGISTRY",
    "CATALOG_REGISTRY",
    "get_system",
    "list_systems",
    "ensure_catalog_registered",
]
