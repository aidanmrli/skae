"""Compatibility imports for :mod:`skae.dynamics.analytic.registry`."""

from skae.dynamics.analytic.registry import (
    ANALYTIC_REGISTRY,
    CATALOG_REGISTRY,
    get_system,
    list_systems,
    register,
)

__all__ = [
    "ANALYTIC_REGISTRY",
    "CATALOG_REGISTRY",
    "register",
    "get_system",
    "list_systems",
]
