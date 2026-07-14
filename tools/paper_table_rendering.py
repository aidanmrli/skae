"""Compatibility imports for paper table rendering."""

from importlib import import_module

_implementation = import_module("experiments.neurips_2026.evidence.table_rendering")


def __getattr__(name: str):
    return getattr(_implementation, name)
