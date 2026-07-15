"""Compatibility entry point for controlled per-system paper tables."""

from importlib import import_module

_implementation = import_module("experiments.neurips_2026.evidence.controlled_tables")
main = _implementation.main


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    main()
