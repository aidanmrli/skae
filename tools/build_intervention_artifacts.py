"""Compatibility entry point for intervention artifacts."""

from importlib import import_module

_implementation = import_module("experiments.neurips_2026.interventions.artifacts")
main = _implementation.main


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    main()
