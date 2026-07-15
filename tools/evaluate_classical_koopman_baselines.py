"""Compatibility entry point for classical Koopman controls."""

from importlib import import_module

_implementation = import_module("experiments.neurips_2026.baselines.classical")
main = _implementation.main


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    main()
