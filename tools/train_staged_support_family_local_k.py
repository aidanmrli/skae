"""Compatibility entry point for the paper's support-routed trainer."""

from importlib import import_module

_implementation = import_module("experiments.neurips_2026.local_operators.train")
main = _implementation.main


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    main()
