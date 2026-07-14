"""Compatibility entry point for paper Dysts cache construction."""

from importlib import import_module

_implementation = import_module("experiments.neurips_2026.workflows.dysts_cache")
main = _implementation.main


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    main()
