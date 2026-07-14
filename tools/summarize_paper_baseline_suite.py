"""Compatibility entry point for standalone-control summaries."""

from importlib import import_module

_implementation = import_module("experiments.neurips_2026.baselines.summarize")
main = _implementation.main


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    main()
