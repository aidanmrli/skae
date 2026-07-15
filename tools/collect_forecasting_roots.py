"""Compatibility entry point for controlled forecasting collection."""

from importlib import import_module

_implementation = import_module(
    "experiments.neurips_2026.workflows.controlled_collection"
)
main = _implementation.main


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    main()
