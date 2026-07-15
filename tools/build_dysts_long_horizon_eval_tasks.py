"""Compatibility entry point for Dysts evaluation task generation."""

from importlib import import_module

_implementation = import_module(
    "experiments.neurips_2026.workflows.dysts_evaluation_tasks"
)
main = _implementation.main


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    main()
