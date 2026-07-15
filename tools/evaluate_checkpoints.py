"""Compatibility entry point for :mod:`skae.cli.evaluate`."""

from skae.cli.evaluate import evaluate_checkpoint, get_device, get_dt_from_config, main

__all__ = ["get_device", "get_dt_from_config", "evaluate_checkpoint", "main"]


if __name__ == "__main__":
    main()
