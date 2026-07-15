"""Compatibility entry point for :mod:`skae.training.plotting`."""

from skae.training.plotting import load_metrics_history, main, plot_metrics

__all__ = ["load_metrics_history", "plot_metrics", "main"]


if __name__ == "__main__":
    main()
