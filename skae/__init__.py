"""Sparse Koopman autoencoders in PyTorch.

    from skae import KoopmanAutoencoder, ModelConfig, fit, trajectory_windows

    windows = trajectory_windows(trajectories, window_length=8)  # [N, T, D] -> [W, 9, D]
    model = KoopmanAutoencoder(ModelConfig(), obs_dim=trajectories.shape[-1])
    fit(model, windows)
    forecast = model.forecast(x0, horizon=100)                      # [B, 100, D]
    forecast = model.forecast(x0, horizon=100, reencode_every=10)   # re-encode every 10 steps
"""

__version__ = "1.0.0"

from skae.config import ModelConfig

__all__ = ["KoopmanAutoencoder", "ModelConfig", "SupportLossConfig", "WindowSimilarity", "fit", "training_loss",
           "trajectory_windows"]


def __getattr__(name: str):
    """Import the torch modules on first use, so that ``skae.config`` loads without torch."""
    if name in ("KoopmanAutoencoder", "training_loss"):
        from skae import models

        return getattr(models, name)
    if name in ("fit", "trajectory_windows"):
        from skae import training

        return getattr(training, name)
    if name in ("SupportLossConfig", "WindowSimilarity"):
        from skae import losses

        return getattr(losses, name)
    raise AttributeError(f"module 'skae' has no attribute {name!r}")
