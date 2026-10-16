"""Fit a sparse Koopman autoencoder to your own trajectories.

    uv run python examples/custom_data.py

Replace ``load_trajectories`` with your data: a float tensor ``[N, T, D]`` of
``N`` trajectories, each with ``T`` equally spaced states of ``D`` coordinates.
"""

import torch

from skae import KoopmanAutoencoder, ModelConfig, fit, trajectory_windows


def load_trajectories() -> torch.Tensor:
    """A damped double-well oscillator, ``[64, 200, 2]``: each trajectory settles into one of two wells."""
    state = 4 * torch.rand(64, 2, generator=torch.Generator().manual_seed(0)) - 2  # (position, velocity)
    states = [state]
    for _ in range(199):
        x, v = state.unbind(-1)
        state = torch.stack([x + 0.05 * v, v + 0.05 * (x - x**3 - 0.5 * v)], dim=-1)
        states.append(state)
    return torch.stack(states, dim=1)


def main() -> None:
    trajectories = load_trajectories()
    train, test = trajectories[:48], trajectories[48:]

    # The paper's multibasin LISTA model, with the observation mean and scale stored in the model
    # (computed on the training trajectories only); encode and decode use the original coordinates.
    torch.manual_seed(0)
    model = KoopmanAutoencoder(ModelConfig(input_normalization=True), obs_dim=train.shape[-1])
    model.set_input_normalization(train.mean(dim=(0, 1)), train.std(dim=(0, 1)).clamp_min(1e-8))
    windows = trajectory_windows(train, window_length=8)  # [W, 9, D]
    history = fit(model, windows, epochs=20, batch_size=256)
    print(f"training loss: {history[0]:.4f} (epoch 1) -> {history[-1]:.4f} (epoch {len(history)})")

    model.eval()
    with torch.no_grad():
        codes = model.encode(test)  # [N, T, latent_dim]
        forecast = model.forecast(test[:, 0], horizon=25)  # [N, 25, D]; reencode_every=m re-encodes every m steps
    rmse = (forecast - test[:, 1:26]).square().mean().sqrt()
    active = (codes.abs() > 1e-3).sum(-1).float().mean()
    print(f"test forecast RMSE over 25 steps: {rmse:.3f} (coordinates have std {train.std():.2f})")
    print(f"active code coordinates per state: {active:.1f} of {model.latent_dim}")


if __name__ == "__main__":
    main()
