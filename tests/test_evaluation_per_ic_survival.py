import torch

from skae.evaluation import (
    _select_best_full_horizon_mode,
    rollout_no_reencode,
    rollout_periodic_reencode,
)


class _SelectiveExplosionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def encode(self, x):
        return x

    def encode_with_prior(self, x, latent_prior=None):
        return x

    def step_latent(self, z):
        return torch.where(z > 10.0, torch.full_like(z, torch.inf), z + 1.0)

    def decode(self, z):
        return z


def test_direct_rollout_keeps_finite_initial_conditions_alive():
    model = _SelectiveExplosionModel()
    prediction = rollout_no_reencode(model, torch.tensor([[0.0], [20.0]]), 4)
    assert torch.isfinite(prediction[:, 0]).all()
    assert torch.isnan(prediction[:, 1]).all()


def test_periodic_rollout_keeps_finite_initial_conditions_alive():
    model = _SelectiveExplosionModel()
    prediction = rollout_periodic_reencode(
        model, torch.tensor([[0.0], [20.0]]), 4, period=2
    )
    assert torch.isfinite(prediction[:, 0]).all()
    assert torch.isnan(prediction[:, 1]).all()


def test_best_mode_prefers_survival_over_low_finite_prefix_error():
    modes = {
        "periodic_explodes": {
            "horizons": {
                "100": {
                    "mean": 1e-9,
                    "strict_full_horizon_mean": 1e-9,
                    "full_horizon_finite_fraction": 0.5,
                }
            }
        },
        "periodic_stable": {
            "horizons": {
                "100": {
                    "mean": 1.0,
                    "strict_full_horizon_mean": 1.0,
                    "full_horizon_finite_fraction": 1.0,
                }
            }
        },
    }
    selected = _select_best_full_horizon_mode(
        modes,
        "100",
        include_mode=lambda name: name.startswith("periodic_"),
    )
    assert selected == "periodic_stable"
