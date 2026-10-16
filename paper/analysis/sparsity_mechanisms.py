"""Where the encoder's zeros are (multibasin): inactive codes and hidden-activation zeros.

Five encoders, from dense to sparse codes: Dense MLP (tanh hidden layers),
Dense MLP with ReLU hidden layers, Sparse MLP without L1, Sparse MLP and LISTA.
Each run is measured on held-out trajectories (64 uniform resets with seeds
``seed + 4,000,000 + i``, 128 transitions each, 8,256 states):

* ``encoded``: the fraction of exactly-zero coordinates of the codes ``E(x)``;
* ``rollout``: the fraction of coordinates with ``|z| <= 1e-3`` in
  ``E(x_0) K^h``, h = 1..8, from each trajectory's first state (no re-encoding);
* hidden zeros: the fraction of exactly-zero activations after each hidden
  layer of the encoder MLP (LISTA: of its pre-code MLP) on the same encoded
  states, per layer and pooled over both (``layerall``).

Non-finite code coordinates count as active. Each number is the mean over
systems of per-system seed means. Writes ``sparsity_mechanisms.json``, with
the number of runs measured per model.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from paper.analysis import finish, pinned_threads, producer_parser
from paper.analysis.support import reset_trajectories
from paper.config import ANALYSIS_TEST_SEED_OFFSET, SEEDS, SUPPORT_THRESHOLD, find_run, systems_for
from paper.io import write_json

MODELS = ("dense_mlp", "dense_mlp_relu", "sparse_mlp_l1_0", "sparse_mlp", "lista")
ROLLOUT_HORIZON = 8


def held_out_trajectories(system, seed: int) -> torch.Tensor:
    """Held-out trajectories ``[64, 129, d]`` for the run seed ``seed``."""
    return reset_trajectories(system, seed + ANALYSIS_TEST_SEED_OFFSET, 64, 128)


def inactive_fraction(codes: torch.Tensor, threshold: float) -> float:
    """Fraction of finite coordinates with ``|z| <= threshold`` (``threshold = 0``: exactly zero)."""
    magnitude = codes.abs().cpu().numpy()
    inactive = np.isfinite(magnitude) & ((magnitude == 0.0) if threshold == 0.0 else (magnitude <= threshold))
    return float(inactive.mean())


def hidden_activations(model) -> List[torch.nn.Module]:
    """The activation module after each hidden layer of the encoder MLP (LISTA: of its pre-code MLP)."""
    network = getattr(model.encoder, "precode", model.encoder).network
    return [network[2 * k + 1] for k in range(len(model.cfg.hidden))]


@torch.no_grad()
def measure_run(model, trajectories: torch.Tensor) -> Dict[str, float]:
    """Inactive code fractions and per-layer zero counts of the hidden activations of one run."""
    model.eval()
    captured: Dict[int, torch.Tensor] = {}
    handles = [module.register_forward_hook(lambda _m, _i, out, k=k: captured.__setitem__(k, out.detach().clone()))
               for k, module in enumerate(hidden_activations(model))]
    try:
        encoded = model.encode(trajectories.reshape(-1, trajectories.shape[-1]))
    finally:
        for handle in handles:
            handle.remove()
    rollout = model.rollout_latent(model.encode(trajectories[:, 0]), ROLLOUT_HORIZON)
    measured = {"encoded": inactive_fraction(encoded, 0.0), "rollout": inactive_fraction(rollout, SUPPORT_THRESHOLD)}
    for k, activations in captured.items():
        measured[f"layer{k + 1}_zeros"] = int((activations == 0.0).sum())
        measured[f"layer{k + 1}_size"] = activations.numel()
    return measured


def run_measurements(runs, device: str = "cpu", include_short: bool = False) -> List[Dict[str, object]]:
    """One row per trained run of the five encoders (systems, seeds and models in roster order)."""
    from paper.evaluate import load_run_model

    rows = []
    for system in systems_for("multibasin"):
        for seed in SEEDS:
            trajectories = None
            for name in MODELS:
                directory = find_run(runs, "multibasin", system, name, seed, include_short)
                if not (directory / "final.pt").exists():
                    continue
                run = load_run_model(directory, device)
                if trajectories is None:
                    trajectories = held_out_trajectories(run.system, seed).to(device)
                rows.append({"model": name, "system": system, "seed": seed, **measure_run(run.model, trajectories)})
    return rows


def _mean_of_system_means(rows: Sequence[Dict[str, object]], value) -> float:
    """Mean over systems of the per-system mean over seeds (both in row order); NaN without rows."""
    by_system = defaultdict(list)
    for row in rows:
        by_system[row["system"]].append(value(row))
    return float(np.mean([np.mean(v) for v in by_system.values()])) if by_system else float("nan")


def summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    """The evidence: ``{model}/{encoded,rollout}`` inactivity, ``{model}/layer{1,2,all}`` hidden zeros and the
    number of runs of each model."""
    inactive, hidden = {}, {}
    for model in MODELS:
        mine = [row for row in rows if row["model"] == model]
        for target in ("encoded", "rollout"):
            inactive[f"{model}/{target}"] = _mean_of_system_means(mine, lambda row: row[target])
        for layer in ("1", "2"):
            hidden[f"{model}/layer{layer}"] = _mean_of_system_means(
                mine, lambda row: row[f"layer{layer}_zeros"] / row[f"layer{layer}_size"])
        hidden[f"{model}/layerall"] = _mean_of_system_means(
            mine, lambda row: (row["layer1_zeros"] + row["layer2_zeros"]) / (row["layer1_size"] + row["layer2_size"]))
    return {"inactive_mean_of_system_means": inactive, "hidden_zero_mean_of_system_means": hidden,
            "runs": {model: sum(row["model"] == model for row in rows) for model in MODELS}}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = producer_parser("sparsity-mechanisms", __doc__).parse_args(argv)
    with pinned_threads():
        rows = run_measurements(args.runs, args.device, args.include_short)
    path = args.evidence / "sparsity_mechanisms.json"
    if rows:
        write_json(path, summarize(rows), sort_keys=True)
    return finish("sparsity-mechanisms", len(rows), len(MODELS) * len(systems_for("multibasin")) * len(SEEDS),
                  args.allow_missing_runs, path)
