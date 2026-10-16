"""Coordinate interventions on the latent support (Table 2, Figure 4).

The LISTA run on Local-linear gates, seed 0, is evaluated on 100
basin-interior starts: basin-balanced high-margin states (trajectory-fate
labels, :func:`paper.analysis.support.high_margin_slice`) that stay in their
basin for the next 21 steps, drawn from 512 trajectories of 65 states. For
each start ``z0 = E(x0)`` is either

* left unchanged (``baseline``),
* modified by zeroing its ``k`` largest-magnitude active coordinates
  (``drop_top_k``, ``k = 1..10``, up to the smallest active count over starts), or
* modified by moving its active values, shuffled, onto randomly chosen
  inactive coordinates (``random_support_r``, 20 repeats),

and then rolled out with ``D(z0 K^h)`` without re-encoding. The accumulated
error at horizon ``H`` is ``sum_{h<=H} mean_d (x_hat - x)^2``. Writes
``interventions.csv.gz`` (``condition, point_id, horizon, cumulative_mse_sum``).
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from paper.analysis import finish, pinned_threads, producer_parser
from paper.analysis.support import fate_labels_and_centers, high_margin_slice, reset_trajectories
from paper.config import SUPPORT_THRESHOLD, find_run
from paper.io import write_gzip

NUM_STARTS = 100
EVALUATION_SEED = 42
HORIZONS = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21)
MAX_DROP = 10
RANDOM_REPEATS = 20
RANDOM_SEED = 123


def candidate_trajectories(system) -> torch.Tensor:
    """``[512, 65, 2]`` candidate trajectories (reset seeds ``42 + i``)."""
    return reset_trajectories(system, EVALUATION_SEED, 512, 64)


def select_starts(labels: np.ndarray, keep: np.ndarray, max_horizon: int, num_points: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Basin-balanced starts ``[(trajectory, time)]`` that stay in their basin for ``max_horizon`` steps."""
    valid = keep.reshape(labels.shape).copy()
    valid[:, labels.shape[1] - max_horizon:] = False
    start_labels = labels[:, :-max_horizon]
    stable = np.ones_like(start_labels, dtype=bool)
    for step in range(1, max_horizon + 1):
        stable &= labels[:, step:step + start_labels.shape[1]] == start_labels
    valid[:, :-max_horizon] &= stable
    pairs = np.argwhere(valid)
    by_basin: Dict[int, List[np.ndarray]] = defaultdict(list)
    for pair in pairs:
        by_basin[int(labels[int(pair[0]), int(pair[1])])].append(pair)
    selected: List[np.ndarray] = []
    used = set()
    basins = sorted(by_basin)
    base, remainder = divmod(num_points, len(basins))
    for offset, basin in enumerate(basins):
        pool = np.asarray(by_basin[basin], dtype=np.int64)
        take = min(base + (1 if offset < remainder else 0), pool.shape[0])
        if take <= 0:
            continue
        for item in pool[np.sort(rng.choice(pool.shape[0], size=take, replace=False))]:
            used.add((int(item[0]), int(item[1])))
            selected.append(item)
    if len(selected) < num_points:  # top up from the other basins' remaining starts
        remaining = np.asarray([p for p in pairs if (int(p[0]), int(p[1])) not in used], dtype=np.int64)
        need = num_points - len(selected)
        selected.extend(remaining[np.sort(rng.choice(remaining.shape[0], size=need, replace=False))])
    chosen = np.asarray(selected[:num_points], dtype=np.int64)
    return chosen[np.lexsort((chosen[:, 1], chosen[:, 0]))]


@torch.no_grad()
def _decode_rollout(model, z0: torch.Tensor, horizon: int) -> torch.Tensor:
    """Predictions ``[starts, horizon, d]`` (CPU) of ``D(z0 K^h)``, h = 1..horizon, without re-encoding."""
    z, predictions = z0, []
    for _ in range(horizon):
        z = model.step_latent(z)
        predictions.append(model.decode(z).cpu())
    return torch.stack(predictions, dim=1)


def _drop_top(z0: torch.Tensor, order: Sequence[Sequence[int]], count: int) -> torch.Tensor:
    """``drop_top_<count>``: each code ``[starts, latent]`` with its ``count`` largest active coordinates zeroed
    (``order``: each start's active coordinates by decreasing magnitude)."""
    z = z0.clone()
    for row, coordinates in enumerate(order):
        z[row, list(coordinates[:count])] = 0.0
    return z


def _random_support(z0: torch.Tensor, active: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
    """``random_support_<r>``: each code's active values, shuffled, on randomly chosen inactive coordinates
    (on random coordinates when fewer are inactive than active); every other coordinate is zero."""
    z_np = z0.cpu().numpy()
    out = np.zeros_like(z_np)
    for row, mask in enumerate(active):
        source = np.flatnonzero(mask)
        values = z_np[row, source].copy()
        rng.shuffle(values)
        inactive = np.flatnonzero(~mask)
        if inactive.size >= source.size:
            destination = rng.choice(inactive, size=source.size, replace=False)
        else:
            destination = rng.permutation(np.arange(z_np.shape[1]))[: source.size]
        out[row, destination] = values
    return torch.from_numpy(out).to(device=z0.device, dtype=z0.dtype)


@torch.no_grad()
def run_interventions(model, system, device: str = "cpu", training_seed: int = 0) -> Dict[str, np.ndarray]:
    """Accumulated squared error per condition, ``{condition: [starts, horizons]}`` (plus the starts ``_x0``)."""
    trajectories = candidate_trajectories(system)
    labels_t, centers = fate_labels_and_centers(system, trajectories)
    labels = labels_t.cpu().numpy()
    keep = high_margin_slice(trajectories, centers, labels)
    horizon = max(HORIZONS)
    chosen = select_starts(labels, keep, horizon, NUM_STARTS, np.random.default_rng(EVALUATION_SEED + training_seed))
    x0 = trajectories[chosen[:, 0].tolist(), chosen[:, 1].tolist()]
    truth = torch.stack([trajectories[int(t), int(s) + 1:int(s) + horizon + 1] for t, s in chosen.tolist()], dim=0)
    z0 = model.encode(x0.to(device, dtype=torch.float32))
    z0_np = z0.cpu().numpy()
    active = np.abs(z0_np) > SUPPORT_THRESHOLD
    order = [np.flatnonzero(m)[np.argsort(-np.abs(r[np.flatnonzero(m)]))].tolist() for r, m in zip(z0_np, active)]

    conditions = {"baseline": z0}
    for count in range(1, min(MAX_DROP, int(active.sum(axis=1).min())) + 1):
        conditions[f"drop_top_{count}"] = _drop_top(z0, order, count)
    for repeat in range(RANDOM_REPEATS):
        rng = np.random.default_rng(RANDOM_SEED + repeat)
        conditions[f"random_support_{repeat}"] = _random_support(z0, active, rng)
    results = {}
    for name, z_init in conditions.items():
        error = ((_decode_rollout(model, z_init, horizon) - truth) ** 2).mean(dim=-1).numpy()
        results[name] = np.stack([error[:, :h].sum(axis=1) for h in HORIZONS], axis=1)
    results["_x0"] = x0.numpy()
    return results


def intervention_rows(results: Dict[str, np.ndarray]) -> List[Dict[str, object]]:
    """Long rows ``condition, point_id, horizon, cumulative_mse_sum`` of :func:`run_interventions` results."""
    return [{"condition": condition, "point_id": point, "horizon": h, "cumulative_mse_sum": float(value)}
            for condition, values in results.items() if not condition.startswith("_")
            for point, per_horizon in enumerate(values) for h, value in zip(HORIZONS, per_horizon)]


def main(argv: Optional[Sequence[str]] = None) -> int:
    from paper.evaluate import load_run_model

    args = producer_parser("interventions", __doc__).parse_args(argv)
    directory = find_run(args.runs, "multibasin", "gated_local_linear", "lista", 0, args.include_short)
    path = args.evidence / "interventions.csv.gz"
    if not (directory / "final.pt").exists():
        return finish("interventions (the LISTA run on gated_local_linear, seed 0)", 0, 1, args.allow_missing_runs,
                      path)
    with pinned_threads():
        run = load_run_model(directory, args.device)
        results = run_interventions(run.model, run.system, device=args.device, training_seed=run.cfg.seed)
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=["condition", "point_id", "horizon", "cumulative_mse_sum"],
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(intervention_rows(results))
    write_gzip(path, text.getvalue())
    return finish("interventions", 1, 1, args.allow_missing_runs, path)
