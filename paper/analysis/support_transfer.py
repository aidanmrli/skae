"""Validation-fit, test-transfer alignment of support families with basins.

For each multibasin run (system, model, training seed ``s``):

1. Draw 64 validation trajectories (reset seeds ``s + 3,000,000 + i``) and 64
   test trajectories (``s + 4,000,000 + i``) of 128 transitions (8,256 test
   states). Basin centers are fitted on the validation trajectories only;
   every trajectory is labelled by the center nearest its own limit point
   (:mod:`paper.analysis.support`).
2. High-margin subset: the per-label 75th percentile of the own-basin margin
   over the validation states is applied unchanged to the test states; a test
   state must also be nearer its own center than any other.
3. Supports ``|z_i| > a`` of the validation states are merged into families
   (Jaccard >= ``j``); each test support joins its most similar validation
   prototype, or stays unmatched (family ``-1``, kept in every metric).
   Primary cell: ``a = 1e-3``, ``j = 0.5``.
4. Score the test states of each subset: ``all``, ``high_margin`` and
   ``high_margin_dedup`` (high-margin states with exact duplicates removed). A
   cell is *informative* if at least two basins are observed; an *exact
   bijection* needs every state matched, as many families as basins and
   ``H(B|F) = H(F|B) = 0``.

Writes ``support_transfer.csv``: one row per run and subset (primary cell);
:mod:`paper.analysis.support_sensitivity` repeats the analysis on a grid of
``(a, j)`` and summarizes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from paper.analysis import finish, pinned_threads, producer_parser
from paper.analysis.support import (
    center_margins,
    conditional_entropy,
    entropy,
    fate_centers,
    fate_labels,
    fit_margin_cutoffs,
    fit_support_families,
    high_margin_mask,
    limit_points,
    reset_trajectories,
    transfer_support_families,
)
from paper.config import (ANALYSIS_TEST_SEED_OFFSET, ANALYSIS_VALIDATION_SEED_OFFSET, MODELS, SEEDS, SUPPORT_THRESHOLD,
                          find_run, systems_for)
from paper.evaluate import load_run_model
from paper.io import write_csv

TRAJECTORIES = 64
TRANSITIONS = 128
PRIMARY_ABSOLUTE_THRESHOLD = SUPPORT_THRESHOLD
PRIMARY_JACCARD_THRESHOLD = 0.5
CHECKPOINT = "final.pt"
_ZERO = 1e-10  # |H| below this counts as zero (the entropies carry a 1e-12 log offset)


def split_trajectories(system, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validation and test trajectories ``[64, 129, 2]`` for training seed ``seed``."""
    return (reset_trajectories(system, seed + ANALYSIS_VALIDATION_SEED_OFFSET, TRAJECTORIES, TRANSITIONS),
            reset_trajectories(system, seed + ANALYSIS_TEST_SEED_OFFSET, TRAJECTORIES, TRANSITIONS))


def fit_labels(system, validation: torch.Tensor, test: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """``(validation_labels, test_labels, centers)``, the centers fitted on the validation trajectories."""
    validation_points, test_points = limit_points(system, validation), limit_points(system, test)
    centers = fate_centers(system, validation_points)
    return (fate_labels(validation, validation_points, centers).numpy(),
            fate_labels(test, test_points, centers).numpy(), centers.to(dtype=validation.dtype))


def first_occurrences(states: torch.Tensor, mask: np.ndarray) -> np.ndarray:
    """``mask`` restricted to the first occurrence of every distinct state value."""
    flat = np.ascontiguousarray(states.reshape(-1, states.shape[-1]).cpu().numpy())
    index = np.flatnonzero(mask)
    keep = np.zeros(mask.size, dtype=bool)
    if index.size:
        _, first = np.unique(flat[index], axis=0, return_index=True)
        keep[index[first]] = True
    return keep


@dataclass
class BasinReference:
    """Model-independent labels of one (system, seed), shared by every model."""

    validation: torch.Tensor          # [64, 129, 2]
    test: torch.Tensor                # [64, 129, 2]
    num_centers: int
    test_labels: np.ndarray           # [64, 129]
    subsets: Dict[str, np.ndarray]    # name -> [8256] mask of scored test states
    distinct: Dict[str, Tuple[int, str]]  # name -> (distinct states, "basin:count;...")


def basin_reference(system, seed: int) -> BasinReference:
    """The basin labels and scored state subsets of training seed ``seed`` (steps 1-2 of the module docstring)."""
    validation, test = split_trajectories(system, seed)
    validation_labels, test_labels, centers = fit_labels(system, validation, test)
    cutoffs = fit_margin_cutoffs(center_margins(validation, centers, validation_labels), validation_labels)
    high_margin = high_margin_mask(center_margins(test, centers, test_labels), test_labels, cutoffs)
    subsets = {"all": np.ones(test_labels.size, dtype=bool), "high_margin": high_margin,
               "high_margin_dedup": first_occurrences(test, high_margin)}
    distinct = {}
    for name, mask in subsets.items():
        unique = first_occurrences(test, mask)
        basins, counts = np.unique(test_labels.reshape(-1)[unique], return_counts=True)
        distinct[name] = (int(unique.sum()), ";".join(f"{b}:{c}" for b, c in zip(basins.tolist(), counts.tolist())))
    return BasinReference(validation, test, int(centers.shape[0]), test_labels, subsets, distinct)


def score_alignment(basin_labels: np.ndarray, family_labels: np.ndarray, mask: np.ndarray) -> Dict[str, object]:
    """Scores of the test states selected by ``mask``; unmatched states (family ``-1``) count as one family."""
    if not np.any(mask):
        raise ValueError("no test states to score: the state subset is empty")
    basins = np.asarray(basin_labels).reshape(-1)[mask].astype(np.int64)
    families = np.asarray(family_labels).reshape(-1)[mask].astype(np.int64)
    basin_list, family_list = basins.tolist(), families.tolist()
    h_basin, h_family = entropy(basin_list), entropy(family_list)
    h_basin_given_family = conditional_entropy(basin_list, family_list)
    h_family_given_basin = conditional_entropy(family_list, basin_list)
    coverage = float(np.mean(families >= 0))
    observed_families = int(np.unique(families[families >= 0]).size)
    observed_basins = int(np.unique(basins).size)
    exact_count = observed_families == observed_basins
    purity = sum(int(np.bincount(basins[families == f] - basins.min()).max()) for f in np.unique(families))
    return {
        "num_states": int(basins.size),
        "coverage": coverage,
        "unmatched_states": int(np.sum(families < 0)),
        "observed_family_count": observed_families,
        "observed_basin_count": observed_basins,
        "informative": observed_basins >= 2,
        "single_basin": observed_basins < 2,
        "exact_count_match": exact_count,
        "exact_bijection": bool(coverage == 1.0 and exact_count and abs(h_basin_given_family) < _ZERO
                                and abs(h_family_given_basin) < _ZERO),
        "h_basin": h_basin,
        "h_family": h_family,
        "h_basin_given_family": h_basin_given_family,
        "h_family_given_basin": h_family_given_basin,
        # One basin (H(B) = 0) or one family (H(F) = 0): the ratio is undefined.
        "normalized_h_basin_given_family": h_basin_given_family / h_basin if observed_basins >= 2 else None,
        "normalized_h_family_given_basin": (h_family_given_basin / h_family
                                            if np.unique(families).size >= 2 else None),
        "purity": purity / basins.size,
    }


@torch.no_grad()
def encode(model, trajectories: torch.Tensor, device: str = "cpu") -> np.ndarray:
    """Codes ``[N, T, latent]`` (NumPy) of trajectories ``[N, T, d]``."""
    flat = trajectories.reshape(-1, trajectories.shape[-1]).to(device)
    return model.encode(flat).reshape(*trajectories.shape[:2], -1).cpu().numpy()


def alignment_rows(reference: BasinReference, validation_latents: np.ndarray, test_latents: np.ndarray,
                   absolute_thresholds: Sequence[float], jaccard_thresholds: Sequence[float]) -> List[Dict]:
    """One row per threshold cell and subset: labelling, cell, subset and score columns."""
    rows = []
    for absolute in absolute_thresholds:
        validation_mask, test_mask = np.abs(validation_latents) > absolute, np.abs(test_latents) > absolute
        for jaccard in jaccard_thresholds:
            prototypes, _ = fit_support_families(validation_mask, jaccard)
            families = transfer_support_families(test_mask, prototypes, jaccard)
            for subset, mask in reference.subsets.items():
                scores = score_alignment(reference.test_labels, families, mask)
                rows.append({
                    "labels_mode": "trajectory_fate", "label_source": "trajectory_fate", "margin_rule": "own_basin",
                    "num_centers": reference.num_centers, "absolute_threshold": float(absolute),
                    "jaccard_threshold": float(jaccard),
                    "primary": absolute == PRIMARY_ABSOLUTE_THRESHOLD and jaccard == PRIMARY_JACCARD_THRESHOLD,
                    "validation_family_count": int(prototypes.shape[0]), "subset": subset,
                    "num_test_states": int(mask.size), "num_states": scores["num_states"],
                    "retained_fraction": scores["num_states"] / mask.size,
                    "distinct_states": reference.distinct[subset][0],
                    "distinct_states_by_basin": reference.distinct[subset][1], **scores,
                })
    return rows


def evaluate(runs: Path, absolute_thresholds: Sequence[float], jaccard_thresholds: Sequence[float],
             device: str = "cpu", include_short: bool = False, models: Sequence[str] = MODELS) -> List[Dict]:
    """Rows of every trained multibasin run of ``models`` (one basin reference per system and seed)."""
    rows = []
    for system_name in systems_for("multibasin"):
        for seed in SEEDS:
            reference = None
            for model_name in models:
                directory = find_run(runs, "multibasin", system_name, model_name, seed, include_short)
                if not (directory / CHECKPOINT).exists():
                    continue
                run = load_run_model(directory, device)
                reference = reference or basin_reference(run.system, seed)
                identity = {"model": model_name, "system": system_name, "seed": seed, "run_id": run.cfg.run_name,
                            "checkpoint": CHECKPOINT, "checkpoint_step": run.step}
                cells = alignment_rows(reference, encode(run.model, reference.validation, device),
                                       encode(run.model, reference.test, device), absolute_thresholds,
                                       jaccard_thresholds)
                rows += [{**identity, **row} for row in cells]
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = producer_parser("support-transfer", __doc__).parse_args(argv)
    with pinned_threads():
        rows = evaluate(args.runs, (PRIMARY_ABSOLUTE_THRESHOLD,), (PRIMARY_JACCARD_THRESHOLD,), args.device,
                        args.include_short)
    path = args.evidence / "support_transfer.csv"
    if rows:
        write_csv(path, rows)
    used = len({(row["model"], row["system"], row["seed"]) for row in rows})
    return finish("support-transfer", used, len(systems_for("multibasin")) * len(MODELS) * len(SEEDS),
                  args.allow_missing_runs, path)
