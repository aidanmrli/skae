#!/usr/bin/env python3
"""Held-out state-space identification of support-selected laws in one global K.

This evaluator deliberately does not fit a latent dynamics operator.  It masks
coordinates discovered without basin labels, applies the checkpoint's unchanged
global Koopman matrix, decodes the resulting K-minus-I update, and asks which of
the three known GatedLocalLinear laws the held-out state-space slope identifies.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from experiments.neurips_2026.global_k_dense_checkpoint_audit import (
    assert_exact_dense_control,
)
from experiments.neurips_2026.global_k_dense_specificity import (
    _load_checkpoint,
    assign_dense_families,
    discover_dense_roster,
    fit_dense_family_codebook,
    matched_topk_masks,
)
from experiments.neurips_2026.global_k_support_invariance import (
    FamilyCodebook,
    RunSpec,
    _encode,
    assert_sign_split_layout,
    assign_families,
    discover_primary_roster,
    fit_family_codebook,
    load_card as load_sparse_card,
    sha256_path,
    sign_pair_permutations,
)
from skae.data import VectorWrapper


DEFAULT_CARD = Path(__file__).with_name("global_k_distinct_laws_card.json")
EPS = 1e-12


@dataclass(frozen=True)
class LocalFit:
    matrix: np.ndarray
    intercept: np.ndarray
    relative_residual: float
    update_rms: float


def _hash_text(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol_card(path: Path = DEFAULT_CARD) -> tuple[dict[str, Any], str]:
    card = json.loads(path.read_text())
    if card.get("protocol_id") != "global_k_distinct_laws_gated_local_linear_v1":
        raise RuntimeError("Unexpected distinct-law protocol ID")
    for key in ("sparse", "dense"):
        source = Path(card["frozen_sources"][f"{key}_card"])
        expected = card["frozen_sources"][f"{key}_card_sha256"]
        actual = _hash_text(source)
        if actual != expected:
            raise RuntimeError(f"{key} card hash drift: {actual} != {expected}")
    return card, _hash_text(path)


def sample_centered_disk(count: int, radius: float, seed: int) -> np.ndarray:
    """Draw deterministic area-uniform 2D offsets, excluding a zero-radius draw."""
    if count < 3 or radius <= 0:
        raise ValueError((count, radius))
    rng = np.random.default_rng(seed)
    radial = radius * np.sqrt(np.maximum(rng.random(count), np.finfo(float).eps))
    angle = 2.0 * np.pi * rng.random(count)
    return np.column_stack((radial * np.cos(angle), radial * np.sin(angle))).astype(
        np.float32
    )


def rk4_step_matrix(matrix: np.ndarray, dt: float) -> np.ndarray:
    """Classical RK4 polynomial for a constant column-vector linear system."""
    matrix = np.asarray(matrix, dtype=np.float64)
    identity = np.eye(matrix.shape[0], dtype=np.float64)
    scaled = float(dt) * matrix
    square = scaled @ scaled
    cube = square @ scaled
    fourth = cube @ scaled
    return identity + scaled + square / 2.0 + cube / 6.0 + fourth / 24.0


def fit_centered_local_matrix(offsets: np.ndarray, updates: np.ndarray) -> LocalFit:
    """Fit update = intercept + offset @ B and return column-vector matrix B.T."""
    offsets = np.asarray(offsets, dtype=np.float64)
    updates = np.asarray(updates, dtype=np.float64)
    if offsets.ndim != 2 or updates.shape != offsets.shape:
        raise ValueError((offsets.shape, updates.shape))
    design = np.column_stack((np.ones(offsets.shape[0]), offsets))
    coefficients, *_ = np.linalg.lstsq(design, updates, rcond=None)
    fitted = design @ coefficients
    residual = math.sqrt(float(np.sum((updates - fitted) ** 2)))
    energy = math.sqrt(float(np.sum(updates**2)))
    return LocalFit(
        matrix=coefficients[1:].T,
        intercept=coefficients[0],
        relative_residual=residual / max(energy, EPS),
        update_rms=energy / math.sqrt(max(1, updates.shape[0])),
    )


def match_families_to_basins(
    assignments: np.ndarray,
    retained: np.ndarray,
    basin_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluation-only one-to-one matching; discovery itself remains label-free."""
    assignments = np.asarray(assignments, dtype=np.int64)
    if assignments.ndim != 2 or assignments.shape[0] != basin_count:
        raise ValueError(assignments.shape)
    retained_ids = np.flatnonzero(np.asarray(retained, dtype=bool))
    if retained_ids.size < basin_count:
        return (
            np.full(basin_count, -1, dtype=np.int64),
            np.zeros(basin_count, dtype=np.float64),
            np.zeros((basin_count, retained_ids.size), dtype=np.int64),
        )
    counts = np.zeros((basin_count, retained_ids.size), dtype=np.int64)
    for basin in range(basin_count):
        for column, family in enumerate(retained_ids):
            counts[basin, column] = int(np.sum(assignments[basin] == family))
    rows, columns = linear_sum_assignment(-counts)
    mapping = np.full(basin_count, -1, dtype=np.int64)
    rates = np.zeros(basin_count, dtype=np.float64)
    for basin, column in zip(rows, columns):
        mapping[basin] = int(retained_ids[column])
        rates[basin] = float(counts[basin, column] / assignments.shape[1])
    return mapping, rates, counts


def law_cost_summary(
    predicted: np.ndarray, true_matrices: np.ndarray,
) -> dict[str, Any]:
    """Score all assignments without changing any predicted source chart."""
    predicted = np.asarray(predicted, dtype=np.float64)
    true_matrices = np.asarray(true_matrices, dtype=np.float64)
    if predicted.shape != true_matrices.shape or predicted.shape[0] != 3:
        raise ValueError((predicted.shape, true_matrices.shape))
    costs = np.empty((3, 3), dtype=np.float64)
    for basin in range(3):
        for law in range(3):
            costs[basin, law] = np.linalg.norm(
                predicted[basin] - true_matrices[law], ord="fro"
            )
    row, col = linear_sum_assignment(costs)
    assignment = np.full(3, -1, dtype=np.int64)
    assignment[row] = col
    permutations = list(itertools.permutations(range(3)))
    assignment_costs = {
        "".join(str(value) for value in perm): float(
            sum(costs[basin, perm[basin]] for basin in range(3))
        )
        for perm in permutations
    }
    identity_cost = assignment_costs["012"]
    best_nonidentity = min(
        value for key, value in assignment_costs.items() if key != "012"
    )
    row_ratios = []
    own_relative = []
    for basin in range(3):
        nearest_wrong = min(costs[basin, law] for law in range(3) if law != basin)
        row_ratios.append(float(costs[basin, basin] / max(nearest_wrong, EPS)))
        own_relative.append(
            float(
                costs[basin, basin]
                / max(np.linalg.norm(true_matrices[basin], ord="fro"), EPS)
            )
        )
    return {
        "cost_matrix": costs.tolist(),
        "optimal_assignment": assignment.tolist(),
        "identity_is_unique_optimum": bool(
            np.array_equal(assignment, np.arange(3))
            and identity_cost + 1e-12 < best_nonidentity
        ),
        "assignment_costs": assignment_costs,
        "identity_over_best_nonidentity": float(
            identity_cost / max(best_nonidentity, EPS)
        ),
        "own_over_nearest_wrong_by_basin": row_ratios,
        "own_relative_error_by_basin": own_relative,
        "max_own_over_nearest_wrong": float(max(row_ratios)),
        "max_own_relative_error": float(max(own_relative)),
    }


def _encode_points(model, points: torch.Tensor, device: str, batch_size: int) -> np.ndarray:
    chunks = []
    with torch.no_grad():
        for start in range(0, points.shape[0], batch_size):
            chunks.append(model.encode(points[start : start + batch_size].to(device)).cpu())
    return torch.cat(chunks).numpy().astype(np.float32)


def _decoded_updates(
    model,
    latent: np.ndarray,
    mask: np.ndarray | None,
    device: str,
    mode: str,
    permutation: np.ndarray | None = None,
) -> np.ndarray:
    z = torch.from_numpy(np.asarray(latent, dtype=np.float32)).to(device)
    k_matrix = model.kmatrix().detach()
    if permutation is not None:
        z = z[:, torch.as_tensor(permutation, device=z.device)]
    with torch.no_grad():
        if mode == "global":
            stepped, identity = z @ k_matrix, z
        else:
            if mask is None:
                raise ValueError("A mask is required outside global mode")
            effective_mask = np.asarray(mask, dtype=bool)
            if permutation is not None:
                effective_mask = effective_mask[permutation]
            p = torch.from_numpy(effective_mask.astype(np.float32)).to(device)
            source = z * p
            stepped = source @ k_matrix
            if mode == "block":
                stepped = stepped * p
            elif mode != "source":
                raise ValueError(mode)
            identity = source
        update = model.decode(stepped) - model.decode(identity)
    return update.cpu().numpy().astype(np.float64)


def _family_assigner(
    arm: str,
    codebook: FamilyCodebook,
    threshold: float,
    jaccard: float,
) -> Callable[[np.ndarray, np.ndarray | None], np.ndarray]:
    if arm == "sparse":
        return lambda latent, _paired: assign_families(
            np.abs(latent) > threshold, codebook, jaccard
        )
    return lambda latent, paired: assign_dense_families(
        matched_topk_masks(latent, paired, threshold), codebook, jaccard
    )


def _score_arm(
    *,
    arm: str,
    model,
    sparse_model,
    env,
    codebook: FamilyCodebook,
    retained: np.ndarray,
    card: dict[str, Any],
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    geometry = card["evaluation_geometry"]
    radius = float(geometry["disk_radius"])
    calibration_offsets = sample_centered_disk(
        int(geometry["calibration_points_per_basin"]),
        radius,
        int(geometry["calibration_seed"]),
    )
    score_offsets = sample_centered_disk(
        int(geometry["score_points_per_basin"]),
        radius,
        int(geometry["score_seed"]),
    )
    centers = env.unwrapped.points_2d.detach().cpu().numpy().astype(np.float32)
    if centers.shape != (3, 2):
        raise AssertionError(centers.shape)
    calibration = centers[:, None, :] + calibration_offsets[None, :, :]
    score = centers[:, None, :] + score_offsets[None, :, :]
    calibration_flat = torch.from_numpy(calibration.reshape(-1, 2))
    score_flat = torch.from_numpy(score.reshape(-1, 2))
    calibration_latent = _encode_points(model, calibration_flat, device, batch_size)
    score_latent = _encode_points(model, score_flat, device, batch_size)
    paired_calibration = None
    paired_score = None
    if arm == "dense":
        paired_calibration = _encode_points(
            sparse_model, calibration_flat, device, batch_size
        )
        paired_score = _encode_points(sparse_model, score_flat, device, batch_size)
    threshold = float(card["family_discovery"]["sparse_support"].split(">")[-1])
    jaccard = float(card["family_discovery"]["jaccard_threshold"])
    assign = _family_assigner(arm, codebook, threshold, jaccard)
    calibration_assignment = assign(calibration_latent, paired_calibration).reshape(3, -1)
    mapping, calibration_rates, count_matrix = match_families_to_basins(
        calibration_assignment, retained, 3
    )
    score_assignment = assign(score_latent, paired_score).reshape(3, -1)
    score_rates = np.asarray(
        [np.mean(score_assignment[basin] == mapping[basin]) for basin in range(3)],
        dtype=np.float64,
    )
    min_calibration = float(geometry["min_calibration_match_rate_per_basin"])
    min_score = float(geometry["min_score_match_rate_per_basin"])
    family_valid = bool(
        np.all(mapping >= 0)
        and len(set(mapping.tolist())) == 3
        and np.all(calibration_rates >= min_calibration)
        and np.all(score_rates >= min_score)
    )
    routing = {
        "retained_family_count": int(np.sum(retained)),
        "retained_family_ids": np.flatnonzero(retained).tolist(),
        "matched_family_by_basin": mapping.tolist(),
        "calibration_match_rate_by_basin": calibration_rates.tolist(),
        "score_match_rate_by_basin": score_rates.tolist(),
        "calibration_count_matrix": count_matrix.tolist(),
        "family_valid": family_valid,
    }
    if not family_valid:
        return {"status": "ineligible_family_mapping", "routing": routing}

    with torch.no_grad():
        region = env.unwrapped.region_label(score_flat).cpu().numpy().reshape(3, -1)
        true_next = env.step(score_flat).cpu().numpy().reshape(3, -1, 2)
    if not all(np.all(region[basin] == basin) for basin in range(3)):
        raise AssertionError("Score disk leaves a basin-local region")
    basin_matrices = env.unwrapped.basin_matrices.detach().cpu().numpy()
    step_matrices = np.stack(
        [rk4_step_matrix(matrix, float(env.unwrapped.dt)) for matrix in basin_matrices]
    )
    true_change = step_matrices - np.eye(2, dtype=np.float64)
    analytic_next = np.stack(
        [
            centers[basin]
            + score_offsets.astype(np.float64) @ step_matrices[basin].T
            for basin in range(3)
        ]
    )
    analytic_disagreement = float(np.max(np.abs(analytic_next - true_next)))

    latent_by_basin = score_latent.reshape(3, -1, score_latent.shape[-1])
    block_fits: list[LocalFit] = []
    source_fits: list[LocalFit] = []
    global_fits: list[LocalFit] = []
    anchor_ratios = []
    block_source_discrepancy = []
    true_update_rms = []
    for basin in range(3):
        mask = codebook.representatives[mapping[basin]]
        latent = latent_by_basin[basin]
        block = _decoded_updates(model, latent, mask, device, "block")
        source = _decoded_updates(model, latent, mask, device, "source")
        global_update = _decoded_updates(model, latent, None, device, "global")
        block_fit = fit_centered_local_matrix(score_offsets, block)
        source_fit = fit_centered_local_matrix(score_offsets, source)
        global_fit = fit_centered_local_matrix(score_offsets, global_update)
        block_fits.append(block_fit)
        source_fits.append(source_fit)
        global_fits.append(global_fit)
        true_updates = score_offsets.astype(np.float64) @ true_change[basin].T
        true_rms = math.sqrt(float(np.mean(np.sum(true_updates**2, axis=1))))
        true_update_rms.append(true_rms)
        anchor_ratios.append(
            float(np.linalg.norm(block_fit.intercept) / max(true_rms, EPS))
        )
        block_source_discrepancy.append(
            float(
                np.linalg.norm(block_fit.matrix - source_fit.matrix, ord="fro")
                / max(np.linalg.norm(source_fit.matrix, ord="fro"), EPS)
            )
        )

    block_matrices = np.stack([fit.matrix for fit in block_fits])
    source_matrices = np.stack([fit.matrix for fit in source_fits])
    global_matrices = np.stack([fit.matrix for fit in global_fits])
    block_laws = law_cost_summary(block_matrices, true_change)
    source_laws = law_cost_summary(source_matrices, true_change)
    global_laws = law_cost_summary(global_matrices, true_change)

    null_identity_ratios = []
    base_dim = score_latent.shape[-1] // 2
    null = card["controls"]["coordinate_null"]
    for permutation in sign_pair_permutations(
        base_dim, int(null["replicates"]), int(null["seed"])
    ):
        null_matrices = []
        for basin in range(3):
            mask = codebook.representatives[mapping[basin]]
            update = _decoded_updates(
                model,
                latent_by_basin[basin],
                mask,
                device,
                "block",
                permutation=permutation,
            )
            null_matrices.append(
                fit_centered_local_matrix(score_offsets, update).matrix
            )
        null_summary = law_cost_summary(np.stack(null_matrices), true_change)
        null_identity_ratios.append(
            float(null_summary["identity_over_best_nonidentity"])
        )

    validity = card["validity_gates"]
    strong = card["strong_distinct_law_gate"]
    geometry_valid = bool(
        analytic_disagreement <= float(validity["max_true_step_analytic_disagreement"])
        and max(fit.relative_residual for fit in block_fits)
        <= float(validity["max_local_fit_relative_residual"])
        and max(anchor_ratios) <= float(validity["max_anchor_update_over_true_rms"])
    )
    strong_pass = bool(
        geometry_valid
        and global_laws["max_own_relative_error"]
        <= float(strong["max_global_change_matrix_relative_error"])
        and block_laws["max_own_relative_error"]
        <= float(strong["max_support_block_change_matrix_relative_error"])
        and block_laws["max_own_over_nearest_wrong"]
        <= float(strong["max_support_block_over_nearest_wrong_true_law_cost"])
        and block_laws["identity_over_best_nonidentity"]
        <= float(strong["max_identity_assignment_over_best_nonidentity_assignment_cost"])
        and block_laws["identity_is_unique_optimum"]
        and max(block_source_discrepancy)
        <= float(strong["max_support_block_vs_source_only_matrix_discrepancy"])
    )
    return {
        "status": "eligible" if geometry_valid else "ineligible_geometry",
        "routing": routing,
        "geometry": {
            "analytic_true_step_max_abs_disagreement": analytic_disagreement,
            "true_change_matrices": true_change.tolist(),
            "true_update_rms_by_basin": true_update_rms,
        },
        "block": {
            "local_matrices": block_matrices.tolist(),
            "intercepts": [fit.intercept.tolist() for fit in block_fits],
            "relative_fit_residual_by_basin": [
                fit.relative_residual for fit in block_fits
            ],
            "anchor_update_over_true_rms_by_basin": anchor_ratios,
            "law_identification": block_laws,
        },
        "source_only": {
            "local_matrices": source_matrices.tolist(),
            "law_identification": source_laws,
        },
        "global": {
            "local_matrices": global_matrices.tolist(),
            "law_identification": global_laws,
        },
        "closure": {
            "block_vs_source_matrix_discrepancy_by_basin": block_source_discrepancy,
            "max_block_vs_source_matrix_discrepancy": max(block_source_discrepancy),
        },
        "coordinate_null": {
            "identity_over_best_nonidentity_replicates": null_identity_ratios,
            "median_identity_over_best_nonidentity": float(
                np.median(null_identity_ratios)
            ),
            "secondary_only": True,
        },
        "geometry_valid": geometry_valid,
        "strong_distinct_law_pass": strong_pass,
    }


def _discover_sparse(seed: int, card: dict[str, Any]) -> RunSpec:
    sparse_card, sparse_hash = load_sparse_card(
        Path(card["frozen_sources"]["sparse_card"])
    )
    if sparse_hash != card["frozen_sources"]["sparse_card_sha256"]:
        raise RuntimeError("Sparse card drift after load")
    matches = [
        spec
        for spec in discover_primary_roster(sparse_card)
        if spec.system_key == "gated_local_linear" and spec.seed == seed
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one sparse GatedLocalLinear seed {seed}: {matches}")
    return matches[0]


def evaluate_seed(
    *,
    arm: str,
    seed: int,
    card: dict[str, Any],
    card_hash: str,
    device: str,
    batch_size: int,
    dense_task_tsv: Path | None,
    dense_base_out: Path | None,
) -> dict[str, Any]:
    started = time.time()
    sparse_spec = _discover_sparse(seed, card)
    sparse_cfg, sparse_env, sparse_model, sparse_path, sparse_checkpoint = _load_checkpoint(
        sparse_spec.run_dir, sparse_spec.system_key, device
    )
    del sparse_checkpoint
    assert_sign_split_layout(sparse_cfg, sparse_model)
    selected_model = sparse_model
    selected_path = sparse_path
    dense_audit = None
    dense_attempts = None
    if arm == "dense":
        if dense_task_tsv is None or dense_base_out is None:
            raise ValueError("Dense arm requires --dense_task_tsv and --dense_base_out")
        dense_card = json.loads(
            Path(card["frozen_sources"]["dense_card"]).read_text()
        )
        dense_matches = [
            spec
            for spec in discover_dense_roster(
                dense_card, dense_task_tsv, dense_base_out
            )
            if spec.system_key == "gated_local_linear" and spec.seed == seed
        ]
        if len(dense_matches) != 1:
            raise RuntimeError(f"Expected one dense GatedLocalLinear seed {seed}")
        dense_spec = dense_matches[0]
        dense_cfg, dense_env, dense_model, dense_path, dense_checkpoint = _load_checkpoint(
            dense_spec.run_dir, dense_spec.system_key, device
        )
        dense_audit = assert_exact_dense_control(
            dense_cfg, dense_model, dense_card, dense_checkpoint
        )
        if dense_env.observation_size != sparse_env.observation_size:
            raise AssertionError("Sparse/dense state dimensions differ")
        selected_model = dense_model
        selected_path = dense_path
        dense_attempts = {
            "attempt_count": dense_spec.attempt_count,
            "incomplete_attempt_count": dense_spec.incomplete_attempt_count,
        }

    corpus = json.loads(
        Path(card["frozen_sources"]["sparse_card"]).read_text()
    )["corpus"]
    trajectories = VectorWrapper(
        sparse_env, int(corpus["num_trajectories"])
    ).generate_sequence_batch(
        rng=torch.Generator().manual_seed(int(corpus["eval_seed"])),
        window_length=int(corpus["trajectory_length"]),
    ).float()
    order = np.random.default_rng(int(corpus["split_seed"])).permutation(
        trajectories.shape[0]
    )
    fit_ids = order[: int(corpus["fit_trajectories"])]
    sparse_latent = _encode(sparse_model, trajectories, device, batch_size)
    sparse_fit = sparse_latent[fit_ids, :-1].reshape(-1, sparse_latent.shape[-1])
    threshold = float(card["family_discovery"]["sparse_support"].split(">")[-1])
    jaccard = float(card["family_discovery"]["jaccard_threshold"])
    if arm == "sparse":
        codebook = fit_family_codebook(np.abs(sparse_fit) > threshold, jaccard)
    else:
        dense_latent = _encode(selected_model, trajectories, device, batch_size)
        dense_fit = dense_latent[fit_ids, :-1].reshape(-1, dense_latent.shape[-1])
        codebook = fit_dense_family_codebook(
            matched_topk_masks(dense_fit, sparse_fit, threshold), jaccard
        )
    retained = codebook.fit_counts >= int(
        card["family_discovery"]["min_fit_source_transitions"]
    )
    result = _score_arm(
        arm=arm,
        model=selected_model,
        sparse_model=sparse_model,
        env=sparse_env,
        codebook=codebook,
        retained=retained,
        card=card,
        device=device,
        batch_size=batch_size,
    )
    return {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "card_sha256": card_hash,
        "arm": arm,
        "seed": seed,
        "result": result,
        "assertions": {
            "global_K_unmodified": True,
            "no_latent_dynamics_fit": True,
            "family_discovery_uses_basin_labels_or_count": False,
            "family_matching_is_evaluation_only": True,
            "wrong_source_projector_not_evaluated": True,
            "state_space_K_minus_I_primary": True,
            "dense_masks_called_cardinality_matched_coordinates": arm == "dense",
        },
        "provenance": {
            "sparse_run_dir": sparse_spec.run_dir,
            "sparse_checkpoint_sha256": sha256_path(sparse_path),
            "selected_checkpoint_path": str(selected_path),
            "selected_checkpoint_sha256": sha256_path(selected_path),
            "dense_attempts": dense_attempts,
            "dense_checkpoint_audit": dense_audit,
            "evaluator_sha256": _hash_text(Path(__file__)),
            "git_commit": os.environ.get("SKAE_GIT_COMMIT", "launcher_not_recorded"),
        },
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("sparse", "dense"), required=True)
    parser.add_argument("--task_index", type=int, required=True)
    parser.add_argument("--dense_task_tsv", type=Path)
    parser.add_argument("--dense_base_out", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()
    card, card_hash = load_protocol_card(args.card)
    seeds = [int(seed) for seed in card["scope"][f"{args.arm}_seeds"]]
    if not 0 <= args.task_index < len(seeds):
        raise IndexError(args.task_index)
    seed = seeds[args.task_index]
    output = args.output_dir / "shards" / f"{args.arm}_seed_{seed}.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    payload = evaluate_seed(
        arm=args.arm,
        seed=seed,
        card=card,
        card_hash=card_hash,
        device=args.device,
        batch_size=args.batch_size,
        dense_task_tsv=args.dense_task_tsv,
        dense_base_out=args.dense_base_out,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "status": payload["result"]["status"]}))


if __name__ == "__main__":
    main()
