"""Frozen sparse-family routing and dense center-projector steelman for V2."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from experiments.neurips_2026.global_k_distinct_laws_v2_math import (
    authenticate_local_geometry,
    match_families_to_basins,
    sample_centered_disk,
)
from experiments.neurips_2026.global_k_support_invariance import (
    FamilyCodebook,
    _encode,
    assign_families,
    fit_family_codebook,
)
from skae.data import VectorWrapper


def _encode_points(model, points: np.ndarray, batch_size: int) -> np.ndarray:
    parameter = next(model.parameters())
    values = torch.as_tensor(points, dtype=parameter.dtype)
    chunks = []
    with torch.no_grad():
        for start in range(0, values.shape[0], batch_size):
            chunk = values[start : start + batch_size].to(parameter.device)
            chunks.append(model.encode(chunk).cpu())
    return torch.cat(chunks).numpy().astype(np.float32)


def _threshold(card: dict[str, Any]) -> float:
    return float(card["label_free_family_discovery"]["sparse_support"].split(">")[-1])


def _sparse_family_assigner(
    codebook: FamilyCodebook, threshold: float, jaccard: float,
) -> Callable[[np.ndarray], np.ndarray]:
    return lambda latent: assign_families(
        np.abs(latent) > threshold, codebook, jaccard
    )


def discover_sparse_families(
    sparse_model, env, card: dict[str, Any], batch_size: int,
) -> tuple[FamilyCodebook, np.ndarray, dict[str, Any]]:
    protocol = card["label_free_family_discovery"]
    trajectories = VectorWrapper(
        env, int(protocol["trajectory_count"])
    ).generate_sequence_batch(
        rng=torch.Generator().manual_seed(int(protocol["corpus_seed"])),
        window_length=int(protocol["trajectory_length"]),
    ).float()
    latent = _encode_points(sparse_model, trajectories, batch_size)
    order = np.random.default_rng(int(protocol["trajectory_split_seed"])).permutation(
        trajectories.shape[0]
    )
    fit_ids = order[: int(protocol["fit_trajectories"])]
    held_ids = order[int(protocol["fit_trajectories"]):]
    fit = latent[fit_ids, :-1].reshape(-1, latent.shape[-1])
    threshold = _threshold(card)
    jaccard = float(protocol["jaccard_threshold"])
    codebook = fit_family_codebook(np.abs(fit) > threshold, jaccard)
    retained = codebook.fit_counts >= int(protocol["min_fit_source_transitions"])
    held = latent[held_ids, :-1].reshape(-1, latent.shape[-1])
    labels = _sparse_family_assigner(codebook, threshold, jaccard)(held)
    valid = labels >= 0
    retained_valid = valid & retained[np.maximum(labels, 0)]
    return codebook, retained, {
        "family_count": int(codebook.representatives.shape[0]),
        "retained_family_count": int(retained.sum()),
        "held_out_assignment_rate": float(valid.mean()),
        "held_out_retained_assignment_rate": float(retained_valid.mean()),
        "arm": "paired_sparse_label_free_support",
        "training_uses_basin_labels_or_count": False,
    }


def match_sparse_routing(
    sparse_model, env, codebook: FamilyCodebook,
    retained: np.ndarray, card: dict[str, Any], batch_size: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    protocol = card["evaluation_only_family_matching"]
    centers = env.unwrapped.points_2d.detach().cpu().numpy().astype(np.float32)
    assign = _sparse_family_assigner(
        codebook, _threshold(card),
        float(card["label_free_family_discovery"]["jaccard_threshold"]),
    )

    def sample(kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        offsets = sample_centered_disk(
            int(protocol[f"{kind}_points_per_basin"]),
            float(protocol[f"{kind}_disk_radius"]),
            int(protocol[f"{kind}_seed"]),
        )
        points = centers[:, None, :] + offsets[None, :, :]
        flat = points.reshape(-1, 2)
        labels = assign(_encode_points(sparse_model, flat, batch_size)).reshape(3, -1)
        authentication = authenticate_local_geometry(
            env, points, category=kind,
            dt=float(card["benchmark"]["dt"]),
            max_abs_error=float(
                card["validity"]["max_analytic_environment_step_disagreement"]
            ),
        )
        return offsets, labels, points, authentication

    _unused, calibration, _unused_points, calibration_auth = sample("calibration")
    verification_offsets, verification, verification_points, verification_auth = sample(
        "verification"
    )
    mapping, calibration_rates, counts = match_families_to_basins(
        calibration, retained, 3
    )
    verification_rates = np.asarray([
        np.mean(verification[basin] == mapping[basin]) for basin in range(3)
    ])
    center_assignment = assign(_encode_points(sparse_model, centers, batch_size))
    matched_counts = [
        int(np.sum(verification[basin] == mapping[basin])) for basin in range(3)
    ]
    mapping_resolves = bool(np.all(mapping >= 0) and len(set(mapping.tolist())) == 3)
    masks = (
        np.stack([codebook.representatives[family] for family in mapping])
        if mapping_resolves else np.empty((0, 0), dtype=bool)
    )
    cardinalities = masks.sum(axis=1).astype(int).tolist() if mapping_resolves else []
    jaccards = []
    if mapping_resolves:
        for left in range(3):
            for right in range(left + 1, 3):
                intersection = int(np.logical_and(masks[left], masks[right]).sum())
                union = int(np.logical_or(masks[left], masks[right]).sum())
                jaccards.append({
                    "left_basin": left, "right_basin": right,
                    "jaccard": intersection / max(union, 1),
                })
    support_valid = bool(
        mapping_resolves
        and all(0 < value < masks.shape[1] for value in cardinalities)
        and all(item["jaccard"] < 0.5 for item in jaccards)
    )
    valid = bool(
        mapping_resolves
        and np.all(calibration_rates >= float(protocol["minimum_calibration_match_rate_each_basin"]))
        and np.all(verification_rates >= float(protocol["minimum_verification_match_rate_each_basin"]))
        and np.array_equal(center_assignment, mapping)
        and min(matched_counts) >= int(card["direct_active_code_cloud_closure"]["minimum_points_per_seed_basin"])
        and support_valid
    )
    routing = {
        "matched_family_by_basin": mapping.tolist(),
        "calibration_match_rate_by_basin": calibration_rates.tolist(),
        "verification_match_rate_by_basin": verification_rates.tolist(),
        "center_family_by_basin": center_assignment.tolist(),
        "calibration_count_matrix": counts.tolist(),
        "matched_verification_point_count_by_basin": matched_counts,
        "matched_support_cardinality_by_basin": cardinalities,
        "matched_support_fraction_by_basin": (
            [value / masks.shape[1] for value in cardinalities]
            if mapping_resolves else []
        ),
        "matched_support_pairwise_jaccards": jaccards,
        "matched_support_diagnostics_valid": support_valid,
        "matched_families_distinct": len(set(mapping.tolist())) == 3,
        "family_valid": valid,
        "labels_and_known_count_used_only_for_scoring_map": True,
        "geometry_authentication": {
            "calibration": calibration_auth,
            "verification": verification_auth,
            "passed": calibration_auth["passed"] and verification_auth["passed"],
        },
    }
    points = (
        [verification_points[b][verification[b] == mapping[b]] for b in range(3)]
        if valid else []
    )
    return routing, (masks if valid else np.empty((0, 0))), centers, verification_offsets, points


def dense_center_projectors(
    dense_model, centers: np.ndarray, sparse_masks: np.ndarray, batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Freeze deterministic center top-k dense masks with exact sparse k."""
    latent = _encode_points(dense_model, centers, batch_size)
    masks = np.zeros_like(latent, dtype=bool)
    sparse_k = sparse_masks.sum(axis=1).astype(int)
    coordinates = np.arange(latent.shape[1])
    for basin, k_value in enumerate(sparse_k):
        order = np.lexsort((coordinates, -np.abs(latent[basin]).astype(np.float64)))
        masks[basin, order[: int(k_value)]] = True
    dense_k = masks.sum(axis=1).astype(int)
    valid = bool(
        np.array_equal(dense_k, sparse_k)
        and all(0 < value < latent.shape[1] for value in dense_k)
        and np.isfinite(latent).all()
    )
    return masks, {
        "projector_source": "top_k_absolute_dense_encoding_at_known_basin_center",
        "tie_break": "ascending_coordinate_index",
        "sparse_projector_cardinality_by_basin": sparse_k.tolist(),
        "dense_projector_cardinality_by_basin": dense_k.tolist(),
        "exact_cardinality_equality_by_basin": (dense_k == sparse_k).tolist(),
        "known_centers_and_sparse_k_used_only_for_evaluation": True,
        "center_projectors_valid": valid,
    }
