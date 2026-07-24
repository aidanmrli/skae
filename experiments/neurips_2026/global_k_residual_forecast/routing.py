"""Label-free support routing and matched coordinate-null construction."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from experiments.neurips_2026.global_k_residual_forecast.protocol import (
    nearest_family,
    select_projectors,
    sha256_array,
    stable_sign_pair_permutations,
)
from experiments.neurips_2026.global_k_support_invariance import fit_family_codebook


def evenly_spaced_indices(
    total_count: int, sample_count: int, *, device: torch.device | None = None,
) -> torch.Tensor:
    if sample_count < 1 or total_count < sample_count:
        raise ValueError((total_count, sample_count))
    indices = torch.linspace(
        0, total_count - 1, steps=sample_count, device=device
    ).round().long()
    if int(torch.unique(indices).numel()) != sample_count:
        raise RuntimeError("Evenly spaced index construction produced a duplicate")
    return indices


@torch.no_grad()
def encode_flat(model, values: torch.Tensor, batch_size: int) -> torch.Tensor:
    flat = values.reshape(-1, values.shape[-1])
    chunks = []
    device = next(model.parameters()).device
    for start in range(0, flat.shape[0], batch_size):
        chunks.append(model.encode(flat[start : start + batch_size].to(device)))
    return torch.cat(chunks, dim=0)


def fit_codebook(
    model, fit_trajectories: torch.Tensor, card: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor]:
    route = card["label_free_support_routing"]
    source = fit_trajectories[:, :-1]
    latent = encode_flat(model, source, int(route["encode_batch_size"])).cpu().numpy()
    support = np.abs(latent) > float(route["support_threshold"])
    codebook = fit_family_codebook(support, float(route["codebook_jaccard_threshold"]))
    retained_ids = np.flatnonzero(
        codebook.fit_counts >= int(route["minimum_fit_transitions"])
    )
    too_many = retained_ids.size > int(route["maximum_retained_families"])
    retained_ids = retained_ids[: int(route["maximum_retained_families"])]
    fallback_used = retained_ids.size == 0
    if fallback_used:
        retained_ids = np.asarray([int(np.argmax(codebook.fit_counts))])
    representatives_np = codebook.representatives[retained_ids].astype(bool, copy=True)
    representatives = torch.from_numpy(representatives_np).to(next(model.parameters()).device)
    retained_coverage = float(codebook.fit_counts[retained_ids].sum() / support.shape[0])
    base_dim = representatives_np.shape[1] // 2
    exclusive = not np.logical_and(
        representatives_np[:, :base_dim], representatives_np[:, base_dim:]
    ).any()
    cardinalities = representatives_np.sum(axis=1).astype(int)
    valid = bool(
        not fallback_used
        and not too_many
        and retained_ids.size >= int(route["minimum_retained_families"])
        and np.all(cardinalities > 0)
        and np.all(cardinalities < representatives_np.shape[1])
        and exclusive
        and retained_coverage >= float(route["minimum_retained_fit_coverage"])
    )
    return representatives, {
        "all_family_count": int(codebook.representatives.shape[0]),
        "retained_family_count": int(retained_ids.size),
        "retained_original_family_ids": retained_ids.tolist(),
        "retained_fit_coverage": retained_coverage,
        "support_cardinalities": cardinalities.tolist(),
        "sign_pair_exclusivity": bool(exclusive),
        "fallback_used": bool(fallback_used),
        "maximum_family_truncation_used": bool(too_many),
        "representatives_sha256": sha256_array(representatives_np),
        "label_or_known_basin_count_used": False,
        "fit_valid": valid,
    }, torch.from_numpy(latent).to(next(model.parameters()).device)


@torch.no_grad()
def audit_routing(
    model, representatives: torch.Tensor, trajectories: torch.Tensor,
    env, card: dict[str, Any],
) -> dict[str, Any]:
    route = card["label_free_support_routing"]
    latent = encode_flat(
        model, trajectories[:, :-1], int(route["encode_batch_size"])
    )
    assignment, similarity = nearest_family(
        latent, representatives, float(route["support_threshold"])
    )
    counts = torch.bincount(assignment, minlength=representatives.shape[0]).cpu().numpy()
    active = int(np.sum(counts >= float(route["minimum_audit_family_fraction"]) * counts.sum()))
    similarities = similarity.cpu().numpy()
    route_valid = bool(
        np.isfinite(similarities).all()
        and float(np.mean(similarities)) >= float(route["minimum_mean_audit_jaccard"])
        and float(np.mean(similarities >= route["minimum_confident_jaccard"]))
        >= float(route["minimum_confident_audit_fraction"])
        and active >= int(route["minimum_retained_families"])
    )

    # Evaluation-only stratification is deliberately downstream of assignments.
    physical = trajectories[:, :-1].reshape(-1, trajectories.shape[-1])
    labels = env.unwrapped.basin_label(physical).cpu().numpy().astype(np.int64)
    assignments = assignment.cpu().numpy().astype(np.int64)
    unique_labels = np.unique(labels)
    contingency = np.asarray(
        [[np.sum((assignments == family) & (labels == label)) for label in unique_labels]
         for family in range(representatives.shape[0])],
        dtype=np.int64,
    )
    purity = float(contingency.max(axis=1).sum() / max(contingency.sum(), 1))
    return {
        "assignment_count_by_family": counts.tolist(),
        "mean_nearest_jaccard": float(np.mean(similarities)),
        "confident_assignment_fraction": float(
            np.mean(similarities >= route["minimum_confident_jaccard"])
        ),
        "active_family_count_at_minimum_fraction": active,
        "label_free_route_audit_valid": route_valid,
        "evaluation_only_alignment": {
            "labels_computed_after_and_not_passed_to_assignment_or_predictor": True,
            "observed_label_values": unique_labels.tolist(),
            "contingency_family_by_label": contingency.tolist(),
            "family_conditional_basin_purity": purity,
        },
    }


@torch.no_grad()
def matched_null_projectors(
    model,
    representatives: torch.Tensor,
    fit_latent: torch.Tensor,
    card: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    null = card["matched_sign_pair_permutation_null"]
    route = card["label_free_support_routing"]
    sample_count = int(null["scale_match_points"])
    sample_indices = evenly_spaced_indices(
        fit_latent.shape[0], sample_count, device=fit_latent.device
    )
    sample = fit_latent.index_select(0, sample_indices)
    assignment, _ = nearest_family(
        sample, representatives, float(route["support_threshold"])
    )
    correct_p = select_projectors(assignment, representatives).to(sample.dtype)
    k_matrix = model.kmatrix()

    def scale(p: torch.Tensor) -> tuple[float, float]:
        source = sample * p
        stepped = (source @ k_matrix) * p
        update = model.decode(stepped) - model.decode(source)
        source_rms = torch.sqrt(source.float().square().sum(dim=1).mean())
        update_rms = torch.sqrt(update.float().square().sum(dim=1).mean())
        return float(source_rms.item()), float(update_rms.item())

    correct_source, correct_update = scale(correct_p)
    permutations = stable_sign_pair_permutations(
        representatives.shape[1],
        int(null["candidate_count"]),
        int(null["candidate_seed"]),
    )
    rows = []
    for candidate_index, permutation in enumerate(permutations):
        index = torch.as_tensor(permutation, device=representatives.device)
        permuted_reps = representatives.index_select(1, index)
        p = select_projectors(assignment, permuted_reps).to(sample.dtype)
        source_rms, update_rms = scale(p)
        source_ratio = source_rms / max(correct_source, 1e-30)
        update_ratio = update_rms / max(correct_update, 1e-30)
        score = abs(math.log(max(source_ratio, 1e-30))) + abs(
            math.log(max(update_ratio, 1e-30))
        )
        eligible = bool(
            float(null["minimum_scale_ratio"]) <= source_ratio
            <= float(null["maximum_scale_ratio"])
            and float(null["minimum_scale_ratio"]) <= update_ratio
            <= float(null["maximum_scale_ratio"])
        )
        rows.append(
            {
                "candidate_index": candidate_index,
                "score": score,
                "source_rms_ratio": source_ratio,
                "update_rms_ratio": update_ratio,
                "eligible": eligible,
            }
        )
    eligible = sorted(
        (row for row in rows if row["eligible"]),
        key=lambda row: (row["score"], row["candidate_index"]),
    )
    needed = int(null["selected_count"])
    selection_valid = len(eligible) >= needed
    selected = eligible[:needed]
    if not selection_valid:
        selected = sorted(rows, key=lambda row: (row["score"], row["candidate_index"]))[:needed]
    selected_indices = [int(row["candidate_index"]) for row in selected]
    banks = []
    for candidate_index in selected_indices:
        index = torch.as_tensor(permutations[candidate_index], device=representatives.device)
        banks.append(representatives.index_select(1, index))
    bank = torch.stack(banks)
    return bank, {
        "candidate_count": len(rows),
        "eligible_candidate_count": len(eligible),
        "selected_candidate_indices": selected_indices,
        "selected_scale_rows": selected,
        "correct_source_rms": correct_source,
        "correct_update_rms": correct_update,
        "permutation_bank_sha256": sha256_array(permutations[selected_indices]),
        "cardinality_and_pairwise_jaccard_exactly_preserved": True,
        "selection_uses_labels_basin_count_or_forecast_truth": False,
        "latent_null_guaranteed_on_encoder_image": False,
        "physical_prediction_reencoded_each_step": True,
        "scale_match_valid": selection_valid,
    }
