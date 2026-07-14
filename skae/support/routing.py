"""Support-family route construction and runtime assignment primitives.

This module intentionally contains only the route construction and runtime
assignment shared by staged training and evaluation.  It is not a standalone
trainer; the frozen paper protocol lives under :mod:`experiments.neurips_2026`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Sequence

import numpy as np
import torch


SUPPORT_DEFINITION = "absolute:0.001"
SUPPORT_SCHEME = "absolute"
SUPPORT_THRESHOLD = 1e-3
FAMILY_JACCARD_THRESHOLD = 0.40
# Historical source-route fitting contract.  The nominal 512-row packet was
# produced by calling ``VectorWrapper.generate_sequence_batch`` twice with one
# generator.  ``VectorWrapper.reset`` reads (but does not advance) that
# generator's initial seed, so the two 256-row batches were bitwise identical.
# Keep that behavior deliberately: a 512-unique-trajectory fit is a new
# protocol and must not silently reuse the published checkpoint/result label.
FIT_CONFIGURED_ROWS = 512
FIT_UNIQUE_TRAJECTORIES = 256
FIT_DUPLICATION_FACTOR = 2
FIT_TRANSITIONS = 192
FIT_STATES = FIT_TRANSITIONS + 1
FIT_SEED_OFFSET = 271_828
FIT_SUPPORTS_CONSIDERED = FIT_CONFIGURED_ROWS * FIT_STATES
FIT_SOURCE_TRANSITIONS = FIT_CONFIGURED_ROWS * FIT_TRANSITIONS
FIT_UNIQUE_SOURCE_TRANSITIONS = FIT_UNIQUE_TRAJECTORIES * FIT_TRANSITIONS

# Compatibility aliases used by historical loaders and downstream utilities.
FIT_NUM_TRAJECTORIES = FIT_CONFIGURED_ROWS
FIT_TRAJECTORY_LENGTH = FIT_TRANSITIONS
MIN_FAMILY_TRANSITIONS = 1
FAMILY_REPRESENTATIVE_RULE = "modal_source_support"
FAMILY_CLUSTERING_RULE = "all_193_states_then_fit_on_first_192_sources"


def _validate_protocol(scheme: str, value: float, family_jaccard_threshold: float) -> None:
    if scheme != SUPPORT_SCHEME or not np.isclose(float(value), SUPPORT_THRESHOLD):
        raise ValueError(
            "The staged paper route is fixed to absolute support threshold 1e-3; "
            f"received {scheme}:{value}."
        )
    if not np.isclose(float(family_jaccard_threshold), FAMILY_JACCARD_THRESHOLD):
        raise ValueError(
            "The staged paper route is fixed to Jaccard threshold 0.40; "
            f"received {family_jaccard_threshold}."
        )


def _support_mask(latents: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(latents)) > SUPPORT_THRESHOLD


def _support_keys(mask: np.ndarray) -> np.ndarray:
    packed = np.packbits(mask.astype(np.uint8), axis=-1)
    flat = packed.reshape(-1, packed.shape[-1])
    return np.asarray([row.tobytes() for row in flat], dtype=object).reshape(mask.shape[:-1])


def _binary_jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    return 1.0 if union == 0.0 else intersection / union


def _support_family_labels(support_mask: np.ndarray) -> np.ndarray:
    """Cluster all states in frequency order, including terminal states.

    The source artifact clustered all 193 states per nominal trajectory before
    restricting map counts, centers, and exact-key routing to the first 192
    source states.  Terminal-state supports therefore affect family formation.
    """
    if support_mask.ndim != 3:
        raise ValueError("support_mask must have shape [trajectories, length, latent_dim]")
    flat_masks = support_mask.reshape(-1, support_mask.shape[-1])
    flat_keys = _support_keys(support_mask).reshape(-1)
    counts = Counter(flat_keys.tolist())
    key_masks: Dict[object, np.ndarray] = {}
    for key, mask in zip(flat_keys.tolist(), flat_masks):
        key_masks.setdefault(key, mask.astype(bool, copy=True))

    prototypes = []
    key_to_family: Dict[object, int] = {}
    for key, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        mask = key_masks[key]
        similarities = [_binary_jaccard(mask, prototype) for prototype in prototypes]
        if similarities and max(similarities) >= FAMILY_JACCARD_THRESHOLD:
            key_to_family[key] = int(np.argmax(similarities))
        else:
            key_to_family[key] = len(prototypes)
            prototypes.append(mask)

    labels = np.asarray([key_to_family[key] for key in flat_keys.tolist()], dtype=np.int64)
    return labels.reshape(support_mask.shape[:-1])


def _prototype_masks(
    family_labels: np.ndarray,
    support_keys: np.ndarray,
    support_masks: np.ndarray,
) -> Dict[object, np.ndarray]:
    """Return each family's modal *source* support, with first-seen tie breaks."""
    key_to_mask: Dict[object, np.ndarray] = {}
    family_support_counts: Dict[object, Counter[object]] = defaultdict(Counter)
    for family_id, key, mask in zip(
        family_labels.tolist(), support_keys.tolist(), support_masks
    ):
        key_to_mask.setdefault(key, mask.astype(bool, copy=True))
        family_support_counts[family_id][key] += 1
    return {
        family_id: key_to_mask[counts.most_common(1)[0][0]]
        for family_id, counts in family_support_counts.items()
    }


def _generate_source_route_fit_batches(
    train_env: object,
    *,
    seed: int,
) -> List[torch.Tensor]:
    """Build the published 512-row route-fit packet from 256 unique paths.

    Only one environment batch is sampled.  It is cloned explicitly to avoid
    depending on the current implementation of ``VectorWrapper.initial_seed``
    behavior while preserving the historical numerical route fit.
    """
    batch_size = int(getattr(train_env, "batch_size", -1))
    if batch_size != FIT_UNIQUE_TRAJECTORIES:
        raise ValueError(
            "The source route-fit protocol requires a 256-row VectorWrapper; "
            f"received batch_size={batch_size}."
        )
    rng = torch.Generator().manual_seed(int(seed))
    unique = train_env.generate_sequence_batch(
        rng,
        window_length=FIT_TRANSITIONS,
    ).float().cpu().contiguous()
    if unique.shape[0] != FIT_UNIQUE_TRAJECTORIES or unique.shape[1] != FIT_STATES:
        raise RuntimeError(
            "Route-fit generation violated the source protocol: expected "
            f"({FIT_UNIQUE_TRAJECTORIES}, {FIT_STATES}, ...), got {tuple(unique.shape)}."
        )
    duplicate = unique.clone()
    if not torch.equal(unique.view(torch.uint8), duplicate.view(torch.uint8)):
        raise RuntimeError("The explicit route-fit duplicate is not bitwise identical.")
    return [unique, duplicate]


def _build_route_codebook(
    fit_latents: np.ndarray,
    *,
    scheme: str = SUPPORT_SCHEME,
    value: float = SUPPORT_THRESHOLD,
    min_operator_transitions: int = MIN_FAMILY_TRANSITIONS,
    family_jaccard_threshold: float = FAMILY_JACCARD_THRESHOLD,
) -> Dict[str, object]:
    """Fit frozen route families and their mean source latent centers."""
    _validate_protocol(scheme, value, family_jaccard_threshold)
    if int(min_operator_transitions) != MIN_FAMILY_TRANSITIONS:
        raise ValueError("The staged paper route retains families with at least one transition.")
    if fit_latents.ndim != 3 or fit_latents.shape[1] < 2:
        raise ValueError("fit_latents must have shape [trajectories, length>=2, latent_dim]")

    support_mask = _support_mask(fit_latents)
    support_keys = _support_keys(support_mask)
    family_labels = _support_family_labels(support_mask)
    sources = fit_latents[:, :-1, :].reshape(-1, fit_latents.shape[-1]).astype(
        np.float32, copy=False
    )
    source_keys = support_keys[:, :-1].reshape(-1).astype(object)
    source_families = family_labels[:, :-1].reshape(-1).astype(object)
    family_counts = Counter(source_families.tolist())
    fitted_family_ids = sorted(family_counts, key=str)
    centers = {
        family_id: sources[source_families == family_id].mean(axis=0).astype(
            np.float32, copy=False
        )
        for family_id in fitted_family_ids
    }
    source_masks = support_mask[:, :-1, :].reshape(-1, support_mask.shape[-1])
    family_prototypes = _prototype_masks(source_families, source_keys, source_masks)
    support_key_to_family: Dict[object, object] = {}
    for key, family_id in zip(source_keys.tolist(), source_families.tolist()):
        support_key_to_family.setdefault(key, family_id)

    return {
        "support_mask": support_mask,
        "family_labels": family_labels,
        "family_counts": family_counts,
        "fitted_family_ids": fitted_family_ids,
        "centers": centers,
        "family_prototypes": family_prototypes,
        "support_key_to_family": support_key_to_family,
        "routing_object": "support_family",
        "runtime_routing_kind": "support_jaccard",
        "route_jaccard_threshold": FAMILY_JACCARD_THRESHOLD,
        "family_representative_rule": FAMILY_REPRESENTATIVE_RULE,
        "family_clustering_rule": FAMILY_CLUSTERING_RULE,
        "clustering_state_count": int(fit_latents.shape[0] * fit_latents.shape[1]),
        "source_transition_count": int(sources.shape[0]),
    }


def _assign_family_ids_np(
    latents: np.ndarray,
    *,
    support_key_to_family: Dict[object, object],
    family_prototypes: Dict[object, np.ndarray],
    family_cache: Dict[object, object],
) -> np.ndarray:
    masks = _support_mask(latents)
    keys = _support_keys(masks)
    assigned = np.empty(masks.shape[0], dtype=object)
    for index, (key, mask) in enumerate(zip(keys.tolist(), masks)):
        family_id = support_key_to_family.get(key)
        if family_id is None and key in family_cache:
            family_id = family_cache[key]
        elif family_id is None:
            best_family = None
            best_similarity = -1.0
            for candidate, prototype in family_prototypes.items():
                similarity = _binary_jaccard(mask, prototype)
                if similarity > best_similarity:
                    best_family = candidate
                    best_similarity = similarity
            family_id = (
                best_family
                if best_family is not None
                and best_similarity >= FAMILY_JACCARD_THRESHOLD
                else None
            )
            family_cache[key] = family_id
        assigned[index] = family_id
    return assigned


def _route_indices_np(
    latents: np.ndarray,
    *,
    scheme: str = SUPPORT_SCHEME,
    value: float = SUPPORT_THRESHOLD,
    family_jaccard_threshold: float = FAMILY_JACCARD_THRESHOLD,
    support_key_to_family: Dict[object, object],
    family_prototypes: Dict[object, np.ndarray],
    family_to_index: Dict[str, int],
    family_cache: Dict[object, object],
) -> np.ndarray:
    _validate_protocol(scheme, value, family_jaccard_threshold)
    family_ids = _assign_family_ids_np(
        latents,
        support_key_to_family=support_key_to_family,
        family_prototypes=family_prototypes,
        family_cache=family_cache,
    )
    route_indices = np.full(family_ids.shape[0], -1, dtype=np.int64)
    for index, family_id in enumerate(family_ids.tolist()):
        if family_id is not None and str(family_id) in family_to_index:
            route_indices[index] = family_to_index[str(family_id)]
    return route_indices


def _step_routes_for_torch(
    z: torch.Tensor,
    *,
    scheme: str,
    value: float,
    family_jaccard_threshold: float,
    support_key_to_family: Dict[object, object],
    family_prototypes: Dict[object, np.ndarray],
    family_to_index: Dict[str, int],
    family_cache: Dict[object, object],
    device: torch.device,
) -> torch.Tensor:
    route_indices = _route_indices_np(
        z.detach().cpu().numpy().astype(np.float32, copy=False),
        scheme=scheme,
        value=value,
        family_jaccard_threshold=family_jaccard_threshold,
        support_key_to_family=support_key_to_family,
        family_prototypes=family_prototypes,
        family_to_index=family_to_index,
        family_cache=family_cache,
    )
    return torch.from_numpy(route_indices).to(device=device, dtype=torch.long)
