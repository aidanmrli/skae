"""Parameter-driven support-family construction and route assignment.

This module contains reusable numerical mechanics only.  Experiment-specific
thresholds, fit packets, and artifact labels belong to the experiment package
that selects them.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict

import numpy as np
import torch


def _validate_support_definition(scheme: str, value: float) -> float:
    if scheme != "absolute":
        raise ValueError(f"Unsupported support scheme {scheme!r}; expected 'absolute'.")
    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("The absolute support threshold must be finite and nonnegative.")
    return threshold


def _validate_jaccard_threshold(value: float) -> float:
    threshold = float(value)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("The family Jaccard threshold must lie in [0, 1].")
    return threshold


def _support_mask(
    latents: np.ndarray,
    *,
    scheme: str,
    value: float,
) -> np.ndarray:
    threshold = _validate_support_definition(scheme, value)
    return np.abs(np.asarray(latents)) > threshold


def _support_keys(mask: np.ndarray) -> np.ndarray:
    packed = np.packbits(mask.astype(np.uint8), axis=-1)
    flat = packed.reshape(-1, packed.shape[-1])
    return np.asarray([row.tobytes() for row in flat], dtype=object).reshape(
        mask.shape[:-1]
    )


def _binary_jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    return 1.0 if union == 0.0 else intersection / union


def _support_family_labels(
    support_mask: np.ndarray,
    *,
    family_jaccard_threshold: float,
) -> np.ndarray:
    """Cluster states in frequency order, including terminal states."""
    threshold = _validate_jaccard_threshold(family_jaccard_threshold)
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
        if similarities and max(similarities) >= threshold:
            key_to_family[key] = int(np.argmax(similarities))
        else:
            key_to_family[key] = len(prototypes)
            prototypes.append(mask)

    labels = np.asarray(
        [key_to_family[key] for key in flat_keys.tolist()], dtype=np.int64
    )
    return labels.reshape(support_mask.shape[:-1])


def _prototype_masks(
    family_labels: np.ndarray,
    support_keys: np.ndarray,
    support_masks: np.ndarray,
) -> Dict[object, np.ndarray]:
    """Return each family's modal source support, with first-seen tie breaks."""
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


def _build_route_codebook(
    fit_latents: np.ndarray,
    *,
    scheme: str,
    value: float,
    min_operator_transitions: int,
    family_jaccard_threshold: float,
    family_representative_rule: str,
    family_clustering_rule: str,
) -> Dict[str, object]:
    """Fit support families and their mean source latent centers."""
    _validate_support_definition(scheme, value)
    threshold = _validate_jaccard_threshold(family_jaccard_threshold)
    minimum = int(min_operator_transitions)
    if minimum < 1:
        raise ValueError("min_operator_transitions must be at least one.")
    if fit_latents.ndim != 3 or fit_latents.shape[1] < 2:
        raise ValueError("fit_latents must have shape [trajectories, length>=2, latent_dim]")

    support_mask = _support_mask(fit_latents, scheme=scheme, value=value)
    support_keys = _support_keys(support_mask)
    family_labels = _support_family_labels(
        support_mask,
        family_jaccard_threshold=threshold,
    )
    sources = fit_latents[:, :-1, :].reshape(-1, fit_latents.shape[-1]).astype(
        np.float32, copy=False
    )
    source_keys = support_keys[:, :-1].reshape(-1).astype(object)
    source_families = family_labels[:, :-1].reshape(-1).astype(object)
    family_counts = Counter(source_families.tolist())
    fitted_family_ids = sorted(
        (
            family_id
            for family_id, count in family_counts.items()
            if count >= minimum
        ),
        key=str,
    )
    centers = {
        family_id: sources[source_families == family_id].mean(axis=0).astype(
            np.float32, copy=False
        )
        for family_id in fitted_family_ids
    }
    source_masks = support_mask[:, :-1, :].reshape(-1, support_mask.shape[-1])
    fitted_set = set(fitted_family_ids)
    retained = np.asarray(
        [family_id in fitted_set for family_id in source_families.tolist()],
        dtype=bool,
    )
    family_prototypes = _prototype_masks(
        source_families[retained],
        source_keys[retained],
        source_masks[retained],
    )
    support_key_to_family: Dict[object, object] = {}
    for key, family_id in zip(source_keys.tolist(), source_families.tolist()):
        if family_id in fitted_set:
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
        "route_jaccard_threshold": threshold,
        "family_representative_rule": family_representative_rule,
        "family_clustering_rule": family_clustering_rule,
        "clustering_state_count": int(fit_latents.shape[0] * fit_latents.shape[1]),
        "source_transition_count": int(sources.shape[0]),
    }


def _assign_family_ids_np(
    latents: np.ndarray,
    *,
    scheme: str,
    value: float,
    family_jaccard_threshold: float,
    support_key_to_family: Dict[object, object],
    family_prototypes: Dict[object, np.ndarray],
    family_cache: Dict[object, object],
) -> np.ndarray:
    threshold = _validate_jaccard_threshold(family_jaccard_threshold)
    masks = _support_mask(latents, scheme=scheme, value=value)
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
                if best_family is not None and best_similarity >= threshold
                else None
            )
            family_cache[key] = family_id
        assigned[index] = family_id
    return assigned


def _route_indices_np(
    latents: np.ndarray,
    *,
    scheme: str,
    value: float,
    family_jaccard_threshold: float,
    support_key_to_family: Dict[object, object],
    family_prototypes: Dict[object, np.ndarray],
    family_to_index: Dict[str, int],
    family_cache: Dict[object, object],
) -> np.ndarray:
    family_ids = _assign_family_ids_np(
        latents,
        scheme=scheme,
        value=value,
        family_jaccard_threshold=family_jaccard_threshold,
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
