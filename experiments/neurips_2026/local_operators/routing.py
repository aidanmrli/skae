"""Apply the frozen paper contract to reusable support-routing mechanics."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from experiments.neurips_2026.local_operators.contract import (
    FAMILY_CLUSTERING_RULE,
    FAMILY_JACCARD_THRESHOLD,
    FAMILY_REPRESENTATIVE_RULE,
    FIT_STATES,
    FIT_TRANSITIONS,
    FIT_UNIQUE_TRAJECTORIES,
    MIN_FAMILY_TRANSITIONS,
    SUPPORT_SCHEME,
    SUPPORT_THRESHOLD,
)
from skae.support.routing import (
    _build_route_codebook as _build_parameterized_route_codebook,
    _route_indices_np as _parameterized_route_indices_np,
)


def _validate_protocol(
    scheme: str,
    value: float,
    family_jaccard_threshold: float,
    min_operator_transitions: int = MIN_FAMILY_TRANSITIONS,
) -> None:
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
    if int(min_operator_transitions) != MIN_FAMILY_TRANSITIONS:
        raise ValueError(
            "The staged paper route retains families with at least one transition."
        )


def _generate_source_route_fit_batches(
    train_env: object,
    *,
    seed: int,
) -> List[torch.Tensor]:
    """Build the published fit packet as two copies of 256 unique paths."""
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
    """Fit the codebook only under the versioned paper route contract."""
    _validate_protocol(
        scheme,
        value,
        family_jaccard_threshold,
        min_operator_transitions,
    )
    return _build_parameterized_route_codebook(
        fit_latents,
        scheme=scheme,
        value=value,
        min_operator_transitions=min_operator_transitions,
        family_jaccard_threshold=family_jaccard_threshold,
        family_representative_rule=FAMILY_REPRESENTATIVE_RULE,
        family_clustering_rule=FAMILY_CLUSTERING_RULE,
    )


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
    _validate_protocol(
        scheme,
        value,
        family_jaccard_threshold,
        MIN_FAMILY_TRANSITIONS,
    )
    return _parameterized_route_indices_np(
        latents,
        scheme=scheme,
        value=value,
        family_jaccard_threshold=family_jaccard_threshold,
        support_key_to_family=support_key_to_family,
        family_prototypes=family_prototypes,
        family_to_index=family_to_index,
        family_cache=family_cache,
    )
