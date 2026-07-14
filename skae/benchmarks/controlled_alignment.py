"""Frozen controlled-benchmark basin/support alignment computation.

Support families are fit over every generated evaluation-trajectory state.
Only after that global-on-evaluation family assignment is fixed do native or
proxy labels and center geometry select the tie-inclusive high-margin slice.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from skae.benchmarks.paper_protocol import (
    CONTROLLED_MODEL_ROW_IDS,
    CONTROLLED_PAPER_PROTOCOL,
)
from skae.benchmarks.transition_rich_basin_partition_manifest import (
    get_transition_rich_basin_count,
)
from skae.checkpoint_compat import load_model_state_dict_compat
from skae.config import Config
from skae.data import VectorWrapper, make_env
from skae.model import make_model


ABSOLUTE_THRESHOLD = 1e-3
FAMILY_JACCARD_THRESHOLD = 0.50
SUPPORT_SCHEME = "absolute:0.001"
# Frozen source packets use this legacy token; explanatory metadata uses the
# exact tie-inclusive high-center-margin semantics.
SUBSET = "deep"
DEFAULT_ROOT_LABELS = ",".join(CONTROLLED_MODEL_ROW_IDS)
OUTPUT_COLUMNS = (
    "root_label",
    "system_name",
    "seed",
    "support_scheme",
    "subset",
    "num_states",
    "observed_label_count",
    "family_jaccard_threshold",
    "family_h_basin_given_family",
    "family_unique_count",
)
EPS = 1e-12
NUM_EVALUATION_TRAJECTORIES = 128
TRAJECTORY_TRANSITIONS = 128
EVALUATION_SEED = 42
ENDPOINT_ROLLOUT_STEPS = 5000
NATIVE_LABEL_SYSTEMS = CONTROLLED_PAPER_PROTOCOL.system_keys[:2]
PROXY_LABEL_SYSTEMS = CONTROLLED_PAPER_PROTOCOL.system_keys[2:]
NATIVE_LABEL_SOURCE = "native_env_basin_label_with_env_points"
PROXY_LABEL_SOURCE = "evaluation_only_nearest_estimated_center_proxy"
ENTROPY_UNITS = "nats"
FAMILY_COUNT_SEMANTICS = "observed_family_ids_on_scored_high_margin_slice"
CENTER_MARGIN_SELECTION_RULE = (
    "margin_greater_than_or_equal_to_empirical_q75_tie_inclusive"
)


def alignment_protocol_metadata() -> Dict[str, object]:
    """Return the machine-readable scientific contract for this diagnostic."""

    return {
        "support_scheme": SUPPORT_SCHEME,
        "family_fit_scope": "all_generated_evaluation_trajectory_states",
        "scoring_scope": "per_observed_label_high_margin_tie_inclusive",
        "family_jaccard_threshold": FAMILY_JACCARD_THRESHOLD,
        "mask_visit_order": (
            "descending_frequency_then_ascending_packbits_bytes"
        ),
        "mask_serialization": "numpy.packbits_default_big_bit_order_then_bytes",
        "family_assignment_tie_break": "earliest_created_family",
        "num_evaluation_trajectories": NUM_EVALUATION_TRAJECTORIES,
        "trajectory_transitions": TRAJECTORY_TRANSITIONS,
        "states_per_trajectory": TRAJECTORY_TRANSITIONS + 1,
        "evaluation_seed": EVALUATION_SEED,
        "native_label_systems": list(NATIVE_LABEL_SYSTEMS),
        "native_label_source": "env.basin_label",
        "native_center_source": "env.points",
        "proxy_label_systems": list(PROXY_LABEL_SYSTEMS),
        "proxy_basin_count_source": "known_benchmark_count_for_evaluation_only",
        "proxy_endpoint_rollout_steps": ENDPOINT_ROLLOUT_STEPS,
        "proxy_center_estimator": (
            "deterministic_farthest_first_kmeans_on_advanced_endpoints"
        ),
        "kmeans_initial_center": "first_advanced_endpoint",
        "kmeans_farthest_tie_break": "first_endpoint_index",
        "kmeans_assignment_tie_break": "first_center_index",
        "kmeans_empty_cluster_rule": "retain_previous_center",
        "kmeans_max_iterations": 25,
        "kmeans_early_stop_rule": "torch.allclose_with_library_defaults",
        "proxy_state_label_rule": "nearest_estimated_center",
        "proxy_state_label_tie_break": "first_center_index",
        "center_margin_definition": (
            "second_nearest_center_distance_minus_nearest"
        ),
        "center_margin_quantile": 0.75,
        "center_margin_quantile_method": "numpy_default_linear",
        "labels_with_fewer_than_four_states": "retain_all_states",
        "center_margin_selection_rule": CENTER_MARGIN_SELECTION_RULE,
        "center_margin_tie_semantics": (
            "retain_margin_greater_than_or_equal_to_q75; ties_can_make_the_"
            "scored_slice_larger_than_25_percent"
        ),
        "entropy_units": ENTROPY_UNITS,
        "entropy_formula": "H(basin,family)-H(family)",
        "entropy_log_offset": EPS,
        "family_count_semantics": FAMILY_COUNT_SEMANTICS,
        "output_columns": list(OUTPUT_COLUMNS),
    }


def _load_checkpoint_model(checkpoint_path: Path, system_key: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = Config.from_dict(checkpoint["config"])
    cfg.ENV.ENV_NAME = system_key
    env = make_env(cfg)
    model = make_model(cfg, env.observation_size)
    load_model_state_dict_compat(model, checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return env, model


def _generate_trajectories(
    env,
    *,
    num_trajectories: int,
    trajectory_length: int,
    eval_seed: int,
) -> torch.Tensor:
    vec_env = VectorWrapper(env, num_trajectories)
    rng = torch.Generator().manual_seed(eval_seed)
    return vec_env.generate_sequence_batch(
        rng=rng,
        window_length=trajectory_length,
    ).float()


def _long_rollout(env, states: torch.Tensor, steps: int) -> torch.Tensor:
    current = states.clone()
    for _ in range(max(0, steps)):
        current = env.step(current)
    return current


def _label_from_native_method(method, sequences: torch.Tensor) -> torch.Tensor:
    flat = sequences.reshape(-1, sequences.shape[-1])
    labels = method(flat)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    return labels.reshape(sequences.shape[:-1]).to(dtype=torch.long)


def _kmeans_centers(
    points: torch.Tensor,
    num_centers: int,
    num_iters: int = 25,
) -> torch.Tensor:
    if points.ndim != 2:
        raise ValueError("points must have shape [N, dim]")
    if points.shape[0] < num_centers:
        raise ValueError("Need at least as many points as centers")

    centers = [points[0]]
    while len(centers) < num_centers:
        current = torch.stack(centers, dim=0)
        min_dists = torch.cdist(points, current).min(dim=1).values
        centers.append(points[min_dists.argmax()])
    current_centers = torch.stack(centers, dim=0).clone()
    for _ in range(num_iters):
        assignments = torch.cdist(points, current_centers).argmin(dim=1)
        updated = []
        for center_idx in range(num_centers):
            mask = assignments == center_idx
            updated.append(
                points[mask].mean(dim=0)
                if bool(mask.any())
                else current_centers[center_idx]
            )
        updated_centers = torch.stack(updated, dim=0)
        if torch.allclose(updated_centers, current_centers):
            break
        current_centers = updated_centers
    return current_centers


def _estimate_basin_centers(
    env,
    trajectories: torch.Tensor,
    basin_count: int,
    endpoint_rollout_steps: int,
) -> torch.Tensor:
    converged = _long_rollout(
        env,
        trajectories[:, -1, :],
        endpoint_rollout_steps,
    )
    return _kmeans_centers(converged, basin_count)


def _assign_nearest_centers(
    sequences: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    flat = sequences.reshape(-1, sequences.shape[-1])
    labels = torch.cdist(flat, centers).argmin(dim=1)
    return labels.reshape(sequences.shape[:-1]).to(dtype=torch.long)


def _label_sequences_and_centers(
    env,
    trajectories: torch.Tensor,
    *,
    system_key: str,
    endpoint_rollout_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    basin_count = int(get_transition_rich_basin_count(system_key))
    points = getattr(env, "points", None)
    centers = (
        points.to(dtype=trajectories.dtype)
        if isinstance(points, torch.Tensor) and points.ndim == 2
        else None
    )
    if system_key in NATIVE_LABEL_SYSTEMS:
        if not callable(getattr(env, "basin_label", None)) or centers is None:
            raise ValueError(
                f"Native-label system {system_key!r} must expose basin_label and points"
            )
        labels = _label_from_native_method(env.basin_label, trajectories)
        return labels, centers, NATIVE_LABEL_SOURCE
    if system_key not in PROXY_LABEL_SYSTEMS:
        raise KeyError(f"System {system_key!r} is not in the controlled paper roster")
    if callable(getattr(env, "basin_label", None)) or centers is not None:
        raise ValueError(
            f"Catalog proxy-label system {system_key!r} unexpectedly exposes "
            "native labels or centers; review the frozen evaluation protocol"
        )
    centers = _estimate_basin_centers(
        env,
        trajectories,
        basin_count,
        endpoint_rollout_steps,
    )
    return (
        _assign_nearest_centers(trajectories, centers),
        centers,
        PROXY_LABEL_SOURCE,
    )


def _encode_trajectories(model, trajectories: torch.Tensor, device: str) -> np.ndarray:
    with torch.no_grad():
        flat = trajectories.reshape(-1, trajectories.shape[-1]).to(device)
        return (
            model.encode(flat)
            .reshape(*trajectories.shape[:2], -1)
            .detach()
            .cpu()
            .numpy()
        )


def absolute_support_mask(latents: np.ndarray) -> np.ndarray:
    """Return the fixed paper support mask ``|z_i| > 1e-3``."""

    return np.abs(latents) > ABSOLUTE_THRESHOLD


def _support_keys(mask: np.ndarray) -> np.ndarray:
    packed = np.packbits(mask.astype(np.uint8), axis=-1)
    flat = packed.reshape(-1, packed.shape[-1])
    keys = np.asarray([row.tobytes() for row in flat], dtype=object)
    return keys.reshape(mask.shape[:-1])


def _binary_jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    return 1.0 if union <= 0.0 else intersection / union


def support_family_labels(
    support_mask: np.ndarray,
    *,
    min_jaccard: float = FAMILY_JACCARD_THRESHOLD,
) -> np.ndarray:
    """Greedily cluster exact supports using the frozen frequency order."""

    if support_mask.ndim != 3:
        raise ValueError(
            "support_mask must have shape [num_trajectories, length, latent_dim]"
        )
    flat_mask = support_mask.reshape(-1, support_mask.shape[-1])
    flat_keys = _support_keys(support_mask).reshape(-1)
    counts = Counter(flat_keys.tolist())
    key_masks: Dict[object, np.ndarray] = {}
    for key, mask in zip(flat_keys.tolist(), flat_mask):
        if key not in key_masks:
            key_masks[key] = mask.astype(bool, copy=True)

    prototypes: List[np.ndarray] = []
    key_to_family: Dict[object, int] = {}
    for key, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        mask = key_masks[key]
        best_family: Optional[int] = None
        best_similarity = -1.0
        for family_id, prototype in enumerate(prototypes):
            similarity = _binary_jaccard(mask, prototype)
            if similarity > best_similarity:
                best_family = family_id
                best_similarity = similarity
        if best_family is not None and best_similarity >= float(min_jaccard):
            key_to_family[key] = best_family
        else:
            key_to_family[key] = len(prototypes)
            prototypes.append(mask)
    labels = np.asarray(
        [key_to_family[key] for key in flat_keys.tolist()],
        dtype=np.int64,
    )
    return labels.reshape(support_mask.shape[:-1])


def tie_inclusive_high_center_margin_mask(
    states: torch.Tensor,
    centers: torch.Tensor,
    basin_labels: np.ndarray,
) -> np.ndarray:
    """Select margins at or above each label's q75, including all ties."""

    flat = states.reshape(-1, states.shape[-1])
    dists = torch.cdist(flat, centers.to(dtype=flat.dtype))
    if dists.shape[1] < 2:
        return np.ones(flat.shape[0], dtype=bool)
    nearest = torch.topk(dists, k=2, largest=False, dim=1).values
    margins = (nearest[:, 1] - nearest[:, 0]).cpu().numpy()
    flat_basins = np.asarray(basin_labels).reshape(-1)
    if flat_basins.size != margins.size:
        raise ValueError("basin labels and flattened states have different sizes")

    deep = np.zeros(margins.size, dtype=bool)
    for basin in np.unique(flat_basins):
        if int(basin) < 0:
            continue
        selected = flat_basins == basin
        basin_margins = margins[selected]
        if basin_margins.size < 4:
            deep[selected] = True
            continue
        cutoff = float(np.quantile(basin_margins, 0.75))
        deep[selected] = basin_margins >= cutoff
    return deep


def _entropy(counter: Counter[object]) -> float:
    total = float(sum(counter.values()))
    if total <= 0.0:
        return 0.0
    value = 0.0
    for count in counter.values():
        probability = float(count) / total
        value -= probability * math.log(probability + EPS)
    return value


def conditional_entropy(x: Sequence[object], y: Sequence[object]) -> float:
    """Return ``H(x | y)`` in natural-log units."""

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    return _entropy(Counter(zip(x, y))) - _entropy(Counter(y))


def _scored_alignment_metrics(
    family_labels: np.ndarray,
    basin_labels: np.ndarray,
    score_mask: np.ndarray,
) -> Tuple[Optional[float], float]:
    """Score entropy in nats and observed families on the high-margin slice."""
    families = family_labels.reshape(-1)[score_mask].tolist()
    basins = [int(value) for value in basin_labels.reshape(-1)[score_mask].tolist()]
    if not families:
        return None, 0.0
    return conditional_entropy(basins, families), float(len(set(families)))


def evaluate_checkpoint_alignment(
    checkpoint_path: Path,
    system_key: str,
    *,
    device: str,
) -> Dict[str, object]:
    """Evaluate one checkpoint under the immutable alignment protocol."""

    env, model = _load_checkpoint_model(checkpoint_path, system_key, device)
    trajectories = _generate_trajectories(
        env,
        num_trajectories=NUM_EVALUATION_TRAJECTORIES,
        trajectory_length=TRAJECTORY_TRANSITIONS,
        eval_seed=EVALUATION_SEED,
    )
    basin_labels, centers, label_source = _label_sequences_and_centers(
        env,
        trajectories,
        system_key=system_key,
        endpoint_rollout_steps=ENDPOINT_ROLLOUT_STEPS,
    )
    latents = _encode_trajectories(model, trajectories, device)
    families = support_family_labels(absolute_support_mask(latents))
    basin_labels_np = basin_labels.cpu().numpy()
    score_mask = tie_inclusive_high_center_margin_mask(
        trajectories, centers, basin_labels_np
    )
    entropy, family_count = _scored_alignment_metrics(
        families, basin_labels_np, score_mask
    )
    return {
        "num_states": int(score_mask.sum()),
        "observed_label_count": int(np.unique(basin_labels_np).size),
        "family_h_basin_given_family": entropy,
        "family_unique_count": family_count,
        "label_source": label_source,
    }
