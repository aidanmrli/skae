"""Evaluate the spatialized reaction-diffusion LISTA smoke result."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from skae.benchmarks.spatialized_reaction_diffusion import (
    field_modal_basin_labels,
    flatten_fields,
    load_dataset,
    reshape_flat_fields,
    spatial_gradient,
    split_fields,
)
from skae.benchmarks.spatialized_conv_koopman import SpatialConvKoopman, SpatialConvKoopmanConfig
from skae.config import Config
from skae.model import make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate forecast MSE and support-family/basin alignment for the PDE smoke benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 8, 12])
    parser.add_argument(
        "--periodic_reencode_periods",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Optional decode/re-encode periods to evaluate in addition to the "
            "default no-reencode rollout. Re-encoding uses the model's own "
            "decoded predictions, never ground-truth future states."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--support_threshold", type=float, default=1e-3)
    parser.add_argument("--family_jaccard", type=float, default=0.5)
    parser.add_argument("--max_validation_reps", type=int, default=256)
    parser.add_argument("--deep_threshold", type=float, default=0.9)
    return parser.parse_args()


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        return "cpu"
    return device_arg


def _valid_reencode_periods(values: Sequence[int]) -> List[int]:
    periods = sorted({int(value) for value in values if int(value) > 0})
    return periods


def load_model(checkpoint_path: Path, observation_size: int, device: str):
    checkpoint = _torch_load(checkpoint_path)
    if checkpoint.get("model_family") == "spatial_conv_koopman":
        model_config = SpatialConvKoopmanConfig.from_mapping(checkpoint["model_config"])
        model = SpatialConvKoopman(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        return model, {"model_family": "spatial_conv_koopman", **model_config.to_dict()}, checkpoint
    cfg = Config.from_dict(checkpoint["config"])
    model = make_model(cfg, observation_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, cfg, checkpoint


def nearest_basin_maps(fields: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Assign batched fields ``[B, X, Y, 2]`` to nearest attractor centers."""

    distances = torch.sum((fields.unsqueeze(-2) - centers.view(1, 1, 1, -1, 2)) ** 2, dim=-1)
    return distances.argmin(dim=-1).to(dtype=torch.int64)


def basin_map_scores(pred_maps: torch.Tensor, true_maps: torch.Tensor, num_basins: int) -> Dict[str, float]:
    pred = pred_maps.reshape(-1)
    truth = true_maps.reshape(-1)
    accuracy = (pred == truth).float().mean().item()
    ious: List[float] = []
    for basin in range(int(num_basins)):
        pred_mask = pred == basin
        true_mask = truth == basin
        union = torch.logical_or(pred_mask, true_mask).sum().item()
        if union == 0:
            continue
        intersection = torch.logical_and(pred_mask, true_mask).sum().item()
        ious.append(float(intersection / max(1, union)))
    return {
        "pixel_basin_accuracy": float(accuracy),
        "pixel_basin_mean_iou": float(np.mean(ious)) if ious else float("nan"),
    }


def fourier_band_mse(pred_fields: torch.Tensor, true_fields: torch.Tensor) -> Dict[str, float]:
    """Return coarse Fourier-shell MSE for fields ``[B, H, X, Y, C]``."""

    diff = pred_fields - true_fields
    spectrum = torch.fft.fft2(diff, dim=(-3, -2))
    power = spectrum.abs().square().mean(dim=(0, 1, -1))
    grid_x, grid_y = power.shape
    fx = torch.fft.fftfreq(grid_x, device=power.device) * float(grid_x)
    fy = torch.fft.fftfreq(grid_y, device=power.device) * float(grid_y)
    radius = (fx[:, None].square() + fy[None, :].square()).sqrt()
    nyquist = float(min(grid_x, grid_y) // 2)
    bands = {
        "fourier_low_mse": radius <= 0.25 * nyquist,
        "fourier_mid_mse": (radius > 0.25 * nyquist) & (radius <= 0.5 * nyquist),
        "fourier_high_mse": radius > 0.5 * nyquist,
    }
    out: Dict[str, float] = {}
    for name, mask in bands.items():
        out[name] = float(power[mask].mean().item()) if bool(mask.any()) else float("nan")
    return out


@torch.no_grad()
def forecast_metrics(
    model,
    fields_flat: torch.Tensor,
    *,
    centers: torch.Tensor,
    grid_size: int,
    true_global_labels: torch.Tensor,
    horizons: Sequence[int],
    batch_size: int,
    device: str,
    reencode_period: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    max_horizon = int(fields_flat.shape[1]) - 1
    too_long = [int(horizon) for horizon in horizons if int(horizon) > max_horizon]
    if too_long:
        raise ValueError(
            "Requested forecast horizons exceed available ground-truth fields: "
            f"max_horizon={max_horizon}, requested_too_long={too_long}. "
            "Increase trajectory_length or remove those horizons; otherwise the "
            "evaluation would silently duplicate the longest available horizon."
        )
    centers = centers.cpu()

    for requested_horizon in horizons:
        horizon = int(requested_horizon)
        if horizon < 1:
            continue
        total_sse = 0.0
        total_count = 0
        total_final_sse = 0.0
        total_final_count = 0
        total_grad_sse = 0.0
        total_grad_count = 0
        pred_labels: List[torch.Tensor] = []
        pred_majority: List[torch.Tensor] = []
        true_horizon_labels: List[torch.Tensor] = []
        true_horizon_majority: List[torch.Tensor] = []
        map_score_accumulator: List[Dict[str, float]] = []
        fourier_accumulator: List[Dict[str, float]] = []

        for start in range(0, fields_flat.shape[0], batch_size):
            batch = fields_flat[start : start + batch_size].to(device)
            truth = batch[:, 1 : horizon + 1, :]
            if reencode_period is None:
                _z_pred, pred = model.rollout_observation_discrete(batch[:, 0, :], horizon=horizon)
            else:
                _z_pred, pred = model.rollout_observation_periodic_reencode(
                    batch[:, 0, :],
                    horizon=horizon,
                    period=int(reencode_period),
                )
            diff = pred - truth
            total_sse += float(diff.square().sum().item())
            total_count += int(diff.numel())

            final_diff = diff[:, -1, :]
            total_final_sse += float(final_diff.square().sum().item())
            total_final_count += int(final_diff.numel())

            pred_field = reshape_flat_fields(pred.detach().cpu(), grid_size)
            truth_field = reshape_flat_fields(truth.detach().cpu(), grid_size)
            pred_gx, pred_gy = spatial_gradient(pred_field)
            true_gx, true_gy = spatial_gradient(truth_field)
            grad_diff_x = pred_gx - true_gx
            grad_diff_y = pred_gy - true_gy
            total_grad_sse += float(grad_diff_x.square().sum().item() + grad_diff_y.square().sum().item())
            total_grad_count += int(grad_diff_x.numel() + grad_diff_y.numel())

            labels, fractions = field_modal_basin_labels(pred_field[:, -1], centers)
            true_labels, true_fractions = field_modal_basin_labels(truth_field[:, -1], centers)
            pred_labels.append(labels)
            pred_majority.append(fractions)
            true_horizon_labels.append(true_labels)
            true_horizon_majority.append(true_fractions)
            pred_maps = nearest_basin_maps(pred_field[:, -1], centers)
            true_maps = nearest_basin_maps(truth_field[:, -1], centers)
            map_score_accumulator.append(basin_map_scores(pred_maps, true_maps, int(centers.shape[0])))
            fourier_accumulator.append(fourier_band_mse(pred_field, truth_field))

        pred_label_tensor = torch.cat(pred_labels, dim=0)
        pred_majority_tensor = torch.cat(pred_majority, dim=0)
        true_horizon_label_tensor = torch.cat(true_horizon_labels, dim=0)
        true_horizon_majority_tensor = torch.cat(true_horizon_majority, dim=0)
        horizon_consistency = (pred_label_tensor == true_horizon_label_tensor).float().mean().item()
        fate_consistency = (pred_label_tensor == true_global_labels.cpu()).float().mean().item()
        map_scores = {
            key: float(np.mean([item[key] for item in map_score_accumulator]))
            for key in ("pixel_basin_accuracy", "pixel_basin_mean_iou")
        }
        fourier_scores = {
            key: float(np.mean([item[key] for item in fourier_accumulator]))
            for key in ("fourier_low_mse", "fourier_mid_mse", "fourier_high_mse")
        }
        results[str(requested_horizon)] = {
            "effective_horizon": int(horizon),
            "rollout_mode": "no_reencode" if reencode_period is None else f"periodic_{int(reencode_period)}",
            "reencode_period": None if reencode_period is None else int(reencode_period),
            "field_mse": total_sse / max(1, total_count),
            "final_field_mse": total_final_sse / max(1, total_final_count),
            "gradient_mse": total_grad_sse / max(1, total_grad_count),
            "same_horizon_modal_basin_consistency": float(horizon_consistency),
            "final_fate_basin_consistency": float(fate_consistency),
            "final_basin_consistency": float(fate_consistency),
            "predicted_final_majority_fraction_mean": float(pred_majority_tensor.mean().item()),
            "true_horizon_majority_fraction_mean": float(true_horizon_majority_tensor.mean().item()),
            "majority_fraction_mae": float((pred_majority_tensor - true_horizon_majority_tensor).abs().mean().item()),
            **map_scores,
            **fourier_scores,
        }
    return results


@torch.no_grad()
def encode_support_masks(
    model,
    states: torch.Tensor,
    *,
    threshold: float,
    batch_size: int,
    device: str,
) -> np.ndarray:
    masks: List[np.ndarray] = []
    for start in range(0, states.shape[0], batch_size):
        batch = states[start : start + batch_size].to(device)
        z = model.encode(batch)
        masks.append((z.abs() > float(threshold)).detach().cpu().numpy().astype(bool))
    return np.concatenate(masks, axis=0)


def build_validation_representatives(
    masks: np.ndarray,
    max_reps: int,
    *,
    family_jaccard: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if masks.ndim != 2:
        raise ValueError("support masks must have shape [num_states, latent_dim].")
    unique, counts = np.unique(masks, axis=0, return_counts=True)
    if unique.shape[0] == 0:
        raise ValueError("no validation support masks available.")
    order = np.argsort(counts)[::-1]
    reps: List[np.ndarray] = []
    rep_counts: List[int] = []
    for index in order.tolist():
        mask = unique[index]
        best = max((_jaccard(mask, rep) for rep in reps), default=-1.0)
        if best < float(family_jaccard):
            reps.append(mask)
            rep_counts.append(int(counts[index]))
        if max_reps > 0 and len(reps) >= max_reps:
            break
    return np.stack(reps, axis=0), np.asarray(rep_counts, dtype=np.int64)


def nearest_representative(mask: np.ndarray, reps: np.ndarray) -> Tuple[int, float]:
    intersection = np.logical_and(reps, mask[None, :]).sum(axis=1).astype(np.float64)
    union = np.logical_or(reps, mask[None, :]).sum(axis=1).astype(np.float64)
    sims = np.where(union == 0.0, 1.0, intersection / np.maximum(union, 1.0))
    best = int(np.argmax(sims))
    return best, float(sims[best])


def _jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    if union <= 0.0:
        return 1.0
    return intersection / union


def _support_key(mask: np.ndarray) -> Tuple[int, ...]:
    """Return the deterministic exact-support key used by the paper objects."""

    return tuple(int(index) for index in np.flatnonzero(mask))


def exact_support_labels(masks: np.ndarray) -> Tuple[np.ndarray, Counter[Tuple[int, ...]]]:
    """Assign deterministic integer IDs to exact Boolean supports."""

    if masks.ndim != 2:
        raise ValueError("support masks must have shape [num_states, latent_dim].")
    keys = [_support_key(mask) for mask in masks]
    counts: Counter[Tuple[int, ...]] = Counter(keys)
    key_to_label = {key: index for index, key in enumerate(sorted(counts))}
    labels = np.asarray([key_to_label[key] for key in keys], dtype=np.int64)
    return labels, counts


def paper_support_family_labels(
    masks: np.ndarray,
    *,
    min_jaccard: float = 0.5,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Construct manuscript-style support families from exact masks.

    The algorithm counts exact masks on the evaluation collection, visits them
    in decreasing frequency with a deterministic key tie-breaker, and assigns
    each exact mask to the nearest fixed representative if its Jaccard overlap
    is at least ``min_jaccard``. Representatives are exact masks and are not
    updated after family creation.
    """

    if masks.ndim != 2:
        raise ValueError("support masks must have shape [num_states, latent_dim].")
    keys = [_support_key(mask) for mask in masks]
    key_counts: Counter[Tuple[int, ...]] = Counter(keys)
    key_masks: Dict[Tuple[int, ...], np.ndarray] = {}
    for key, mask in zip(keys, masks):
        if key not in key_masks:
            key_masks[key] = mask.astype(bool, copy=True)

    prototypes: List[np.ndarray] = []
    family_counts: List[int] = []
    key_to_family: Dict[Tuple[int, ...], int] = {}
    for key, count in sorted(key_counts.items(), key=lambda item: (-item[1], item[0])):
        mask = key_masks[key]
        best_family = None
        best_similarity = -1.0
        for family_id, prototype in enumerate(prototypes):
            similarity = _jaccard(mask, prototype)
            if similarity > best_similarity:
                best_similarity = similarity
                best_family = family_id
        if best_family is not None and best_similarity >= float(min_jaccard):
            key_to_family[key] = best_family
            family_counts[best_family] += int(count)
        else:
            key_to_family[key] = len(prototypes)
            prototypes.append(mask.astype(bool, copy=True))
            family_counts.append(int(count))

    labels = np.asarray([key_to_family[key] for key in keys], dtype=np.int64)
    return labels, prototypes, np.asarray(family_counts, dtype=np.int64)


def _mode_int(values: np.ndarray) -> int:
    if values.size == 0:
        return -1
    counts = np.bincount(values.astype(np.int64))
    return int(np.argmax(counts))


def entropy(labels: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    _unique, counts = np.unique(labels, return_counts=True)
    probs = counts.astype(np.float64) / float(counts.sum())
    return float(-(probs * np.log(np.maximum(probs, 1e-12))).sum())


def conditional_entropy(target: np.ndarray, condition: np.ndarray) -> float:
    if target.size == 0:
        return float("nan")
    total = float(target.size)
    out = 0.0
    for value in np.unique(condition):
        mask = condition == value
        out += float(mask.sum()) / total * entropy(target[mask])
    return float(out)


def purity_score(labels: np.ndarray, families: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    hits = 0
    for family in np.unique(families):
        subset = labels[families == family]
        _unique, counts = np.unique(subset, return_counts=True)
        hits += int(counts.max())
    return float(hits / labels.size)


def clustering_scores(labels: np.ndarray, families: np.ndarray) -> Dict[str, float]:
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    except ImportError:
        return {"nmi": float("nan"), "ari": float("nan")}
    return {
        "nmi": float(normalized_mutual_info_score(labels, families)),
        "ari": float(adjusted_rand_score(labels, families)),
    }


def dominant_object_mass_per_basin(objects: np.ndarray, basins: np.ndarray) -> float:
    if objects.size == 0:
        return float("nan")
    object_by_basin: Dict[int, Counter[int]] = defaultdict(Counter)
    for obj, basin in zip(objects.tolist(), basins.tolist()):
        object_by_basin[int(basin)][int(obj)] += 1
    masses: List[float] = []
    for counter in object_by_basin.values():
        total = float(sum(counter.values()))
        if total > 0.0:
            masses.append(float(max(counter.values()) / total))
    return float(np.mean(masses)) if masses else float("nan")


def object_purity_diagnostics(labels: np.ndarray, objects: np.ndarray) -> Dict[str, float]:
    if labels.size == 0:
        return {
            "object_pure_fraction_unweighted": float("nan"),
            "object_pure_fraction_weighted": float("nan"),
        }
    pure_objects = 0
    pure_states = 0
    object_ids = np.unique(objects)
    for obj in object_ids:
        keep = objects == obj
        unique_labels = np.unique(labels[keep])
        if unique_labels.size == 1:
            pure_objects += 1
            pure_states += int(keep.sum())
    return {
        "object_pure_fraction_unweighted": float(pure_objects / max(1, object_ids.size)),
        "object_pure_fraction_weighted": float(pure_states / max(1, labels.size)),
    }


def summarize_alignment(labels: np.ndarray, families: np.ndarray, jaccards: np.ndarray) -> Dict[str, float]:
    if labels.size == 0:
        return {
            "num_test_trajectories": 0,
            "num_test_families": 0,
            "h_basin_given_family": float("nan"),
            "h_family_given_basin": float("nan"),
            "purity": float("nan"),
            "nmi": float("nan"),
            "ari": float("nan"),
            "mean_state_to_rep_jaccard": float("nan"),
        }
    scores = clustering_scores(labels, families)
    return {
        "num_test_trajectories": int(labels.size),
        "num_test_families": int(np.unique(families).size),
        "h_basin_given_family": conditional_entropy(labels, families),
        "h_family_given_basin": conditional_entropy(families, labels),
        "purity": purity_score(labels, families),
        "nmi": scores["nmi"],
        "ari": scores["ari"],
        "mean_state_to_rep_jaccard": float(np.mean(jaccards)) if jaccards.size else float("nan"),
    }


def summarize_paper_support_collection(
    labels: np.ndarray,
    masks: np.ndarray,
    *,
    support_threshold: float = 1e-3,
    family_jaccard: float = 0.5,
) -> Dict[str, object]:
    """Summarize paper-defined ``S_abs`` and ``F_abs`` on one state collection."""

    if labels.size != masks.shape[0]:
        raise ValueError("labels and masks must have the same number of states.")
    support_sizes = masks.sum(axis=1).astype(np.int64, copy=False)
    if labels.size == 0:
        empty_scores = {
            "num_states": 0,
            "num_represented_basins": 0,
            "basin_entropy": float("nan"),
            "support_threshold": float(support_threshold),
            "family_jaccard": float(family_jaccard),
            "s_abs": {
                "exact_support_count": 0,
                "h_basin_given_s_abs": float("nan"),
                "h_s_abs_given_basin": float("nan"),
                "u_exact": float("nan"),
                "support_size_mean": float("nan"),
                "support_size_median": float("nan"),
                "support_size_min": float("nan"),
                "support_size_max": float("nan"),
                "zero_support_fraction": float("nan"),
            },
            "f_abs": {
                "family_count": 0,
                "h_basin_given_f_abs": float("nan"),
                "h_f_abs_given_basin": float("nan"),
                "purity": float("nan"),
                "nmi": float("nan"),
                "ari": float("nan"),
            },
        }
        return empty_scores

    exact_labels, exact_counts = exact_support_labels(masks)
    family_labels, family_prototypes, family_counts = paper_support_family_labels(
        masks,
        min_jaccard=family_jaccard,
    )
    exact_scores = clustering_scores(labels, exact_labels)
    family_scores = clustering_scores(labels, family_labels)
    exact_purity = object_purity_diagnostics(labels, exact_labels)
    family_purity = object_purity_diagnostics(labels, family_labels)
    support_counts_by_basin = []
    family_counts_by_basin = []
    for basin in np.unique(labels):
        keep = labels == basin
        support_counts_by_basin.append(int(np.unique(exact_labels[keep]).size))
        family_counts_by_basin.append(int(np.unique(family_labels[keep]).size))
    representative_sizes = (
        np.asarray([prototype.sum() for prototype in family_prototypes], dtype=np.float64)
        if family_prototypes
        else np.asarray([], dtype=np.float64)
    )

    return {
        "num_states": int(labels.size),
        "num_represented_basins": int(np.unique(labels).size),
        "basin_entropy": entropy(labels),
        "support_threshold": float(support_threshold),
        "family_jaccard": float(family_jaccard),
        "s_abs": {
            "exact_support_count": int(len(exact_counts)),
            "h_basin_given_s_abs": conditional_entropy(labels, exact_labels),
            "h_s_abs_given_basin": conditional_entropy(exact_labels, labels),
            "u_exact": dominant_object_mass_per_basin(exact_labels, labels),
            "purity": purity_score(labels, exact_labels),
            "nmi": exact_scores["nmi"],
            "ari": exact_scores["ari"],
            **exact_purity,
            "support_size_mean": float(np.mean(support_sizes)),
            "support_size_median": float(np.median(support_sizes)),
            "support_size_min": int(np.min(support_sizes)),
            "support_size_max": int(np.max(support_sizes)),
            "zero_support_fraction": float(np.mean(support_sizes == 0)),
            "top_exact_support_counts": [
                int(count) for _key, count in exact_counts.most_common(10)
            ],
            "exact_supports_per_basin_mean": float(np.mean(support_counts_by_basin)),
            "exact_supports_per_basin_max": int(np.max(support_counts_by_basin)),
        },
        "f_abs": {
            "family_count": int(family_counts.size),
            "h_basin_given_f_abs": conditional_entropy(labels, family_labels),
            "h_f_abs_given_basin": conditional_entropy(family_labels, labels),
            "u_family": dominant_object_mass_per_basin(family_labels, labels),
            "purity": purity_score(labels, family_labels),
            "nmi": family_scores["nmi"],
            "ari": family_scores["ari"],
            **family_purity,
            "top_family_counts": [int(count) for count in family_counts[:10].tolist()],
            "families_per_basin_mean": float(np.mean(family_counts_by_basin)),
            "families_per_basin_max": int(np.max(family_counts_by_basin)),
            "representative_support_size_mean": float(np.mean(representative_sizes))
            if representative_sizes.size
            else float("nan"),
            "representative_support_size_median": float(np.median(representative_sizes))
            if representative_sizes.size
            else float("nan"),
        },
    }


@torch.no_grad()
def paper_support_object_metrics(
    model,
    test_fields_flat: torch.Tensor,
    *,
    test_global_labels: torch.Tensor,
    test_majority_fractions: torch.Tensor,
    threshold: float = 1e-3,
    family_jaccard: float = 0.5,
    batch_size: int,
    deep_threshold: float,
    device: str,
) -> Dict[str, object]:
    """Evaluate manuscript support objects on held-out PDE states.

    Labels are the trajectory-level fate basin labels stored by the benchmark
    for evaluation only. The support objects themselves are constructed only
    from encoder outputs on the selected held-out state collection.
    """

    test_states = test_fields_flat.reshape(-1, test_fields_flat.shape[-1])
    masks_all = encode_support_masks(
        model,
        test_states,
        threshold=threshold,
        batch_size=batch_size,
        device=device,
    ).reshape(test_fields_flat.shape[0], test_fields_flat.shape[1], -1)
    labels_by_trajectory = test_global_labels.cpu().numpy().astype(np.int64)
    deep_trajectory_mask = test_majority_fractions.cpu().numpy() >= float(deep_threshold)

    labels_all = np.repeat(labels_by_trajectory, test_fields_flat.shape[1])
    masks_flat = masks_all.reshape(-1, masks_all.shape[-1])
    repeated_deep_mask = np.repeat(deep_trajectory_mask, test_fields_flat.shape[1])

    final_masks = masks_all[:, -1, :]
    final_labels = labels_by_trajectory

    collections = {
        "all_test_states": (labels_all, masks_flat),
        "deep_test_states": (labels_all[repeated_deep_mask], masks_flat[repeated_deep_mask]),
        "final_test_states": (final_labels, final_masks),
        "deep_final_test_states": (
            final_labels[deep_trajectory_mask],
            final_masks[deep_trajectory_mask],
        ),
    }
    return {
        "support_rule": "S_abs",
        "support_threshold": float(threshold),
        "family_rule": "F_abs",
        "family_jaccard": float(family_jaccard),
        "family_algorithm": "frequency_ordered_greedy_jaccard_fixed_representatives",
        "label_source": "evaluation-only trajectory fate basin labels",
        "deep_slice": {
            "criterion": "trajectory final fate majority fraction",
            "threshold": float(deep_threshold),
            "num_deep_trajectories": int(deep_trajectory_mask.sum()),
            "num_test_trajectories": int(labels_by_trajectory.size),
        },
        "collections": {
            name: summarize_paper_support_collection(
                labels,
                masks,
                support_threshold=threshold,
                family_jaccard=family_jaccard,
            )
            for name, (labels, masks) in collections.items()
        },
    }


@torch.no_grad()
def support_alignment_metrics(
    model,
    val_fields_flat: torch.Tensor,
    test_fields_flat: torch.Tensor,
    *,
    test_global_labels: torch.Tensor,
    test_majority_fractions: torch.Tensor,
    threshold: float,
    family_jaccard: float,
    max_validation_reps: int,
    batch_size: int,
    deep_threshold: float,
    device: str,
) -> Dict[str, object]:
    val_states = val_fields_flat.reshape(-1, val_fields_flat.shape[-1])
    val_masks = encode_support_masks(
        model,
        val_states,
        threshold=threshold,
        batch_size=batch_size,
        device=device,
    )
    reps, rep_counts = build_validation_representatives(
        val_masks,
        max_validation_reps,
        family_jaccard=family_jaccard,
    )

    test_states = test_fields_flat.reshape(-1, test_fields_flat.shape[-1])
    test_masks = encode_support_masks(
        model,
        test_states,
        threshold=threshold,
        batch_size=batch_size,
        device=device,
    ).reshape(test_fields_flat.shape[0], test_fields_flat.shape[1], -1)
    val_support_sizes = val_masks.sum(axis=1)
    test_support_sizes = test_masks.reshape(-1, test_masks.shape[-1]).sum(axis=1)

    trajectory_families: List[int] = []
    trajectory_mean_jaccard: List[float] = []
    for traj_masks in test_masks:
        state_families = []
        state_jaccards = []
        for state_mask in traj_masks:
            family, jaccard = nearest_representative(state_mask, reps)
            state_families.append(family)
            state_jaccards.append(jaccard)
        trajectory_families.append(_mode_int(np.asarray(state_families)))
        trajectory_mean_jaccard.append(float(np.mean(state_jaccards)))

    labels = test_global_labels.cpu().numpy().astype(np.int64)
    families = np.asarray(trajectory_families, dtype=np.int64)
    jaccards = np.asarray(trajectory_mean_jaccard, dtype=np.float64)
    deep_mask = test_majority_fractions.cpu().numpy() >= float(deep_threshold)

    return {
        "support_threshold": float(threshold),
        "family_jaccard": float(family_jaccard),
        "validation_state_count": int(val_masks.shape[0]),
        "validation_unique_supports_before_cap": int(np.unique(val_masks, axis=0).shape[0]),
        "validation_representative_count": int(reps.shape[0]),
        "validation_representative_count_cap": int(max_validation_reps),
        "validation_representative_top_counts": [int(x) for x in rep_counts[:10].tolist()],
        "validation_support_size_mean": float(np.mean(val_support_sizes)),
        "validation_support_size_median": float(np.median(val_support_sizes)),
        "test_support_size_mean": float(np.mean(test_support_sizes)),
        "test_support_size_median": float(np.median(test_support_sizes)),
        "all_test": summarize_alignment(labels, families, jaccards),
        "deep_test": summarize_alignment(labels[deep_mask], families[deep_mask], jaccards[deep_mask]),
        "deep_threshold": float(deep_threshold),
    }


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    metadata = dataset["metadata"]
    test_indices = dataset["split_indices"]["test"]
    val_fields_flat = flatten_fields(split_fields(dataset, "val")).float()
    test_fields = split_fields(dataset, "test").float()
    test_fields_flat = flatten_fields(test_fields).float()
    observation_size = int(test_fields_flat.shape[-1])
    grid_size = int(metadata["grid_size"])
    centers = dataset["attractor_centers"].float()
    device = resolve_device(args.device)

    model, cfg_or_info, checkpoint = load_model(Path(args.checkpoint), observation_size, device)
    test_global_labels = dataset["global_basin_labels"][test_indices]
    test_majority_fractions = dataset["majority_fractions"][test_indices]

    forecast = forecast_metrics(
        model,
        test_fields_flat,
        centers=centers,
        grid_size=grid_size,
        true_global_labels=test_global_labels,
        horizons=args.horizons,
        batch_size=int(args.batch_size),
        device=device,
    )
    reencode_periods = _valid_reencode_periods(args.periodic_reencode_periods)
    forecast_modes: Dict[str, Dict[str, Dict[str, float]]] = {"no_reencode": forecast}
    for period in reencode_periods:
        forecast_modes[f"periodic_{period}"] = forecast_metrics(
            model,
            test_fields_flat,
            centers=centers,
            grid_size=grid_size,
            true_global_labels=test_global_labels,
            horizons=args.horizons,
            batch_size=int(args.batch_size),
            device=device,
            reencode_period=period,
        )
    support = support_alignment_metrics(
        model,
        val_fields_flat,
        test_fields_flat,
        test_global_labels=test_global_labels,
        test_majority_fractions=test_majority_fractions,
        threshold=float(args.support_threshold),
        family_jaccard=float(args.family_jaccard),
        max_validation_reps=int(args.max_validation_reps),
        batch_size=int(args.batch_size),
        deep_threshold=float(args.deep_threshold),
        device=device,
    )
    paper_support = paper_support_object_metrics(
        model,
        test_fields_flat,
        test_global_labels=test_global_labels,
        test_majority_fractions=test_majority_fractions,
        threshold=1e-3,
        family_jaccard=0.5,
        batch_size=int(args.batch_size),
        deep_threshold=float(args.deep_threshold),
        device=device,
    )

    results = {
        "status": "completed",
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "device": device,
        "source_system": metadata.get("source_system_name"),
        "grid_size": grid_size,
        "diffusion": float(metadata["diffusion"]),
        "stored_dt": float(metadata["stored_dt"]),
        "model": {
            "model_family": checkpoint.get("model_family", "skae_flat"),
            "config_env_name": getattr(getattr(cfg_or_info, "ENV", object()), "ENV_NAME", None)
            if not isinstance(cfg_or_info, dict)
            else cfg_or_info.get("model_family"),
            "model_name": getattr(getattr(cfg_or_info, "MODEL", object()), "MODEL_NAME", "SpatialConvKoopman")
            if not isinstance(cfg_or_info, dict)
            else "SpatialConvKoopman",
            "target_size": int(getattr(getattr(cfg_or_info, "MODEL", object()), "TARGET_SIZE", cfg_or_info.get("z_dim", 0) if isinstance(cfg_or_info, dict) else 0)),
            "lista_num_loops": int(getattr(getattr(getattr(cfg_or_info, "MODEL", object()), "ENCODER", object()), "LISTA", object()).NUM_LOOPS)
            if not isinstance(cfg_or_info, dict)
            else int(cfg_or_info.get("lista_loops", 0)),
        },
        "forecast": forecast,
        "forecast_modes": forecast_modes,
        "periodic_reencode_periods": reencode_periods,
        "support_family_alignment": support,
        "paper_support_objects": paper_support,
        "label_policy": metadata.get("training_label_policy"),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
