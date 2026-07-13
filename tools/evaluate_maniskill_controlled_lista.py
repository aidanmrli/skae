"""Evaluate one-seed controlled LISTA/SKAE ManiSkill insertion results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from skae.benchmarks.maniskill_controlled_lista import (
    denormalize_observations,
    load_model_from_checkpoint,
    normalize_actions,
    normalize_observations,
    write_json,
)
from skae.benchmarks.maniskill_insertion_dataset import (
    CompactManiSkillDataset,
    feature_group_indices,
    load_compact_dataset,
)


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Compact ManiSkill .npz dataset")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--horizons", default="10,25,50,100")
    parser.add_argument(
        "--periodic_reencode_periods",
        default="",
        help="Comma-separated periodic re-encoding intervals. Empty keeps no-reencoding only.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--support_threshold", type=float, default=1e-3)
    parser.add_argument("--family_jaccard", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dataset = load_compact_dataset(args.dataset)
    model, stats, checkpoint = load_model_from_checkpoint(args.checkpoint, device=device)

    observations_norm = normalize_observations(dataset.observations, stats)
    actions_norm = normalize_actions(dataset.actions, stats)
    horizons = [int(item) for item in args.horizons.split(",") if item.strip()]
    periodic_reencode_periods = [
        int(item) for item in args.periodic_reencode_periods.split(",") if item.strip()
    ]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rollout_metrics, rollout_modes, best_periodic_rollout = evaluate_rollouts(
        model,
        dataset,
        observations_norm,
        actions_norm,
        stats=stats,
        split=args.split,
        horizons=horizons,
        periodic_reencode_periods=periodic_reencode_periods,
        batch_size=args.batch_size,
        device=device,
    )
    support_metrics, support_rows = evaluate_support_alignment(
        model,
        dataset,
        observations_norm,
        split=args.split,
        support_threshold=args.support_threshold,
        family_jaccard=args.family_jaccard,
        batch_size=args.batch_size,
        device=device,
    )
    summary = {
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "split": args.split,
        "device": str(device),
        "horizons": horizons,
        "periodic_reencode_periods": periodic_reencode_periods,
        "support_threshold": float(args.support_threshold),
        "family_jaccard": float(args.family_jaccard),
        "labels_used_for_training": False,
        "rollout": rollout_metrics,
        "rollout_modes": rollout_modes,
        "best_periodic_rollout": best_periodic_rollout,
        "support_alignment": support_metrics,
    }
    write_json(output_dir / "metrics_summary.json", summary)
    write_support_rows_csv(output_dir / "support_assignments.csv", support_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


@torch.no_grad()
def evaluate_rollouts(
    model,
    dataset: CompactManiSkillDataset,
    observations_norm: np.ndarray,
    actions_norm: np.ndarray,
    *,
    stats,
    split: str,
    horizons: Sequence[int],
    periodic_reencode_periods: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[str, object], Dict[str, Dict[str, object]], Dict[str, object]]:
    modes: List[Tuple[str, Optional[int]]] = [("no_reencode", None)]
    modes.extend((f"periodic_{int(period)}", int(period)) for period in periodic_reencode_periods)

    rollout_modes: Dict[str, Dict[str, object]] = {}
    for mode_name, period in modes:
        rollout_modes[mode_name] = _evaluate_rollout_mode(
            model,
            dataset,
            observations_norm,
            actions_norm,
            stats=stats,
            split=split,
            horizons=horizons,
            batch_size=batch_size,
            device=device,
            reencode_period=period,
        )

    best_periodic_rollout = _select_best_periodic_rollout(
        rollout_modes,
        horizons=horizons,
        periodic_reencode_periods=periodic_reencode_periods,
    )
    return rollout_modes["no_reencode"], rollout_modes, best_periodic_rollout


@torch.no_grad()
def _evaluate_rollout_mode(
    model,
    dataset: CompactManiSkillDataset,
    observations_norm: np.ndarray,
    actions_norm: np.ndarray,
    *,
    stats,
    split: str,
    horizons: Sequence[int],
    batch_size: int,
    device: torch.device,
    reencode_period: Optional[int],
) -> Dict[str, object]:
    indices = dataset.indices_for_split(split)
    group_indices = feature_group_indices(dataset.feature_names)
    metrics: Dict[str, object] = {
        "episode_count": int(indices.size),
        "rollout_mode": "no_reencode" if reencode_period is None else f"periodic_{int(reencode_period)}",
        "reencode_period": None if reencode_period is None else int(reencode_period),
    }
    if indices.size == 0:
        metrics["status"] = "empty_split"
        return metrics

    lengths = dataset.valid[indices].sum(axis=1).astype(np.int64)
    max_horizon = max(horizons) if horizons else 0
    eligible = indices[lengths >= 1]
    if eligible.size == 0:
        metrics["status"] = "no_valid_transitions"
        return metrics

    for horizon in horizons:
        horizon = int(horizon)
        horizon_indices = indices[lengths >= horizon]
        if horizon_indices.size == 0:
            metrics[f"h{horizon}/episode_count"] = 0
            continue

        state_sse = 0.0
        state_count = 0
        final_sse = 0.0
        final_count = 0
        group_sse: Dict[str, float] = defaultdict(float)
        group_count: Dict[str, int] = defaultdict(int)
        outcome_state_sse: Dict[int, float] = defaultdict(float)
        outcome_state_count: Dict[int, int] = defaultdict(int)
        outcome_final_sse: Dict[int, float] = defaultdict(float)
        outcome_final_count: Dict[int, int] = defaultdict(int)
        outcome_episode_count: Counter[int] = Counter()

        for batch_indices in _batches(horizon_indices, batch_size):
            x0 = torch.as_tensor(
                observations_norm[batch_indices, 0],
                dtype=torch.float32,
                device=device,
            )
            action = torch.as_tensor(
                actions_norm[batch_indices, :horizon],
                dtype=torch.float32,
                device=device,
            )
            if reencode_period is None:
                pred_norm_tensor = model.rollout_observations(x0, action)
            else:
                pred_norm_tensor = model.rollout_observations_periodic_reencode(
                    x0,
                    action,
                    period=int(reencode_period),
                )
            pred_norm = pred_norm_tensor.detach().cpu().numpy()
            pred = denormalize_observations(pred_norm, stats)
            true = dataset.observations[batch_indices, 1 : horizon + 1]
            diff = pred - true
            state_sse += float(np.square(diff).sum())
            state_count += int(np.prod(diff.shape))
            final_diff = diff[:, -1]
            final_sse += float(np.square(final_diff).sum())
            final_count += int(np.prod(final_diff.shape))
            batch_outcomes = dataset.outcome[batch_indices]
            for row_index, outcome_label in enumerate(batch_outcomes.tolist()):
                if int(outcome_label) < 0:
                    continue
                label = int(outcome_label)
                row_diff = diff[row_index]
                row_final_diff = final_diff[row_index]
                outcome_episode_count[label] += 1
                outcome_state_sse[label] += float(np.square(row_diff).sum())
                outcome_state_count[label] += int(np.prod(row_diff.shape))
                outcome_final_sse[label] += float(np.square(row_final_diff).sum())
                outcome_final_count[label] += int(np.prod(row_final_diff.shape))
            for group_name, cols in group_indices.items():
                group_diff = diff[..., cols]
                group_sse[group_name] += float(np.square(group_diff).sum())
                group_count[group_name] += int(np.prod(group_diff.shape))

        metrics[f"h{horizon}/episode_count"] = int(horizon_indices.size)
        metrics[f"h{horizon}/state_mse"] = state_sse / max(1, state_count)
        metrics[f"h{horizon}/final_state_mse"] = final_sse / max(1, final_count)
        for group_name in sorted(group_sse.keys()):
            metrics[f"h{horizon}/{group_name}_mse"] = group_sse[group_name] / max(1, group_count[group_name])
        for outcome_label in sorted(outcome_episode_count.keys()):
            outcome_name = _outcome_metric_name(dataset, outcome_label)
            metrics[f"h{horizon}/outcome_{outcome_name}_episode_count"] = int(
                outcome_episode_count[outcome_label]
            )
            metrics[f"h{horizon}/outcome_{outcome_name}_state_mse"] = (
                outcome_state_sse[outcome_label] / max(1, outcome_state_count[outcome_label])
            )
            metrics[f"h{horizon}/outcome_{outcome_name}_final_state_mse"] = (
                outcome_final_sse[outcome_label] / max(1, outcome_final_count[outcome_label])
            )

    metrics["max_requested_horizon"] = int(max_horizon)
    return metrics


def _select_best_periodic_rollout(
    rollout_modes: Mapping[str, Mapping[str, object]],
    *,
    horizons: Sequence[int],
    periodic_reencode_periods: Sequence[int],
) -> Dict[str, object]:
    metrics: Dict[str, object] = {
        "rollout_mode": "best_periodic",
        "periodic_reencode_periods": [int(period) for period in periodic_reencode_periods],
    }
    if "no_reencode" in rollout_modes:
        metrics["episode_count"] = rollout_modes["no_reencode"].get("episode_count", 0)

    for horizon in horizons:
        htag = f"h{int(horizon)}"
        candidates: List[Tuple[float, str, Mapping[str, object]]] = []
        for period in periodic_reencode_periods:
            mode_name = f"periodic_{int(period)}"
            mode_metrics = rollout_modes.get(mode_name)
            if mode_metrics is None:
                continue
            value = mode_metrics.get(f"{htag}/state_mse")
            if value is None:
                continue
            value_float = float(value)
            if not math.isfinite(value_float):
                continue
            candidates.append((value_float, mode_name, mode_metrics))

        if not candidates:
            metrics[f"{htag}/episode_count"] = 0
            metrics[f"{htag}/selected_mode"] = None
            metrics[f"{htag}/selected_period"] = None
            continue

        _value, selected_mode, selected_metrics = min(candidates, key=lambda item: item[0])
        prefix = f"{htag}/"
        for key, value in selected_metrics.items():
            if key.startswith(prefix):
                metrics[key] = value
        metrics[f"{htag}/selected_mode"] = selected_mode
        metrics[f"{htag}/selected_period"] = int(selected_mode.removeprefix("periodic_"))

    if horizons:
        metrics["max_requested_horizon"] = int(max(horizons))
    return metrics


def _outcome_metric_name(dataset: CompactManiSkillDataset, outcome_label: int) -> str:
    names = dataset.metadata.get("outcome_names", [])
    if isinstance(names, list) and 0 <= int(outcome_label) < len(names):
        raw = str(names[int(outcome_label)])
    else:
        raw = f"label_{int(outcome_label)}"
    safe_chars = [char.lower() if char.isalnum() else "_" for char in raw]
    return "".join(safe_chars).strip("_") or f"label_{int(outcome_label)}"


@torch.no_grad()
def evaluate_support_alignment(
    model,
    dataset: CompactManiSkillDataset,
    observations_norm: np.ndarray,
    *,
    split: str,
    support_threshold: float,
    family_jaccard: float,
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    val_indices = dataset.indices_for_split("val")
    target_indices = dataset.indices_for_split(split)
    if val_indices.size == 0:
        val_indices = dataset.indices_for_split("train")

    val_masks, _val_keys = encode_support_masks(
        model,
        dataset,
        observations_norm,
        val_indices,
        support_threshold=support_threshold,
        batch_size=batch_size,
        device=device,
    )
    prototypes = build_frozen_family_prototypes(val_masks, min_jaccard=family_jaccard)
    target_masks, target_state_keys = encode_support_masks(
        model,
        dataset,
        observations_norm,
        target_indices,
        support_threshold=support_threshold,
        batch_size=batch_size,
        device=device,
    )
    family_labels, similarity = assign_to_frozen_families(target_masks, prototypes)

    rows = build_support_assignment_rows(
        dataset,
        target_indices,
        target_state_keys,
        family_labels,
        similarity,
    )
    metrics: Dict[str, object] = {
        "status": "ok",
        "validation_state_count": int(val_masks.shape[0]),
        "evaluated_state_count": int(target_masks.shape[0]),
        "family_count": int(len(prototypes)),
        "mean_family_similarity": _nanmean(similarity),
        "unassigned_state_fraction": float(np.mean(family_labels < 0)) if family_labels.size else None,
        "mean_support_size": float(target_masks.sum(axis=1).mean()) if target_masks.size else None,
    }
    if target_masks.size == 0:
        metrics["status"] = "empty_split"
        return metrics, rows

    outcome_metrics = trajectory_outcome_alignment(dataset, target_indices, family_labels)
    contact_metrics = contact_phase_alignment(dataset, target_indices, family_labels)
    metrics.update(outcome_metrics)
    metrics.update(contact_metrics)
    return metrics, rows


@torch.no_grad()
def encode_support_masks(
    model,
    dataset: CompactManiSkillDataset,
    observations_norm: np.ndarray,
    episode_indices: np.ndarray,
    *,
    support_threshold: float,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    states: List[np.ndarray] = []
    keys: List[Tuple[int, int]] = []
    for episode_index in episode_indices.tolist():
        valid_transitions = dataset.valid[episode_index]
        state_valid = np.zeros((dataset.max_transitions + 1,), dtype=bool)
        state_valid[:-1] |= valid_transitions
        state_valid[1:] |= valid_transitions
        time_indices = np.nonzero(state_valid)[0]
        for time_index in time_indices.tolist():
            states.append(observations_norm[episode_index, time_index])
            keys.append((int(episode_index), int(time_index)))
    if not states:
        return np.zeros((0, int(model.cfg.z_dim)), dtype=bool), keys

    masks = []
    state_array = np.stack(states, axis=0).astype(np.float32, copy=False)
    for start in range(0, len(state_array), int(batch_size)):
        batch = torch.as_tensor(state_array[start : start + int(batch_size)], dtype=torch.float32, device=device)
        masks.append(model.support_mask(batch, threshold=support_threshold).detach().cpu().numpy())
    return np.concatenate(masks, axis=0).astype(bool), keys


def build_frozen_family_prototypes(support_mask: np.ndarray, *, min_jaccard: float) -> List[np.ndarray]:
    """Greedy validation-only support-family representatives."""

    key_counts = Counter(_support_key(mask) for mask in support_mask)
    key_masks: Dict[Tuple[int, ...], np.ndarray] = {}
    for mask in support_mask:
        key = _support_key(mask)
        if key not in key_masks:
            key_masks[key] = mask.astype(bool, copy=True)

    prototypes: List[np.ndarray] = []
    for key, _count in sorted(key_counts.items(), key=lambda item: (-item[1], item[0])):
        mask = key_masks[key]
        best = max((_binary_jaccard(mask, proto) for proto in prototypes), default=-1.0)
        if best < float(min_jaccard):
            prototypes.append(mask)
    return prototypes


def assign_to_frozen_families(
    support_mask: np.ndarray,
    prototypes: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.full((support_mask.shape[0],), -1, dtype=np.int64)
    similarity = np.full((support_mask.shape[0],), np.nan, dtype=np.float32)
    if not prototypes:
        return labels, similarity
    for index, mask in enumerate(support_mask):
        scores = np.asarray([_binary_jaccard(mask, prototype) for prototype in prototypes], dtype=np.float32)
        best = int(np.argmax(scores))
        labels[index] = best
        similarity[index] = float(scores[best])
    return labels, similarity


def trajectory_outcome_alignment(
    dataset: CompactManiSkillDataset,
    episode_indices: np.ndarray,
    state_family_labels: np.ndarray,
) -> Dict[str, object]:
    outcomes = dataset.outcome[episode_indices]
    if not np.any(outcomes >= 0):
        return {"outcome_alignment_available": False}

    trajectory_families = []
    trajectory_outcomes = []
    offset = 0
    for local_index, episode_index in enumerate(episode_indices.tolist()):
        state_count = int(dataset.valid[episode_index].sum()) + 1
        labels = state_family_labels[offset : offset + state_count]
        offset += state_count
        labels = labels[labels >= 0]
        if labels.size == 0 or outcomes[local_index] < 0:
            continue
        trajectory_families.append(_mode_int(labels.tolist()))
        trajectory_outcomes.append(int(outcomes[local_index]))

    if not trajectory_families:
        return {"outcome_alignment_available": False}
    return {
        "outcome_alignment_available": True,
        "outcome_state": class_alignment_metrics(trajectory_families, trajectory_outcomes, prefix="outcome"),
    }


def contact_phase_alignment(
    dataset: CompactManiSkillDataset,
    episode_indices: np.ndarray,
    state_family_labels: np.ndarray,
) -> Dict[str, object]:
    if dataset.contact_phase is None:
        return {"contact_phase_alignment_available": False}

    phases: List[int] = []
    families: List[int] = []
    offset = 0
    for episode_index in episode_indices.tolist():
        state_count = int(dataset.valid[episode_index].sum()) + 1
        labels = state_family_labels[offset : offset + state_count]
        phase = dataset.contact_phase[episode_index, :state_count]
        offset += state_count
        keep = (labels >= 0) & (phase >= 0)
        families.extend(labels[keep].astype(int).tolist())
        phases.extend(phase[keep].astype(int).tolist())
    if not families:
        return {"contact_phase_alignment_available": False}
    return {
        "contact_phase_alignment_available": True,
        "contact_phase": class_alignment_metrics(families, phases, prefix="contact_phase"),
    }


def class_alignment_metrics(classes: Sequence[int], labels: Sequence[int], *, prefix: str) -> Dict[str, float]:
    classes = [int(item) for item in classes]
    labels = [int(item) for item in labels]
    result = {
        f"h_{prefix}_given_family": conditional_entropy(labels, classes),
        f"h_family_given_{prefix}": conditional_entropy(classes, labels),
        f"{prefix}_family_nmi": normalized_mutual_information(classes, labels),
        f"{prefix}_family_purity": purity(classes, labels),
        f"{prefix}_sample_count": float(len(classes)),
        "unique_family_count": float(len(set(classes))),
        f"unique_{prefix}_count": float(len(set(labels))),
    }
    try:
        from sklearn.metrics import adjusted_rand_score

        result[f"{prefix}_family_ari"] = float(adjusted_rand_score(labels, classes))
    except Exception:
        result[f"{prefix}_family_ari"] = float("nan")
    return result


def conditional_entropy(values: Sequence[int], given: Sequence[int]) -> float:
    joint = Counter(zip(given, values))
    given_counts = Counter(given)
    total = float(len(values))
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for given_value, given_count in given_counts.items():
        local_total = float(given_count)
        local_entropy = 0.0
        for (_class, value), count in joint.items():
            if _class != given_value:
                continue
            p = float(count) / local_total
            local_entropy -= p * math.log(max(p, EPS))
        entropy += (local_total / total) * local_entropy
    return float(entropy)


def normalized_mutual_information(x: Sequence[int], y: Sequence[int]) -> float:
    hx = entropy(x)
    hy = entropy(y)
    if hx <= 0.0 or hy <= 0.0:
        return 0.0
    hxy = entropy(list(zip(x, y)))
    return float((hx + hy - hxy) / max(math.sqrt(hx * hy), EPS))


def entropy(values: Sequence[object]) -> float:
    counts = Counter(values)
    total = float(sum(counts.values()))
    if total <= 0.0:
        return 0.0
    out = 0.0
    for count in counts.values():
        p = float(count) / total
        out -= p * math.log(max(p, EPS))
    return float(out)


def purity(classes: Sequence[int], labels: Sequence[int]) -> float:
    by_class: Dict[int, Counter[int]] = defaultdict(Counter)
    for class_label, label in zip(classes, labels):
        by_class[int(class_label)][int(label)] += 1
    total = sum(sum(counter.values()) for counter in by_class.values())
    if total <= 0:
        return 0.0
    hits = sum(max(counter.values()) for counter in by_class.values())
    return float(hits / total)


def build_support_assignment_rows(
    dataset: CompactManiSkillDataset,
    episode_indices: np.ndarray,
    state_keys: Sequence[Tuple[int, int]],
    family_labels: np.ndarray,
    similarity: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row_index, ((episode_index, time_index), family, sim) in enumerate(
        zip(state_keys, family_labels.tolist(), similarity.tolist())
    ):
        outcome = int(dataset.outcome[episode_index]) if episode_index < len(dataset.outcome) else -1
        contact = -1
        if dataset.contact_phase is not None and time_index < dataset.contact_phase.shape[1]:
            contact = int(dataset.contact_phase[episode_index, time_index])
        rows.append(
            {
                "row_index": row_index,
                "episode_index": int(episode_index),
                "episode_id": int(dataset.episode_ids[episode_index]),
                "time_index": int(time_index),
                "family": int(family),
                "family_similarity": float(sim) if np.isfinite(sim) else "",
                "outcome": outcome,
                "contact_phase": contact,
            }
        )
    return rows


def write_support_rows_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _support_key(mask: np.ndarray) -> Tuple[int, ...]:
    return tuple(np.nonzero(np.asarray(mask, dtype=bool))[0].astype(int).tolist())


def _binary_jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    if union <= 0.0:
        return 1.0
    return inter / union


def _mode_int(values: Sequence[int]) -> int:
    return int(Counter(int(item) for item in values).most_common(1)[0][0])


def _batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), int(batch_size)):
        yield indices[start : start + int(batch_size)]


def _nanmean(values: np.ndarray) -> Optional[float]:
    if values.size == 0 or np.all(~np.isfinite(values)):
        return None
    return float(np.nanmean(values))


if __name__ == "__main__":
    main()
