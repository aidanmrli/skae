#!/usr/bin/env python3
"""Frozen-checkpoint test of support-specific invariant subspaces in one global K."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from skae.checkpoint_compat import load_model_state_dict_compat
from skae.config import Config
from skae.data import VectorWrapper, make_env
from skae.model import make_model

EPS = 1e-12
DEFAULT_CARD = Path(__file__).with_name("global_k_support_invariance_card.json")


@dataclass(frozen=True)
class RunSpec:
    root_label: str
    system_key: str
    system_name: str
    seed: int
    run_dir: str


@dataclass(frozen=True)
class FamilyCodebook:
    representatives: np.ndarray
    exact_key_to_family: dict[bytes, int]
    fit_counts: np.ndarray


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_card(path: Path = DEFAULT_CARD) -> tuple[dict[str, Any], str]:
    card = json.loads(path.read_text())
    return card, sha256_path(path)


def _timestamp_key(run_dir: str) -> tuple[str, str]:
    stem = Path(run_dir).name
    return (stem if re.fullmatch(r"\d{8}-\d{6}", stem) else "", run_dir)


def discover_primary_roster(card: dict[str, Any]) -> list[RunSpec]:
    frozen = card["primary_sparse"]
    rows_csv = Path(frozen["rows_csv"])
    actual_hash = sha256_path(rows_csv)
    if actual_hash != frozen["rows_csv_sha256"]:
        raise RuntimeError(f"Roster CSV hash mismatch: {actual_hash}")
    systems, seeds = set(frozen["systems"]), set(frozen["seeds"])
    best: dict[tuple[str, int], dict[str, str]] = {}
    with rows_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["root_label"] != frozen["root_label"]:
                continue
            key = (row["system_key"], int(row["seed"]))
            if key[0] not in systems or key[1] not in seeds:
                continue
            incumbent = best.get(key)
            if incumbent is None or _timestamp_key(row["run_dir"]) > _timestamp_key(incumbent["run_dir"]):
                best[key] = row
    expected_keys = {(system, seed) for system in systems for seed in seeds}
    if set(best) != expected_keys:
        missing = sorted(expected_keys - set(best))
        extra = sorted(set(best) - expected_keys)
        raise RuntimeError(f"Frozen roster mismatch; missing={missing}, extra={extra}")
    specs = [
        RunSpec(
            root_label=row["root_label"],
            system_key=row["system_key"],
            system_name=row["system_name"],
            seed=int(row["seed"]),
            run_dir=row["run_dir"],
        )
        for row in best.values()
    ]
    specs.sort(key=lambda item: (item.system_key, item.seed))
    if len(specs) != int(frozen["expected_run_count"]):
        raise RuntimeError(f"Expected {frozen['expected_run_count']} runs, found {len(specs)}")
    missing_checkpoints = [str(Path(spec.run_dir) / "checkpoint.pt") for spec in specs
                           if not (Path(spec.run_dir) / "checkpoint.pt").is_file()]
    if missing_checkpoints:
        raise FileNotFoundError(f"Missing checkpoints: {missing_checkpoints}")
    return specs


def _load_model(spec: RunSpec, device: str):
    checkpoint_path = Path(spec.run_dir) / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = Config.from_dict(checkpoint["config"])
    cfg.ENV.ENV_NAME = spec.system_key
    env = make_env(cfg)
    model = make_model(cfg, env.observation_size)
    load_model_state_dict_compat(model, checkpoint["model_state_dict"])
    return cfg, env, model.to(device).eval(), checkpoint_path


def assert_sign_split_layout(cfg: Config, model) -> int:
    encoder_kind = str(cfg.MODEL.ENCODER.ENCODER_TYPE).lower()
    final_op = str(cfg.MODEL.ENCODER.LISTA.FINAL_OP).lower()
    latent_dim = int(cfg.MODEL.TARGET_SIZE)
    if encoder_kind != "lista" or final_op != "sign_split" or latent_dim % 2:
        raise AssertionError((encoder_kind, final_op, latent_dim))
    if not bool(getattr(model, "_uses_sign_split_latent", False)):
        raise AssertionError("Loaded model does not declare a sign-split latent")
    encoder = model.encoder
    base_dim = latent_dim // 2
    if int(encoder.base_zdim) != base_dim or not hasattr(encoder, "_split_sign"):
        raise AssertionError("Encoder sign-split dimensions/method do not match config")
    probe = torch.linspace(-2.0, 2.0, base_dim, device=model.dict.device, dtype=model.dict.dtype)
    with torch.no_grad():
        observed = encoder._split_sign(probe)
    expected = torch.cat([F.relu(probe), F.relu(-probe)], dim=-1)
    if observed.shape[-1] != latent_dim or not torch.equal(observed, expected):
        raise AssertionError("Sign-split ordering is not [relu(u), relu(-u)]")
    return base_dim


def _encode(model, trajectories: torch.Tensor, device: str, batch_size: int) -> np.ndarray:
    flat = trajectories.reshape(-1, trajectories.shape[-1])
    chunks = []
    with torch.no_grad():
        for start in range(0, flat.shape[0], batch_size):
            chunks.append(model.encode(flat[start:start + batch_size].to(device)).cpu())
    return torch.cat(chunks).reshape(*trajectories.shape[:2], -1).numpy().astype(np.float32)


def _mask_keys(mask: np.ndarray) -> list[bytes]:
    packed = np.packbits(mask.astype(np.uint8), axis=-1).reshape(-1, (mask.shape[-1] + 7) // 8)
    return [row.tobytes() for row in packed]


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.logical_or(left, right).sum())
    return 1.0 if union == 0 else float(np.logical_and(left, right).sum() / union)


def fit_family_codebook(mask: np.ndarray, min_jaccard: float) -> FamilyCodebook:
    if mask.ndim != 2:
        raise ValueError("mask must have shape [states, latent_dim]")
    keys = _mask_keys(mask)
    counts = Counter(keys)
    key_mask: dict[bytes, np.ndarray] = {}
    for key, row in zip(keys, mask):
        key_mask.setdefault(key, row.astype(bool, copy=True))
    representatives: list[np.ndarray] = []
    mapping: dict[bytes, int] = {}
    for key, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        similarities = [_jaccard(key_mask[key], rep) for rep in representatives]
        best = int(np.argmax(similarities)) if similarities else -1
        if best >= 0 and similarities[best] >= min_jaccard:
            mapping[key] = best
        else:
            mapping[key] = len(representatives)
            representatives.append(key_mask[key])
    labels = np.asarray([mapping[key] for key in keys], dtype=np.int64)
    fit_counts = np.bincount(labels, minlength=len(representatives)).astype(np.int64)
    return FamilyCodebook(np.stack(representatives), mapping, fit_counts)


def assign_families(mask: np.ndarray, codebook: FamilyCodebook, min_jaccard: float) -> np.ndarray:
    labels = np.full(mask.shape[0], -1, dtype=np.int64)
    cache: dict[bytes, int] = {}
    for index, (key, row) in enumerate(zip(_mask_keys(mask), mask)):
        family = codebook.exact_key_to_family.get(key)
        if family is None:
            family = cache.get(key)
        if family is None:
            similarities = np.asarray([_jaccard(row, rep) for rep in codebook.representatives])
            best = int(similarities.argmax())
            family = best if similarities[best] >= min_jaccard else -1
            cache[key] = family
        labels[index] = family
    return labels


def sign_pair_permutations(base_dim: int, count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    permutations = []
    for _ in range(count):
        base = rng.permutation(base_dim)
        permutations.append(np.concatenate([base, base + base_dim]))
    return permutations


def _sum_sq(array: np.ndarray) -> float:
    return float(np.sum(np.square(array, dtype=np.float64), dtype=np.float64))


def _rms_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    den = _sum_sq(denominator)
    return None if den <= EPS else math.sqrt(_sum_sq(numerator) / den)


def _median_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    num = np.linalg.norm(numerator, axis=1)
    den = np.linalg.norm(denominator, axis=1)
    valid = den > EPS
    return float(np.median(num[valid] / den[valid])) if np.any(valid) else None


def transition_metrics(z: np.ndarray, z_next: np.ndarray, masks: np.ndarray, k_matrix: np.ndarray) -> dict[str, Any]:
    mask = masks.astype(np.float32, copy=False)
    source = z * mask
    gated = source @ k_matrix
    inside = gated * mask
    outside = gated * (1.0 - mask)
    source_change = gated - source
    global_pred = z @ k_matrix
    target_inside = z_next * mask
    target_outside = z_next * (1.0 - mask)
    global_inside = global_pred * mask
    metrics = {
        "transition_count": int(z.shape[0]),
        "activity_k_leakage_rms": _rms_ratio(outside, gated),
        "activity_k_leakage_median": _median_ratio(outside, gated),
        "activity_k_closure_rms": _rms_ratio(inside, gated),
        "activity_k_closure_median": _median_ratio(inside, gated),
        "activity_k_change_leakage_rms": _rms_ratio(outside, source_change),
        "activity_k_change_leakage_median": _median_ratio(outside, source_change),
        "source_capture_rms": _rms_ratio(source, z),
        "encoded_next_outside_rms": _rms_ratio(target_outside, z_next),
        "global_latent_residual_rms": _rms_ratio(global_pred - z_next, z_next),
        "identity_latent_residual_rms": _rms_ratio(z - z_next, z_next),
        "source_gated_full_residual_rms": _rms_ratio(gated - z_next, z_next),
        "posthoc_pkp_full_residual_rms": _rms_ratio(inside - z_next, z_next),
        "posthoc_pkp_inside_residual_rms": _rms_ratio(inside - target_inside, target_inside),
        "unmodified_k_projected_inside_residual_rms": _rms_ratio(global_inside - target_inside, target_inside),
    }
    identity = metrics["identity_latent_residual_rms"]
    global_residual = metrics["global_latent_residual_rms"]
    projected = metrics["unmodified_k_projected_inside_residual_rms"]
    restricted = metrics["posthoc_pkp_inside_residual_rms"]
    metrics["global_k_over_identity_residual"] = None if not identity else global_residual / identity
    metrics["posthoc_pkp_over_unmodified_k_inside_residual"] = None if not projected else restricted / projected
    leak, closure = metrics["activity_k_leakage_rms"], metrics["activity_k_closure_rms"]
    metrics["closure_pythagorean_error"] = None if leak is None else abs(leak * leak + closure * closure - 1.0)
    return metrics


def matrix_metrics(mask: np.ndarray, k_matrix: np.ndarray) -> dict[str, float | None]:
    source_rows = k_matrix * mask[:, None]
    inside = source_rows * mask[None, :]
    outside = source_rows * (~mask)[None, :]
    source_change = source_rows - np.diag(mask.astype(k_matrix.dtype, copy=False))
    return {
        "matrix_k_leakage_fro": _rms_ratio(outside, source_rows),
        "matrix_k_closure_fro": _rms_ratio(inside, source_rows),
        "matrix_k_change_leakage_fro": _rms_ratio(outside, source_change),
        "restricted_operator_fro": math.sqrt(_sum_sq(inside)),
    }


def operator_differentiation(representatives: np.ndarray, k_matrix: np.ndarray) -> dict[str, Any]:
    operators = [k_matrix * mask[:, None] * mask[None, :] for mask in representatives]
    cosine, distance = [], []
    for left_index in range(len(operators)):
        for right_index in range(left_index + 1, len(operators)):
            left, right = operators[left_index], operators[right_index]
            left_norm, right_norm = math.sqrt(_sum_sq(left)), math.sqrt(_sum_sq(right))
            if left_norm > EPS and right_norm > EPS:
                cosine.append(1.0 - float(np.sum(left * right, dtype=np.float64)) / (left_norm * right_norm))
                distance.append(math.sqrt(_sum_sq(left - right)) / (0.5 * (left_norm + right_norm)))
    return {
        "family_count": len(operators),
        "pair_count": len(distance),
        "mean_cosine_dissimilarity": float(np.mean(cosine)) if cosine else None,
        "mean_symmetric_normalized_frobenius_distance": float(np.mean(distance)) if distance else None,
    }


def evaluate_regime(z: np.ndarray, z_next: np.ndarray, labels: np.ndarray, representatives: np.ndarray,
                    fit_counts: np.ndarray, k_matrix: np.ndarray, min_family_score: int) -> dict[str, Any]:
    keep = labels >= 0
    selected_z, selected_next, selected_labels = z[keep], z_next[keep], labels[keep]
    masks = representatives[selected_labels]
    aggregate = transition_metrics(selected_z, selected_next, masks, k_matrix)
    family_rows, operator_ids = [], []
    score_counts = Counter(selected_labels.tolist())
    for family in sorted(score_counts):
        family_keep = selected_labels == family
        row = {
            "family": int(family),
            "fit_source_count": int(fit_counts[family]),
            "score_transition_count": int(family_keep.sum()),
            "representative_cardinality": int(representatives[family].sum()),
            **matrix_metrics(representatives[family], k_matrix),
            **transition_metrics(selected_z[family_keep], selected_next[family_keep],
                                 masks[family_keep], k_matrix),
        }
        family_rows.append(row)
        if int(family_keep.sum()) >= min_family_score:
            operator_ids.append(family)
    weights = np.asarray([row["score_transition_count"] for row in family_rows], dtype=np.float64)
    for metric in ("matrix_k_leakage_fro", "matrix_k_closure_fro", "matrix_k_change_leakage_fro"):
        values = np.asarray([row[metric] for row in family_rows], dtype=np.float64)
        aggregate[f"{metric}_activity_weighted_mean"] = (
            float(np.average(values, weights=weights)) if weights.size and weights.sum() > 0 else None
        )
    operator = operator_differentiation(representatives[operator_ids], k_matrix)
    return {"aggregate": aggregate, "families": family_rows, "operator": operator}


def _distribution_summary(records: Sequence[dict[str, Any]], section: str) -> dict[str, Any]:
    keys = sorted(set.intersection(*(set(record[section]) for record in records))) if records else []
    summary = {}
    for key in keys:
        values = [record[section][key] for record in records]
        clean = np.asarray([value for value in values if isinstance(value, (int, float)) and math.isfinite(value)],
                           dtype=np.float64)
        if clean.size:
            summary[key] = {
                "median": float(np.median(clean)),
                "q025": float(np.quantile(clean, 0.025)),
                "q975": float(np.quantile(clean, 0.975)),
            }
    return summary


def evaluate_checkpoint(spec: RunSpec, card: dict[str, Any], card_hash: str, device: str,
                        encode_batch_size: int, task_index: int) -> dict[str, Any]:
    started = time.time()
    cfg, env, model, checkpoint_path = _load_model(spec, device)
    base_dim = assert_sign_split_layout(cfg, model)
    corpus, support, eligibility, null = (card[key] for key in ("corpus", "support", "eligibility", "null"))
    trajectories = VectorWrapper(env, int(corpus["num_trajectories"])).generate_sequence_batch(
        rng=torch.Generator().manual_seed(int(corpus["eval_seed"])),
        window_length=int(corpus["trajectory_length"]),
    ).float()
    order = np.random.default_rng(int(corpus["split_seed"])).permutation(trajectories.shape[0])
    fit_ids, score_ids = order[:int(corpus["fit_trajectories"])], order[int(corpus["fit_trajectories"]):]
    latents = _encode(model, trajectories, device, encode_batch_size)
    fit_source = latents[fit_ids, :-1].reshape(-1, latents.shape[-1])
    threshold, jaccard = float(support["threshold"]), float(support["family_jaccard_threshold"])
    codebook = fit_family_codebook(np.abs(fit_source) > threshold, jaccard)
    retained = codebook.fit_counts >= int(support["min_fit_source_transitions"])
    score = latents[score_ids]
    z, z_next = score[:, :-1].reshape(-1, latents.shape[-1]), score[:, 1:].reshape(-1, latents.shape[-1])
    current = assign_families(np.abs(z) > threshold, codebook, jaccard)
    following = assign_families(np.abs(z_next) > threshold, codebook, jaccard)
    current_valid = (current >= 0) & retained[np.maximum(current, 0)]
    following_valid = (following >= 0) & retained[np.maximum(following, 0)]
    all_labels = np.where(current_valid, current, -1)
    persistent = current_valid & following_valid & (current == following)
    persistent_labels = np.where(persistent, current, -1)
    coverage = float(current_valid.mean())
    persistent_coverage = float(persistent.mean())
    eligible = (int(retained.sum()) >= int(eligibility["min_retained_families"])
                and coverage >= float(eligibility["min_current_coverage"])
                and persistent_coverage >= float(eligibility["min_persistent_coverage"]))
    k_matrix = model.kmatrix().detach().cpu().numpy().astype(np.float32)
    regimes = {}
    permutations = sign_pair_permutations(base_dim, int(null["replicates"]), int(null["seed"]))
    for name, labels in (("all_current", all_labels), ("persistent_family", persistent_labels)):
        true = evaluate_regime(z, z_next, labels, codebook.representatives, codebook.fit_counts,
                               k_matrix, int(support["min_family_score_transitions"]))
        null_records = []
        for permutation in permutations:
            permuted = evaluate_regime(z[:, permutation], z_next[:, permutation], labels,
                                       codebook.representatives[:, permutation], codebook.fit_counts,
                                       k_matrix, int(support["min_family_score_transitions"]))
            null_records.append({"aggregate": permuted["aggregate"], "operator": permuted["operator"]})
        regimes[name] = {
            "true": true,
            "null_replicates": null_records,
            "null_summary": {
                "aggregate": _distribution_summary(null_records, "aggregate"),
                "operator": _distribution_summary(null_records, "operator"),
            },
        }
    return {
        "schema_version": 1,
        "status": "eligible" if eligible else "ineligible",
        "task_index": task_index,
        "card_sha256": card_hash,
        "provenance": {
            "root_label": spec.root_label,
            "system_key": spec.system_key,
            "system_name": spec.system_name,
            "seed": spec.seed,
            "run_dir": spec.run_dir,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_path(checkpoint_path),
            "git_commit": os.environ.get("SKAE_GIT_COMMIT", "launcher_not_recorded"),
        },
        "assertions": {
            "sign_split_order": "positive_half_then_negative_half",
            "base_dim": base_dim,
            "latent_dim": 2 * base_dim,
            "global_k_unmodified": True,
            "dynamics_parameters_fit": False,
            "basin_labels_or_counts_used": False,
            "pkp_is_posthoc_restriction_not_fitted_map": True,
            "near_identity_change_normalized_guard_emitted": True,
        },
        "routing": {
            "family_count_total": int(codebook.representatives.shape[0]),
            "family_count_retained": int(retained.sum()),
            "current_coverage": coverage,
            "persistent_coverage": persistent_coverage,
        },
        "regimes": regimes,
        "elapsed_seconds": time.time() - started,
    }


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--task_index", type=int, required=True)
    parser.add_argument("--systems", default="", help="Pilot-only roster subset")
    parser.add_argument("--seeds", default="", help="Pilot-only comma-separated seeds")
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--encode_batch_size", type=int, default=4096)
    args = parser.parse_args()
    card, card_hash = load_card(args.card)
    roster = discover_primary_roster(card)
    systems, seeds = set(_parse_csv(args.systems)), {int(item) for item in _parse_csv(args.seeds)}
    if systems:
        roster = [spec for spec in roster if spec.system_key in systems]
    if seeds:
        roster = [spec for spec in roster if spec.seed in seeds]
    if not 0 <= args.task_index < len(roster):
        raise IndexError(f"task_index {args.task_index} outside selected roster of {len(roster)}")
    output_path = args.output_dir / "shards" / f"task_{args.task_index:03d}.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    result = evaluate_checkpoint(roster[args.task_index], card, card_hash, args.device,
                                 args.encode_batch_size, args.task_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output_path), "status": result["status"],
                      "elapsed_seconds": result["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
