#!/usr/bin/env python3
"""Matched dense top-k specificity control for the frozen sparse global-K audit."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.neurips_2026.global_k_dense_checkpoint_audit import (
    assert_exact_dense_control,
)
from experiments.neurips_2026.global_k_dense_zero_wd_tasks import load_card, sha256_path
from experiments.neurips_2026.global_k_support_invariance import (
    FamilyCodebook,
    RunSpec,
    _distribution_summary,
    _encode,
    assert_sign_split_layout,
    discover_primary_roster,
    evaluate_regime,
    load_card as load_sparse_card,
    sign_pair_permutations,
)
from skae.checkpoint_compat import load_model_state_dict_compat
from skae.config import Config
from skae.data import VectorWrapper, make_env
from skae.model import make_model


DENSE_EVALUATOR_PATH = Path(__file__)
DENSE_CHECKPOINT_AUDIT_PATH = Path(__file__).with_name(
    "global_k_dense_checkpoint_audit.py"
)
SPARSE_EVALUATOR_PATH = Path(__file__).with_name("global_k_support_invariance.py")
TASK_MODULE_PATH = Path(__file__).with_name("global_k_dense_zero_wd_tasks.py")


@dataclass(frozen=True)
class DenseRun:
    system_key: str
    system_name: str
    seed: int
    run_dir: str
    sparse_run_dir: str
    attempt_count: int
    incomplete_attempt_count: int


def _tagify(value: str) -> str:
    return value.replace("-", "m").replace(".", "p")


def discover_dense_roster(
    card: dict[str, Any], task_tsv: Path, base_out: Path,
) -> list[DenseRun]:
    training = card["training"]
    expected = {
        (system, int(seed))
        for system in training["systems"]
        for seed in training["seeds"]
    }
    rows: dict[tuple[str, int], dict[str, str]] = {}
    with task_tsv.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["system_key"], int(row["seed"]))
            if key in rows:
                raise RuntimeError(f"Duplicate dense task-table key: {key}")
            rows[key] = row
    if set(rows) != expected:
        raise RuntimeError(
            f"Dense task roster mismatch; missing={sorted(expected - set(rows))}, "
            f"extra={sorted(set(rows) - expected)}"
        )

    sparse_card_path = Path(card["frozen_sparse_reference"]["card"])
    sparse_card, sparse_hash = load_sparse_card(sparse_card_path)
    if sparse_hash != card["frozen_sparse_reference"]["card_sha256"]:
        raise RuntimeError("Sparse-card hash drift")
    sparse = {
        (spec.system_key, spec.seed): spec for spec in discover_primary_roster(sparse_card)
    }
    specs = []
    for key in sorted(expected):
        row = rows[key]
        parent = (
            base_out
            / row["phase"]
            / row["model_variant"]
            / row["system_slug"]
            / f"dt_{_tagify(row['env_dt'])}"
            / f"seed_{row['seed']}"
        )
        candidates = sorted(
            path for path in parent.glob("20*")
            if (path / "checkpoint.pt").is_file()
        )
        completed = [
            path for path in candidates
            if (path / "evaluation_summary.json").is_file()
        ]
        if len(completed) != 1:
            raise RuntimeError(
                f"Expected one completed dense checkpoint under {parent}, "
                f"found completed={completed}, all_attempts={candidates}"
            )
        sparse_spec = sparse[key]
        specs.append(
            DenseRun(
                system_key=key[0],
                system_name=sparse_spec.system_name,
                seed=key[1],
                run_dir=str(completed[0]),
                sparse_run_dir=sparse_spec.run_dir,
                attempt_count=len(candidates),
                incomplete_attempt_count=len(candidates) - len(completed),
            )
        )
    return specs


def _load_checkpoint(run_dir: str, system_key: str, device: str):
    checkpoint_path = Path(run_dir) / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = Config.from_dict(checkpoint["config"])
    cfg.ENV.ENV_NAME = system_key
    env = make_env(cfg)
    model = make_model(cfg, env.observation_size)
    load_model_state_dict_compat(model, checkpoint["model_state_dict"])
    return cfg, env, model.to(device).eval(), checkpoint_path, checkpoint


def matched_topk_masks(dense: np.ndarray, sparse: np.ndarray, threshold: float) -> np.ndarray:
    if dense.shape != sparse.shape or dense.ndim != 2:
        raise ValueError("Dense and sparse state matrices must have the same [states, latent] shape")
    cardinalities = (np.abs(sparse) > threshold).sum(axis=1)
    order = np.argsort(-np.abs(dense), axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(order.shape[0])[:, None], order] = np.arange(order.shape[1])
    return ranks < cardinalities[:, None]


def fit_dense_family_codebook(mask: np.ndarray, min_jaccard: float) -> FamilyCodebook:
    """Exact greedy Jaccard codebook with packed-vector scoring for dense masks."""
    if mask.ndim != 2:
        raise ValueError("mask must have shape [states, latent_dim]")
    packed = np.packbits(mask.astype(np.uint8), axis=1)
    keys = [row.tobytes() for row in packed]
    counts: dict[bytes, int] = {}
    key_mask: dict[bytes, np.ndarray] = {}
    key_packed: dict[bytes, np.ndarray] = {}
    for key, row, packed_row in zip(keys, mask, packed):
        counts[key] = counts.get(key, 0) + 1
        key_mask.setdefault(key, row.astype(bool, copy=True))
        key_packed.setdefault(key, packed_row.copy())
    ordered = sorted(counts, key=lambda key: (-counts[key], key))
    rep_packed = np.empty((len(ordered), packed.shape[1]), dtype=np.uint8)
    rep_cardinality = np.empty(len(ordered), dtype=np.int32)
    representatives: list[np.ndarray] = []
    mapping: dict[bytes, int] = {}
    popcount = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)
    for key in ordered:
        candidate = key_packed[key]
        candidate_cardinality = int(key_mask[key].sum())
        rep_count = len(representatives)
        if rep_count:
            intersection = popcount[
                np.bitwise_and(rep_packed[:rep_count], candidate)
            ].sum(axis=1, dtype=np.int32)
            union = rep_cardinality[:rep_count] + candidate_cardinality - intersection
            similarities = np.divide(
                intersection,
                union,
                out=np.ones(rep_count, dtype=np.float64),
                where=union != 0,
            )
            best = int(similarities.argmax())
        else:
            similarities = np.empty(0, dtype=np.float64)
            best = -1
        if best >= 0 and similarities[best] >= min_jaccard:
            mapping[key] = best
        else:
            mapping[key] = rep_count
            representatives.append(key_mask[key])
            rep_packed[rep_count] = candidate
            rep_cardinality[rep_count] = candidate_cardinality
    labels = np.asarray([mapping[key] for key in keys], dtype=np.int64)
    fit_counts = np.bincount(labels, minlength=len(representatives)).astype(np.int64)
    return FamilyCodebook(np.stack(representatives), mapping, fit_counts)


def assign_dense_families(
    mask: np.ndarray, codebook: FamilyCodebook, min_jaccard: float,
) -> np.ndarray:
    """Exact nearest-representative assignment using packed Jaccard scoring."""
    packed = np.packbits(mask.astype(np.uint8), axis=1)
    representatives = np.packbits(codebook.representatives.astype(np.uint8), axis=1)
    rep_cardinality = codebook.representatives.sum(axis=1).astype(np.int32)
    popcount = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)
    labels = np.full(mask.shape[0], -1, dtype=np.int64)
    cache: dict[bytes, int] = {}
    for index, (row, packed_row) in enumerate(zip(mask, packed)):
        key = packed_row.tobytes()
        family = codebook.exact_key_to_family.get(key)
        if family is None:
            family = cache.get(key)
        if family is None:
            intersection = popcount[
                np.bitwise_and(representatives, packed_row)
            ].sum(axis=1, dtype=np.int32)
            union = rep_cardinality + int(row.sum()) - intersection
            similarity = np.divide(
                intersection,
                union,
                out=np.ones(len(representatives), dtype=np.float64),
                where=union != 0,
            )
            best = int(similarity.argmax())
            family = best if similarity[best] >= min_jaccard else -1
            cache[key] = family
        labels[index] = family
    return labels


def evaluate_checkpoint(
    spec: DenseRun, card: dict[str, Any], card_hash: str, device: str,
    encode_batch_size: int, task_index: int, task_tsv_hash: str,
) -> dict[str, Any]:
    started = time.time()
    dense_cfg, dense_env, dense_model, dense_path, dense_checkpoint = _load_checkpoint(
        spec.run_dir, spec.system_key, device
    )
    sparse_cfg, sparse_env, sparse_model, sparse_path, _sparse_checkpoint = _load_checkpoint(
        spec.sparse_run_dir, spec.system_key, device
    )
    optimizer_audit = assert_exact_dense_control(
        dense_cfg, dense_model, card, dense_checkpoint
    )
    assert_sign_split_layout(sparse_cfg, sparse_model)
    if dense_env.observation_size != sparse_env.observation_size:
        raise AssertionError("Sparse/dense observation dimensions differ")
    matched = card["matched_evaluation"]
    corpus = matched["corpus"]
    trajectories = VectorWrapper(
        sparse_env, int(corpus["num_trajectories"])
    ).generate_sequence_batch(
        rng=torch.Generator().manual_seed(int(corpus["eval_seed"])),
        window_length=int(corpus["trajectory_length"]),
    ).float()
    sparse_latent = _encode(sparse_model, trajectories, device, encode_batch_size)
    dense_latent = _encode(dense_model, trajectories, device, encode_batch_size)
    order = np.random.default_rng(int(corpus["split_seed"])).permutation(trajectories.shape[0])
    fit_ids = order[: int(corpus["fit_trajectories"])]
    score_ids = order[int(corpus["fit_trajectories"]):]
    threshold = float(matched["sparse_support"].split(">")[-1].strip())
    fit_dense = dense_latent[fit_ids, :-1].reshape(-1, dense_latent.shape[-1])
    fit_sparse = sparse_latent[fit_ids, :-1].reshape(-1, sparse_latent.shape[-1])
    jaccard = float(matched["family_jaccard_threshold"])
    codebook = fit_dense_family_codebook(
        matched_topk_masks(fit_dense, fit_sparse, threshold), jaccard
    )
    retained = codebook.fit_counts >= int(matched["min_fit_source_transitions"])

    dense_score, sparse_score = dense_latent[score_ids], sparse_latent[score_ids]
    z = dense_score[:, :-1].reshape(-1, dense_latent.shape[-1])
    z_next = dense_score[:, 1:].reshape(-1, dense_latent.shape[-1])
    sparse_z = sparse_score[:, :-1].reshape(-1, sparse_latent.shape[-1])
    sparse_next = sparse_score[:, 1:].reshape(-1, sparse_latent.shape[-1])
    current = assign_dense_families(
        matched_topk_masks(z, sparse_z, threshold), codebook, jaccard
    )
    following = assign_dense_families(
        matched_topk_masks(z_next, sparse_next, threshold), codebook, jaccard
    )
    current_valid = (current >= 0) & retained[np.maximum(current, 0)]
    following_valid = (following >= 0) & retained[np.maximum(following, 0)]
    persistent = current_valid & following_valid & (current == following)
    labels = np.where(persistent, current, -1)
    coverage = float(current_valid.mean())
    persistent_coverage = float(persistent.mean())
    eligibility = matched["eligibility"]
    eligible = (
        int(retained.sum()) >= int(eligibility["min_retained_families"])
        and coverage >= float(eligibility["min_current_coverage"])
        and persistent_coverage >= float(eligibility["min_persistent_coverage"])
    )
    k_matrix = dense_model.kmatrix().detach().cpu().numpy().astype(np.float32)
    true = evaluate_regime(
        z, z_next, labels, codebook.representatives, codebook.fit_counts,
        k_matrix, int(matched["min_family_score_transitions"]),
    )
    null_records = []
    null = matched["null"]
    for permutation in sign_pair_permutations(
        dense_latent.shape[-1] // 2, int(null["replicates"]), int(null["seed"])
    ):
        permuted = evaluate_regime(
            z[:, permutation], z_next[:, permutation], labels,
            codebook.representatives[:, permutation], codebook.fit_counts,
            k_matrix, int(matched["min_family_score_transitions"]),
        )
        null_records.append({"aggregate": permuted["aggregate"], "operator": permuted["operator"]})
    return {
        "schema_version": 1,
        "status": "eligible" if eligible else "ineligible",
        "task_index": task_index,
        "card_sha256": card_hash,
        "task_tsv_sha256": task_tsv_hash,
        "authenticated_sources": {
            "dense_evaluator": {
                "path": "experiments/neurips_2026/global_k_dense_specificity.py",
                "sha256": sha256_path(DENSE_EVALUATOR_PATH),
            },
            "dense_checkpoint_audit": {
                "path": "experiments/neurips_2026/global_k_dense_checkpoint_audit.py",
                "sha256": sha256_path(DENSE_CHECKPOINT_AUDIT_PATH),
            },
            "imported_sparse_evaluator": {
                "path": "experiments/neurips_2026/global_k_support_invariance.py",
                "sha256": sha256_path(SPARSE_EVALUATOR_PATH),
            },
            "dense_task_module": {
                "path": "experiments/neurips_2026/global_k_dense_zero_wd_tasks.py",
                "sha256": sha256_path(TASK_MODULE_PATH),
            },
        },
        "provenance": {
            "system_key": spec.system_key,
            "system_name": spec.system_name,
            "seed": spec.seed,
            "dense_run_dir": spec.run_dir,
            "dense_checkpoint_sha256": sha256_path(dense_path),
            "sparse_run_dir": spec.sparse_run_dir,
            "sparse_checkpoint_sha256": sha256_path(sparse_path),
            "training_attempt_count": spec.attempt_count,
            "incomplete_training_attempt_count": spec.incomplete_attempt_count,
            "git_commit": os.environ.get("SKAE_GIT_COMMIT", "launcher_not_recorded"),
        },
        "assertions": {
            "exact_dense_zero_wd_control": True,
            "same_physical_states": True,
            "paired_sparse_cardinality": True,
            "global_k_unmodified": True,
            "basin_labels_or_counts_used": False,
            "mechanism_hyperparameters_tuned_after_sparse_outcome": False,
            "dense_masks_reported_as_cardinality_matched_coordinates": True,
        },
        "optimizer_audit": optimizer_audit,
        "mask_terminology": "sparse-cardinality-matched top-k coordinate masks",
        "null_caveat": (
            "The identical sign-pair-preserving restricted permutation set is "
            "used for matching; dense coordinates have no natural sign-pair semantics."
        ),
        "routing": {
            "family_count_total": int(codebook.representatives.shape[0]),
            "family_count_retained": int(retained.sum()),
            "current_coverage": coverage,
            "persistent_coverage": persistent_coverage,
        },
        "regime": {
            "true": true,
            "null_replicates": null_records,
            "null_summary": {
                "aggregate": _distribution_summary(null_records, "aggregate"),
                "operator": _distribution_summary(null_records, "operator"),
            },
        },
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--source_lock", type=Path, required=True)
    parser.add_argument("--task_tsv", type=Path, required=True)
    parser.add_argument("--base_out", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--task_index", type=int, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--encode_batch_size", type=int, default=4096)
    args = parser.parse_args()
    card, card_hash = load_card(args.card)
    roster = discover_dense_roster(card, args.task_tsv, args.base_out)
    if not 0 <= args.task_index < len(roster):
        raise IndexError(args.task_index)
    output = args.output_dir / "shards" / f"task_{args.task_index:03d}.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    payload = evaluate_checkpoint(
        roster[args.task_index], card, card_hash, args.device,
        args.encode_batch_size, args.task_index, sha256_path(args.task_tsv),
    )
    source_lock = json.loads(args.source_lock.read_text())
    if source_lock.get("protocol_id") != card.get("protocol_id"):
        raise RuntimeError("Source lock/card protocol mismatch")
    payload["source_lock"] = {
        "path": str(args.source_lock),
        "sha256": sha256_path(args.source_lock),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
