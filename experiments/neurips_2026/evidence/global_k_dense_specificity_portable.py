#!/usr/bin/env python3
"""Build portable paper evidence from the frozen dense-specificity outputs.

This adapter intentionally leaves the source-locked evaluator, reducer, card,
and original evidence builder unchanged.  The frozen shard schema contains
both positive and negative assertions, so assertion validity is defined by an
explicit expected-value contract rather than by ``all(assertions.values())``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.neurips_2026.evidence import global_k_dense_specificity as frozen


ASSERTION_CONTRACT = {
    "basin_labels_or_counts_used": False,
    "dense_masks_reported_as_cardinality_matched_coordinates": True,
    "exact_dense_zero_wd_control": True,
    "global_k_unmodified": True,
    "mechanism_hyperparameters_tuned_after_sparse_outcome": False,
    "paired_sparse_cardinality": True,
    "same_physical_states": True,
}


def validate_assertion_contract(assertions: Any, shard_path: Path) -> None:
    """Fail closed unless a shard has exactly the frozen assertion polarities."""
    if not isinstance(assertions, dict):
        raise RuntimeError(f"Malformed artifact assertions: {shard_path}")
    expected_keys = set(ASSERTION_CONTRACT)
    actual_keys = set(assertions)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        raise RuntimeError(
            f"Artifact assertion schema mismatch: {shard_path}; "
            f"missing={missing}, extra={extra}"
        )
    non_boolean = sorted(key for key, value in assertions.items() if type(value) is not bool)
    if non_boolean:
        raise RuntimeError(
            f"Artifact assertions are not boolean: {shard_path}; keys={non_boolean}"
        )
    mismatched = {
        key: {"expected": expected, "actual": assertions[key]}
        for key, expected in ASSERTION_CONTRACT.items()
        if assertions[key] is not expected
    }
    if mismatched:
        raise RuntimeError(
            f"Artifact assertion polarity mismatch: {shard_path}; "
            f"mismatched={mismatched}"
        )


def _assert_record(path: Path, record: dict[str, Any], label: str) -> None:
    expected = record.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError(f"Malformed authenticated digest: {label}")
    if not path.is_file() or frozen.sha256_path(path) != expected:
        raise RuntimeError(f"Authenticated artifact drift: {label}: {path}")


def _load_authenticated_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], str, str]:
    card = json.loads(args.card.read_text())
    card_hash = frozen.sha256_path(args.card)
    source_lock = json.loads(args.source_lock.read_text())
    if source_lock.get("protocol_id") != card.get("protocol_id"):
        raise RuntimeError("Source lock/card protocol mismatch")
    for name, record in source_lock.get("sources", {}).items():
        _assert_record(Path(str(record.get("path", ""))), record, f"locked source {name}")
    for name, record in source_lock.get("external_inputs", {}).items():
        _assert_record(
            Path(str(record.get("path", ""))), record, f"external input {name}"
        )
    _assert_record(
        args.card,
        source_lock.get("sources", {}).get("frozen_card", {}),
        "frozen card",
    )
    _assert_record(
        args.task_manifest,
        source_lock.get("external_inputs", {}).get("full_task_manifest", {}),
        "full task manifest",
    )
    _assert_record(
        args.smoke_decision,
        source_lock.get("external_inputs", {}).get("smoke_decision", {}),
        "smoke decision",
    )

    smoke = json.loads(args.smoke_decision.read_text())
    manifest = json.loads(args.task_manifest.read_text())
    if smoke.get("passed") is not True or smoke.get("card_sha256") != card_hash:
        raise RuntimeError("Smoke decision is not a card-matched pass")
    if manifest.get("card_sha256") != card_hash:
        raise RuntimeError("Task manifest/card mismatch")
    if manifest.get("mode") != "full" or int(manifest.get("task_count", -1)) != 45:
        raise RuntimeError("Task manifest is not the frozen 45-run full roster")
    task_tsv_hash = manifest.get("task_tsv_sha256")
    task_tsv_path = Path(str(manifest.get("task_tsv", "")))
    if not isinstance(task_tsv_hash, str) or len(task_tsv_hash) != 64:
        raise RuntimeError("Task manifest lacks a valid task-table digest")
    if not task_tsv_path.is_file() or frozen.sha256_path(task_tsv_path) != task_tsv_hash:
        raise RuntimeError("Task manifest points to a drifted task table")
    locked_tsv = source_lock.get("external_inputs", {}).get("full_task_tsv", {})
    if locked_tsv.get("sha256") != task_tsv_hash:
        raise RuntimeError("Source lock/task-table mismatch")
    return card, card_hash, task_tsv_hash


def _authenticate_shards(
    args: argparse.Namespace,
    card: dict[str, Any],
    card_hash: str,
    task_tsv_hash: str,
) -> list[dict[str, Any]]:
    shard_paths = sorted((args.evaluation_dir / "shards").glob("task_*.json"))
    if len(shard_paths) != 45:
        raise RuntimeError(f"Expected 45 evaluation shards, found {len(shard_paths)}")
    expected_pairs = {
        (system, int(seed))
        for system in card["training"]["systems"]
        for seed in card["training"]["seeds"]
    }
    source_lock_hash = frozen.sha256_path(args.source_lock)
    records = []
    actual_pairs = set()
    for shard_path in shard_paths:
        shard = json.loads(shard_path.read_text())
        if shard.get("card_sha256") != card_hash:
            raise RuntimeError(f"Shard/card mismatch: {shard_path}")
        if shard.get("task_tsv_sha256") != task_tsv_hash:
            raise RuntimeError(f"Shard/task-table mismatch: {shard_path}")
        if shard.get("source_lock", {}).get("sha256") != source_lock_hash:
            raise RuntimeError(f"Shard/source-lock mismatch: {shard_path}")
        validate_assertion_contract(shard.get("assertions"), shard_path)
        if shard.get("mask_terminology") != (
            "sparse-cardinality-matched top-k coordinate masks"
        ):
            raise RuntimeError(f"Dense-mask terminology guard failed: {shard_path}")
        weight_decays = shard.get("optimizer_audit", {}).get(
            "reconstructed_param_group_weight_decays", []
        )
        if not weight_decays or any(float(value) != 0.0 for value in weight_decays):
            raise RuntimeError(f"Invalid reconstructed optimizer weight decay: {shard_path}")

        provenance = shard.get("provenance", {})
        pair = (provenance.get("system_key"), int(provenance.get("seed", -1)))
        actual_pairs.add(pair)
        dense_checkpoint = Path(str(provenance.get("dense_run_dir", ""))) / "checkpoint.pt"
        sparse_checkpoint = Path(str(provenance.get("sparse_run_dir", ""))) / "checkpoint.pt"
        if (
            not dense_checkpoint.is_file()
            or frozen.sha256_path(dense_checkpoint)
            != provenance.get("dense_checkpoint_sha256")
        ):
            raise RuntimeError(f"Dense checkpoint hash mismatch: {dense_checkpoint}")
        if (
            not sparse_checkpoint.is_file()
            or frozen.sha256_path(sparse_checkpoint)
            != provenance.get("sparse_checkpoint_sha256")
        ):
            raise RuntimeError(f"Sparse checkpoint hash mismatch: {sparse_checkpoint}")
        records.append(
            {
                "system_key": pair[0],
                "seed": pair[1],
                "dense_checkpoint": frozen._source_record(dense_checkpoint),
                "sparse_checkpoint": frozen._source_record(sparse_checkpoint),
                "evaluation_shard": frozen._source_record(shard_path),
            }
        )
    if actual_pairs != expected_pairs or len(actual_pairs) != len(records):
        raise RuntimeError("Evaluation shard roster is not the exact frozen 45-run roster")
    records.sort(key=lambda row: (row["system_key"], int(row["seed"])))
    return records


def build(args: argparse.Namespace) -> None:
    targets = [
        args.output_data_dir / name
        for name in (
            frozen.RUN_ROWS,
            frozen.SYSTEM_ROWS,
            frozen.DECISION,
            frozen.CARD,
            frozen.PROVENANCE,
        )
    ] + [args.output_table]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite evidence artifacts: {existing}")

    card, card_hash, task_tsv_hash = _load_authenticated_inputs(args)
    checkpoint_records = _authenticate_shards(
        args, card, card_hash, task_tsv_hash
    )
    decision_source = json.loads((args.summary_dir / "decision.json").read_text())
    decision_sources = decision_source.get("authenticated_sources")
    if not isinstance(decision_sources, dict):
        raise RuntimeError("Summary decision lacks authenticated evaluator/reducer sources")
    if decision_sources.get("source_lock", {}).get("sha256") != frozen.sha256_path(
        args.source_lock
    ):
        raise RuntimeError("Summary decision used a different source lock")

    args.output_data_dir.mkdir(parents=True, exist_ok=True)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    normalized = frozen._normalize_summary(args.summary_dir, args.output_data_dir)
    card_target = args.output_data_dir / frozen.CARD
    card_target.write_bytes(args.card.read_bytes())
    decision = json.loads(normalized["decision"].read_text())
    args.output_table.write_text(frozen.render_table(decision))

    source_lock = json.loads(args.source_lock.read_text())
    frozen_builder = source_lock["sources"]["evidence_builder"]
    source_artifacts = {
        "scratch_run_rows": frozen._source_record(args.summary_dir / "run_rows.csv"),
        "scratch_system_rows": frozen._source_record(args.summary_dir / "system_rows.csv"),
        "scratch_decision": frozen._source_record(args.summary_dir / "decision.json"),
        "frozen_card": frozen._source_record(args.card),
        "task_manifest": frozen._source_record(args.task_manifest),
        "smoke_decision": frozen._source_record(args.smoke_decision),
    }
    normalized_artifacts = {
        "run_rows": frozen._source_record(normalized["run_rows"]),
        "system_rows": frozen._source_record(normalized["system_rows"]),
        "decision": frozen._source_record(normalized["decision"]),
        "card": frozen._source_record(card_target),
        "table": frozen._source_record(args.output_table),
    }
    provenance = {
        "schema_version": 1,
        "packet_id": frozen.PREFIX,
        "card_sha256": card_hash,
        "decision": decision["decision"],
        "assertion_contract": ASSERTION_CONTRACT,
        "authenticated_sources": {
            **decision_sources,
            "frozen_evidence_builder": frozen_builder,
            "portable_evidence_builder": frozen._source_record(Path(__file__)),
            "source_lock": frozen._source_record(args.source_lock),
        },
        "source_artifacts": source_artifacts,
        "normalized_artifacts": normalized_artifacts,
        "checkpoints_and_shards": checkpoint_records,
        "complete_recipe_architecture_caveat": (
            "This is a complete-recipe comparison between the sparse LISTA sign-split "
            "model and the dense tanh MLP zero-weight-decay model. Encoder architecture, "
            "sign splitting, decoder parameterization/normalization, and optimization "
            "differ, so it does not identify a sparsity-only causal effect. Physical "
            "states, latent width, system/seed roster, per-state support cardinality, "
            "estimands, and nulls are matched. Dense objects are sparse-cardinality-"
            "matched top-k coordinate masks, not natural dense supports; their null is "
            "the identical restricted sign-pair-preserving permutation set even though "
            "dense coordinates have no natural sign-pair semantics."
        ),
    }
    provenance_path = args.output_data_dir / frozen.PROVENANCE
    frozen._write_json(provenance_path, provenance)
    try:
        check_packet(provenance_path)
    except Exception:
        for path in targets:
            path.unlink(missing_ok=True)
        raise


def check_packet(provenance_path: Path) -> None:
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("assertion_contract") != ASSERTION_CONTRACT:
        raise RuntimeError("Portable packet assertion contract is missing or drifted")
    frozen.check_packet(provenance_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--card", type=Path, required=True)
    build_parser.add_argument("--source_lock", type=Path, required=True)
    build_parser.add_argument("--task_manifest", type=Path, required=True)
    build_parser.add_argument("--smoke_decision", type=Path, required=True)
    build_parser.add_argument("--evaluation_dir", type=Path, required=True)
    build_parser.add_argument("--summary_dir", type=Path, required=True)
    build_parser.add_argument("--output_data_dir", type=Path, required=True)
    build_parser.add_argument("--output_table", type=Path, required=True)
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--provenance", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build(args)
    else:
        check_packet(args.provenance)
    print(json.dumps({"command": args.command, "status": "ok"}, sort_keys=True))


if __name__ == "__main__":
    main()
