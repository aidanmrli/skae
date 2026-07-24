#!/usr/bin/env python3
"""Build and check compact paper evidence for dense global-K specificity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


PREFIX = "global_k_dense_zero_wd_specificity"
RUN_ROWS = f"{PREFIX}_run_rows.csv"
SYSTEM_ROWS = f"{PREFIX}_system_rows.csv"
DECISION = f"{PREFIX}_decision.json"
CARD = f"{PREFIX}_card.json"
PROVENANCE = f"{PREFIX}_provenance.json"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _normalize_summary(summary_dir: Path, output_data_dir: Path) -> dict[str, Path]:
    run_fields, run_rows = _read_csv(summary_dir / "run_rows.csv")
    system_fields, system_rows = _read_csv(summary_dir / "system_rows.csv")
    if len(run_rows) != 45 or len(system_rows) != 15:
        raise RuntimeError(f"Expected 45 run rows and 15 system rows, got {len(run_rows)}, {len(system_rows)}")
    run_keys = {(row["system_key"], int(row["seed"])) for row in run_rows}
    if len(run_keys) != 45:
        raise RuntimeError("Run evidence has duplicate system/seed keys")
    run_rows.sort(key=lambda row: (row["system_key"], int(row["seed"])))
    system_rows.sort(key=lambda row: row["system_key"])
    decision = json.loads((summary_dir / "decision.json").read_text())
    if decision.get("decision") not in {
        "sparse_support_specific", "not_sparse_specific", "invalid_dense_control"
    }:
        raise RuntimeError("Unknown dense-specificity decision")
    targets = {
        "run_rows": output_data_dir / RUN_ROWS,
        "system_rows": output_data_dir / SYSTEM_ROWS,
        "decision": output_data_dir / DECISION,
    }
    _write_csv(targets["run_rows"], run_fields, run_rows)
    _write_csv(targets["system_rows"], system_fields, system_rows)
    _write_json(targets["decision"], decision)
    return targets


def render_table(decision: dict[str, Any]) -> str:
    sparse = decision["sparse_system_medians"]
    dense = decision["dense_system_medians"]
    ratio = decision["sparse_over_dense"]
    rows = (
        (
            "Raw-$K$ activity leakage / null",
            sparse["activity_leakage_true_over_null"],
            dense["activity_leakage_true_over_null"],
            ratio["activity_leakage_null_ratio"],
        ),
        (
            "Post-hoc restricted residual / null",
            sparse["restricted_residual_true_over_null"],
            dense["restricted_residual_true_over_null"],
            ratio["restricted_residual_null_ratio"],
        ),
    )
    def fmt(value: Any) -> str:
        if value is None:
            return "--"
        numeric = float(value)
        return f"{numeric:.3f}" if math.isfinite(numeric) else "--"

    body = "\n".join(
        f"{label} & {fmt(sparse_value)} & {fmt(dense_value)} & {fmt(comparison)} \\\\"
        for label, sparse_value, dense_value, comparison in rows
    )
    return (
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "Metric & Sparse / null & Dense / null & Sparse / dense \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def _source_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_path(path)}


def build(args: argparse.Namespace) -> None:
    targets = [
        args.output_data_dir / name
        for name in (RUN_ROWS, SYSTEM_ROWS, DECISION, CARD, PROVENANCE)
    ] + [args.output_table]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite evidence artifacts: {existing}")
    args.output_data_dir.mkdir(parents=True, exist_ok=True)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)

    card_payload = json.loads(args.card.read_text())
    card_hash = sha256_path(args.card)
    source_lock_payload = json.loads(args.source_lock.read_text())
    if source_lock_payload.get("protocol_id") != card_payload.get("protocol_id"):
        raise RuntimeError("Source lock/card protocol mismatch")
    if source_lock_payload.get("sources", {}).get("frozen_card", {}).get("sha256") != card_hash:
        raise RuntimeError("Source lock/frozen-card digest mismatch")
    smoke = json.loads(args.smoke_decision.read_text())
    task_manifest = json.loads(args.task_manifest.read_text())
    if smoke.get("passed") is not True or smoke.get("card_sha256") != card_hash:
        raise RuntimeError("Smoke decision is not a card-matched pass")
    if task_manifest.get("card_sha256") != card_hash:
        raise RuntimeError("Task manifest/card mismatch")
    if task_manifest.get("mode") != "full" or int(task_manifest.get("task_count", -1)) != 45:
        raise RuntimeError("Task manifest is not the frozen 45-run full roster")
    task_tsv_hash = task_manifest.get("task_tsv_sha256")
    if not isinstance(task_tsv_hash, str) or len(task_tsv_hash) != 64:
        raise RuntimeError("Task manifest lacks a valid task-table digest")

    shard_paths = sorted((args.evaluation_dir / "shards").glob("task_*.json"))
    if len(shard_paths) != 45:
        raise RuntimeError(f"Expected 45 evaluation shards, found {len(shard_paths)}")
    checkpoint_records = []
    for shard_path in shard_paths:
        shard = json.loads(shard_path.read_text())
        if shard.get("card_sha256") != card_hash:
            raise RuntimeError(f"Shard/card mismatch: {shard_path}")
        if shard.get("task_tsv_sha256") != task_tsv_hash:
            raise RuntimeError(f"Shard/task-table mismatch: {shard_path}")
        if not all(shard.get("assertions", {}).values()):
            raise RuntimeError(f"Failed artifact assertion: {shard_path}")
        if shard.get("mask_terminology") != "sparse-cardinality-matched top-k coordinate masks":
            raise RuntimeError(f"Dense-mask terminology guard failed: {shard_path}")
        optimizer_audit = shard.get("optimizer_audit", {})
        if any(
            float(value) != 0.0
            for value in optimizer_audit.get("reconstructed_param_group_weight_decays", [])
        ):
            raise RuntimeError(f"Nonzero reconstructed optimizer weight decay: {shard_path}")
        provenance = shard["provenance"]
        dense_checkpoint = Path(provenance["dense_run_dir"]) / "checkpoint.pt"
        sparse_checkpoint = Path(provenance["sparse_run_dir"]) / "checkpoint.pt"
        if sha256_path(dense_checkpoint) != provenance["dense_checkpoint_sha256"]:
            raise RuntimeError(f"Dense checkpoint hash mismatch: {dense_checkpoint}")
        if sha256_path(sparse_checkpoint) != provenance["sparse_checkpoint_sha256"]:
            raise RuntimeError(f"Sparse checkpoint hash mismatch: {sparse_checkpoint}")
        checkpoint_records.append(
            {
                "system_key": provenance["system_key"],
                "seed": provenance["seed"],
                "dense_checkpoint": _source_record(dense_checkpoint),
                "sparse_checkpoint": _source_record(sparse_checkpoint),
                "evaluation_shard": _source_record(shard_path),
            }
        )
    checkpoint_records.sort(key=lambda row: (row["system_key"], int(row["seed"])))

    normalized = _normalize_summary(args.summary_dir, args.output_data_dir)
    card_target = args.output_data_dir / CARD
    _write_json(card_target, card_payload)
    decision = json.loads(normalized["decision"].read_text())
    decision_sources = decision.get("authenticated_sources")
    if not isinstance(decision_sources, dict):
        raise RuntimeError("Summary decision lacks authenticated evaluator/reducer sources")
    if decision_sources.get("source_lock", {}).get("sha256") != sha256_path(args.source_lock):
        raise RuntimeError("Summary decision used a different source lock")
    args.output_table.write_text(render_table(decision))
    source_artifacts = {
        "scratch_run_rows": _source_record(args.summary_dir / "run_rows.csv"),
        "scratch_system_rows": _source_record(args.summary_dir / "system_rows.csv"),
        "scratch_decision": _source_record(args.summary_dir / "decision.json"),
        "frozen_card": _source_record(args.card),
        "task_manifest": _source_record(args.task_manifest),
        "smoke_decision": _source_record(args.smoke_decision),
    }
    normalized_artifacts = {
        "run_rows": _source_record(normalized["run_rows"]),
        "system_rows": _source_record(normalized["system_rows"]),
        "decision": _source_record(normalized["decision"]),
        "card": _source_record(card_target),
        "table": _source_record(args.output_table),
    }
    provenance = {
        "schema_version": 1,
        "packet_id": PREFIX,
        "card_sha256": card_hash,
        "decision": decision["decision"],
        "authenticated_sources": {
            **decision_sources,
            "evidence_builder": _source_record(Path(__file__)),
            "source_lock": _source_record(args.source_lock),
        },
        "source_artifacts": source_artifacts,
        "normalized_artifacts": normalized_artifacts,
        "checkpoints_and_shards": checkpoint_records,
        "complete_recipe_architecture_caveat": (
            "This is a complete-recipe comparison between the sparse LISTA sign-split "
            "model and the dense tanh MLP zero-weight-decay model. Encoder architecture, "
            "sign splitting, decoder parameterization/normalization, and optimization "
            "differ, so it does not identify a "
            "sparsity-only causal effect. Physical states, latent width, system/seed "
            "roster, per-state support cardinality, estimands, and nulls are matched. "
            "Dense objects are sparse-cardinality-matched top-k coordinate masks, not "
            "natural dense supports; their null is the identical restricted "
            "sign-pair-preserving permutation set even though dense coordinates have "
            "no natural sign-pair semantics."
        ),
    }
    _write_json(args.output_data_dir / PROVENANCE, provenance)
    check_packet(args.output_data_dir / PROVENANCE)


def check_packet(provenance_path: Path) -> None:
    provenance = json.loads(provenance_path.read_text())
    for name, record in provenance["normalized_artifacts"].items():
        path = Path(record["path"])
        if sha256_path(path) != record["sha256"]:
            raise RuntimeError(f"Evidence hash mismatch: normalized_artifacts.{name}")
    for group in ("source_artifacts", "authenticated_sources"):
        for name, record in provenance[group].items():
            digest = record.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise RuntimeError(f"Malformed recorded source digest: {group}.{name}")
    if "complete-recipe comparison" not in provenance["complete_recipe_architecture_caveat"]:
        raise RuntimeError("Required complete-recipe/architecture caveat is missing")
    normalized = provenance["normalized_artifacts"]
    card_path = Path(normalized["card"]["path"])
    if sha256_path(card_path) != provenance["card_sha256"]:
        raise RuntimeError("Compact frozen card does not match top-level card digest")
    card = json.loads(card_path.read_text())
    expected_pairs = {
        (system, int(seed))
        for system in card["training"]["systems"]
        for seed in card["training"]["seeds"]
    }
    expected_systems = set(card["training"]["systems"])
    run_fields, run_rows = _read_csv(Path(normalized["run_rows"]["path"]))
    del run_fields
    _, system_rows = _read_csv(Path(normalized["system_rows"]["path"]))
    actual_pairs = {(row["system_key"], int(row["seed"])) for row in run_rows}
    actual_systems = {row["system_key"] for row in system_rows}
    if (
        len(run_rows) != 45
        or len(system_rows) != 15
        or actual_pairs != expected_pairs
        or actual_systems != expected_systems
        or any(int(row["run_count"]) != 3 for row in system_rows)
    ):
        raise RuntimeError("Normalized evidence roster is not the exact frozen 45/15 roster")
    decision_path = Path(normalized["decision"]["path"])
    decision = json.loads(decision_path.read_text())
    if decision.get("decision") != provenance.get("decision"):
        raise RuntimeError("Compact decision disagrees with provenance")
    table_path = Path(normalized["table"]["path"])
    if table_path.read_text() != render_table(decision):
        raise RuntimeError("Compact table is not the deterministic decision rendering")
    checkpoint_rows = provenance.get("checkpoints_and_shards", [])
    checkpoint_pairs = {
        (row["system_key"], int(row["seed"])) for row in checkpoint_rows
    }
    if len(checkpoint_rows) != 45 or checkpoint_pairs != expected_pairs:
        raise RuntimeError("Recorded checkpoint/shard provenance roster is incomplete")
    for index, row in enumerate(checkpoint_rows):
        for key in ("dense_checkpoint", "sparse_checkpoint", "evaluation_shard"):
            record = row.get(key, {})
            digest = record.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise RuntimeError(
                    f"Malformed recorded checkpoint/shard digest: row={index} key={key}"
                )


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
