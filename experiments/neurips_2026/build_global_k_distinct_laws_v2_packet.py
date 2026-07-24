#!/usr/bin/env python3
"""Build a fail-closed compact evidence packet from all distinct-law V2 rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.neurips_2026.global_k_distinct_laws_v2_source_lock import (
    verify_source_lock,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_tasks import (
    load_card,
    sha256_path,
)
from experiments.neurips_2026.summarize_global_k_distinct_laws_v2 import (
    _load_shards,
)


def build_packet(
    *, card_path: Path, source_lock_path: Path, expected_source_lock_sha: str,
    task_tsv: Path, audit_summary_path: Path, evaluation_dir: Path,
    decision_path: Path, telemetry_path: Path,
) -> dict[str, Any]:
    if sha256_path(source_lock_path) != expected_source_lock_sha:
        raise RuntimeError("Expected source-lock hash mismatch")
    lock = verify_source_lock(source_lock_path)
    card, card_hash = load_card(card_path)
    task_hash = sha256_path(task_tsv)
    if task_hash != lock["external_inputs"]["full_task_tsv"]["sha256"]:
        raise RuntimeError("Packet task-table/source-lock mismatch")
    rows = _load_shards(
        evaluation_dir, card, card_hash, task_hash, expected_source_lock_sha
    )
    audit = json.loads(audit_summary_path.read_text())
    if (
        audit.get("status") != "passed"
        or audit.get("protocol_id") != card["protocol_id"]
        or audit.get("card_sha256") != card_hash
        or audit.get("task_tsv_sha256") != task_hash
        or audit.get("task_count") != 20
        or audit.get("passed_count") != 20
    ):
        raise RuntimeError("Packet checkpoint-audit authentication failed")
    audit_by_task = {int(row["task_id"]): row for row in audit["rows"]}
    if set(audit_by_task) != set(range(20)) or any(
        audit_by_task[int(row["task_id"])]["checkpoint_sha256"]
        != row["provenance"]["selected_checkpoint_sha256"]
        for row in rows
    ):
        raise RuntimeError("Packet audit/evaluation checkpoint mismatch")
    decision = json.loads(decision_path.read_text())
    provenance = decision.get("provenance", {})
    if (
        decision.get("protocol_id") != card["protocol_id"]
        or provenance.get("card_sha256") != card_hash
        or provenance.get("task_tsv_sha256") != task_hash
        or provenance.get("source_lock_sha256") != expected_source_lock_sha
        or len(decision.get("seed_rows", [])) != 20
    ):
        raise RuntimeError("Packet decision authentication failed")
    telemetry = json.loads(telemetry_path.read_text())
    telemetry_provenance = telemetry.get("provenance", {})
    if (
        not telemetry.get("assessment_complete")
        or telemetry.get("outcomes_inspected") is not False
        or telemetry_provenance.get("card_sha256") != card_hash
        or telemetry_provenance.get("task_tsv_sha256") != task_hash
        or telemetry_provenance.get("source_lock_sha256")
        != expected_source_lock_sha
    ):
        raise RuntimeError("Packet scientific telemetry authentication failed")
    return {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "decision": {
            key: decision[key] for key in (
                "mechanism_tier", "mechanism_text", "relative_specificity_tier",
                "relative_specificity_text", "mandatory_caveat", "validity",
                "sparse_gates", "sparse_distributions", "specificity",
                "audited_parameter_counts_by_arm", "seed_rows", "bootstrap",
            )
        },
        "claim_boundary": {
            "supported_scope": "family-projected codes zP_b from matched verification clouds are approximately closed; tangent/local-law interpretation only through joint H/G gates",
            "forbidden_scope": "containment of unprojected z in P_b or algebraic invariance of the entire support-coordinate subspace",
            "source_containment_of_unprojected_codes_not_tested": True,
            "static_P_K_I_minus_P_ratios_are_descriptive_only": True,
            "checkpoint_selector_reencodes_physical_state_every_step": True,
            "checkpoint_selector_is_not_autonomous_repeated_K": True,
        },
        "scientific_gpu_assessment": telemetry,
        "provenance": {
            "card": {"path": str(card_path), "sha256": card_hash},
            "task_tsv": {"path": str(task_tsv), "sha256": task_hash},
            "source_lock": {
                "path": str(source_lock_path), "sha256": expected_source_lock_sha,
                "sources": lock["sources"],
            },
            "decision": {"path": str(decision_path), "sha256": sha256_path(decision_path)},
            "checkpoint_audit": {
                "path": str(audit_summary_path),
                "sha256": sha256_path(audit_summary_path),
            },
            "telemetry_assessment": {
                "path": str(telemetry_path), "sha256": sha256_path(telemetry_path),
            },
            "evaluation_shards": [
                {
                    "task_id": row["task_id"], "arm": row["arm"],
                    "seed": row["seed"],
                    "path": str(evaluation_dir / "shards" / f"task_{int(row['task_id']):02d}.json"),
                    "sha256": sha256_path(
                        evaluation_dir / "shards" / f"task_{int(row['task_id']):02d}.json"
                    ),
                    "checkpoint_sha256": row["provenance"]["selected_checkpoint_sha256"],
                }
                for row in rows
            ],
            "packet_builder_sha256": sha256_path(Path(__file__)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--source_lock", type=Path, required=True)
    parser.add_argument("--expected_source_lock_sha", required=True)
    parser.add_argument("--task_tsv", type=Path, required=True)
    parser.add_argument("--audit_summary", type=Path, required=True)
    parser.add_argument("--evaluation_dir", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--telemetry_assessment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    payload = build_packet(
        card_path=args.card, source_lock_path=args.source_lock,
        expected_source_lock_sha=args.expected_source_lock_sha,
        task_tsv=args.task_tsv, audit_summary_path=args.audit_summary,
        evaluation_dir=args.evaluation_dir, decision_path=args.decision,
        telemetry_path=args.telemetry_assessment,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output), "sha256": sha256_path(args.output),
        "mechanism_tier": payload["decision"]["mechanism_tier"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
