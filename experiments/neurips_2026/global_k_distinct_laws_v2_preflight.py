#!/usr/bin/env python3
"""Dependency-light, fail-closed preflights for distinct-law V2 SLURM stages."""

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


PROTOCOL_ID = "global_k_distinct_laws_gated_local_linear_v2_new_seeds"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Distinct-law V2 preflight failed: {message}")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    _require(isinstance(payload, dict), f"expected JSON object: {path}")
    return payload


def _authenticate_roots(
    card_path: Path, source_lock_path: Path,
    expected_source_lock_sha: str | None,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    source_lock_sha = sha256_path(source_lock_path)
    if expected_source_lock_sha is not None:
        _require(
            source_lock_sha == expected_source_lock_sha,
            "source-lock root hash mismatch",
        )
    lock = verify_source_lock(source_lock_path)
    card, card_sha = load_card(card_path)
    _require(card["protocol_id"] == PROTOCOL_ID, "protocol ID mismatch")
    _require(lock["card_sha256"] == card_sha, "card/source-lock mismatch")
    return card, card_sha, lock, source_lock_sha


def _authenticate_task_inputs(
    *, mode: str, task_tsv: Path, task_manifest: Path,
    card: dict[str, Any], card_sha: str, lock: dict[str, Any],
) -> tuple[str, str]:
    _require(mode in {"smoke", "full"}, f"invalid mode {mode!r}")
    task_sha = sha256_path(task_tsv)
    manifest_sha = sha256_path(task_manifest)
    locked_task = lock["external_inputs"][f"{mode}_task_tsv"]
    locked_manifest = lock["external_inputs"][f"{mode}_manifest"]
    _require(task_sha == locked_task["sha256"], "task/source-lock mismatch")
    _require(
        manifest_sha == locked_manifest["sha256"],
        "manifest/source-lock mismatch",
    )
    manifest = _read_json(task_manifest)
    expected_quarantine = mode == "smoke"
    checks = {
        "protocol": manifest.get("protocol_id") == card["protocol_id"],
        "mode": manifest.get("mode") == mode,
        "card": manifest.get("card_sha256") == card_sha,
        "task_hash": manifest.get("task_tsv_sha256") == task_sha,
        "task_count": manifest.get("task_count") == 20,
        "arm_count": len(manifest.get("arms", [])) == 20,
        "seed_count": len(manifest.get("seeds", [])) == 20,
        "quarantine": manifest.get("outcomes_quarantined") is expected_quarantine,
    }
    _require(all(checks.values()), f"{mode} manifest fields: {checks}")
    return task_sha, manifest_sha


def preflight_mixed_pack(
    *, mode: str, card_path: Path, source_lock_path: Path,
    expected_source_lock_sha: str, task_tsv: Path, task_manifest: Path,
) -> dict[str, Any]:
    card, card_sha, lock, source_lock_sha = _authenticate_roots(
        card_path, source_lock_path, expected_source_lock_sha
    )
    task_sha, manifest_sha = _authenticate_task_inputs(
        mode=mode, task_tsv=task_tsv, task_manifest=task_manifest,
        card=card, card_sha=card_sha, lock=lock,
    )
    return {
        "status": "passed", "stage": f"mixed_{mode}",
        "outcomes_inspected": False, "card_sha256": card_sha,
        "source_lock_sha256": source_lock_sha,
        "task_tsv_sha256": task_sha, "task_manifest_sha256": manifest_sha,
    }


def preflight_scientific_queue(
    *, card_path: Path, source_lock_path: Path, smoke_decision: Path,
    task_tsv: Path, task_manifest: Path,
) -> dict[str, Any]:
    card, card_sha, lock, source_lock_sha = _authenticate_roots(
        card_path, source_lock_path, None
    )
    task_sha, manifest_sha = _authenticate_task_inputs(
        mode="full", task_tsv=task_tsv, task_manifest=task_manifest,
        card=card, card_sha=card_sha, lock=lock,
    )
    decision = _read_json(smoke_decision)
    provenance = decision.get("provenance", {})
    required_checks = set(
        card["gpu_utilization_and_schedule"]["smoke"][
            "required_decision_check_keys"
        ]
    )
    checks = decision.get("checks")
    _require(
        isinstance(checks, dict) and set(checks) == required_checks
        and all(value is True for value in checks.values()),
        "smoke decision has missing, extra, non-boolean, or failed checks",
    )
    roots = {
        "passed": decision.get("passed") is True,
        "outcome_quarantine": decision.get("outcomes_inspected") is False,
        "protocol": decision.get("protocol_id") == card["protocol_id"],
        "card": provenance.get("card_sha256") == card_sha,
        "source_lock": provenance.get("source_lock_sha256") == source_lock_sha,
        "smoke_task": provenance.get("task_tsv_sha256")
        == lock["external_inputs"]["smoke_task_tsv"]["sha256"],
        "assessor": provenance.get("assessor_sha256")
        == lock["sources"][
            "experiments/neurips_2026/assess_global_k_distinct_laws_v2_smoke.py"
        ]["sha256"],
    }
    _require(all(roots.values()), f"smoke decision roots: {roots}")
    return {
        "status": "passed", "stage": "scientific_queue",
        "outcomes_inspected": False, "card_sha256": card_sha,
        "source_lock_sha256": source_lock_sha,
        "task_tsv_sha256": task_sha, "task_manifest_sha256": manifest_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    mixed = subparsers.add_parser("mixed")
    mixed.add_argument("--mode", choices=("smoke", "full"), required=True)
    mixed.add_argument("--expected_source_lock_sha", required=True)
    queue = subparsers.add_parser("queue")
    for child in (mixed, queue):
        child.add_argument("--card", type=Path, required=True)
        child.add_argument("--source_lock", type=Path, required=True)
        child.add_argument("--task_tsv", type=Path, required=True)
        child.add_argument("--task_manifest", type=Path, required=True)
    queue.add_argument("--smoke_decision", type=Path, required=True)
    args = parser.parse_args()
    common = {
        "card_path": args.card, "source_lock_path": args.source_lock,
        "task_tsv": args.task_tsv, "task_manifest": args.task_manifest,
    }
    if args.stage == "mixed":
        payload = preflight_mixed_pack(
            mode=args.mode,
            expected_source_lock_sha=args.expected_source_lock_sha,
            **common,
        )
    else:
        payload = preflight_scientific_queue(
            smoke_decision=args.smoke_decision, **common
        )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
