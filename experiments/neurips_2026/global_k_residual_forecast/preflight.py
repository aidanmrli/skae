"""Outcome-blind CPU preflight for the complete residual forecast bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from experiments.neurips_2026.global_k_residual_forecast.protocol import (
    DEFAULT_CARD,
    DEFAULT_SOURCES,
    DEFAULT_TASKS,
    authenticate_checkpoint_roster,
    authenticate_v2_inputs,
    load_frozen_protocol,
)


def authenticate_complete_bundle(
    *,
    card_path: Path,
    task_path: Path,
    source_manifest_path: Path,
    expected_card_sha256: str,
    expected_task_sha256: str,
    expected_source_manifest_sha256: str,
    stage: str,
    output_root: Path,
    mode: str | None = None,
    task_index: int | None = None,
) -> dict[str, Any]:
    card, tasks, freeze = load_frozen_protocol(
        card_path=card_path,
        task_path=task_path,
        source_manifest_path=source_manifest_path,
        expected_card_sha256=expected_card_sha256,
        expected_task_sha256=expected_task_sha256,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
    )
    if Path(card["freeze"]["output_root"]) != output_root:
        raise RuntimeError("Output root differs from the frozen card")
    if stage not in {"queue", "prepare", "forecast", "telemetry", "summary"}:
        raise RuntimeError(f"Unknown preflight stage: {stage}")
    if stage == "forecast":
        if mode not in {"smoke", "scientific"}:
            raise RuntimeError("Forecast preflight requires a valid mode")
        expected_tasks = {0} if mode == "smoke" else set(range(10))
        if task_index not in expected_tasks:
            raise RuntimeError("Forecast task identity is invalid for its mode")
    elif stage == "telemetry":
        if mode not in {"smoke", "scientific"} or task_index is not None:
            raise RuntimeError("Telemetry preflight stage identity is invalid")
    elif mode is not None or task_index is not None:
        raise RuntimeError(f"{stage} preflight accepts no mode or task index")
    v2 = authenticate_v2_inputs(card)
    checkpoint_count = authenticate_checkpoint_roster(tasks)
    return {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "artifact_role": "outcome_blind_complete_bundle_preflight",
        "stage": stage,
        "mode": mode,
        "task_index": task_index,
        "freeze": freeze,
        "v2_protocol_id": v2["card"]["protocol_id"],
        "checkpoint_count": checkpoint_count,
        "outcomes_inspected": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--expected-card-sha256", required=True)
    parser.add_argument("--expected-task-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument(
        "--stage", choices=("queue", "prepare", "forecast", "telemetry", "summary"),
        required=True,
    )
    parser.add_argument("--mode", choices=("smoke", "scientific"))
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError("CPU preflight requires CUDA_VISIBLE_DEVICES to be empty")
    payload = authenticate_complete_bundle(
        card_path=args.card,
        task_path=args.tasks,
        source_manifest_path=args.sources,
        expected_card_sha256=args.expected_card_sha256,
        expected_task_sha256=args.expected_task_sha256,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        stage=args.stage,
        output_root=args.output_root,
        mode=args.mode,
        task_index=args.task_index,
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
