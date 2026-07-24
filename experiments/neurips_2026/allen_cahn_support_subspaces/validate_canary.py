"""Validate seed-64 lineage and telemetry before releasing remaining shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.neurips_2026.allen_cahn_support_subspaces.evaluation_helpers import (
    load_profile_decision,
    verify_source_manifest,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.io import (
    CARD_PATH,
    checkpoint_roster,
    load_card,
    sha256_path,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.summarize_gpu_telemetry import (
    TELEMETRY_RECEIPT_CHECK_KEYS,
    telemetry_receipt_checks,
)


SOURCE_MANIFEST = Path(__file__).with_name("source_manifest.sha256")
LINEAGE_CHECK_KEYS = frozenset({
    "status", "metric_keys_absent", "seed", "task", "card", "source",
    "profile", "source_current", "profile_current", "shard_path", "shard_hash",
    "historical_replay", "historical_batch", "historical_sequence", "firewall",
    "sparse_checkpoint", "dense_checkpoint",
})
TELEMETRY_CHECK_KEYS = TELEMETRY_RECEIPT_CHECK_KEYS


def _exact_checks_pass(receipt: dict, group: str, expected_keys: frozenset) -> bool:
    checks = receipt.get("checks")
    if not isinstance(checks, dict):
        return False
    group_checks = checks.get(group)
    return bool(
        isinstance(group_checks, dict)
        and group_checks
        and set(group_checks) == expected_keys
        and all(value is True for value in group_checks.values())
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--expected_card_sha256", required=True)
    parser.add_argument("--expected_source_manifest_sha256", required=True)
    parser.add_argument("--expected_profile_decision_sha256", required=True)
    parser.add_argument("--card", type=Path, default=CARD_PATH)
    return parser.parse_args()


def validate_release_receipt(
    path: Path, *, card_hash: str, source_hash: str, profile_hash: str
) -> dict:
    receipt = json.loads(path.read_text())
    valid = bool(
        receipt.get("status") == "passed"
        and receipt.get("mechanism_metrics_deserialized") is False
        and receipt.get("mechanism_metrics_used_for_release") is False
        and int(receipt.get("seed", -1)) == 64
        and receipt.get("card_sha256") == card_hash
        and receipt.get("source_manifest_sha256") == source_hash
        and receipt.get("profile_decision_sha256") == profile_hash
        and _exact_checks_pass(receipt, "lineage", LINEAGE_CHECK_KEYS)
        and _exact_checks_pass(receipt, "telemetry", TELEMETRY_CHECK_KEYS)
    )
    if not valid:
        raise RuntimeError("Canary release receipt failed frozen provenance checks")
    return receipt


def main() -> None:
    args = parse_args()
    output = args.output_root / "canary" / "validation.json"
    if output.exists():
        raise FileExistsError(output)
    card, card_hash = load_card(args.card)
    if card_hash != args.expected_card_sha256:
        raise RuntimeError("Canary card root mismatch")
    source_hash = verify_source_manifest(SOURCE_MANIFEST)
    if source_hash != args.expected_source_manifest_sha256:
        raise RuntimeError("Canary source root mismatch")
    profile_path = args.output_root / "profile" / "decision.json"
    profile_raw = json.loads(profile_path.read_text())
    selected_batch = int(profile_raw["selected_batch_size"])
    _, profile_hash = load_profile_decision(
        profile_path,
        selected_batch,
        card=card,
        card_hash=card_hash,
        source_manifest_hash=source_hash,
    )
    if profile_hash != args.expected_profile_decision_sha256:
        raise RuntimeError("Canary profile root mismatch")

    seed = int(card["roster"]["model_seeds"][0])
    if seed != 64:
        raise RuntimeError("Frozen canary must be model seed 64")
    shard_path = args.output_root / "shards" / f"seed_{seed}.json"
    lineage_path = args.output_root / "lineage" / f"seed_{seed}.json"
    telemetry_path = args.output_root / "telemetry" / f"seed_{seed}.json"
    lineage = json.loads(lineage_path.read_text())
    telemetry = json.loads(telemetry_path.read_text())
    roster = checkpoint_roster(card)
    lineage_checks = {
        "status": lineage.get("status") == "lineage_complete",
        "metric_keys_absent": lineage.get("mechanism_metric_keys_included") is False,
        "seed": int(lineage.get("seed", -1)) == seed,
        "task": int(lineage.get("task_index", -1)) == 0,
        "card": lineage.get("card_sha256") == card_hash,
        "source": lineage.get("source_manifest_sha256") == source_hash,
        "profile": lineage.get("profile_decision_sha256") == profile_hash,
        "source_current": sha256_path(Path(lineage["source_manifest_path"])) == source_hash,
        "profile_current": sha256_path(Path(lineage["profile_decision_path"])) == profile_hash,
        "shard_path": Path(lineage.get("scientific_shard", "")) == shard_path,
        "shard_hash": sha256_path(shard_path) == lineage.get("scientific_shard_sha256"),
        "historical_replay": lineage.get("historical_reproduction_passed") is True,
        "historical_batch": int(lineage.get(
            "historical_reproduction_batch_size", -1
        )) == int(card["inputs"]["ordinary_forecast_seed_rows"][
            "historical_reproduction_batch_size"
        ]),
        "historical_sequence": lineage.get(
            "historical_evaluator_horizon_sequence"
        ) == card["inputs"]["ordinary_forecast_seed_rows"][
            "historical_evaluator_horizon_sequence"
        ],
        "firewall": lineage.get("information_firewall", {}).get(
            "future_states_used_for_routing"
        ) is False,
    }
    for arm in ("sparse", "dense"):
        expected = roster[(arm, seed)]
        observed = lineage.get("provenance", {}).get(arm, {})
        lineage_checks[f"{arm}_checkpoint"] = (
            observed.get("checkpoint_path") == str(expected.path)
            and observed.get("checkpoint_sha256") == expected.sha256
        )
    telemetry_checks = telemetry_receipt_checks(
        telemetry,
        args.output_root / "telemetry",
        card_hash=card_hash,
        source_hash=source_hash,
        seed=seed,
        slurm_job_id=str(lineage.get("slurm_job_id")),
        evaluator_scope=lineage.get("gpu_telemetry_scope", {}),
    )
    check_rosters_match = (
        set(lineage_checks) == LINEAGE_CHECK_KEYS
        and set(telemetry_checks) == TELEMETRY_CHECK_KEYS
    )
    if (
        not check_rosters_match
        or not all(value is True for value in lineage_checks.values())
        or not all(value is True for value in telemetry_checks.values())
    ):
        raise RuntimeError(
            f"Canary failed: lineage={lineage_checks}, telemetry={telemetry_checks}"
        )
    payload = {
        "schema_version": 1,
        "status": "passed",
        "semantic_scope": "lineage, historical replay, firewall, and GPU telemetry only",
        "mechanism_metrics_deserialized": False,
        "mechanism_metrics_used_for_release": False,
        "seed": seed,
        "checks": {"lineage": lineage_checks, "telemetry": telemetry_checks},
        "shard_sha256": sha256_path(shard_path),
        "lineage_receipt_sha256": sha256_path(lineage_path),
        "telemetry_sha256": sha256_path(telemetry_path),
        "raw_telemetry_sha256": telemetry["raw_telemetry_sha256"],
        "gpu_start_marker_sha256": telemetry["gpu_start_marker_sha256"],
        "gpu_done_marker_sha256": telemetry["gpu_done_marker_sha256"],
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "profile_decision_sha256": profile_hash,
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    validate_release_receipt(
        output, card_hash=card_hash, source_hash=source_hash, profile_hash=profile_hash
    )
    print(json.dumps({"status": "passed", "seed": seed, "output": str(output)}))


if __name__ == "__main__":
    main()
