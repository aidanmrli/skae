"""Receipt-gated CPU reduction of the sole primary and descriptive curves."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.neurips_2026.allen_cahn_forecast_replication.io import (
    CARD_PATH,
    MANIFEST_PATH,
    assert_runtime_values_safe,
    duplicate_safe_json,
    load_card,
    sha256_path,
    verify_source_manifest,
    write_json_once,
)
from experiments.neurips_2026.allen_cahn_forecast_replication.statistics import (
    summarize_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, default=CARD_PATH)
    parser.add_argument("--expected-card-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--outcome-guard-receipt", type=Path, required=True)
    parser.add_argument("--expected-outcome-guard-receipt-sha256", required=True)
    return parser.parse_args()


def verify_outcome_guard_receipt(
    path: Path,
    *,
    expected_sha256: str,
    card_hash: str,
    source_hash: str,
) -> dict[str, Any]:
    observed = sha256_path(path)
    if observed != expected_sha256:
        raise RuntimeError(f"Outcome-guard receipt hash mismatch: {observed} != {expected_sha256}")
    receipt = duplicate_safe_json(path)
    roster = receipt.get("checkpoint_roster")
    if not isinstance(roster, list) or len(roster) != 20:
        raise RuntimeError("Outcome-guard receipt lacks the explicit 20-row roster")
    roster_keys = {(str(row["arm"]), int(row["seed"])) for row in roster}
    encoded_roster = json.dumps(
        roster, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    roster_hash = hashlib.sha256(encoded_roster).hexdigest()
    if (
        receipt.get("status") != "authorized_for_dependent_cpu_summary"
        or receipt.get("card_sha256") != card_hash
        or receipt.get("source_manifest_sha256") != source_hash
        or receipt.get("crossed_cells") != 60
        or receipt.get("scientific_payload_opened") is not False
        or len(roster_keys) != 20
        or roster_hash != receipt.get("checkpoint_roster_sha256")
        or not str(receipt.get("gpu_uuid", "")).startswith("GPU-")
    ):
        raise RuntimeError("Outcome-guard receipt failed its fixed lineage fields")
    bound = (
        ("runtime_lineage_path", "runtime_lineage_sha256"),
        ("dataset_manifest_path", "dataset_manifest_sha256"),
        ("telemetry_audit_path", "telemetry_audit_sha256"),
        ("scientific_payload_path", "scientific_payload_sha256"),
    )
    for path_key, hash_key in bound:
        candidate = Path(str(receipt[path_key]))
        if sha256_path(candidate) != receipt[hash_key]:
            raise RuntimeError(f"Receipt-bound artifact failed: {candidate}")
    runtime = duplicate_safe_json(Path(str(receipt["runtime_lineage_path"])))
    telemetry = duplicate_safe_json(Path(str(receipt["telemetry_audit_path"])))
    if (
        runtime.get("status") != "scientific_payload_written_but_not_authorized_for_summary"
        or runtime.get("scientific_payload_sha256") != receipt["scientific_payload_sha256"]
        or runtime.get("checkpoint_roster_sha256") != receipt["checkpoint_roster_sha256"]
        or runtime.get("checkpoint_roster") != roster
        or runtime.get("scientific_metrics_printed") is not False
    ):
        raise RuntimeError("Metric-free runtime lineage differs from receipt")
    if (
        telemetry.get("status") != "passed"
        or telemetry.get("scientific_payload_opened") is not False
        or not all(telemetry.get("evaluation_checks", {}).values())
        or telemetry.get("gpu_uuid") != receipt["gpu_uuid"]
        or telemetry.get("slurm_job_id") != receipt.get("slurm_job_id")
    ):
        raise RuntimeError("Telemetry audit did not authorize outcome access")
    runtime_slurm = str(runtime.get("environment", {}).get("slurm_job_id", "not_recorded"))
    if runtime_slurm != str(receipt.get("slurm_job_id", "not_recorded")):
        raise RuntimeError("Runtime and receipt SLURM job lineage differ")
    return receipt


def main() -> None:
    args = parse_args()
    assert_runtime_values_safe(
        [
            args.card,
            args.source_manifest,
            args.output_root,
            args.outcome_guard_receipt,
            args.expected_card_sha256,
            args.expected_source_manifest_sha256,
            args.expected_outcome_guard_receipt_sha256,
        ]
    )
    card, card_hash = load_card(args.card, expected_sha256=args.expected_card_sha256)
    source_hash = verify_source_manifest(
        card,
        path=args.source_manifest,
        expected_sha256=args.expected_source_manifest_sha256,
    )
    if args.output_root != Path(card["prospective_datasets"]["output_root"]):
        raise RuntimeError("Summary output root differs from the frozen card")
    receipt = verify_outcome_guard_receipt(
        args.outcome_guard_receipt,
        expected_sha256=args.expected_outcome_guard_receipt_sha256,
        card_hash=card_hash,
        source_hash=source_hash,
    )

    # This is deliberately the first scientific-payload deserialization.
    scientific_path = Path(str(receipt["scientific_payload_path"]))
    scientific = duplicate_safe_json(scientific_path)
    if (
        scientific.get("protocol_id") != card["protocol_id"]
        or scientific.get("card_sha256") != card_hash
        or scientific.get("source_manifest_sha256") != source_hash
        or scientific.get("checkpoint_roster_sha256") != receipt["checkpoint_roster_sha256"]
        or scientific.get("crossed_cells") != 60
    ):
        raise RuntimeError("Scientific payload lineage failed after receipt authorization")
    summary = summarize_rows(scientific["rows"], card)
    summary["provenance"] = {
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "outcome_guard_receipt_path": str(args.outcome_guard_receipt),
        "outcome_guard_receipt_sha256": args.expected_outcome_guard_receipt_sha256,
        "scientific_payload_path": str(scientific_path),
        "scientific_payload_sha256": receipt["scientific_payload_sha256"],
        "inference_unit": "paired_model_seed",
        "curve_inference_policy": "pointwise_descriptive_only",
    }
    output = args.output_root / "summary" / "decision.json"
    write_json_once(output, summary)
    print(
        json.dumps(
            {
                "status": "summary_written_after_valid_receipt",
                "summary_path": str(output),
                "summary_sha256": sha256_path(output),
                "scientific_metrics_printed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
