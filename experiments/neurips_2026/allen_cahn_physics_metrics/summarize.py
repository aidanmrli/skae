"""CPU-only guarded summary of all frozen Allen--Cahn physics metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.neurips_2026.allen_cahn_physics_metrics.core import validate_score_record
from experiments.neurips_2026.allen_cahn_physics_metrics.io import (
    CARD_PATH,
    MANIFEST_PATH,
    assert_paths_sealed,
    duplicate_safe_json,
    load_card,
    sha256_path,
    verify_file,
    verify_source_manifest,
    write_json_once,
)
from experiments.neurips_2026.allen_cahn_physics_metrics.statistics import summarize_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, default=CARD_PATH)
    parser.add_argument("--expected-card-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--outcome-receipt", type=Path, required=True)
    parser.add_argument("--expected-outcome-receipt-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_paths_sealed(vars(args).values())
    card, card_hash = load_card(args.card, expected_sha256=args.expected_card_sha256)
    source_hash = verify_source_manifest(
        args.source_manifest, expected_sha256=args.expected_source_manifest_sha256
    )
    if args.output_root != Path(card["execution"]["output_root"]):
        raise RuntimeError("Summary output root differs from the frozen card")
    verify_file(args.outcome_receipt, args.expected_outcome_receipt_sha256)
    receipt = duplicate_safe_json(args.outcome_receipt)
    if (
        receipt.get("status") != "authorized_for_dependent_cpu_summary"
        or receipt.get("card_sha256") != card_hash
        or receipt.get("source_manifest_sha256") != source_hash
        or receipt.get("row_count") != 63
        or receipt.get("scientific_payload_opened") is not False
    ):
        raise RuntimeError("Outcome receipt does not authorize this exact summary")
    runtime_path = Path(receipt["runtime_lineage_path"])
    verify_file(runtime_path, receipt["runtime_lineage_sha256"])
    payload_path = Path(receipt["scientific_payload_path"])
    verify_file(payload_path, receipt["scientific_payload_sha256"])
    payload = duplicate_safe_json(payload_path)
    if (
        payload.get("protocol_id") != card["protocol_id"]
        or payload.get("card_sha256") != card_hash
        or payload.get("source_manifest_sha256") != source_hash
        or payload.get("outcomes_printed") is not False
    ):
        raise RuntimeError("Physics payload lineage failed")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 63:
        raise RuntimeError("Physics payload has the wrong row count")
    for row in rows:
        validate_score_record(row)
    inference = card["inference_and_reporting"]
    summary = summarize_rows(
        rows,
        bootstrap_replicates=int(inference["bootstrap_replicates"]),
        bootstrap_seeds_by_metric=inference["bootstrap_seeds_by_metric"],
    )
    write_json_once(
        args.output_root / "summary" / "physics_metrics_summary.json",
        {
            "schema_version": 1,
            "protocol_id": card["protocol_id"],
            "card_sha256": card_hash,
            "source_manifest_sha256": source_hash,
            "outcome_receipt_sha256": args.expected_outcome_receipt_sha256,
            "evidence_grade": "outcome_aware_same_checkpoint_secondary",
            "original_field_mse_inference_reclassified": False,
            "all_frozen_metrics_and_horizons_reported": True,
            **summary,
        },
    )


if __name__ == "__main__":
    main()
