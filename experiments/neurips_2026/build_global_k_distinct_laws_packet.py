#!/usr/bin/env python3
"""Build an authenticated sparse-first packet for the distinct-law protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.neurips_2026.global_k_distinct_laws import load_protocol_card
from experiments.neurips_2026.summarize_global_k_distinct_laws import adjudicate


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_complete_arm(
    output_dir: Path,
    arm: str,
    card: dict[str, Any],
    card_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    expected = {int(seed) for seed in card["scope"][f"{arm}_seeds"]}
    observed: dict[int, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for path in sorted((output_dir / "shards").glob(f"{arm}_seed_*.json")):
        payload = json.loads(path.read_text())
        if payload.get("protocol_id") != card["protocol_id"]:
            raise RuntimeError(f"Protocol mismatch in {path}")
        if payload.get("card_sha256") != card_hash:
            raise RuntimeError(f"Card mismatch in {path}")
        if payload.get("arm") != arm:
            raise RuntimeError(f"Arm mismatch in {path}")
        seed = int(payload["seed"])
        if seed in observed:
            raise RuntimeError(f"Duplicate {arm} seed {seed}")
        observed[seed] = payload
        hashes[str(path)] = sha256_path(path)
    if set(observed) != expected:
        raise RuntimeError(
            f"Refusing partial {arm} packet: missing={sorted(expected - set(observed))}, "
            f"extra={sorted(set(observed) - expected)}"
        )
    payloads = [observed[seed] for seed in sorted(expected)]
    evaluator_hashes = {
        row["provenance"]["evaluator_sha256"] for row in payloads
    }
    if len(evaluator_hashes) != 1:
        raise RuntimeError(f"Evaluator hash drift across {arm} shards")
    return payloads, hashes


def sparse_phase_decision(rows: list[dict[str, Any]], card: dict[str, Any]) -> dict[str, Any]:
    valid = all(row["result"]["status"] == "eligible" for row in rows)
    strong = valid and all(
        bool(row["result"]["strong_distinct_law_pass"]) for row in rows
    )
    threshold = float(
        card["strong_distinct_law_gate"]["max_global_change_matrix_relative_error"]
    )
    global_positive = valid and all(
        float(
            row["result"]["global"]["law_identification"][
                "max_own_relative_error"
            ]
        )
        <= threshold
        for row in rows
    )
    if not valid:
        decision = "invalid"
        claim = "At least one sparse seed failed a frozen validity gate; no distinct-law conclusion is permitted."
    elif strong:
        decision = "strong_distinct_laws_pending_dense_specificity"
        claim = (
            "All sparse seeds pass the frozen controlled distinct-law gate. The one-K "
            "capacity claim is permitted on GatedLocalLinear; sparse-specific causation "
            "remains pending the exact dense arm."
        )
    elif global_positive:
        decision = "global_map_only"
        claim = card["decision_branches"]["global_map_only"]
    else:
        decision = "failed"
        claim = card["decision_branches"]["failed"]
    return {
        "decision": decision,
        "allowed_claim": claim,
        "sparse_valid_3_of_3": valid,
        "sparse_strong_3_of_3": strong,
        "global_positive_3_of_3": global_positive,
        "dense_specificity": "pending",
    }


def compact_seed_row(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload["result"]
    row: dict[str, Any] = {
        "arm": payload["arm"],
        "seed": payload["seed"],
        "status": result["status"],
        "checkpoint_sha256": payload["provenance"]["selected_checkpoint_sha256"],
    }
    if "block" not in result:
        row["routing"] = result.get("routing", {})
        return row
    block = result["block"]["law_identification"]
    global_law = result["global"]["law_identification"]
    row.update(
        {
            "geometry_valid": result["geometry_valid"],
            "strong_distinct_law_pass": result["strong_distinct_law_pass"],
            "matched_family_by_basin": result["routing"]["matched_family_by_basin"],
            "calibration_match_rate_by_basin": result["routing"][
                "calibration_match_rate_by_basin"
            ],
            "score_match_rate_by_basin": result["routing"][
                "score_match_rate_by_basin"
            ],
            "block_max_own_relative_error": block["max_own_relative_error"],
            "block_max_own_over_nearest_wrong": block[
                "max_own_over_nearest_wrong"
            ],
            "block_identity_over_best_nonidentity": block[
                "identity_over_best_nonidentity"
            ],
            "block_identity_is_unique_optimum": block[
                "identity_is_unique_optimum"
            ],
            "global_max_own_relative_error": global_law["max_own_relative_error"],
            "max_local_fit_relative_residual": max(
                result["block"]["relative_fit_residual_by_basin"]
            ),
            "max_anchor_update_over_true_rms": max(
                result["block"]["anchor_update_over_true_rms_by_basin"]
            ),
            "max_block_vs_source_matrix_discrepancy": result["closure"][
                "max_block_vs_source_matrix_discrepancy"
            ],
            "analytic_true_step_max_abs_disagreement": result["geometry"][
                "analytic_true_step_max_abs_disagreement"
            ],
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, default=Path(__file__).with_name("global_k_distinct_laws_card.json"))
    parser.add_argument("--sparse_output_dir", type=Path, required=True)
    parser.add_argument("--dense_output_dir", type=Path)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--sparse_job_id", default="not_recorded")
    parser.add_argument("--dense_job_id", default="pending")
    args = parser.parse_args()
    if args.output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_path}")
    card, card_hash = load_protocol_card(args.card)
    sparse, sparse_hashes = load_complete_arm(
        args.sparse_output_dir, "sparse", card, card_hash
    )
    sparse_decision = sparse_phase_decision(sparse, card)
    dense: list[dict[str, Any]] = []
    dense_hashes: dict[str, str] = {}
    final_decision = None
    if args.dense_output_dir is not None:
        dense, dense_hashes = load_complete_arm(
            args.dense_output_dir, "dense", card, card_hash
        )
        final_decision = adjudicate({"sparse": sparse, "dense": dense}, card)
    packet = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "phase": "paired_complete" if dense else "sparse_complete_dense_pending",
        "card": {"path": str(args.card), "sha256": card_hash},
        "jobs": {
            "sparse": args.sparse_job_id,
            "dense": args.dense_job_id if dense else "pending",
        },
        "sparse_phase_decision": sparse_decision,
        "final_paired_decision": final_decision,
        "seed_rows": [compact_seed_row(row) for row in sparse + dense],
        "shard_sha256": {**sparse_hashes, **dense_hashes},
        "source_sha256": {
            "evaluator": sparse[0]["provenance"]["evaluator_sha256"],
            "packet_builder": sha256_path(Path(__file__)),
        },
        "claim_guard": (
            "Sparse completion cannot establish sparse-specific causation. The paired "
            "decision is emitted only after all three exact dense shards authenticate."
        ),
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(packet, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output_path),
                "phase": packet["phase"],
                "decision": sparse_decision["decision"],
            }
        )
    )


if __name__ == "__main__":
    main()
