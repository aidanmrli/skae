#!/usr/bin/env python3
"""Outcome-blind supplemental integrity audit for distinct-local-laws V2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from experiments.neurips_2026.global_k_distinct_laws_v2_checkpoint_audit import (
    discover_trained_runs,
    load_trained_model,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_supplemental_claim_guards import (
    adverse_specificity_guard as _adverse_specificity_guard,
    finite_radius_integrity as _finite_radius_integrity,
    per_basin_counts as _per_basin_counts,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_source_lock import (
    verify_source_lock,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_tasks import (
    load_card,
    sha256_path,
)
from experiments.neurips_2026.summarize_global_k_distinct_laws_v2 import (
    _load_shards,
    adjudicate,
)
from skae.data import VectorWrapper
from skae.training.runner import evaluate, generate_sequence_batch_for_device


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ID = "global_k_distinct_laws_v2_supplemental_integrity_audit"
SUPPLEMENTAL_SOURCES = {
    "experiments/neurips_2026/global_k_distinct_laws_v2_supplemental_audit.py",
    "experiments/neurips_2026/global_k_distinct_laws_v2_supplemental_audit_card.json",
    "experiments/neurips_2026/global_k_distinct_laws_v2_supplemental_claim_guards.py",
    "scripts/neurips_2026/global_k_distinct_laws_v2/run_supplemental_audit.sh",
    "tests/test_global_k_distinct_laws_v2_supplemental_audit.py",
}
EVALUATOR_SOURCE = "experiments/neurips_2026/global_k_distinct_laws_v2.py"
SUMMARIZER_SOURCE = (
    "experiments/neurips_2026/summarize_global_k_distinct_laws_v2.py"
)


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _load_supplemental_lock(
    lock_path: Path, supplemental_card_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = json.loads(lock_path.read_text())
    card = json.loads(supplemental_card_path.read_text())
    if lock.get("protocol_id") != PROTOCOL_ID or card.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Unexpected supplemental-audit protocol")
    if card.get("status") != (
        "amended_and_refrozen_before_scientific_outcome_access_by_auditor"
    ):
        raise RuntimeError("Supplemental card is not frozen")
    sources = lock.get("supplemental_sources", {})
    if set(sources) != SUPPLEMENTAL_SOURCES:
        raise RuntimeError("Supplemental source roster drift")
    failures = []
    for name, expected in sources.items():
        source = REPOSITORY_ROOT / name
        actual = sha256_path(source) if source.is_file() else "missing"
        if expected.get("path") != name or expected.get("sha256") != actual:
            failures.append(f"{name}: {actual} != {expected.get('sha256')}")
    card_name = (
        "experiments/neurips_2026/"
        "global_k_distinct_laws_v2_supplemental_audit_card.json"
    )
    if sources[card_name]["sha256"] != sha256_path(supplemental_card_path):
        failures.append("caller supplemental card differs from frozen card")
    if failures:
        raise RuntimeError("Supplemental source-lock failure:\n" + "\n".join(failures))
    return lock, card


def _finite_trajectory_counts(
    initial: torch.Tensor, truth: torch.Tensor, prediction: torch.Tensor,
) -> dict[str, Any]:
    if truth.shape != prediction.shape or truth.ndim != 3:
        raise RuntimeError(
            f"Unexpected validation trajectory shapes: {truth.shape}, {prediction.shape}"
        )
    if initial.ndim != 2 or initial.shape != truth.shape[1:]:
        raise RuntimeError(f"Unexpected validation initial-state shape: {initial.shape}")
    initial_ok = torch.isfinite(initial).all(dim=1)
    truth_ok = torch.isfinite(truth).all(dim=(0, 2))
    prediction_ok = torch.isfinite(prediction).all(dim=(0, 2))
    joint = initial_ok & truth_ok & prediction_ok
    return {
        "trajectory_count": int(truth.shape[1]),
        "initial_finite_count": int(initial_ok.sum().item()),
        "truth_finite_count": int(truth_ok.sum().item()),
        "prediction_finite_count": int(prediction_ok.sum().item()),
        "joint_finite_count": int(joint.sum().item()),
        "joint_finite_by_trajectory": [bool(value) for value in joint.tolist()],
    }


def _serialized_selector_score(
    checkpoint: dict[str, Any], audit_row: dict[str, Any],
) -> tuple[float | None, str | None]:
    names = ("selected_validation_final_error", "best_eval_final_error")
    for name in names:
        value = checkpoint.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value), f"checkpoint.{name}"
    selector = audit_row.get("checkpoint_selector", {})
    for name in names:
        value = selector.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value), f"checkpoint_audit.{name}"
    return None, None


def _checkpoint_hashes_authenticate(
    current: str, audit: str, evaluation: str,
) -> bool:
    return current == audit == evaluation


def _rerun_selected_checkpoint_validation(
    *, task_tsv: Path, base_out: Path, v2_card: dict[str, Any],
    audit_summary: dict[str, Any], supplemental_card: dict[str, Any],
    evaluation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    specs = discover_trained_runs(task_tsv, base_out, v2_card)
    audits = {int(row["task_id"]): row for row in audit_summary.get("rows", [])}
    evaluations = {int(row["task_id"]): row for row in evaluation_rows}
    expected_runs = int(
        supplemental_card["selected_checkpoint_validation"][
            "required_selected_checkpoints"
        ]
    )
    required_finite = int(
        supplemental_card["selected_checkpoint_validation"][
            "required_joint_finite_trajectories_per_checkpoint"
        ]
    )
    expected_tasks = set(range(expected_runs))
    if (
        len(specs) != expected_runs
        or set(audits) != expected_tasks
        or set(evaluations) != expected_tasks
    ):
        raise RuntimeError("Validation rerun/checkpoint-audit roster is incomplete")
    rows = []
    for spec in specs:
        cfg, base_env, model, checkpoint, checkpoint_path = load_trained_model(spec)
        eval_env = VectorWrapper(base_env, int(cfg.TRAIN.BATCH_SIZE))
        validation_rng = torch.Generator().manual_seed(int(cfg.SEED) + 999999)
        validation_sequence = generate_sequence_batch_for_device(
            eval_env,
            validation_rng,
            window_length=max(1, int(cfg.TRAIN.SEQUENCE_LENGTH)),
            device="cpu",
        )
        initial = validation_sequence[:16, 0, :]
        result = evaluate(
            model,
            initial,
            lambda state: eval_env.step(state),
            num_steps=int(cfg.TRAIN.EVAL_NUM_STEPS),
        )
        truth = result["true_trajectory"]
        prediction = result["pred_trajectory"]
        finite = _finite_trajectory_counts(initial, truth, prediction)
        direct_score = float(
            torch.linalg.vector_norm(prediction[-1] - truth[-1], dim=-1).mean().item()
        )
        recomputed_score = float(result["final_error"])
        internal_equal = math.isclose(
            recomputed_score, direct_score, rel_tol=1e-12, abs_tol=1e-12
        )
        historical_score, historical_source = _serialized_selector_score(
            checkpoint, audits[spec.task_id]
        )
        historical_equal = (
            None
            if historical_score is None
            else math.isclose(
                recomputed_score, historical_score, rel_tol=1e-12, abs_tol=1e-12
            )
        )
        checkpoint_hash = sha256_path(checkpoint_path)
        audit_hash = audits[spec.task_id]["checkpoint_sha256"]
        evaluation_hash = evaluations[spec.task_id]["provenance"][
            "selected_checkpoint_sha256"
        ]
        checkpoint_authenticated = _checkpoint_hashes_authenticate(
            checkpoint_hash, audit_hash, evaluation_hash
        )
        passed = bool(
            finite["trajectory_count"] == required_finite
            and finite["joint_finite_count"] == required_finite
            and math.isfinite(recomputed_score)
            and internal_equal
            and historical_equal is not False
            and checkpoint_authenticated
        )
        rows.append(
            {
                "task_id": spec.task_id,
                "arm": spec.arm,
                "seed": spec.seed,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_sha256_from_audit": audit_hash,
                "checkpoint_sha256_from_evaluation_shard": evaluation_hash,
                "checkpoint_authenticated_across_all_sources": checkpoint_authenticated,
                "selector_rollout": "every_step_physical_reencoding",
                "selector_horizon_steps": int(cfg.TRAIN.EVAL_NUM_STEPS),
                "finite_trajectories": finite,
                "recomputed_final_error": recomputed_score,
                "direct_finite_mean_final_error": direct_score,
                "internal_score_equality": internal_equal,
                "serialized_historical_score_available": historical_score is not None,
                "serialized_historical_score_source": historical_source,
                "serialized_historical_score": historical_score,
                "historical_score_equality": historical_equal,
                "historical_score_limit": (
                    None
                    if historical_score is not None
                    else supplemental_card["selected_checkpoint_validation"][
                        "known_serialization_limit"
                    ]
                ),
                "passed": passed,
            }
        )
    return {
        "passed": len(rows) == expected_runs and all(row["passed"] for row in rows),
        "required_checkpoint_count": expected_runs,
        "passed_checkpoint_count": sum(row["passed"] for row in rows),
        "required_joint_finite_trajectories_each": required_finite,
        "rows": rows,
    }


def _reproduce_decision(
    *, rows: list[dict[str, Any]], v2_card: dict[str, Any],
    audit_summary: dict[str, Any], decision_path: Path, card_hash: str,
    task_hash: str, source_lock_hash: str, evaluator_hash: str,
    summarizer_hash: str,
) -> dict[str, Any]:
    evaluator_hashes = {
        row.get("provenance", {}).get("evaluator_sha256") for row in rows
    }
    if evaluator_hashes != {evaluator_hash}:
        raise RuntimeError(
            f"Shard evaluator/source-lock mismatch: {sorted(map(str, evaluator_hashes))}"
        )
    original_bytes = decision_path.read_bytes()
    original = json.loads(original_bytes)
    expected_provenance = {
        "card_sha256": card_hash,
        "task_tsv_sha256": task_hash,
        "source_lock_sha256": source_lock_hash,
        "summarizer_sha256": summarizer_hash,
    }
    if original.get("provenance") != expected_provenance:
        raise RuntimeError("Decision provenance differs from source-locked provenance")
    reproduced = adjudicate(rows, v2_card, audit_summary)
    reproduced["provenance"] = expected_provenance
    reproduced_bytes = _canonical_bytes(reproduced)
    return {
        "passed": bool(original == reproduced and original_bytes == reproduced_bytes),
        "parsed_value_equality": original == reproduced,
        "byte_equality": original_bytes == reproduced_bytes,
        "original_decision_sha256": sha256_path(decision_path),
        "reproduced_decision_sha256": __import__("hashlib").sha256(
            reproduced_bytes
        ).hexdigest(),
        "evaluator_sha256": evaluator_hash,
        "summarizer_sha256": summarizer_hash,
        "decision": reproduced,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplemental_lock", type=Path, required=True)
    parser.add_argument("--supplemental_card", type=Path, required=True)
    parser.add_argument("--v2_card", type=Path, required=True)
    parser.add_argument("--source_lock", type=Path, required=True)
    parser.add_argument("--task_tsv", type=Path, required=True)
    parser.add_argument("--base_out", type=Path, required=True)
    parser.add_argument("--audit_dir", type=Path, required=True)
    parser.add_argument("--evaluation_dir", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    supplemental_lock, supplemental_card = _load_supplemental_lock(
        args.supplemental_lock, args.supplemental_card
    )
    parent = supplemental_card["parent_protocol"]
    frozen_parent = supplemental_lock["frozen_parent_inputs"]
    if frozen_parent != parent:
        raise RuntimeError("Supplemental lock and card parent inputs differ")
    if sha256_path(args.source_lock) != parent["source_lock_sha256"]:
        raise RuntimeError("Caller V2 source-lock hash mismatch")
    source_lock = verify_source_lock(args.source_lock)
    v2_card, card_hash = load_card(args.v2_card)
    task_hash = sha256_path(args.task_tsv)
    if card_hash != parent["card_sha256"] or task_hash != parent["full_task_tsv_sha256"]:
        raise RuntimeError("Caller V2 card/task hash mismatch")
    evaluator_hash = source_lock["sources"][EVALUATOR_SOURCE]["sha256"]
    summarizer_hash = source_lock["sources"][SUMMARIZER_SOURCE]["sha256"]
    if (
        evaluator_hash != parent["evaluator_sha256"]
        or summarizer_hash != parent["summarizer_sha256"]
    ):
        raise RuntimeError("Evaluator/summarizer differs from supplemental freeze")
    source_lock_hash = parent["source_lock_sha256"]
    rows = _load_shards(
        args.evaluation_dir, v2_card, card_hash, task_hash, source_lock_hash
    )
    audit_summary = json.loads((args.audit_dir / "summary.json").read_text())
    validation = _rerun_selected_checkpoint_validation(
        task_tsv=args.task_tsv,
        base_out=args.base_out,
        v2_card=v2_card,
        audit_summary=audit_summary,
        supplemental_card=supplemental_card,
        evaluation_rows=rows,
    )
    reproduction = _reproduce_decision(
        rows=rows,
        v2_card=v2_card,
        audit_summary=audit_summary,
        decision_path=args.decision,
        card_hash=card_hash,
        task_hash=task_hash,
        source_lock_hash=source_lock_hash,
        evaluator_hash=evaluator_hash,
        summarizer_hash=summarizer_hash,
    )
    minimum = int(
        supplemental_card["per_basin_replication"][
            "minimum_adequate_passes_per_basin"
        ]
    )
    decision = reproduction.pop("decision")
    radius_integrity = _finite_radius_integrity(rows, v2_card)
    per_basin = _per_basin_counts(
        rows, v2_card, decision["mechanism_tier"], minimum
    )
    specificity_guard = _adverse_specificity_guard(decision, v2_card)
    integrity_passed = bool(
        validation["passed"]
        and reproduction["passed"]
        and radius_integrity["passed"]
    )
    wording_permitted = per_basin["blanket_three_law_wording_permitted"]
    relative_claim_frozen = (
        decision["relative_specificity_tier"]
        == "sparse_recipe_support_basis_specific"
    )
    relative_wording_permitted = bool(
        not relative_claim_frozen
        or specificity_guard["positive_relative_specificity_claim_permitted"]
    )
    recommended_text = decision["mechanism_text"]
    if wording_permitted is False:
        recommended_text = (
            "Per-basin replication is imbalanced for this aggregate tier; report "
            "the basin-specific numerators and do not state blanket recovery of "
            "all three local laws."
        )
    recommended_relative_text = decision["relative_specificity_text"]
    if not relative_wording_permitted:
        recommended_relative_text = (
            "Sparse-versus-dense relative specificity is unresolved under the "
            "fixed-ten-seed adverse completion; do not claim recipe specificity."
        )
    output = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": (
            "passed"
            if integrity_passed and wording_permitted is not False
            and relative_wording_permitted
            else "passed_with_wording_restriction"
            if integrity_passed
            else "failed"
        ),
        "integrity_passed": integrity_passed,
        "selected_checkpoint_validation": validation,
        "independent_adjudication_reproduction": reproduction,
        "finite_radius_integrity": radius_integrity,
        "per_basin_replication": per_basin,
        "relative_specificity_adverse_completion": specificity_guard,
        "frozen_mechanism_tier": decision["mechanism_tier"],
        "frozen_mechanism_text": decision["mechanism_text"],
        "recommended_mechanism_text": recommended_text,
        "frozen_relative_specificity_tier": decision["relative_specificity_tier"],
        "frozen_relative_specificity_text": decision["relative_specificity_text"],
        "recommended_relative_specificity_text": recommended_relative_text,
        "limitations": supplemental_card["limitations"],
        "provenance": {
            "supplemental_lock_sha256": sha256_path(args.supplemental_lock),
            "supplemental_card_sha256": sha256_path(args.supplemental_card),
            "v2_card_sha256": card_hash,
            "source_lock_sha256": source_lock_hash,
            "task_tsv_sha256": task_hash,
            "evaluator_sha256": evaluator_hash,
            "summarizer_sha256": summarizer_hash,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(output))
    print(json.dumps({"output": str(args.output), "status": output["status"]}))
    if not integrity_passed:
        raise RuntimeError("Supplemental V2 integrity audit failed")


if __name__ == "__main__":
    main()
