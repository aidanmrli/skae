"""Generate, select, and evaluate the frozen Allen--Cahn periodic policy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Any

import torch

from experiments.neurips_2026.allen_cahn_periodic_reencoding.generator import (
    generate_fields,
    integrate_initial_fields,
    realized_rng_streams,
    sample_initial_fields,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.h400_full_grid import (
    validate_full_grid_h400_rows,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.io import (
    CARD_PATH,
    MANIFEST_PATH,
    checkpoint_specs,
    duplicate_safe_json,
    load_card,
    load_checkpoint_model,
    sha256_path,
    verify_file,
    verify_source_manifest,
    write_json_once,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.runtime_helpers import (
    cadence_grid as _cadence_grid,
    evaluate_cross as _evaluate_cross,
    save_datasets as _save_datasets,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.lineage import (
    canonical_digest,
    write_runtime_lineage,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.p200_one_refresh import (
    P200,
    validate_p200_h400_rows,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.statistics import (
    select_recipe_cadences,
    validate_primary_test_rows,
    validate_test_rows,
    validate_validation_rows,
    validation_candidate_scores,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.stress import (
    evaluate_stress_cross,
    truth_difficulty,
    validate_stress_prefix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, default=CARD_PATH)
    parser.add_argument("--expected-card-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--smoke-receipt", type=Path, required=True)
    parser.add_argument("--expected-smoke-receipt-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _require_cuda(card: dict[str, Any]) -> torch.device:
    hardware = card["hardware_plan"]
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The frozen run requires exactly one visible CUDA GPU")
    name = torch.cuda.get_device_name(0)
    if "A100" not in name or torch.cuda.get_device_properties(0).total_memory < 70 * 2**30:
        raise RuntimeError(f"The frozen run requires one A100 80GB-class GPU, got {name}")
    if int(hardware["visible_gpu_count"]) != 1:
        raise RuntimeError("Card visible-GPU contract drifted")
    return torch.device("cuda:0")


def _marker(
    root: Path,
    stage: str,
    *,
    card_hash: str,
    source_hash: str,
    epoch_seconds: float | None = None,
    scientific_payload_sha256: str | None = None,
) -> Path:
    if epoch_seconds is None and torch.cuda.is_available():
        torch.cuda.synchronize()
    path = root / "markers" / f"{stage}.json"
    payload = {
        "schema_version": 1,
        "stage": stage,
        "epoch_seconds": time.time() if epoch_seconds is None else float(epoch_seconds),
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_recorded"),
    }
    if scientific_payload_sha256 is not None:
        payload["scientific_payload_sha256"] = scientific_payload_sha256
    write_json_once(
        path,
        payload,
    )
    return path


def _configure_precision() -> dict[str, Any]:
    torch.set_float32_matmul_precision("high")
    observed = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    expected = {
        "float32_matmul_precision": "high",
        "cuda_matmul_allow_tf32": True,
        "cudnn_allow_tf32": True,
    }
    if observed != expected:
        raise RuntimeError(f"Float32 precision contract drifted: {observed} != {expected}")
    return observed


def _seeds(card: dict[str, Any], role: str) -> list[int]:
    values = [int(record["seed"]) for record in card["prospective_datasets"][role]]
    if len(values) != 3 or len(set(values)) != 3:
        raise RuntimeError(f"Expected three distinct {role} seeds")
    return values


def _integrity_failure(tier: str, error: Exception) -> dict[str, Any]:
    return {
        "tier": tier,
        "status": "strict_h400_integrity_failure",
        "error_type": type(error).__name__,
        "error": str(error),
        "finite_prefix_scored": False,
    }


def main() -> None:
    args = parse_args()
    card, card_hash = load_card(args.card, expected_sha256=args.expected_card_sha256)
    source_hash = verify_source_manifest(
        card,
        path=args.source_manifest,
        expected_sha256=args.expected_source_manifest_sha256,
    )
    verify_file(args.smoke_receipt, args.expected_smoke_receipt_sha256)
    smoke_receipt = duplicate_safe_json(args.smoke_receipt)
    if (
        smoke_receipt.get("status") != "passed_outcome_free_gpu_smoke"
        or smoke_receipt.get("card_sha256") != card_hash
        or smoke_receipt.get("source_manifest_sha256") != source_hash
        or smoke_receipt.get("scientific_outcomes_accessed") is not False
    ):
        raise RuntimeError("The required outcome-free smoke receipt is invalid")
    expected_root = Path(card["prospective_datasets"]["output_root"])
    if args.output_root != expected_root:
        raise RuntimeError(f"Output root drifted: {args.output_root} != {expected_root}")
    if args.output_root.exists():
        raise FileExistsError(f"Refusing pre-existing output root {args.output_root}")
    args.output_root.mkdir(parents=False, exist_ok=False)
    device = _require_cuda(card)
    torch.set_grad_enabled(False)
    torch.cuda.reset_peak_memory_stats(device)
    precision = _configure_precision()
    rng_proof = realized_rng_streams(card)
    validation_seeds = _seeds(card, "validation")
    test_seeds = _seeds(card, "test")
    cadence_grid = _cadence_grid(card)

    _marker(args.output_root, "job_start", card_hash=card_hash, source_hash=source_hash)
    validation_fields = generate_fields(
        card,
        seeds=validation_seeds,
        horizon=200,
        device=device,
    )
    validation_manifest_path, validation_manifest_hash = _save_datasets(
        validation_fields,
        card,
        role="validation",
        seeds=validation_seeds,
        root=args.output_root,
    )
    validation_fields = validation_fields.reshape(3, 256, 201, 512)

    specs_and_models = [
        (spec, load_checkpoint_model(spec, card, device=device))
        for spec in checkpoint_specs(card)
    ]
    if len(specs_and_models) != 20:
        raise RuntimeError("Did not preload the exact twenty-checkpoint roster")

    _marker(
        args.output_root,
        "selection_start",
        card_hash=card_hash,
        source_hash=source_hash,
    )
    validation_rows = _evaluate_cross(
        specs_and_models,
        validation_fields,
        dataset_seeds=validation_seeds,
        cadences=cadence_grid,
        horizon=200,
        batch_size=int(card["hardware_plan"]["validation_batch"]),
        max_decode_segment=int(card["hardware_plan"]["maximum_latent_segment"]),
    )
    validate_validation_rows(validation_rows, card)
    selected = select_recipe_cadences(validation_rows, card)
    candidate_scores = validation_candidate_scores(validation_rows, card)
    selection_path = args.output_root / "selection_decision.json"
    write_json_once(
        selection_path,
        {
            "schema_version": 1,
            "protocol_id": card["protocol_id"],
            "card_sha256": card_hash,
            "source_manifest_sha256": source_hash,
            "selection_endpoint": "H200 cumulative field MSE",
            "selection_scope": "one recipe-level cadence per arm",
            "selected_cadences": selected,
            "candidate_scores": candidate_scores,
            "validation_rows_sha256": canonical_digest(validation_rows),
        },
    )
    selection_hash = sha256_path(selection_path)
    _marker(
        args.output_root,
        "selection_end",
        card_hash=card_hash,
        source_hash=source_hash,
    )
    del validation_fields
    torch.cuda.empty_cache()

    test_initial = sample_initial_fields(card, seeds=test_seeds, horizon=400)
    _marker(
        args.output_root,
        "evaluation_start",
        card_hash=card_hash,
        source_hash=source_hash,
    )
    test_fields = integrate_initial_fields(
        card, test_initial, horizon=400, device=device
    )
    del test_initial
    test_fields = test_fields.reshape(3, 256, 401, 512)
    primary_test_rows = _evaluate_cross(
        specs_and_models,
        test_fields,
        dataset_seeds=test_seeds,
        cadences=cadence_grid,
        horizon=200,
        batch_size=int(card["hardware_plan"]["heldout_batch"]),
        max_decode_segment=int(card["hardware_plan"]["maximum_latent_segment"]),
    )
    validate_primary_test_rows(primary_test_rows, card)
    required_stress_cadences = [
        value
        for value in cadence_grid
        if value in {"direct", selected["dense"], selected["sparse"]}
    ]
    h400_cadences = [*cadence_grid, P200]
    stress_rows, stress_failures = evaluate_stress_cross(
        _evaluate_cross,
        specs_and_models,
        test_fields,
        dataset_seeds=test_seeds,
        cadences=h400_cadences,
        batch_size=int(card["hardware_plan"]["heldout_batch"]),
        max_decode_segment=int(card["hardware_plan"]["maximum_latent_segment"]),
    )
    required_stress_failures = [
        row for row in stress_failures
        if row["cadence"] in set(required_stress_cadences)
    ]
    required_stress_rows = [
        row for row in stress_rows
        if row["cadence"] in set(required_stress_cadences)
    ]
    grid_stress_failures = [
        row for row in stress_failures if row["cadence"] in set(cadence_grid)
    ]
    grid_stress_rows = [
        row for row in stress_rows if row["cadence"] in set(cadence_grid)
    ]
    p200_failures = [row for row in stress_failures if row["cadence"] == P200]
    p200_rows = [row for row in stress_rows if row["cadence"] == P200]
    if not required_stress_failures:
        try:
            validate_test_rows(required_stress_rows, card, selected)
            validate_stress_prefix(primary_test_rows, required_stress_rows)
        except (AssertionError, FloatingPointError, ValueError) as error:
            required_stress_failures.append(_integrity_failure("required", error))
    if not grid_stress_failures:
        try:
            validate_full_grid_h400_rows(grid_stress_rows, card)
            validate_stress_prefix(primary_test_rows, grid_stress_rows)
        except (AssertionError, FloatingPointError, ValueError) as error:
            grid_stress_failures.append(_integrity_failure("full_grid", error))
    if not p200_failures:
        h200_direct_rows = [
            row for row in primary_test_rows if row["cadence"] == "direct"
        ]
        try:
            validate_p200_h400_rows(p200_rows, h200_direct_rows, card)
        except (AssertionError, FloatingPointError, ValueError) as error:
            p200_failures.append(_integrity_failure("p200", error))
    all_stress_failures = [
        *stress_failures,
        *[row for row in required_stress_failures if row not in stress_failures],
        *[row for row in grid_stress_failures if row not in stress_failures],
        *[row for row in p200_failures if row not in stress_failures],
    ]
    truth_difficulty_rows = truth_difficulty(test_fields)
    if _configure_precision() != precision:
        raise RuntimeError("Precision contract changed during evaluation")
    torch.cuda.synchronize(device)
    evaluation_end_epoch = time.time()
    _marker(
        args.output_root,
        "evaluation_compute_end",
        card_hash=card_hash,
        source_hash=source_hash,
        epoch_seconds=evaluation_end_epoch,
    )

    test_manifest_path, test_manifest_hash = _save_datasets(
        test_fields.reshape(3, 256, 401, 16, 16, 2),
        card,
        role="test",
        seeds=test_seeds,
        root=args.output_root,
    )
    del test_fields
    scientific_path = args.output_root / "scientific_payload.json"
    scientific_payload = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "smoke_receipt_path": str(args.smoke_receipt),
        "smoke_receipt_sha256": args.expected_smoke_receipt_sha256,
        "validation_data_manifest_path": str(validation_manifest_path),
        "validation_data_manifest_sha256": validation_manifest_hash,
        "test_data_manifest_path": str(test_manifest_path),
        "test_data_manifest_sha256": test_manifest_hash,
        "selection_decision_path": str(selection_path),
        "selection_decision_sha256": selection_hash,
        "selected_cadences": selected,
        "validation_rows": validation_rows,
        "primary_test_rows": primary_test_rows,
        "stress_rows": stress_rows,
        "stress_failures": all_stress_failures,
        "grid_stress_failures": grid_stress_failures,
        "p200_failures": p200_failures,
        "required_stress_cadences": required_stress_cadences,
        "required_stress_failures": required_stress_failures,
        "truth_difficulty": truth_difficulty_rows,
    }
    write_json_once(scientific_path, scientific_payload)
    scientific_hash = sha256_path(scientific_path)
    _marker(
        args.output_root,
        "evaluation_end",
        card_hash=card_hash,
        source_hash=source_hash,
        epoch_seconds=evaluation_end_epoch,
        scientific_payload_sha256=scientific_hash,
    )

    write_runtime_lineage(
        root=args.output_root,
        card=card,
        card_hash=card_hash,
        source_hash=source_hash,
        smoke_receipt=args.smoke_receipt,
        smoke_hash=args.expected_smoke_receipt_sha256,
        selection_path=selection_path,
        selection_hash=selection_hash,
        validation_manifest_path=validation_manifest_path,
        validation_manifest_hash=validation_manifest_hash,
        test_manifest_path=test_manifest_path,
        test_manifest_hash=test_manifest_hash,
        scientific_path=scientific_path,
        scientific_hash=scientific_hash,
        specs_and_models=specs_and_models,
        rng_proof=rng_proof,
        precision=precision,
        device=device,
    )
    _marker(args.output_root, "job_end", card_hash=card_hash, source_hash=source_hash)


if __name__ == "__main__":
    main()
