"""Build and check compact Allen--Cahn new-initial-condition evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.neurips_2026.evidence.allen_cahn_forecast_replication_curves import (
    build_curve_companion,
    check_curve_companion,
    curve_output_paths,
    curve_source_hashes,
)
from experiments.neurips_2026.evidence.allen_cahn_forecast_replication_rendering import (
    render_replication_figure,
)
from experiments.neurips_2026.paths import PAPER_DATA_DIR, PAPER_EVIDENCE_DIR, REPO_ROOT


PACKET_ID = "allen_cahn_new_ic_replication"
DEFAULT_SOURCE_ROOT = Path(
    "/network/scratch/l/lia/skae/allen_cahn_forecast_replication_v1_20260720"
)
CARD = REPO_ROOT / "experiments/neurips_2026/allen_cahn_forecast_replication/prediction_card.json"
SOURCE_MANIFEST = REPO_ROOT / "experiments/neurips_2026/allen_cahn_forecast_replication/source_manifest.sha256"
SUMMARY = PAPER_DATA_DIR / f"{PACKET_ID}_summary.json"
SEED_ROWS = PAPER_DATA_DIR / f"{PACKET_ID}_seed_rows.csv"
DATASET_ROWS = PAPER_DATA_DIR / f"{PACKET_ID}_dataset_rows.csv"
EVIDENCE_MANIFEST = PAPER_DATA_DIR / f"{PACKET_ID}_evidence_manifest.json"
FIGURE_PDF = PAPER_EVIDENCE_DIR / f"fig_{PACKET_ID}.pdf"
FIGURE_PNG = PAPER_EVIDENCE_DIR / f"fig_{PACKET_ID}.png"

EXPECTED_HASHES = {
    "decision": "fde59ff99cc407270c5b9e6a8eaaa1730a0f0639d10abf968f8db2d2fce5583d",
    "telemetry": "fca5ab4bb8434a344be01bc71b510d3c8f4e2526a083fa425ad802dea48c373b",
    "receipt": "0ba035b5d48eafcfa67e22d2ab5cfd31bb642d9341c632e678a9d216b26ecac4",
    "card": "5519644cbbc8992a356045e68ff496818dceed500300432fd985febf80a555de",
    "source_manifest": "8add4eb16eea0f1e4b6d1483bf96149e092549f20977d93cb94b566502587595",
}
MODEL_SEEDS = tuple(range(64, 74))
DATASET_SEEDS = (1_775_404_171, 74_732_421, 293_789_188)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(), object_pairs_hook=_reject_duplicate_pairs)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _assert_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_path(path)
    _require(actual == expected, f"{label} SHA-256 mismatch: {actual}")


def _studentized_statistic(differences: np.ndarray) -> float:
    mean = float(differences.mean())
    standard_error = float(differences.std(ddof=1) / math.sqrt(differences.size))
    if standard_error == 0.0:
        return math.copysign(math.inf, mean) if mean else 0.0
    return mean / standard_error


def exact_sign_flip_p(differences: np.ndarray) -> float:
    observed = _studentized_statistic(differences)
    exceedances = 0
    for signs in itertools.product((-1.0, 1.0), repeat=differences.size):
        permuted = differences * np.asarray(signs, dtype=np.float64)
        exceedances += _studentized_statistic(permuted) >= observed
    return exceedances / (2**differences.size)


def _validate_source_links(
    decision: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    receipt: Mapping[str, Any],
    card: Mapping[str, Any],
) -> None:
    _require(decision.get("schema_version") == 1, "Unexpected decision schema")
    _require(decision.get("protocol_id") == "allen_cahn_global_k_new_ic_replication_v1", "Decision protocol drifted")
    _require(decision.get("status") == "strong_replication", "Decision is not the strong-replication branch")
    _require(decision.get("claim_boundary") == card.get("claim_boundary"), "Decision/card claim boundary drifted")
    _require(decision["primary"].get("strong_gate_passed") is True, "Primary strong gate did not pass")
    _require(telemetry.get("status") == "passed", "Evaluation telemetry did not pass")
    _require(all(telemetry.get("evaluation_checks", {}).values()), "A telemetry guard failed")
    _require(telemetry.get("evaluation_gates_every_retained_sample_including_zeros") is True, "Telemetry omitted zero-utilization samples")
    _require(receipt.get("status") == "authorized_for_dependent_cpu_summary", "Receipt did not authorize the summary")
    _require(receipt.get("scientific_payload_opened") is False, "GPU job opened its scientific payload")
    for payload in (decision["provenance"], telemetry, receipt):
        _require(payload.get("card_sha256") == EXPECTED_HASHES["card"], "Card linkage drifted")
        _require(payload.get("source_manifest_sha256") == EXPECTED_HASHES["source_manifest"], "Source-manifest linkage drifted")
    _require(receipt.get("telemetry_audit_sha256") == EXPECTED_HASHES["telemetry"], "Receipt/telemetry linkage drifted")
    _require(decision["provenance"].get("outcome_guard_receipt_sha256") == EXPECTED_HASHES["receipt"], "Decision/receipt linkage drifted")
    _require(receipt.get("scientific_payload_sha256") == decision["provenance"].get("scientific_payload_sha256"), "Receipt/decision payload linkage drifted")
    _require(int(receipt.get("crossed_cells", -1)) == 60, "Crossed evaluation roster is incomplete")
    roster = receipt.get("checkpoint_roster", [])
    roster_keys = {(row["arm"], int(row["seed"])) for row in roster}
    expected_roster = {(arm, seed) for arm in ("dense", "sparse") for seed in MODEL_SEEDS}
    _require(len(roster) == 20 and roster_keys == expected_roster, "Checkpoint roster is not the fixed 20-run roster")
    _require(card.get("protocol_id") == decision.get("protocol_id"), "Card protocol drifted")
    _require(card["evaluation"].get("model_seeds") == list(MODEL_SEEDS), "Model-seed roster drifted")
    _require(card["evaluation"].get("dataset_seeds") == list(DATASET_SEEDS), "Dataset-seed roster drifted")
    _require(card["evaluation"].get("primary_horizon") == 200, "Primary horizon drifted")
    rollout = card["evaluation"].get("rollout", "")
    _require(
        all(
            phrase in rollout
            for phrase in (
                "encoder exactly once",
                "exactly 200 times",
                "There is no periodic, every-step, event-triggered, terminal, or hidden reencoding",
            )
        ),
        "Rollout contract drifted",
    )
    _require(card["system_and_generator"].get("physical_horizon") == 20.0, "Physical horizon drifted")
    _require(card["original_outcome_aware_context"].get("four_cell_gate_passed") is False, "Original failed gate disclosure drifted")
    _require("previously observed 5.48%" in card.get("expected_outcome", ""), "Card-stated development cumulative effect is missing")


def load_authenticated_source(
    source_root: Path,
    card_path: Path = CARD,
    source_manifest_path: Path = SOURCE_MANIFEST,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = {
        "decision": source_root / "summary/decision.json",
        "telemetry": source_root / "telemetry_audit.json",
        "receipt": source_root / "outcome_guard_receipt.json",
        "card": card_path,
        "source_manifest": source_manifest_path,
    }
    for label, path in paths.items():
        _require(path.is_file(), f"Missing authenticated {label}: {path}")
        _assert_hash(path, EXPECTED_HASHES[label], label)
    decision = read_json(paths["decision"])
    telemetry = read_json(paths["telemetry"])
    receipt = read_json(paths["receipt"])
    card = read_json(paths["card"])
    _validate_source_links(decision, telemetry, receipt, card)
    return decision, telemetry, receipt, card


def _h200_seed_rows(decision: Mapping[str, Any]) -> list[dict[str, object]]:
    curves = decision["descriptive_curves"]
    _require(curves.get("horizons") == list(range(1, 201)), "Horizon curve is incomplete")
    by_arm = curves["paired_model_seed_curves_after_three_dataset_average"]
    rows = []
    for index, seed in enumerate(MODEL_SEEDS):
        dense_cumulative = float(by_arm["dense"]["cumulative_field_mse"][index][-1])
        sparse_cumulative = float(by_arm["sparse"]["cumulative_field_mse"][index][-1])
        dense_terminal = float(by_arm["dense"]["instantaneous_field_mse"][index][-1])
        sparse_terminal = float(by_arm["sparse"]["instantaneous_field_mse"][index][-1])
        rows.append(
            {
                "model_seed": seed,
                "dense_h200_cumulative_field_mse": dense_cumulative,
                "sparse_h200_cumulative_field_mse": sparse_cumulative,
                "h200_cumulative_sparse_over_dense": sparse_cumulative / dense_cumulative,
                "h200_cumulative_sparse_better": sparse_cumulative < dense_cumulative,
                "dense_h200_terminal_field_mse": dense_terminal,
                "sparse_h200_terminal_field_mse": sparse_terminal,
                "h200_terminal_sparse_over_dense": sparse_terminal / dense_terminal,
                "h200_terminal_sparse_better": sparse_terminal < dense_terminal,
            }
        )
    primary = decision["primary"]["paired_model_seed_values"]
    np.testing.assert_allclose([row["dense_h200_cumulative_field_mse"] for row in rows], primary["dense"], rtol=0, atol=1e-15)
    np.testing.assert_allclose([row["sparse_h200_cumulative_field_mse"] for row in rows], primary["sparse"], rtol=0, atol=1e-15)
    return rows


def _dataset_rows(decision: Mapping[str, Any]) -> list[dict[str, object]]:
    effects = {int(row["dataset_seed"]): row for row in decision["primary"]["dataset_effects"]}
    _require(set(effects) == set(DATASET_SEEDS), "Dataset-specific evidence roster drifted")
    return [
        {
            "dataset_seed": seed,
            "role": "prospective_new_initial_conditions",
            "dense_h200_cumulative_field_mse": float(effects[seed]["dense_mean"]),
            "sparse_h200_cumulative_field_mse": float(effects[seed]["sparse_mean"]),
            "relative_reduction": float(effects[seed]["relative_reduction"]),
            "sparse_better": float(effects[seed]["sparse_mean"]) < float(effects[seed]["dense_mean"]),
        }
        for seed in DATASET_SEEDS
    ]


def build_summary(
    decision: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    receipt: Mapping[str, Any],
    card: Mapping[str, Any],
    seed_rows: Sequence[Mapping[str, object]],
    dataset_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    primary = decision["primary"]
    terminal = decision["secondary"]["h200_terminal"]
    dense = np.asarray([float(row["dense_h200_cumulative_field_mse"]) for row in seed_rows])
    sparse = np.asarray([float(row["sparse_h200_cumulative_field_mse"]) for row in seed_rows])
    observed_p = exact_sign_flip_p(dense - sparse)
    _require(observed_p == float(primary["exact_sign_flip"]["one_sided_exact_p"]), "Exact sign-flip result did not reproduce")
    bootstrap = primary["bootstrap"]
    relative = 1.0 - float(sparse.mean() / dense.mean())
    checks = {
        "relative_reduction_at_least_5_percent": bool(relative >= 0.05),
        "paired_bootstrap_lower_above_zero": bool(float(bootstrap["ci95_lower"]) > 0.0),
        "one_sided_exact_sign_flip_p_at_most_0p05": bool(observed_p <= 0.05),
        "at_least_8_of_10_paired_seed_wins": bool(sum(sparse < dense) >= 8),
        "sparse_mean_lower_on_all_three_new_datasets": all(bool(row["sparse_better"]) for row in dataset_rows),
    }
    _require(all(checks.values()), "Recomputed strong-replication gate failed")
    evaluation_window = telemetry["windows"]["evaluation_validity"]
    context = card["original_outcome_aware_context"]
    return {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "status": "strong_same_checkpoint_new_initial_condition_replication",
        "protocol": {
            "system": "Allen--Cahn spatialized reaction-diffusion PDE",
            "state_dimension": int(card["system_and_generator"]["state_dim"]),
            "latent_dimension": int(card["system_and_generator"]["latent_dim"]),
            "horizon_steps": 200,
            "physical_time": 20.0,
            "rollout": "one encode followed by 200 repeated global-K steps; no re-encoding",
            "model_seeds": list(MODEL_SEEDS),
            "new_dataset_seeds": list(DATASET_SEEDS),
            "trajectories_per_dataset": 256,
            "same_fixed_checkpoints": True,
            "retraining_or_reselection": False,
        },
        "primary": {
            "endpoint": "H200 through-horizon mean field MSE over steps 1--200",
            "dense_mean": float(primary["dense_mean"]),
            "sparse_mean": float(primary["sparse_mean"]),
            "relative_reduction_of_arm_means": relative,
            "ci95_lower": float(bootstrap["ci95_lower"]),
            "ci95_upper": float(bootstrap["ci95_upper"]),
            "bootstrap_replicates": int(bootstrap["replicates"]),
            "bootstrap_unit": "paired model seed after averaging three datasets",
            "one_sided_exact_sign_flip_p": observed_p,
            "paired_model_seed_wins": int(sum(sparse < dense)),
            "dataset_effects": list(dataset_rows),
            "strong_gate_checks": checks,
        },
        "secondary": {
            "h200_terminal": {
                "status": "descriptive_only_no_test_no_interval_no_rescue",
                "dense_mean": float(terminal["dense_mean"]),
                "sparse_mean": float(terminal["sparse_mean"]),
                "relative_reduction_of_arm_means": float(terminal["relative_reduction_of_arm_means"]),
                "paired_model_seed_wins": int(terminal["paired_model_seed_wins"]),
            }
        },
        "development_context": {
            "dataset_seed": int(context["source_dataset_seed"]),
            "role": context["source_dataset_role"],
            "h200_cumulative_reduction_card_rounded": 0.0548,
            "h200_terminal_reduction": float(context["h200_terminal_reduction_percent"]) / 100.0,
            "four_cell_gate_passed": False,
            "h200_terminal_ci95_percent": context["h200_terminal_ci95_percent"],
            "comparison_policy": "show beside prospective aggregate only; never pool or call the opened result confirmatory",
        },
        "gpu_evaluation": {
            "status": telemetry["status"],
            "mean_retained_utilization_percent": float(evaluation_window["mean_retained_all_window_gpu_utilization_percent"]),
            "p10_retained_utilization_percent": float(evaluation_window["p10_retained_all_window_gpu_utilization_percent"]),
            "retained_samples": int(evaluation_window["retained_all_window_samples"]),
            "zero_utilization_samples_retained": int(evaluation_window["zero_utilization_retained_samples_descriptive"]),
            "peak_memory_fraction": float(evaluation_window["peak_memory_fraction"]),
            "no_padding": True,
        },
        "authentication": {
            "source_artifact_sha256": dict(EXPECTED_HASHES),
            "scientific_payload_sha256": receipt["scientific_payload_sha256"],
            "checkpoint_roster_sha256": receipt["checkpoint_roster_sha256"],
            "dataset_manifest_sha256": receipt["dataset_manifest_sha256"],
            "runtime_lineage_sha256": receipt["runtime_lineage_sha256"],
            "raw_telemetry_sha256": telemetry["raw_telemetry_sha256"],
            "crossed_model_by_dataset_cells": int(receipt["crossed_cells"]),
            "slurm_job_id": receipt["slurm_job_id"],
            "gpu_uuid": receipt["gpu_uuid"],
        },
        "claim_boundary": decision["claim_boundary"],
        "mandatory_disclosure": decision["mandatory_branch_disclosure"],
        "not_supported": [
            "repair of the original failed four-cell gate",
            "terminal superiority",
            "sparsity-component-only causal attribution",
            "support-alignment mediation",
            "physics or system generalization",
        ],
        "figure_caption": (
            "Same-checkpoint Allen--Cahn new-initial-condition replication. At H=200 "
            "(physical time T=20), the joint sparse recipe lowers mean direct-rollout field "
            "MSE over steps 1--200 by 5.86% across three new 256-trajectory datasets (ten paired model "
            "seeds; 95% paired-seed bootstrap CI 2.40--9.21%; exact one-sided sign-flip "
            "p=0.0088). Dataset-specific effects are descriptive and all positive. H200 "
            "terminal error is descriptive. Opened development and prospective results are "
            "shown separately and never pooled. Checkpoints were neither retrained nor reselected."
        ),
    }


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    from io import StringIO

    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        normalized = {}
        for field in fields:
            value = row[field]
            if isinstance(value, bool):
                normalized[field] = "true" if value else "false"
            elif isinstance(value, float):
                normalized[field] = format(value, ".17g")
            else:
                normalized[field] = value
        writer.writerow(normalized)
    return handle.getvalue().encode("utf-8")


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _output_paths(data_dir: Path, figure_dir: Path) -> dict[str, Path]:
    paths = {
        "summary": data_dir / SUMMARY.name,
        "seed_rows": data_dir / SEED_ROWS.name,
        "dataset_rows": data_dir / DATASET_ROWS.name,
        "figure_pdf": figure_dir / FIGURE_PDF.name,
        "figure_png": figure_dir / FIGURE_PNG.name,
        "manifest": data_dir / EVIDENCE_MANIFEST.name,
    }
    paths.update(curve_output_paths(data_dir, figure_dir))
    return paths


def _builder_source_hashes() -> dict[str, str]:
    return {
        "evidence_builder_sha256": sha256_path(Path(__file__)),
        "rendering_sha256": sha256_path(Path(render_replication_figure.__code__.co_filename)),
        **curve_source_hashes(),
    }


def build(source_root: Path, data_dir: Path, figure_dir: Path) -> dict[str, Path]:
    paths = _output_paths(data_dir, figure_dir)
    existing = [str(path) for path in paths.values() if path.exists()]
    _require(not existing, f"Refusing to overwrite evidence artifacts: {existing}")
    decision, telemetry, receipt, card = load_authenticated_source(source_root)
    seed_rows = _h200_seed_rows(decision)
    dataset_rows = _dataset_rows(decision)
    summary = build_summary(decision, telemetry, receipt, card, seed_rows, dataset_rows)
    validate_compact(summary, seed_rows, dataset_rows)
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_bytes(_json_bytes(summary))
    seed_fields = list(seed_rows[0])
    dataset_fields = list(dataset_rows[0])
    paths["seed_rows"].write_bytes(_csv_bytes(seed_rows, seed_fields))
    paths["dataset_rows"].write_bytes(_csv_bytes(dataset_rows, dataset_fields))
    render_replication_figure(summary, seed_rows, dataset_rows, paths["figure_pdf"], paths["figure_png"])
    build_curve_companion(source_root, decision, receipt, card, paths)
    manifest = {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "source_artifacts": dict(EXPECTED_HASHES),
        "builder_sources": _builder_source_hashes(),
        "outputs": {
            key: {"path": path.name, "sha256": sha256_path(path)}
            for key, path in paths.items()
            if key != "manifest"
        },
        "reproduction": "uv run skae-paper build allen-cahn-new-ic-replication --check",
    }
    paths["manifest"].write_bytes(_json_bytes(manifest))
    check_packet(data_dir, figure_dir)
    return paths


def validate_compact(
    summary: Mapping[str, Any],
    seed_rows: Sequence[Mapping[str, object]],
    dataset_rows: Sequence[Mapping[str, object]],
) -> None:
    _require(summary.get("packet_id") == PACKET_ID, "Compact packet ID drifted")
    _require(summary.get("status") == "strong_same_checkpoint_new_initial_condition_replication", "Compact status drifted")
    _require(summary["protocol"].get("physical_time") == 20.0, "Physical-time label drifted")
    _require(summary["protocol"].get("retraining_or_reselection") is False, "Retraining disclosure drifted")
    _require(len(seed_rows) == 10, "Compact seed roster is incomplete")
    seeds = [int(row["model_seed"]) for row in seed_rows]
    _require(seeds == list(MODEL_SEEDS), "Compact seed order drifted")
    _require(len(dataset_rows) == 3, "Compact dataset roster is incomplete")
    datasets = [int(row["dataset_seed"]) for row in dataset_rows]
    _require(datasets == list(DATASET_SEEDS), "Compact dataset order drifted")
    dense = np.asarray([float(row["dense_h200_cumulative_field_mse"]) for row in seed_rows])
    sparse = np.asarray([float(row["sparse_h200_cumulative_field_mse"]) for row in seed_rows])
    primary = summary["primary"]
    np.testing.assert_allclose(dense.mean(), float(primary["dense_mean"]), rtol=0, atol=1e-15)
    np.testing.assert_allclose(sparse.mean(), float(primary["sparse_mean"]), rtol=0, atol=1e-15)
    np.testing.assert_allclose(1.0 - sparse.mean() / dense.mean(), float(primary["relative_reduction_of_arm_means"]), rtol=0, atol=1e-15)
    _require(sum(sparse < dense) == int(primary["paired_model_seed_wins"]) == 8, "Primary paired wins drifted")
    _require(all(float(row["relative_reduction"]) > 0.0 for row in dataset_rows), "A prospective dataset effect is not positive")
    terminal = summary["secondary"]["h200_terminal"]
    _require(terminal.get("status") == "descriptive_only_no_test_no_interval_no_rescue", "Terminal claim boundary drifted")
    _require(int(terminal["paired_model_seed_wins"]) == 7, "Terminal paired wins drifted")
    _require(summary["development_context"].get("comparison_policy").startswith("show beside"), "Development no-pooling guard drifted")
    _require(summary["development_context"].get("four_cell_gate_passed") is False, "Original gate disclosure drifted")
    _require("support-alignment mediation" in summary.get("not_supported", []), "Mechanism caveat is missing")
    _require(summary["authentication"].get("source_artifact_sha256") == EXPECTED_HASHES, "Compact source authentication drifted")


def check_packet(data_dir: Path = PAPER_DATA_DIR, figure_dir: Path = PAPER_EVIDENCE_DIR) -> None:
    paths = _output_paths(data_dir, figure_dir)
    manifest = read_json(paths["manifest"])
    _require(manifest.get("packet_id") == PACKET_ID, "Evidence manifest ID drifted")
    for key, record in manifest.get("outputs", {}).items():
        path = paths[key]
        _require(path.name == record["path"], f"Manifest path drifted: {key}")
        _assert_hash(path, record["sha256"], f"compact output {key}")
    summary = read_json(paths["summary"])
    seed_rows = _read_csv(paths["seed_rows"])
    dataset_rows = _read_csv(paths["dataset_rows"])
    validate_compact(summary, seed_rows, dataset_rows)
    check_curve_companion(paths)
    _require(manifest.get("source_artifacts") == EXPECTED_HASHES, "Manifest source hashes drifted")
    _require(manifest.get("builder_sources") == _builder_source_hashes(), "Builder source drifted")
    with TemporaryDirectory() as temporary:
        temp = Path(temporary)
        pdf = temp / FIGURE_PDF.name
        png = temp / FIGURE_PNG.name
        render_replication_figure(summary, seed_rows, dataset_rows, pdf, png)
        _require(pdf.read_bytes() == paths["figure_pdf"].read_bytes(), "PDF rendering is not deterministic")
        _require(png.read_bytes() == paths["figure_png"].read_bytes(), "PNG rendering is not deterministic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-data-dir", type=Path, default=PAPER_DATA_DIR)
    parser.add_argument("--output-figure-dir", type=Path, default=PAPER_EVIDENCE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        check_packet(args.output_data_dir, args.output_figure_dir)
        result = {"command": "check", "status": "ok"}
    else:
        paths = build(args.source_root, args.output_data_dir, args.output_figure_dir)
        result = {"command": "build", "status": "ok", "outputs": {key: str(path) for key, path in paths.items()}}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
