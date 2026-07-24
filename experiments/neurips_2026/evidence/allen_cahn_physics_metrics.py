"""Build and fail-closed check compact Allen--Cahn physics evidence."""

from __future__ import annotations

import argparse
import csv
import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.neurips_2026.allen_cahn_physics_metrics.core import (
    HORIZON,
    METRIC_NAMES,
    METRIC_SPECS,
    validate_score_record,
)
from experiments.neurips_2026.allen_cahn_physics_metrics.io import (
    CARD_PATH,
    MANIFEST_PATH,
    duplicate_safe_json,
    load_card,
    sha256_path,
    verify_file,
    verify_source_manifest,
)
from experiments.neurips_2026.allen_cahn_physics_metrics.statistics import summarize_rows
from experiments.neurips_2026.evidence.allen_cahn_physics_metrics_rendering import (
    SHORT_LABELS,
    physics_table_bytes,
    render_physics_figure,
)
from experiments.neurips_2026.paths import (
    PAPER_DATA_DIR,
    PAPER_EVIDENCE_DIR,
    PAPER_TABLE_DIR,
)

PACKET_ID = "allen_cahn_physics_metrics"
DEFAULT_SOURCE_ROOT = Path(
    "/network/scratch/l/lia/skae/allen_cahn_physics_metrics_v1_20260721"
)
DEFAULT_DATA_DIR = PAPER_DATA_DIR / PACKET_ID
DEFAULT_FIGURE_DIR = PAPER_EVIDENCE_DIR
DEFAULT_TABLE_DIR = PAPER_TABLE_DIR
EXPECTED_SOURCE_HASHES = {
    "card": "d4748ec37aaf6d10de0c02eb988c5278840eacbbcf649007e1675a0c788bcb88",
    "source_manifest": "2c0af1fac15f182b3dd6538d909b77b68e123566a46f488a93b89068c31c3221",
    "outcome_receipt": "b0cf579a38ed472e6ae40ec225141e3ee86e76b2393d85b6c14763307798bc8c",
    "scientific_payload": "3ab7d3ee5fc1b155bad66571d353a867eb48896306b48378f5076e2d7af45e43",
    "snapshot": "955c4ae968df66908ede5e7c9bdd7c94be2016736b766a70adf17eac7909b079",
    "telemetry_audit": "0a48e9c153bbc47e981909f0bd057e6fc32b36be58249c8df43ec30e6d843a84",
    "runtime_lineage": "11cfa65eee98ccb70d0cc5d0dfa209851c2454fac10a0f1f091278ff2cb1b018",
    "raw_gpu_telemetry": "1df11d171374b213d357ff626fc0526565275ddda222ba77832d3bf5e7ad2646",
    "scratch_summary": "de0e8b84fab79d2198ffd2807924957d4a5d1986520b2eca67034c66f052c128",
}

# Independently regenerated roots make ``--check`` portable without scratch.
EXPECTED_RELEASE_HASHES = {
    "summary": "e5aa8c07dc6b4f23311176519c3880446435f72ae4eebe1d09be8a1a24870dfb",
    "seed_rows": "ce2616b85b1fa282aa1cfc10d1eeff098a3da27b213fea2c05a35f8a7960f80f",
    "curve_rows": "da6f0fca2df76aee11f14789f6a90df811071b443fbfd13c2132b35241cd34dd",
    "tie_rows": "2c928c7424e7e490b4410558265015bb0b5fdf1cc3bbda320f1f3678d945b8b2",
    "provenance": "002bcb87c0c7362562d1551d19f2feb73d65ca0e50b4a14248aadeeee65a139d",
    "manifest": "329948b8334ecc19cb988ff2347c233f76fa2d1d4e8149c81a23e3d6d02b352e",
    "table": "8508130bbc4cb81121297def4829d9f740dcd6d2ea3608d88f9e68de7daf21db",
    "figure_pdf": "3dad4d126b7fbadb0bef8066804a2a55a18889f5243d5cd5a94c93f5cd0cae3a",
    "figure_png": "e5ee5e82ca9ddb7304e3db2f2b747fe500093443c43c1cea1305e150abde8c1a",
}
FILE_NAMES = {
    "summary": "summary.json",
    "seed_rows": "h200_paired_seed_rows.csv",
    "curve_rows": "all_metric_curve_rows.csv",
    "tie_rows": "modal_tie_curve_rows.csv",
    "provenance": "provenance.json",
    "manifest": "evidence_manifest.json",
    "table": "table_allen_cahn_physics_metrics.tex",
    "figure_pdf": "fig_allen_cahn_physics_metrics.pdf",
    "figure_png": "fig_allen_cahn_physics_metrics.png",
}

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        normalized: dict[str, object] = {}
        for field in fields:
            value = row[field]
            if isinstance(value, bool):
                normalized[field] = "true" if value else "false"
            elif isinstance(value, float):
                normalized[field] = format(value, ".17g")
            else:
                normalized[field] = value
        writer.writerow(normalized)
    return handle.getvalue().encode()


def _paths(data_dir: Path, figure_dir: Path, table_dir: Path) -> dict[str, Path]:
    return {
        "summary": data_dir / FILE_NAMES["summary"],
        "seed_rows": data_dir / FILE_NAMES["seed_rows"],
        "curve_rows": data_dir / FILE_NAMES["curve_rows"],
        "tie_rows": data_dir / FILE_NAMES["tie_rows"],
        "provenance": data_dir / FILE_NAMES["provenance"],
        "manifest": data_dir / FILE_NAMES["manifest"],
        "table": table_dir / FILE_NAMES["table"],
        "figure_pdf": figure_dir / FILE_NAMES["figure_pdf"],
        "figure_png": figure_dir / FILE_NAMES["figure_png"],
    }


def _load_authenticated_source(source_root: Path) -> tuple[dict[str, Any], ...]:
    card, card_hash = load_card(CARD_PATH, expected_sha256=EXPECTED_SOURCE_HASHES["card"])
    source_hash = verify_source_manifest(
        MANIFEST_PATH, expected_sha256=EXPECTED_SOURCE_HASHES["source_manifest"]
    )
    source_paths = {
        "outcome_receipt": source_root / "outcome_guard_receipt.json",
        "scientific_payload": source_root / "scientific_physics_curves.json",
        "snapshot": source_root / "visualization_snapshots.pt",
        "telemetry_audit": source_root / "telemetry_audit.json",
        "runtime_lineage": source_root / "runtime_lineage.json",
        "raw_gpu_telemetry": source_root / "raw_gpu_telemetry.csv",
        "scratch_summary": source_root / "summary/physics_metrics_summary.json",
    }
    for name, path in source_paths.items():
        verify_file(path, EXPECTED_SOURCE_HASHES[name])
    receipt = duplicate_safe_json(source_paths["outcome_receipt"])
    telemetry = duplicate_safe_json(source_paths["telemetry_audit"])
    runtime = duplicate_safe_json(source_paths["runtime_lineage"])
    payload = duplicate_safe_json(source_paths["scientific_payload"])
    scratch_summary = duplicate_safe_json(source_paths["scratch_summary"])
    _require(receipt.get("status") == "authorized_for_dependent_cpu_summary", "Receipt is not authorized")
    _require(receipt.get("scientific_payload_opened") is False, "GPU job opened its scientific payload")
    _require(telemetry.get("status") == "passed", "GPU telemetry audit failed")
    _require(all(telemetry.get("evaluation_checks", {}).values()), "A GPU telemetry guard failed")
    _require(telemetry.get("every_retained_sample_including_zero_used") is True, "Telemetry filtered samples")
    for item in (receipt, telemetry, runtime, payload, scratch_summary):
        _require(item.get("card_sha256") == card_hash, "Card lineage drifted")
        _require(item.get("source_manifest_sha256") == source_hash, "Source lineage drifted")
    _require(receipt.get("scientific_payload_sha256") == EXPECTED_SOURCE_HASHES["scientific_payload"], "Receipt/payload linkage drifted")
    _require(receipt.get("snapshot_sha256") == EXPECTED_SOURCE_HASHES["snapshot"], "Receipt/snapshot linkage drifted")
    _require(receipt.get("telemetry_audit_sha256") == EXPECTED_SOURCE_HASHES["telemetry_audit"], "Receipt/telemetry linkage drifted")
    _require(telemetry.get("raw_telemetry_sha256") == EXPECTED_SOURCE_HASHES["raw_gpu_telemetry"], "Raw telemetry linkage drifted")
    rows = payload.get("rows")
    _require(isinstance(rows, list) and len(rows) == 63, "Scientific payload does not contain 63 cells")
    for row in rows:
        validate_score_record(row)
    inference = card["inference_and_reporting"]
    recomputed = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "outcome_receipt_sha256": EXPECTED_SOURCE_HASHES["outcome_receipt"],
        "evidence_grade": "outcome_aware_same_checkpoint_secondary",
        "original_field_mse_inference_reclassified": False,
        "all_frozen_metrics_and_horizons_reported": True,
        **summarize_rows(
            rows,
            bootstrap_replicates=int(inference["bootstrap_replicates"]),
            bootstrap_seeds_by_metric=inference["bootstrap_seeds_by_metric"],
        ),
    }
    _require(_json_bytes(recomputed) == source_paths["scratch_summary"].read_bytes(), "Scratch summary does not reproduce from the authenticated payload")
    return card, receipt, telemetry, runtime, payload, scratch_summary


def _compact_payloads(
    card: Mapping[str, Any],
    receipt: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    runtime: Mapping[str, Any],
    scratch_summary: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, Any]]:
    metric_summaries: list[dict[str, Any]] = []
    seed_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    render_metrics: list[dict[str, Any]] = []
    times = (0.1 * np.arange(1, HORIZON + 1)).tolist()
    for metric_index, spec in enumerate(METRIC_SPECS):
        source = scratch_summary["metrics"][spec.name]
        dense_values = source["h200_cumulative_paired_seed_dense"]
        sparse_values = source["h200_cumulative_paired_seed_sparse"]
        improvements = source["h200_cumulative_paired_seed_improvement"]
        endpoint = source["mandatory_endpoints"]["h200_cumulative"]
        metric = {
            "name": spec.name,
            "display_name": SHORT_LABELS[spec.name],
            "family": spec.family,
            "direction": spec.direction,
            "h200_cumulative": endpoint,
            "arm_mean_effect": source["arm_mean_effect"],
            "paired_seed_oriented_absolute_effect": improvements,
            "h200_cumulative_seed_wins": source["h200_cumulative_seed_wins"],
            "paired_bootstrap": source["paired_bootstrap"],
            "raw_one_sided_p": source["exact_secondary_test"]["one_sided_exact_p"],
            "holm_p": source["exact_secondary_test"]["holm_adjusted_p_across_seven_metrics"],
            "holm_significant_0p05": source["exact_secondary_test"]["holm_adjusted_p_across_seven_metrics"] <= 0.05,
            "mandatory_endpoints": source["mandatory_endpoints"],
        }
        metric_summaries.append(metric)
        for seed_offset, model_seed in enumerate(range(64, 74)):
            seed_rows.append(
                {
                    "metric_order": metric_index,
                    "metric_name": spec.name,
                    "family": spec.family,
                    "direction": spec.direction,
                    "model_seed": model_seed,
                    "dense_h200_cumulative": float(dense_values[seed_offset]),
                    "sparse_h200_cumulative": float(sparse_values[seed_offset]),
                    "oriented_absolute_improvement": float(improvements[seed_offset]),
                    "sparse_better": float(improvements[seed_offset]) > 0.0,
                }
            )
        curves = source["full_curves"]
        for horizon_index, physical_time in enumerate(times):
            curve_rows.append(
                {
                    "metric_order": metric_index,
                    "metric_name": spec.name,
                    "family": spec.family,
                    "direction": spec.direction,
                    "horizon_step": horizon_index + 1,
                    "physical_time": float(physical_time),
                    "dense_instantaneous": float(curves["dense_instantaneous_mean"][horizon_index]),
                    "sparse_instantaneous": float(curves["sparse_instantaneous_mean"][horizon_index]),
                    "persistence_instantaneous": float(curves["persistence_instantaneous"][horizon_index]),
                    "dense_cumulative": float(curves["dense_cumulative_mean"][horizon_index]),
                    "sparse_cumulative": float(curves["sparse_cumulative_mean"][horizon_index]),
                    "persistence_cumulative": float(curves["persistence_cumulative"][horizon_index]),
                }
            )
        render_metrics.append(
            {
                **metric,
                "physical_time": times,
                "cumulative_curves": {
                    "dense": curves["dense_cumulative_mean"],
                    "sparse": curves["sparse_cumulative_mean"],
                    "persistence": curves["persistence_cumulative"],
                },
            }
        )
    ties = scratch_summary["modal_tie_diagnostics"]
    tie_rows = [
        {
            "horizon_step": index + 1,
            "physical_time": float(times[index]),
            "truth_modal_tie_rate": float(ties["truth_modal_tie_rate"][index]),
            "dense_modal_tie_rate": float(ties["dense_modal_tie_rate"][index]),
            "sparse_modal_tie_rate": float(ties["sparse_modal_tie_rate"][index]),
            "persistence_modal_tie_rate": float(ties["persistence_modal_tie_rate"][index]),
        }
        for index in range(HORIZON)
    ]
    window = telemetry["evaluation_window"]
    summary = {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "status": "broad_secondary_concordance",
        "evidence_grade": "outcome_aware_same_checkpoint_secondary",
        "protocol": {
            "system": "synthetic spatialized Allen--Cahn reaction--diffusion PDE",
            "state_dimension": 512,
            "latent_dimension": 2048,
            "horizon_steps": 200,
            "physical_horizon": 20.0,
            "time_step": 0.1,
            "rollout": "one encode followed by 200 repeated global-K steps; no reencoding",
            "model_seeds": list(range(64, 74)),
            "dataset_seeds": card["matching_contract"]["dataset_seeds"],
            "trajectories_per_dataset": 256,
            "arms": ["sparse", "dense", "persistence"],
            "inference_unit": card["inference_and_reporting"]["inference_unit"],
        },
        "metrics": metric_summaries,
        "multiplicity": scratch_summary["multiplicity"],
        "secondary_pattern": scratch_summary["secondary_pattern"],
        "gpu_evaluation": {
            "status": telemetry["status"],
            "gpu_name": runtime["environment"]["gpu_name"],
            "mean_retained_utilization_percent": window["mean_retained_all_window_gpu_utilization_percent"],
            "p10_retained_utilization_percent": window["p10_retained_all_window_gpu_utilization_percent"],
            "retained_samples": window["retained_all_window_samples"],
            "peak_memory_fraction": window["peak_memory_fraction"],
            "no_padding": True,
        },
        "interpretation": (
            "Sparse is directionally better in all seven H200 cumulative physical metrics and "
            "covers all three frozen families. Five metrics survive Holm correction; modal-well "
            "accuracy and potential-energy absolute error remain descriptive."
        ),
        "claim_boundary": (
            "This outcome-aware same-checkpoint secondary analysis translates the existing field-MSE "
            "result within one fixed synthetic PDE condition. It does not reclassify the field-MSE "
            "endpoint, isolate sparsity causally, establish terminal superiority, or show physics/system generalization."
        ),
        "figure_caption": (
            "Outcome-aware same-checkpoint Allen--Cahn physics analysis showing all seven frozen metrics "
            "at the H200 cumulative endpoint. Curves trace those metrics across physical time for sparse, exact-dense, and "
            "persistence forecasts. The top panel shows ten paired model-seed effects and frozen 95% "
            "paired-seed bootstrap intervals. Filled markers survive Holm correction across all seven "
            "H200 metrics; hollow markers are descriptive. Three fixed datasets are averaged within seed."
        ),
    }
    provenance = {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "generated_by": "experiments.neurips_2026.evidence.allen_cahn_physics_metrics",
        "display_scope": "all seven frozen metrics at the H200 cumulative endpoint",
        "external_source_root": str(DEFAULT_SOURCE_ROOT),
        "source_artifacts": {
            "prediction_card": {"path": str(CARD_PATH), "sha256": EXPECTED_SOURCE_HASHES["card"]},
            "source_manifest": {"path": str(MANIFEST_PATH), "sha256": EXPECTED_SOURCE_HASHES["source_manifest"]},
            "outcome_guard_receipt": {"path": str(DEFAULT_SOURCE_ROOT / "outcome_guard_receipt.json"), "sha256": EXPECTED_SOURCE_HASHES["outcome_receipt"]},
            "scientific_payload": {"path": str(DEFAULT_SOURCE_ROOT / "scientific_physics_curves.json"), "sha256": EXPECTED_SOURCE_HASHES["scientific_payload"]},
            "visualization_snapshots": {"path": str(DEFAULT_SOURCE_ROOT / "visualization_snapshots.pt"), "sha256": EXPECTED_SOURCE_HASHES["snapshot"]},
            "telemetry_audit": {"path": str(DEFAULT_SOURCE_ROOT / "telemetry_audit.json"), "sha256": EXPECTED_SOURCE_HASHES["telemetry_audit"]},
            "runtime_lineage": {"path": str(DEFAULT_SOURCE_ROOT / "runtime_lineage.json"), "sha256": EXPECTED_SOURCE_HASHES["runtime_lineage"]},
            "raw_gpu_telemetry": {"path": str(DEFAULT_SOURCE_ROOT / "raw_gpu_telemetry.csv"), "sha256": EXPECTED_SOURCE_HASHES["raw_gpu_telemetry"]},
            "scratch_summary": {"path": str(DEFAULT_SOURCE_ROOT / "summary/physics_metrics_summary.json"), "sha256": EXPECTED_SOURCE_HASHES["scratch_summary"]},
        },
        "authentication": {
            "raw_external_artifacts_modified": False,
            "summary_recomputed_exactly_from_authenticated_payload": True,
            "card_sha256": receipt["card_sha256"],
            "source_manifest_sha256": receipt["source_manifest_sha256"],
            "checkpoint_roster_sha256": receipt["checkpoint_roster_sha256"],
            "scientific_row_count": receipt["row_count"],
            "slurm_job_id": receipt["slurm_job_id"],
            "gpu_uuid": receipt["gpu_uuid"],
        },
        "reproduction": {
            "portable_check": "uv run skae-paper build allen-cahn-physics-metrics --check",
            "source_regeneration_check": "uv run skae-paper build allen-cahn-physics-metrics --check-sources",
        },
    }
    return summary, seed_rows, curve_rows, tie_rows, {"provenance": provenance, "render_metrics": render_metrics}


def write_packet(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    data_dir: Path = DEFAULT_DATA_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    table_dir: Path = DEFAULT_TABLE_DIR,
) -> dict[str, str]:
    paths = _paths(data_dir, figure_dir, table_dir)
    existing = [str(path) for path in paths.values() if path.exists()]
    _require(not existing, f"Refusing to overwrite evidence artifacts: {existing}")
    card, receipt, telemetry, runtime, _payload, scratch_summary = _load_authenticated_source(source_root)
    summary, seed_rows, curve_rows, tie_rows, extras = _compact_payloads(
        card, receipt, telemetry, runtime, scratch_summary
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_bytes(_json_bytes(summary))
    paths["seed_rows"].write_bytes(_csv_bytes(seed_rows, list(seed_rows[0])))
    paths["curve_rows"].write_bytes(_csv_bytes(curve_rows, list(curve_rows[0])))
    paths["tie_rows"].write_bytes(_csv_bytes(tie_rows, list(tie_rows[0])))
    paths["provenance"].write_bytes(_json_bytes(extras["provenance"]))
    paths["table"].write_bytes(physics_table_bytes(summary))
    render_physics_figure(extras["render_metrics"], paths["figure_pdf"], paths["figure_png"])
    output_hashes = {
        name: sha256_path(path) for name, path in paths.items() if name != "manifest"
    }
    manifest = {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "status": "broad_secondary_concordance",
        "evidence_grade": "outcome_aware_same_checkpoint_secondary",
        "metric_roster": list(METRIC_NAMES),
        "all_seven_metrics_rendered": True,
        "holm_significant_metrics": [
            metric["name"] for metric in summary["metrics"] if metric["holm_significant_0p05"]
        ],
        "descriptive_metrics": [
            metric["name"] for metric in summary["metrics"] if not metric["holm_significant_0p05"]
        ],
        "outputs": output_hashes,
        "source_artifact_hashes": EXPECTED_SOURCE_HASHES,
        "portable_check": "uv run skae-paper build allen-cahn-physics-metrics --check",
        "source_check": "uv run skae-paper build allen-cahn-physics-metrics --check-sources",
    }
    paths["manifest"].write_bytes(_json_bytes(manifest))
    return {name: sha256_path(path) for name, path in paths.items()}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_packet(
    data_dir: Path = DEFAULT_DATA_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    table_dir: Path = DEFAULT_TABLE_DIR,
) -> dict[str, Any]:
    paths = _paths(data_dir, figure_dir, table_dir)
    _require(set(EXPECTED_RELEASE_HASHES) == set(paths), "Released hash roster is incomplete")
    for name, path in paths.items():
        _require(path.is_file(), f"Missing physics evidence artifact: {path}")
        _require(sha256_path(path) == EXPECTED_RELEASE_HASHES[name], f"Released artifact hash drifted: {name}")
    summary = duplicate_safe_json(paths["summary"])
    metrics = summary.get("metrics", [])
    _require([metric["name"] for metric in metrics] == list(METRIC_NAMES), "Metric roster/order drifted")
    _require(summary.get("evidence_grade") == "outcome_aware_same_checkpoint_secondary", "Evidence grade drifted")
    _require(summary.get("status") == "broad_secondary_concordance", "Pattern classification drifted")
    _require(sum(metric["holm_significant_0p05"] for metric in metrics) == 5, "Holm result count drifted")
    _require(all(float(metric["paired_bootstrap"]["oriented_absolute_improvement"]) > 0 for metric in metrics), "A frozen directional result reversed")
    seed_rows = _read_csv(paths["seed_rows"])
    curve_rows = _read_csv(paths["curve_rows"])
    tie_rows = _read_csv(paths["tie_rows"])
    _require(len(seed_rows) == 70, "Paired-seed row count drifted")
    _require(len(curve_rows) == 7 * HORIZON, "All-metric curve row count drifted")
    _require(len(tie_rows) == HORIZON, "Modal-tie row count drifted")
    for metric_index, name in enumerate(METRIC_NAMES):
        selected = [row for row in curve_rows if row["metric_name"] == name]
        _require([int(row["horizon_step"]) for row in selected] == list(range(1, HORIZON + 1)), f"Incomplete curve for {name}")
        _require(np.allclose([float(row["physical_time"]) for row in selected], 0.1 * np.arange(1, HORIZON + 1)), f"Physical-time drift for {name}")
        seeds = [row for row in seed_rows if row["metric_name"] == name]
        _require([int(row["model_seed"]) for row in seeds] == list(range(64, 74)), f"Seed roster drifted for {name}")
        _require(all(int(row["metric_order"]) == metric_index for row in selected + seeds), f"Metric order drifted for {name}")
    manifest = duplicate_safe_json(paths["manifest"])
    _require(manifest.get("metric_roster") == list(METRIC_NAMES), "Manifest metric roster drifted")
    _require(manifest.get("all_seven_metrics_rendered") is True, "Manifest permits metric omission")
    _require(manifest.get("outputs") == {name: EXPECTED_RELEASE_HASHES[name] for name in paths if name != "manifest"}, "Manifest/output roots drifted")
    provenance = duplicate_safe_json(paths["provenance"])
    _require(provenance.get("authentication", {}).get("raw_external_artifacts_modified") is False, "Raw-artifact mutation disclosure drifted")
    _require(provenance.get("authentication", {}).get("summary_recomputed_exactly_from_authenticated_payload") is True, "Summary reproduction guard drifted")
    table = paths["table"].read_text(encoding="utf-8")
    _require("Outcome-aware, same-checkpoint secondary" in table, "Table evidence grade is missing")
    _require(all(SHORT_LABELS[name] in table for name in METRIC_NAMES), "Table omits a frozen metric")
    return {
        "status": summary["status"],
        "metric_count": len(metrics),
        "holm_significant_count": 5,
        "paired_seed_rows": len(seed_rows),
        "curve_rows": len(curve_rows),
    }


def validate_sources(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    data_dir: Path = DEFAULT_DATA_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    table_dir: Path = DEFAULT_TABLE_DIR,
) -> dict[str, Any]:
    validate_packet(data_dir, figure_dir, table_dir)
    released = _paths(data_dir, figure_dir, table_dir)
    with TemporaryDirectory(prefix="allen_cahn_physics_packet_") as temporary:
        root = Path(temporary)
        regenerated = _paths(root / "data", root / "figures", root / "tables")
        write_packet(source_root, root / "data", root / "figures", root / "tables")
        for name in released:
            _require(released[name].read_bytes() == regenerated[name].read_bytes(), f"Source regeneration drifted: {name}")
    return {"source_artifacts": len(EXPECTED_SOURCE_HASHES), "deterministic_outputs": len(released)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-sources", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require(sum((args.write, args.check, args.check_sources)) == 1, "Choose exactly one action")
    if args.write:
        result: dict[str, Any] = {"written_hashes": write_packet(args.source_root, args.data_dir, args.figure_dir, args.table_dir)}
    elif args.check_sources:
        result = validate_sources(args.source_root, args.data_dir, args.figure_dir, args.table_dir)
    else:
        result = validate_packet(args.data_dir, args.figure_dir, args.table_dir)
    print(json.dumps({"status": "passed", **result}, sort_keys=True))

if __name__ == "__main__":
    main()
