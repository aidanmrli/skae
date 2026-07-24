"""Verify the frozen one-seed Allen--Cahn half-global/half-local negative."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from experiments.neurips_2026.paths import PAPER_DATA_DIR


PACKET_ID = "allen_cahn_half_global_half_local_negative"
DEFAULT_PACKET_DIR = PAPER_DATA_DIR / PACKET_ID
FILES = (
    "protocol.json",
    "result.json",
    "provenance.json",
    "evidence_manifest.json",
)
EXPECTED_HASHES = {
    "protocol.json": "ebcd3120ea035dd7bda9c320453f9debc05df17010e957ff9e86ff3bb30c11eb",
    "result.json": "e8ed176a00a2bbbbb6d7593d6f3f46145a1541557caa98cf62264290b74b865a",
    "provenance.json": "92eeb72a81ee596dd7dae28cde5b06750ab736e3bd7fbfb443fc8558f5bbd6d0",
    "evidence_manifest.json": "77ba4a62abcdb1caacb1191f7c2043bba4617d505b411a976036400ef21318f8",
}
SOURCE_GIT_COMMIT = "4d384ac976b520f49e53be1ded9c6909aa5335df"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"Expected a JSON object: {path}")
    return payload


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    _require(
        protocol.get("status") == "frozen_before_dataset_generation_or_training",
        "Half-local protocol is not the frozen preregistration",
    )
    _require(protocol.get("model_seed") == 61, "Half-local seed drifted")
    model = protocol["model"]
    _require(
        model["state_dimension"] == 512
        and model["latent_dimension"] == 2048
        and model["latent_dimension"] >= 4 * model["state_dimension"],
        "Half-local spatial or overcomplete dimensions drifted",
    )
    stage = protocol["stage_contract"]
    expected_stage = {
        "autoencoder_pretraining_updates": 2000,
        "stage1_joint_updates": 1750,
        "stage2_updates": 1750,
        "sequence_length": 80,
        "batch_size": 8,
        "transitions_per_update": 640,
        "stage2_transitions": 1_120_000,
        "spatial_augmentation": True,
        "matched_stage2_generator_state": True,
        "freeze_encoder_decoder_global_k_for_local": True,
    }
    _require(stage == expected_stage, "Half-local stage contract drifted")
    _require(
        protocol["checkpoint_partition"] == "validation_even"
        and protocol["report_partition"] == "validation_odd",
        "Half-local selection/report split drifted",
    )
    _require(
        protocol["eval_horizons"] == [80, 120, 160, 200],
        "Half-local horizon roster drifted",
    )
    _require(
        set(protocol["routed_arms"])
        == {
            "sparse_support_full_delta",
            "sparse_cosine_kmeans_full_delta",
            "dense_cosine_kmeans_full_delta",
        }
        and protocol["local_learning_rates"] == [0.00001, 0.00003],
        "Half-local arm or learning-rate roster drifted",
    )
    _require(
        protocol["routing"]["label_policy"]
        == "Training fields and model latents only; no labels or basin count.",
        "Half-local label-free routing contract drifted",
    )


def _validate_result(result: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, float]:
    _require(
        result.get("status") == "failed"
        and result.get("decision") == "do_not_promote",
        "Half-local negative decision drifted",
    )
    _require(
        result["selected_recipes"]
        == {
            "dense_kmeans": "lr1e5",
            "sparse_kmeans": "lr1e5",
            "sparse_support": "lr1e5",
        },
        "Half-local selected recipe drifted",
    )
    comparisons = result["comparisons"]
    for reference in ("sparse_global", "dense_global", "dense_routed"):
        cells = comparisons[reference]["cells"]
        _require(
            not comparisons[reference]["all_primary_cells_lower"]
            and all(float(cell["relative_reduction"]) < 0.0 for cell in cells.values()),
            f"Half-local negative comparison drifted for {reference}",
        )
    _require(
        all(
            float(cell["relative_reduction"]) == 0.0
            for cell in comparisons["sparse_kmeans"]["cells"].values()
        ),
        "Support and cosine-routed zero-update forecasts no longer match",
    )

    h200_mean = comparisons["sparse_global"]["cells"]["h200_field_mse"]
    h200_terminal = comparisons["sparse_global"]["cells"]["h200_final_field_mse"]
    mean_local = float(h200_mean["sparse_support"])
    mean_global = float(h200_mean["reference"])
    terminal_local = float(h200_terminal["sparse_support"])
    terminal_global = float(h200_terminal["reference"])
    mean_ratio = mean_local / mean_global
    terminal_ratio = terminal_local / terminal_global
    _require(
        math.isclose(mean_ratio, 1.2178167362420764, rel_tol=0.0, abs_tol=1e-15)
        and math.isclose(
            terminal_ratio, 1.1849323659826243, rel_tol=0.0, abs_tol=1e-15
        ),
        "Half-local H200 effect sizes drifted",
    )
    coverage = float(result["minimum_predicted_rollout_route_coverage"])
    _require(
        coverage == 0.83984375
        and coverage < float(protocol["screen_gate"]["minimum_predicted_route_coverage"]),
        "Half-local route-coverage failure drifted",
    )
    transition = result["transition_diagnostics"]
    _require(
        transition["local_parameterization"] == "full_delta"
        and transition["global_map_frozen"] is True
        and transition["full_local_matrix_count"] == 4
        and transition["full_local_slope_parameter_count"] == 16_777_216
        and transition["delta_frobenius_norm_by_route"] == [0.0] * 4
        and transition["bias_l2_norm_by_route"] == [0.0] * 4,
        "Half-local exact full-matrix zero-update result drifted",
    )
    latent = result["route_codebook"]["latent_sparsity"]
    _require(
        latent["pair_exclusivity_fraction"] == 1.0
        and math.isclose(
            float(latent["pre_split_group_active_density"]),
            0.9982335567474365,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "Half-local near-dense ancestor disclosure drifted",
    )
    return {
        "mean_local_over_global": mean_ratio,
        "terminal_local_over_global": terminal_ratio,
        "route_coverage": coverage,
    }


def _validate_provenance(provenance: Mapping[str, Any]) -> None:
    _require(
        provenance.get("schema_version") == 1
        and provenance.get("packet_id") == PACKET_ID
        and provenance.get("status") == "negative_one_seed_noncausal_control",
        "Half-local provenance identity drifted",
    )
    _require(
        provenance.get("source_git_commit_recorded_by_jobs") == SOURCE_GIT_COMMIT,
        "Half-local source commit drifted",
    )
    _require(
        provenance.get("paper_packet_roots")
        == {
            "protocol.json": EXPECTED_HASHES["protocol.json"],
            "result.json": EXPECTED_HASHES["result.json"],
        },
        "Half-local packet roots drifted",
    )
    source_files = provenance["source_files"]
    _require(
        source_files["source_worktree"]
        ["configs/allen_cahn_exact_full_local_k_screen_20260719.json"]
        == EXPECTED_HASHES["protocol.json"]
        and source_files["experiment"]
        ["exact_full_local_k/summary/selection.json"]
        == EXPECTED_HASHES["result.json"],
        "Half-local raw protocol/result roots drifted",
    )
    _require(
        source_files["experiment"]["data/allen_cahn_4_grid16_dt0p1_t20_dev.pt"]
        == "4a8a0846ee4ecd7d0bc8cac94a41fb55b1c4efad31073b4a8b88e1a9c5429236",
        "Half-local dataset root drifted",
    )
    audit = provenance["source_validation_audit"]
    expected_pairs = {
        (recipe, arm)
        for recipe in ("lr1e5", "lr3e5")
        for arm in ("sparse_support", "sparse_kmeans", "dense_kmeans")
    }
    _require(
        len(audit) == 6
        and {(row["recipe"], row["arm"]) for row in audit} == expected_pairs
        and all(
            row["best_step"] == -1
            and row["gpu_gate_at_least_85_percent"] is True
            and float(row["mean_active_gpu_utilization_percent"]) >= 85.0
            for row in audit
        ),
        "Half-local six-arm zero-update/GPU audit drifted",
    )
    for recipe, arm in expected_pairs:
        prefix = f"exact_full_local_k/stage2/{recipe}/{arm}/seed_61"
        for name in ("checkpoint.pt", "training_args.json", "validation_evaluation.json"):
            _require(
                f"{prefix}/{name}" in source_files["experiment"],
                f"Half-local source roster lacks {prefix}/{name}",
            )


def validate_packet(directory: Path = DEFAULT_PACKET_DIR) -> dict[str, Any]:
    for name in FILES:
        path = directory / name
        _require(path.is_file(), f"Missing half-local evidence: {path}")
        _require(
            sha256_path(path) == EXPECTED_HASHES[name],
            f"Released half-local evidence hash drifted: {name}",
        )
    manifest = _load_json(directory / "evidence_manifest.json")
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("packet_id") == PACKET_ID
        and manifest.get("files")
        == {name: EXPECTED_HASHES[name] for name in FILES if name != "evidence_manifest.json"},
        "Half-local evidence manifest drifted",
    )
    protocol = _load_json(directory / "protocol.json")
    result = _load_json(directory / "result.json")
    provenance = _load_json(directory / "provenance.json")
    _validate_protocol(protocol)
    metrics = _validate_result(result, protocol)
    _validate_provenance(provenance)
    return {
        "decision": result["decision"],
        "model_seed_count": 1,
        "all_six_local_recipes_selected_zero_update": True,
        **metrics,
    }


def validate_source_files(directory: Path = DEFAULT_PACKET_DIR) -> dict[str, int]:
    """Rehash the external source roster when the reviewed scratch roots exist."""
    validate_packet(directory)
    provenance = _load_json(directory / "provenance.json")
    checked: dict[str, int] = {}
    for group, files in provenance["source_files"].items():
        root = Path(provenance["source_roots"][group])
        count = 0
        for relative_path, expected_hash in files.items():
            path = root / relative_path
            _require(path.is_file(), f"Missing half-local raw source: {path}")
            _require(
                sha256_path(path) == expected_hash,
                f"Half-local raw source hash drifted: {path}",
            )
            count += 1
        checked[group] = count
    return checked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET_DIR)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-sources", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.check and not args.check_sources:
        raise ValueError("Use --check for the portable packet or --check-sources for raw files")
    result = validate_packet(args.packet_dir)
    if args.check_sources:
        result["source_files_checked"] = validate_source_files(args.packet_dir)
    print(json.dumps({"status": "passed", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
