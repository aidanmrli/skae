"""Authenticate quarantined forecast shards before statistical reduction."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from experiments.neurips_2026.global_k_residual_forecast.diagnostic_recompute import (
    EXPECTED_PARAMETER_COUNTS,
    recompute_route_diagnostics,
    validate_count_weighted_basin_h200,
)
from experiments.neurips_2026.global_k_residual_forecast.protocol import (
    load_json,
    sha256_path,
    verify_sha,
)
from experiments.neurips_2026.global_k_residual_forecast.validation import (
    ALL_METHODS,
    DENSE_METHODS,
    PREDICTOR_ASSERTIONS,
    SPARSE_METHODS,
    exact_keys,
    finite_number,
    require_sha256,
    validate_method,
)


def _validate_route_structures(shard: dict[str, Any], card: dict[str, Any]) -> int:
    fit = exact_keys(
        shard["label_free_family_fit"],
        {
            "all_family_count", "retained_family_count", "retained_original_family_ids",
            "retained_fit_coverage", "support_cardinalities", "sign_pair_exclusivity",
            "fallback_used", "maximum_family_truncation_used", "representatives_sha256",
            "label_or_known_basin_count_used", "fit_valid",
        },
        "route fit",
    )
    retained = fit["retained_family_count"]
    if not isinstance(retained, int) or not 1 <= retained <= 64:
        raise RuntimeError("Retained family count is invalid")
    bool_keys = (
        "sign_pair_exclusivity", "fallback_used", "maximum_family_truncation_used",
        "fit_valid",
    )
    if (
        not isinstance(fit["all_family_count"], int)
        or fit["all_family_count"] < retained
        or len(fit["retained_original_family_ids"]) != retained
        or len(fit["support_cardinalities"]) != retained
        or any(
            not isinstance(value, int) or not 0 < value < 256
            for value in fit["support_cardinalities"]
        )
        or not 0 <= finite_number(fit["retained_fit_coverage"], "fit coverage") <= 1
        or fit["label_or_known_basin_count_used"] is not False
        or not all(isinstance(fit[key], bool) for key in bool_keys)
    ):
        raise RuntimeError("Route-fit structure is invalid")
    require_sha256(fit["representatives_sha256"], "representative digest")

    audit = exact_keys(
        shard["held_out_route_audit"],
        {
            "assignment_count_by_family", "mean_nearest_jaccard",
            "confident_assignment_fraction", "active_family_count_at_minimum_fraction",
            "label_free_route_audit_valid", "evaluation_only_alignment",
        },
        "route audit",
    )
    counts = audit["assignment_count_by_family"]
    route_spec = card["outcome_free_trajectory_corpora"]["route_audit"]
    expected_audit = int(route_spec["trajectory_count"]) * int(
        route_spec["horizon_steps"]
    )
    if (
        not isinstance(counts, list)
        or len(counts) != retained
        or any(not isinstance(value, int) or value < 0 for value in counts)
        or sum(counts) != expected_audit
        or not 0 <= finite_number(audit["mean_nearest_jaccard"], "audit Jaccard") <= 1
        or not 0 <= finite_number(
            audit["confident_assignment_fraction"], "audit confidence"
        ) <= 1
        or not isinstance(audit["active_family_count_at_minimum_fraction"], int)
        or not isinstance(audit["label_free_route_audit_valid"], bool)
    ):
        raise RuntimeError("Route-audit structure is invalid")
    alignment = exact_keys(
        audit["evaluation_only_alignment"],
        {
            "labels_computed_after_and_not_passed_to_assignment_or_predictor",
            "observed_label_values", "contingency_family_by_label",
            "family_conditional_basin_purity",
        },
        "route alignment",
    )
    contingency = alignment["contingency_family_by_label"]
    if (
        alignment["labels_computed_after_and_not_passed_to_assignment_or_predictor"]
        is not True
        or alignment["observed_label_values"] != [0, 1, 2]
        or not isinstance(contingency, list)
        or len(contingency) != retained
        or any(not isinstance(row, list) or len(row) != 3 for row in contingency)
        or any(
            not isinstance(value, int) or value < 0
            for row in contingency
            for value in row
        )
        or sum(sum(row) for row in contingency) != expected_audit
        or not 0 <= finite_number(
            alignment["family_conditional_basin_purity"], "basin purity"
        ) <= 1
    ):
        raise RuntimeError("Evaluation-only route alignment is malformed")

    null = exact_keys(
        shard["matched_coordinate_null"],
        {
            "candidate_count", "eligible_candidate_count", "selected_candidate_indices",
            "selected_scale_rows", "correct_source_rms", "correct_update_rms",
            "permutation_bank_sha256", "cardinality_and_pairwise_jaccard_exactly_preserved",
            "selection_uses_labels_basin_count_or_forecast_truth",
            "latent_null_guaranteed_on_encoder_image", "physical_prediction_reencoded_each_step",
            "scale_match_valid",
        },
        "matched coordinate null",
    )
    selected = null["selected_candidate_indices"]
    scale_rows = null["selected_scale_rows"]
    if (
        null["candidate_count"] != 256
        or not isinstance(null["eligible_candidate_count"], int)
        or not 0 <= null["eligible_candidate_count"] <= 256
        or not isinstance(selected, list)
        or len(selected) != len(set(selected))
        or len(selected) != 32
        or any(not isinstance(value, int) or not 0 <= value < 256 for value in selected)
        or not isinstance(scale_rows, list)
        or len(scale_rows) != 32
        or null["cardinality_and_pairwise_jaccard_exactly_preserved"] is not True
        or null["selection_uses_labels_basin_count_or_forecast_truth"] is not False
        or null["latent_null_guaranteed_on_encoder_image"] is not False
        or null["physical_prediction_reencoded_each_step"] is not True
        or not isinstance(null["scale_match_valid"], bool)
    ):
        raise RuntimeError("Matched-null structure is invalid")
    row_keys = {
        "candidate_index", "score", "source_rms_ratio", "update_rms_ratio", "eligible"
    }
    for index, row in zip(selected, scale_rows):
        exact_keys(row, row_keys, "selected null scale row")
        if row["candidate_index"] != index or not isinstance(row["eligible"], bool):
            raise RuntimeError("Selected null row identity drifted")
        finite_number(row["score"], "null scale score", minimum=0.0)
        if finite_number(row["source_rms_ratio"], "null source ratio") <= 0:
            raise RuntimeError("Null source ratio is not positive")
        if finite_number(row["update_rms_ratio"], "null update ratio") <= 0:
            raise RuntimeError("Null update ratio is not positive")
    if finite_number(null["correct_source_rms"], "correct source RMS") <= 0:
        raise RuntimeError("Correct source RMS is not positive")
    if finite_number(null["correct_update_rms"], "correct update RMS") <= 0:
        raise RuntimeError("Correct update RMS is not positive")
    require_sha256(null["permutation_bank_sha256"], "permutation bank digest")
    recompute_route_diagnostics(shard, card)
    return retained


def _validate_data_manifest(
    path: Path, card: dict[str, Any], freeze: dict[str, Any]
) -> str:
    manifest = exact_keys(
        load_json(path),
        {
            "schema_version", "protocol_id", "artifact_role", "freeze", "rows",
            "forbidden_content_assertion",
        },
        "outcome-free data manifest",
    )
    if (
        manifest["schema_version"] != 1
        or manifest["protocol_id"] != card["protocol_id"]
        or manifest["artifact_role"] != "outcome_free_physical_trajectory_manifest"
        or manifest["freeze"] != freeze
    ):
        raise RuntimeError("Outcome-free data-manifest identity failed")
    corpora = card["outcome_free_trajectory_corpora"]
    expected = [
        ("route_fit", None, corpora["route_fit"]["seed"], "route_fit.pt"),
        ("route_audit", None, corpora["route_audit"]["seed"], "route_audit.pt"),
        *[
            ("evaluation", index, seed, f"evaluation_{index}.pt")
            for index, seed in enumerate(corpora["evaluation"]["seeds"])
        ],
        (
            "smoke_evaluation",
            corpora["smoke_evaluation"]["dataset_index"],
            corpora["smoke_evaluation"]["seed"],
            "smoke_evaluation_0.pt",
        ),
    ]
    rows = manifest["rows"]
    if not isinstance(rows, list) or len(rows) != 6:
        raise RuntimeError("Outcome-free data roster is not exactly six rows")
    for row, (role, index, seed, filename) in zip(rows, expected):
        keys = {"role", "seed", "path", "sha256"}
        if index is not None:
            keys.add("dataset_index")
        exact_keys(row, keys, "outcome-free data row")
        expected_path = path.parent / filename
        if (
            row["role"] != role
            or row["seed"] != seed
            or (index is not None and row["dataset_index"] != index)
            or Path(row["path"]) != expected_path
        ):
            raise RuntimeError("Outcome-free data row identity drifted")
        verify_sha(
            expected_path,
            require_sha256(row["sha256"], "trajectory artifact digest"),
            role,
        )
    return sha256_path(path)


def _validate_provenance(
    shard: dict[str, Any],
    task: dict[str, Any],
    tasks: dict[str, Any],
    *,
    expected_data_path: Path,
    expected_evaluator: str,
) -> str:
    provenance = exact_keys(
        shard["provenance"],
        {
            "sparse", "dense", "data_manifest_path", "data_manifest_sha256",
            "evaluator_sha256", "git_commit", "gpu",
        },
        "shard provenance",
    )
    expected_counts = tasks["provenance_contract"]
    if expected_counts != EXPECTED_PARAMETER_COUNTS:
        raise RuntimeError("Frozen task parameter-count contract drifted")
    for arm in ("sparse", "dense"):
        row = exact_keys(
            provenance[arm],
            {
                "checkpoint_path", "checkpoint_sha256",
                "v2_exact_checkpoint_audit_passed", "checkpoint_step",
                "trainable_parameter_counts",
            },
            f"{arm} checkpoint provenance",
        )
        checkpoint = task[f"{arm}_checkpoint"]
        counts = expected_counts[f"{arm}_trainable_parameter_counts"]
        if (
            row["checkpoint_path"] != checkpoint["path"]
            or row["checkpoint_sha256"] != checkpoint["sha256"]
            or row["v2_exact_checkpoint_audit_passed"] is not True
            or not isinstance(row["checkpoint_step"], int)
            or row["checkpoint_step"] <= 0
            or row["trainable_parameter_counts"] != counts
        ):
            raise RuntimeError(f"{arm} checkpoint provenance drifted")
    data_hash = require_sha256(
        provenance["data_manifest_sha256"], "data manifest digest"
    )
    if (
        Path(provenance["data_manifest_path"]) != expected_data_path
        or provenance["evaluator_sha256"] != expected_evaluator
        or re.fullmatch(r"[0-9a-f]{40}", str(provenance["git_commit"])) is None
    ):
        raise RuntimeError("Common runtime provenance drifted")
    gpu = exact_keys(
        provenance["gpu"], {"name", "total_memory_bytes"}, "GPU provenance"
    )
    if (
        "A100" not in str(gpu["name"])
        or not isinstance(gpu["total_memory_bytes"], int)
        or gpu["total_memory_bytes"] < 75 * 1024**3
    ):
        raise RuntimeError("Shard GPU provenance is below the A100 requirement")
    return data_hash


def _validate_dataset(
    dataset: dict[str, Any],
    *,
    index: int,
    seed: int,
    count: int,
    retained: int,
    card: dict[str, Any],
) -> tuple[int, int, str, int]:
    exact_keys(
        dataset,
        {
            "dataset_index", "dataset_seed", "trajectory_sha256", "trajectory_count",
            "sparse", "dense", "evaluation_only_basin_stratification",
        },
        "dataset row",
    )
    digest = require_sha256(dataset["trajectory_sha256"], "trajectory-array digest")
    if (
        dataset["dataset_index"] != index
        or dataset["dataset_seed"] != seed
        or dataset["trajectory_count"] != count
    ):
        raise RuntimeError("Dataset roster identity drifted")
    sparse = exact_keys(
        dataset["sparse"],
        {"methods", "routing_during_sparse_routed_residual"},
        "sparse methods",
    )
    dense = exact_keys(dataset["dense"], {"methods"}, "dense methods")
    if (
        set(sparse["methods"]) != set(SPARSE_METHODS)
        or set(dense["methods"]) != set(DENSE_METHODS)
    ):
        raise RuntimeError("Exact 41-method roster or order drifted")
    methods = {**sparse["methods"], **dense["methods"]}
    for name, method in methods.items():
        validate_method(name, method)
    routing = exact_keys(
        sparse["routing_during_sparse_routed_residual"],
        {
            "mean_nearest_jaccard", "confident_assignment_fraction",
            "family_switch_fraction", "assignment_count_by_family",
        },
        "forecast routing",
    )
    usage = routing["assignment_count_by_family"]
    horizon = int(card["forecast_protocol"]["stress_horizon_steps"])
    if (
        not isinstance(usage, list)
        or len(usage) != retained
        or any(not isinstance(value, int) or value < 0 for value in usage)
        or sum(usage) != count * horizon
        or any(
            not 0 <= finite_number(routing[key], f"routing {key}") <= 1
            for key in (
                "mean_nearest_jaccard", "confident_assignment_fraction",
                "family_switch_fraction",
            )
        )
    ):
        raise RuntimeError("Forecast routing structure is invalid")
    basin = exact_keys(
        dataset["evaluation_only_basin_stratification"],
        {
            "labels_are_evaluation_only_and_were_not_passed_to_any_predictor",
            "label_source", "rows",
        },
        "basin stratification",
    )
    if (
        basin["labels_are_evaluation_only_and_were_not_passed_to_any_predictor"]
        is not True
        or basin["label_source"] != "ground-truth initial-state basin_label"
        or set(basin["rows"]) != {"0", "1", "2"}
    ):
        raise RuntimeError("Evaluation-only basin stratification drifted")
    basin_count = 0
    for label, basin_row in basin["rows"].items():
        exact_keys(
            basin_row,
            {"trajectory_count", "through_h200_mse_by_method"},
            f"basin {label}",
        )
        if (
            not isinstance(basin_row["trajectory_count"], int)
            or basin_row["trajectory_count"] <= 0
        ):
            raise RuntimeError("Basin trajectory count is invalid")
        basin_count += basin_row["trajectory_count"]
        values = basin_row["through_h200_mse_by_method"]
        if not isinstance(values, dict) or set(values) != set(ALL_METHODS):
            raise RuntimeError("Basin method roster drifted")
        for value in values.values():
            if value is not None:
                finite_number(value, "basin H200 MSE", minimum=0.0)
    if basin_count != count:
        raise RuntimeError("Basin counts do not cover the trajectory panel")
    validate_count_weighted_basin_h200(methods, basin["rows"], count)
    return index, seed, digest, count


def validate_scientific_shards(
    raw_shards: list[dict[str, Any]],
    *,
    tasks: dict[str, Any],
    card: dict[str, Any],
    freeze: dict[str, Any],
    output_root: Path,
    evaluator_path: Path,
) -> None:
    if len(raw_shards) != 10:
        raise RuntimeError("Exactly ten scientific shards are required")
    expected_evaluator = sha256_path(evaluator_path)
    expected_data_path = output_root / "outcome_free_data" / "manifest.json"
    common_data_hash: str | None = None
    common_roster: list[tuple[int, int, str, int]] | None = None
    shard_keys = {
        "schema_version", "protocol_id", "artifact_role", "task_id", "model_seed",
        "freeze", "provenance", "predictor_assertions", "label_free_family_fit",
        "held_out_route_audit", "matched_coordinate_null", "dataset_rows",
        "compute_elapsed_seconds",
    }
    for task, shard in zip(tasks["tasks"], raw_shards):
        exact_keys(shard, shard_keys, "scientific shard")
        if (
            shard["schema_version"] != 1
            or shard["protocol_id"] != card["protocol_id"]
            or shard["artifact_role"] != "quarantined_scientific_seed_shard"
            or shard["task_id"] != task["task_id"]
            or shard["model_seed"] != task["model_seed"]
            or shard["freeze"] != freeze
        ):
            raise RuntimeError("Scientific shard identity failed")
        if finite_number(
            shard["compute_elapsed_seconds"], "compute elapsed seconds"
        ) <= 0:
            raise RuntimeError("Compute elapsed time is not positive")
        data_hash = _validate_provenance(
            shard,
            task,
            tasks,
            expected_data_path=expected_data_path,
            expected_evaluator=expected_evaluator,
        )
        common_data_hash = data_hash if common_data_hash is None else common_data_hash
        if data_hash != common_data_hash:
            raise RuntimeError("Scientific shards do not share one data manifest")
        assertions = exact_keys(
            shard["predictor_assertions"],
            PREDICTOR_ASSERTIONS,
            "predictor assertions",
        )
        if any(value is not True for value in assertions.values()):
            raise RuntimeError("A frozen predictor assertion failed")
        retained = _validate_route_structures(shard, card)
        datasets = shard["dataset_rows"]
        seeds = card["outcome_free_trajectory_corpora"]["evaluation"]["seeds"]
        count = int(
            card["outcome_free_trajectory_corpora"]["evaluation"]
            ["trajectory_count_each"]
        )
        if not isinstance(datasets, list) or len(datasets) != 3:
            raise RuntimeError("Shard does not contain exactly three datasets")
        roster = [
            _validate_dataset(
                dataset,
                index=index,
                seed=int(seed),
                count=count,
                retained=retained,
                card=card,
            )
            for index, (dataset, seed) in enumerate(zip(datasets, seeds))
        ]
        common_roster = roster if common_roster is None else common_roster
        if roster != common_roster:
            raise RuntimeError("Dataset roster differs across model seeds")
    observed_data_hash = _validate_data_manifest(expected_data_path, card, freeze)
    if observed_data_hash != common_data_hash:
        raise RuntimeError("Data-manifest hash differs from shard provenance")
