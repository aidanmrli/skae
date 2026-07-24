from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from .features import (
    field_summary,
    matched_topk_masks,
    well_area_fractions,
)
from .io import (
    checkpoint_specs,
    duplicate_safe_json,
    load_card,
    load_field_roster,
    load_task_manifest,
    load_training_labels,
    sha256_path,
    torch_load,
    verify_source_manifest,
)
from .packet import write_invalid_packet, write_valid_packet
from .probes import (
    classification_metrics,
    fit_probe,
    require_class_counts,
)
from .reduction_utils import (
    relative_pass,
    split_time_matrix,
)
from .statistics import (
    absolute_permutation_p,
    contrast_summary,
    holm_adjust,
)
from .validity import evaluate_target_validity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-card-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-task-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-telemetry-receipt-sha256", required=True)
    args = parser.parse_args()

    card, card_sha = load_card(expected_sha256=args.expected_card_sha256)
    source_sha = verify_source_manifest(
        card, expected_sha256=args.expected_source_manifest_sha256
    )
    task, task_sha = load_task_manifest(
        card, expected_sha256=args.expected_task_manifest_sha256
    )
    if Path(task["output_root"]) != args.output_root:
        raise RuntimeError("Summary output root differs from frozen task")
    if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise RuntimeError("The label-aware summary must not request a GPU")

    receipt_path = args.output_root / "field_only" / "telemetry_receipt.json"
    if sha256_path(receipt_path) != args.expected_telemetry_receipt_sha256:
        raise RuntimeError("Telemetry receipt root mismatch")
    receipt = duplicate_safe_json(receipt_path)
    roots = {
        "card_sha256": card_sha,
        "source_manifest_sha256": source_sha,
        "task_manifest_sha256": task_sha,
    }
    if receipt.get("status") != "authenticated" or any(
        receipt.get(key) != value for key, value in roots.items()
    ):
        raise RuntimeError("GPU receipt is not authenticated against launch roots")

    features_path = args.output_root / "field_only" / "features.pt"
    if sha256_path(features_path) != receipt["features_sha256"]:
        raise RuntimeError("Field-only feature artifact changed after telemetry")
    encoded = torch_load(features_path)
    if encoded.get("status") != "prospective_field_only_frozen_before_semantic_access":
        raise RuntimeError("Feature artifact status failed")
    if any(encoded.get(key) != value for key, value in roots.items()):
        raise RuntimeError("Feature roots differ from launch roots")
    if encoded.get("semantic_target_keys_accessed") != 0 or encoded.get(
        "semantic_ties_or_labels_derived"
    ) != 0:
        raise RuntimeError("GPU semantic firewall was violated")
    specs = checkpoint_specs(card)
    expected_checkpoints = {
        f"{arm}_seed_{seed}": specs[(arm, seed)].sha256
        for arm in card["roster"]["arms"]
        for seed in card["roster"]["model_seeds"]
    }
    if encoded["checkpoint_sha256"] != expected_checkpoints:
        raise RuntimeError("Encoded checkpoint roster drifted")

    # The semantic outcome is first derived only after all roots above authenticate.
    if encoded.get("dataset_manifest_sha256") != receipt["dataset_manifest_sha256"]:
        raise RuntimeError("Feature and receipt dataset roots differ")
    train_fields, test_fields, _ = load_field_roster(
        card, expected_manifest_sha256=receipt["dataset_manifest_sha256"]
    )
    stored_train_labels, stored_train_final = load_training_labels(card)
    adjudication = evaluate_target_validity(
        card,
        train_fields,
        test_fields,
        stored_train_labels,
        stored_train_final,
    )
    train_labels = adjudication["train_labels"]
    test_labels = adjudication["test_labels"]
    train_eligible = adjudication["train_mask"]
    test_eligible = adjudication["test_masks"]
    eligibility = adjudication["report"]
    terminal_diagnostics = eligibility["terminal_diagnostics"]
    validity = card["validity"]
    if adjudication["reasons"]:
        write_invalid_packet(
            args.output_root,
            card=card,
            roots=roots,
            receipt_path=receipt_path,
            features_path=features_path,
            reasons=adjudication["reasons"],
            validity=eligibility,
        )
        return
    minimum_test = int(
        validity["minimum_test_count_per_class_per_dataset_after_exclusion"]
    )
    observed_train_counts = require_class_counts(
        train_labels,
        minimum=int(validity["minimum_training_count_per_class_after_exclusion"]),
    )
    test_counts = [
        require_class_counts(values, minimum=minimum_test) for values in test_labels
    ]
    fate_terminology_passed = adjudication["fate_terminology_passed"]

    times = card["roster"]["observation_times"]
    observation_indices = card["roster"]["observation_indices"]
    seeds = card["roster"]["model_seeds"]
    dataset_seeds = card["roster"]["dataset_seeds"]
    feature_names = [
        "sparse_support",
        "sparse_values",
        "dense_values",
        "dense_topk",
    ]
    learned = {
        name: [np.full((10, 3), np.nan) for _ in times] for name in feature_names
    }
    physical = {
        name: [np.full(3, np.nan) for _ in times]
        for name in ("raw_x", "well_area_fractions", "field_summary", "current_modal")
    }
    rows: list[dict[str, object]] = []
    support_diagnostics: list[dict[str, object]] = []
    primary_predictions: list[list[np.ndarray]] = [[] for _ in seeds]
    layout = encoded["layout"]
    threshold = float(card["feature_protocol"]["support_threshold"])
    probe = card["probe"]

    def record_fit(
        feature_name: str,
        time_position: int,
        model_position: int | None,
        train_x: np.ndarray,
        tests_x: list[np.ndarray],
    ) -> object:
        result = fit_probe(
            train_x,
            train_labels,
            tests_x,
            test_labels,
            alphas=probe["alpha_grid"],
            n_splits=int(probe["selection_folds"]),
            split_seed=int(probe["split_seed"]),
            minimum_test_count=minimum_test,
        )
        for dataset_position, metrics in enumerate(result.metrics):
            row = {
                "feature": feature_name,
                "observation_index": observation_indices[time_position],
                "observation_time": times[time_position],
                "model_seed": "" if model_position is None else seeds[model_position],
                "dataset_seed": dataset_seeds[dataset_position],
                "eligible_rows": len(test_labels[dataset_position]),
                "alpha": result.alpha,
                "cv_balanced_accuracy": result.cv_balanced_accuracy,
                **metrics,
            }
            rows.append(row)
            if model_position is None:
                physical[feature_name][time_position][dataset_position] = metrics[
                    "balanced_accuracy"
                ]
            else:
                learned[feature_name][time_position][model_position, dataset_position] = metrics[
                    "balanced_accuracy"
                ]
        return result

    # Physical steelmen are fit once per time and never duplicated as model seeds.
    trajectory_sets = [train_fields, *test_fields]
    eligibility_masks = [train_eligible, *test_eligible]
    physical_diagnostics = []
    for time_position, observation_index in enumerate(observation_indices):
        current_sets = [values[:, observation_index] for values in trajectory_sets]
        raw_sets = [
            values[mask].numpy()
            for values, mask in zip(current_sets, eligibility_masks)
        ]
        fraction_sets = [
            well_area_fractions(values).numpy()[mask]
            for values, mask in zip(current_sets, eligibility_masks)
        ]
        summary_sets = [
            field_summary(values).numpy()[mask]
            for values, mask in zip(current_sets, eligibility_masks)
        ]
        record_fit("raw_x", time_position, None, raw_sets[0], raw_sets[1:])
        record_fit(
            "well_area_fractions",
            time_position,
            None,
            fraction_sets[0],
            fraction_sets[1:],
        )
        record_fit("field_summary", time_position, None, summary_sets[0], summary_sets[1:])
        current_test_labels = [values.argmax(1) for values in fraction_sets[1:]]
        diagnostic_row = {"observation_time": times[time_position], "datasets": []}
        for dataset_position, (truth, current, fractions) in enumerate(
            zip(test_labels, current_test_labels, fraction_sets[1:])
        ):
            metrics = classification_metrics(truth, current)
            physical["current_modal"][time_position][dataset_position] = metrics[
                "balanced_accuracy"
            ]
            ordered = np.sort(fractions, axis=1)[:, ::-1]
            diagnostic_row["datasets"].append(
                {
                    "dataset_seed": dataset_seeds[dataset_position],
                    "current_to_final_label_change_fraction": float(np.mean(current != truth)),
                    "mean_current_top_occupancy": float(ordered[:, 0].mean()),
                    "mean_current_top1_minus_top2_margin": float(
                        (ordered[:, 0] - ordered[:, 1]).mean()
                    ),
                    **metrics,
                }
            )
            rows.append(
                {
                    "feature": "current_modal",
                    "observation_index": observation_index,
                    "observation_time": times[time_position],
                    "model_seed": "",
                    "dataset_seed": dataset_seeds[dataset_position],
                    "eligible_rows": len(truth),
                    "alpha": "",
                    "cv_balanced_accuracy": "",
                    **metrics,
                }
            )
        physical_diagnostics.append(diagnostic_row)

    # Learned frozen representations are crossed over ten paired model seeds.
    for model_position, seed in enumerate(seeds):
        sparse_matrix = encoded["latents"][f"sparse_seed_{seed}"].numpy()
        dense_matrix = encoded["latents"][f"dense_seed_{seed}"].numpy()
        if sparse_matrix.shape != (6400, 2048) or dense_matrix.shape != (6400, 2048):
            raise RuntimeError("Latent feature shape drifted")
        for time_position in range(len(times)):
            sparse_train, sparse_tests = split_time_matrix(
                sparse_matrix, time_index=time_position, layout=layout
            )
            dense_train, dense_tests = split_time_matrix(
                dense_matrix, time_index=time_position, layout=layout
            )
            sparse_sets = [
                values[mask]
                for values, mask in zip(
                    [sparse_train, *sparse_tests], eligibility_masks
                )
            ]
            dense_sets = [
                values[mask]
                for values, mask in zip(
                    [dense_train, *dense_tests], eligibility_masks
                )
            ]
            sparse_all = np.concatenate(sparse_sets, axis=0)
            dense_all = np.concatenate(dense_sets, axis=0)
            sparse_masks = np.abs(sparse_all) > threshold
            dense_masks = matched_topk_masks(dense_all, sparse_masks)
            if not np.array_equal(sparse_masks.sum(1), dense_masks.sum(1)):
                raise RuntimeError("Dense top-k cardinality parity failed")
            split_points = np.cumsum([len(values) for values in sparse_sets[:-1]])
            sparse_mask_sets = np.split(sparse_masks, split_points)
            dense_mask_sets = np.split(dense_masks, split_points)
            feature_sets = {
                "sparse_support": sparse_mask_sets,
                "sparse_values": sparse_sets,
                "dense_values": dense_sets,
                "dense_topk": dense_mask_sets,
            }
            for feature_name, values in feature_sets.items():
                result = record_fit(
                    feature_name,
                    time_position,
                    model_position,
                    values[0],
                    values[1:],
                )
                if feature_name == "sparse_support" and time_position == 0:
                    primary_predictions[model_position] = list(result.predictions)
            support_diagnostics.append(
                {
                    "observation_index": observation_indices[time_position],
                    "observation_time": times[time_position],
                    "model_seed": seed,
                    "eligible_rows": int(sparse_masks.shape[0]),
                    "mean_active_fraction": float(sparse_masks.mean()),
                    "minimum_active_coordinates": int(sparse_masks.sum(1).min()),
                    "maximum_active_coordinates": int(sparse_masks.sum(1).max()),
                }
            )

    if any(not np.isfinite(matrix).all() for values in learned.values() for matrix in values):
        raise RuntimeError("Learned crossed roster contains omissions")
    if any(not np.isfinite(vector).all() for values in physical.values() for vector in values):
        raise RuntimeError("Physical baseline roster contains omissions")
    if any(len(values) != 3 for values in primary_predictions):
        raise RuntimeError("Primary prediction roster contains omissions")

    bootstrap = card["statistics"]["two_way_bootstrap"]
    early_contrasts = []
    for time_position in range(len(times)):
        contrast = contrast_summary(
            learned["sparse_support"][time_position]
            - learned["dense_topk"][time_position],
            bootstrap_replicates=int(bootstrap["replicates"]),
            bootstrap_seed=int(bootstrap["seed"]),
        )
        contrast["observation_time"] = times[time_position]
        early_contrasts.append(contrast)
    early_adjusted = holm_adjust(
        [item["one_sided_exact_sign_flip_p"] for item in early_contrasts]
    )
    for item, adjusted in zip(early_contrasts, early_adjusted):
        item["holm_adjusted_p"] = adjusted
        item["strong_secondary_relative_gate"] = relative_pass(item, adjusted)

    x0_right = {
        "dense_topk": learned["dense_topk"][0],
        "dense_values": learned["dense_values"][0],
        "raw_x": np.broadcast_to(physical["raw_x"][0][None, :], (10, 3)),
        "current_modal": np.broadcast_to(
            physical["current_modal"][0][None, :], (10, 3)
        ),
    }
    tier_contrasts = {}
    for name, right in x0_right.items():
        tier_contrasts[name] = contrast_summary(
            learned["sparse_support"][0] - right,
            bootstrap_replicates=int(bootstrap["replicates"]),
            bootstrap_seed=int(bootstrap["seed"]),
        )
    tier_adjusted = holm_adjust(
        [
            tier_contrasts[name]["one_sided_exact_sign_flip_p"]
            for name in x0_right
        ]
    )
    for name, adjusted in zip(x0_right, tier_adjusted):
        tier_contrasts[name]["holm_adjusted_p"] = adjusted
        tier_contrasts[name]["claim_tier_relative_gate"] = relative_pass(
            tier_contrasts[name], adjusted
        )

    observed_absolute, absolute_p = absolute_permutation_p(
        test_labels,
        primary_predictions,
        replicates=int(card["statistics"]["absolute_null"]["replicates"]),
        seed=int(card["statistics"]["absolute_null"]["seed"]),
    )
    support_information = bool(
        observed_absolute
        >= card["primary_gate"]["minimum_sparse_support_x0_balanced_accuracy"]
        and absolute_p <= card["primary_gate"]["maximum_absolute_permutation_p"]
    )
    primary = tier_contrasts["dense_topk"]
    primary_gate = bool(
        support_information
        and primary["mean_difference"]
        >= card["primary_gate"]["minimum_sparse_support_minus_dense_topk_x0"]
        and primary["model_seed_wins"]
        >= card["primary_gate"]["minimum_model_seed_wins"]
        and primary["dataset_seed_wins"]
        >= card["primary_gate"]["minimum_dataset_seed_wins"]
        and primary["two_way_bootstrap_interval"][0]
        > card["primary_gate"]["two_way_bootstrap_lower_above"]
        and primary["one_sided_exact_sign_flip_p"]
        <= card["primary_gate"]["maximum_one_sided_exact_sign_flip_p"]
    )
    claim_tiers = {
        "support_information": support_information,
        "coordinate_identity": primary_gate,
        "better_than_dense_representation": bool(
            primary_gate and tier_contrasts["dense_values"]["claim_tier_relative_gate"]
        ),
        "accessibility_beyond_physical_state": bool(
            primary_gate and tier_contrasts["raw_x"]["claim_tier_relative_gate"]
        ),
        "future_information_beyond_initial_occupancy": bool(
            primary_gate
            and tier_contrasts["current_modal"]["claim_tier_relative_gate"]
        ),
    }
    any_secondary = any(
        item["strong_secondary_relative_gate"] for item in early_contrasts[1:]
    )
    any_continuous = bool(
        max(
            learned["sparse_values"][0].mean(),
            learned["dense_values"][0].mean(),
            physical["raw_x"][0].mean(),
            physical["well_area_fractions"][0].mean(),
            physical["field_summary"][0].mean(),
        )
        >= 0.60
    )
    if primary_gate:
        branch = "strong_x0_support_readout"
    elif support_information:
        branch = "absolute_support_information_without_dense_binary_advantage"
    elif any_secondary:
        branch = "early_only"
    elif any_continuous:
        branch = "continuous_or_physical_only"
    else:
        branch = "failed"

    decision = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "status": "valid_complete",
        "decision_branch": branch,
        "target_wording": (
            "decisive T20 modal-well fate"
            if fate_terminology_passed
            else "decisive T20 modal-well occupancy"
        ),
        "fate_terminology_construct_gate_passed": fate_terminology_passed,
        "fixed_dataset_inference_only": True,
        "eligibility": eligibility,
        "no_v3": True,
        "claim_tiers": claim_tiers,
        "primary": {
            "sparse_support_mean_balanced_accuracy": observed_absolute,
            "absolute_joint_label_permutation_p": absolute_p,
            "full_gate_passed": primary_gate,
            **primary,
        },
        "x0_tier_contrasts": tier_contrasts,
        "early_sparse_support_vs_dense_topk": early_contrasts,
        "terminal_diagnostics": terminal_diagnostics,
        "observation_diagnostics": physical_diagnostics,
        "support_density_diagnostics": support_diagnostics,
        "training_class_counts": observed_train_counts.tolist(),
        "test_class_counts": [values.tolist() for values in test_counts],
        "claim_boundary": card["claim_boundary"],
    }
    write_valid_packet(
        args.output_root,
        card=card,
        roots=roots,
        receipt_path=receipt_path,
        features_path=features_path,
        decision=decision,
        rows=rows,
    )


if __name__ == "__main__":
    main()
