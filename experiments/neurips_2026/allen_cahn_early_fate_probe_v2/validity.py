"""Pre-fit target eligibility and four-class validity adjudication."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .probes import stratified_folds
from .reduction_utils import labels_and_eligibility, occupancy_diagnostics


def evaluate_target_validity(
    card: dict[str, Any],
    train_fields: torch.Tensor,
    test_fields: list[torch.Tensor],
    stored_train_labels: torch.Tensor,
    stored_train_final: torch.Tensor,
) -> dict[str, Any]:
    target_index = int(card["system"]["final_target_index"])
    expected_final = train_fields[:, target_index].reshape(512, 16, 16, 2)
    if not torch.equal(stored_train_final, expected_final):
        raise RuntimeError("Training final-field reload mismatch")
    train_labels_tensor, train_mask_tensor = labels_and_eligibility(
        train_fields[:, target_index]
    )
    test_pairs = [
        labels_and_eligibility(values[:, target_index]) for values in test_fields
    ]
    train_mask = train_mask_tensor.numpy()
    test_masks = [pair[1].numpy() for pair in test_pairs]
    train_labels = train_labels_tensor.numpy()[train_mask]
    test_labels = [
        pair[0].numpy()[mask] for pair, mask in zip(test_pairs, test_masks)
    ]
    train_counts = np.bincount(train_labels, minlength=4)
    test_counts = [np.bincount(values, minlength=4) for values in test_labels]
    terminal = {
        "training": occupancy_diagnostics(train_fields[:, target_index]),
        "tests": [
            occupancy_diagnostics(values[:, target_index]) for values in test_fields
        ],
    }
    report = {
        "training_fraction": float(train_mask.mean()),
        "test_fractions": [float(mask.mean()) for mask in test_masks],
        "pooled_test_fraction": float(np.concatenate(test_masks).mean()),
        "all_rows_fraction": float(
            np.concatenate([train_mask, *test_masks]).mean()
        ),
        "training_class_counts_after_exclusion": train_counts.tolist(),
        "test_class_counts_after_exclusion": [values.tolist() for values in test_counts],
        "terminal_diagnostics": terminal,
    }
    frozen = card["validity"]
    reasons = []
    if report["training_fraction"] < frozen["minimum_eligible_fraction_training"]:
        reasons.append("training_eligibility_below_0p95")
    if any(
        value < frozen["minimum_eligible_fraction_each_test_dataset"]
        for value in report["test_fractions"]
    ):
        reasons.append("test_dataset_eligibility_below_0p95")
    if report["pooled_test_fraction"] < frozen["minimum_eligible_fraction_pooled_test"]:
        reasons.append("pooled_test_eligibility_below_0p95")
    if report["all_rows_fraction"] < frozen["minimum_eligible_fraction_all_rows"]:
        reasons.append("all_row_eligibility_below_0p95")
    if np.any(train_counts < frozen["minimum_training_count_per_class_after_exclusion"]):
        reasons.append("training_class_count_gate_failed")
    if any(
        np.any(values < frozen["minimum_test_count_per_class_per_dataset_after_exclusion"])
        for values in test_counts
    ):
        reasons.append("test_class_count_gate_failed")
    if not np.array_equal(stored_train_labels.numpy()[train_mask], train_labels):
        reasons.append("eligible_training_stored_label_mismatch")
    try:
        stratified_folds(
            train_labels,
            n_splits=int(card["probe"]["selection_folds"]),
            seed=int(card["probe"]["split_seed"]),
        )
    except ValueError:
        reasons.append("post_exclusion_training_fold_gate_failed")
    fate_wording = all(
        item["eligible_fraction_top_occupancy_at_least_half"] >= 0.90
        for item in terminal["tests"]
    )
    return {
        "train_labels": train_labels,
        "test_labels": test_labels,
        "train_mask": train_mask,
        "test_masks": test_masks,
        "report": report,
        "reasons": reasons,
        "fate_terminology_passed": fate_wording,
    }
