"""Training-frozen, same-cardinality wrong-support controls."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _select_same_cardinality_prototype(
    train_masks: np.ndarray,
    train_assignments: np.ndarray,
    representatives: np.ndarray,
    *,
    target_index: int,
    donor_index: int,
) -> tuple[np.ndarray, bool]:
    """Resize a donor prototype using only its assigned training supports."""

    target = representatives[target_index]
    donor = representatives[donor_index]
    cardinality = int(target.sum())
    members = train_masks[train_assignments == donor_index]
    frequency = members.mean(0) if members.shape[0] else donor.astype(np.float64)
    coordinate = np.arange(target.size)
    # Prefer donor-frequency, donor membership, and then coordinates outside the
    # target. The last preference makes an otherwise tied control distinct.
    order = np.lexsort(
        (coordinate, target.astype(np.int8), -donor.astype(np.int8), -frequency)
    )
    result = np.zeros(target.size, dtype=bool)
    result[order[:cardinality]] = True
    if np.array_equal(result, target) and 0 < cardinality < target.size:
        removable = np.flatnonzero(result & target)
        addable = np.flatnonzero(~result & ~target)
        if removable.size and addable.size:
            remove = removable[np.argmin(frequency[removable])]
            add = addable[np.argmax(frequency[addable])]
            result[remove] = False
            result[add] = True
    return result, bool(int(result.sum()) == cardinality and not np.array_equal(result, target))


def build_wrong_support_codebook(
    train_masks: np.ndarray,
    train_assignments: np.ndarray,
    representatives: np.ndarray,
    fit_counts: np.ndarray,
) -> dict[str, Any]:
    """Build one label-free wrong support for every train-fit family.

    Donors are chosen by closest support cardinality, then largest training
    count, then stable family index. A donor is resized with coordinate
    frequencies from its training members, preserving the target cardinality.
    """

    masks = np.asarray(train_masks, dtype=bool)
    assignments = np.asarray(train_assignments, dtype=np.int64)
    reps = np.asarray(representatives, dtype=bool)
    counts = np.asarray(fit_counts, dtype=np.int64)
    if masks.ndim != 2 or reps.ndim != 2 or masks.shape[1] != reps.shape[1]:
        raise ValueError("Training masks and representatives must share a coordinate axis")
    if assignments.shape != (masks.shape[0],) or counts.shape != (reps.shape[0],):
        raise ValueError("Training assignments or family counts have the wrong shape")

    family_count = reps.shape[0]
    wrong = np.zeros_like(reps)
    donors = np.full(family_count, -1, dtype=np.int64)
    valid = np.zeros(family_count, dtype=bool)
    resized = np.zeros(family_count, dtype=bool)
    cardinalities = reps.sum(1).astype(np.int64)
    for target_index in range(family_count):
        candidates = [index for index in range(family_count) if index != target_index]
        if not candidates:
            continue
        donor_index = min(
            candidates,
            key=lambda index: (
                abs(int(cardinalities[index]) - int(cardinalities[target_index])),
                -int(counts[index]),
                int(index),
            ),
        )
        donors[target_index] = donor_index
        wrong[target_index], valid[target_index] = _select_same_cardinality_prototype(
            masks,
            assignments,
            reps,
            target_index=target_index,
            donor_index=donor_index,
        )
        resized[target_index] = cardinalities[donor_index] != cardinalities[target_index]

    wrong_cardinalities = wrong.sum(1).astype(np.int64)
    return {
        "construction": "train-only donor prototype resized to target cardinality",
        "future_or_fate_information_used": False,
        "representatives": torch.from_numpy(wrong),
        "donor_family_indices": torch.from_numpy(donors),
        "target_cardinalities": torch.from_numpy(cardinalities),
        "wrong_cardinalities": torch.from_numpy(wrong_cardinalities),
        "donor_was_resized": torch.from_numpy(resized),
        "valid_and_distinct": torch.from_numpy(valid),
        "all_valid_are_cardinality_matched": bool(
            np.array_equal(wrong_cardinalities[valid], cardinalities[valid])
        ),
    }
