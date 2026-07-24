"""Selection-aware within-arm periodic-policy generalization bootstrap."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from experiments.neurips_2026.allen_cahn_periodic_reencoding.pipeline_inference import (
    _full_h200_rows,
    _risk_cube,
    _select_indices,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.statistics import (
    ARMS,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    DIRECT,
    _integer,
)


def summarize_policy_pipeline_bootstrap(
    validation_rows: Sequence[Mapping[str, Any]],
    primary_test_rows: Sequence[Mapping[str, Any]],
    card: Mapping[str, Any],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    chunk_size: int = 2_000,
) -> dict[str, Any]:
    """Rerun each arm's selector and compare its held-out risk with direct."""

    validation, test, frozen = _full_h200_rows(
        validation_rows, primary_test_rows, card
    )
    validation_seed = _risk_cube(
        validation, frozen, dataset_key="validation_seeds"
    ).mean(axis=2)
    test_seed = _risk_cube(test, frozen, dataset_key="test_seeds").mean(axis=2)
    repetitions = _integer(bootstrap_replicates, name="bootstrap replicates")
    random_seed = _integer(bootstrap_seed, name="bootstrap seed")
    chunk = _integer(chunk_size, name="bootstrap chunk size")
    if repetitions <= 0 or random_seed < 0 or chunk <= 0:
        raise ValueError("Bootstrap count/chunk must be positive and seed nonnegative")
    direct_index = frozen["cadence_grid"].index(DIRECT)
    result: dict[str, Any] = {}
    for arm_index, arm in enumerate(ARMS):
        # Use the identical paired-seed resample stream for both arms.
        rng = np.random.default_rng(random_seed)
        point_index = int(
            _select_indices(
                validation_seed[arm_index].mean(axis=0), frozen["cadence_grid"]
            )
        )
        direct = test_seed[arm_index, :, direct_index]
        selected = test_seed[arm_index, :, point_index]
        if direct.mean() <= 0.0:
            raise ValueError("A direct H200 policy mean is nonpositive")
        reductions = np.empty(repetitions, dtype=np.float64)
        counts = np.zeros(len(frozen["cadence_grid"]), dtype=np.int64)
        written = 0
        while written < repetitions:
            size = min(chunk, repetitions - written)
            indices = rng.integers(0, 10, size=(size, 10))
            choices = _select_indices(
                validation_seed[arm_index][indices].mean(axis=1),
                frozen["cadence_grid"],
            )
            batch = np.arange(size)[:, None]
            selected_values = test_seed[arm_index][indices, choices[:, None]]
            direct_values = test_seed[arm_index][indices, direct_index]
            direct_means = direct_values.mean(axis=1)
            if np.any(direct_means <= 0.0):
                raise ValueError("A policy bootstrap direct mean is nonpositive")
            reductions[written:written + size] = (
                1.0 - selected_values.mean(axis=1) / direct_means
            )
            counts += np.bincount(choices, minlength=len(counts))
            written += size
        lower, upper = np.quantile(reductions, (0.025, 0.975))
        result[arm] = {
            "selected_cadence": frozen["cadence_grid"][point_index],
            "heldout_point_relative_reduction": float(
                1.0 - selected.mean() / direct.mean()
            ),
            "heldout_point_selected_seed_wins": int(np.sum(selected < direct)),
            "selection_aware_bootstrap_ci95_lower": float(lower),
            "selection_aware_bootstrap_ci95_upper": float(upper),
            "replicates": repetitions,
            "seed": random_seed,
            "shared_resample_stream_across_arms": True,
            "selection_frequencies": [
                {
                    "cadence": cadence,
                    "count": int(counts[index]),
                    "frequency": float(counts[index] / repetitions),
                }
                for index, cadence in enumerate(frozen["cadence_grid"])
            ],
        }
    return {
        "arms": result,
        "resampling_unit": "paired_model_seed_entire_validation_and_test_pipeline",
        "selector_rerun_for_every_replicate": True,
        "interpretation": (
            "The held-out point direction diagnoses cadence generalization; the "
            "selection-aware interval propagates selector instability."
        ),
    }
