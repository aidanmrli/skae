"""Fixed same-cadence H200 diagnostics for the sealed cadence grid."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from experiments.neurips_2026.allen_cahn_periodic_reencoding.forecast_skill import (
    deployment_cost,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.pipeline_inference import (
    _risk_cube,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.statistics import (
    ARMS,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    VALIDATION_HORIZON,
    _frozen_card,
    _prepare_rows,
    exact_one_sided_studentized_sign_flip,
    paired_ratio_bootstrap,
)


def summarize_same_cadence_h200(
    rows: Sequence[Mapping[str, Any]],
    card: Mapping[str, Any],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Report every fixed sparse-versus-dense cadence as a diagnostic."""

    frozen = _frozen_card(card)
    prepared = _prepare_rows(
        rows,
        model_seeds=frozen["model_seeds"],
        dataset_seeds=frozen["test_seeds"],
        cadences=frozen["cadence_grid"],
        horizon=VALIDATION_HORIZON,
        allow_nonfinite=False,
    )
    cube = _risk_cube(prepared, frozen, dataset_key="test_seeds")
    result: dict[str, Any] = {}
    for cadence_index, cadence in enumerate(frozen["cadence_grid"]):
        seed_risks = cube[:, :, :, cadence_index].mean(axis=2)
        dense, sparse = seed_risks[0], seed_risks[1]
        dense_mean, sparse_mean = float(dense.mean()), float(sparse.mean())
        if dense_mean <= 0.0:
            raise ValueError("A fixed-cadence dense H200 mean is nonpositive")
        dataset_effects = []
        for dataset_index, dataset_seed in enumerate(frozen["test_seeds"]):
            dense_dataset = cube[0, :, dataset_index, cadence_index]
            sparse_dataset = cube[1, :, dataset_index, cadence_index]
            baseline = float(dense_dataset.mean())
            if baseline <= 0.0:
                raise ValueError("A fixed-cadence dataset baseline is nonpositive")
            dataset_effects.append(
                {
                    "dataset_seed": int(dataset_seed),
                    "dense_mean": baseline,
                    "sparse_mean": float(sparse_dataset.mean()),
                    "relative_reduction_of_arm_means": float(
                        1.0 - sparse_dataset.mean() / baseline
                    ),
                    "sparse_seed_wins": int(np.sum(sparse_dataset < dense_dataset)),
                }
            )
        label = "direct" if cadence == "direct" else f"period_{cadence}"
        result[label] = {
            "cadence": cadence,
            **deployment_cost(cadence, VALIDATION_HORIZON),
            "dense_paired_seed_values": dense.tolist(),
            "sparse_paired_seed_values": sparse.tolist(),
            "dense_mean": dense_mean,
            "sparse_mean": sparse_mean,
            "relative_reduction_of_arm_means": 1.0 - sparse_mean / dense_mean,
            "sparse_seed_wins": int(np.sum(sparse < dense)),
            "exact_one_sided_studentized_sign_flip": (
                exact_one_sided_studentized_sign_flip(dense - sparse)
            ),
            "paired_ratio_bootstrap": paired_ratio_bootstrap(
                dense,
                sparse,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + cadence_index,
            ),
            "per_dataset_effects": dataset_effects,
            "inference_role": "mandatory_descriptive_sensitivity",
        }
    return result
