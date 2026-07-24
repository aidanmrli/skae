"""Recipe-level cadence selection and paired Allen--Cahn forecast inference."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from experiments.neurips_2026.allen_cahn_periodic_reencoding.numeric_serialization import (
    json_safe_statistic,
)
ARMS = ("dense", "sparse")
DIRECT = "direct"
VALIDATION_HORIZON = 200
TEST_HORIZON = 400
DEFAULT_BOOTSTRAP_REPLICATES = 100_000
DEFAULT_BOOTSTRAP_SEED = 20_260_721

Cadence = str | int
PreparedRow = tuple[np.ndarray, np.ndarray]
Frozen = dict[str, tuple[Any, ...]]
RowKey = tuple[str, int, int, Cadence]
def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)
def _cadence(value: Any) -> Cadence:
    if value == DIRECT and isinstance(value, str):
        return DIRECT
    period = _integer(value, name="cadence")
    if period <= 0:
        raise ValueError("A periodic-reencoding cadence must be positive")
    return period
def _frozen_card(card: Mapping[str, Any]) -> Frozen:
    """Read the canonical nested card; flat keys support only unit fixtures."""

    try:
        roster = card.get("roster")
        if roster is None:
            model_values = card["model_seeds"]
        else:
            if tuple(roster["arms"]) != ARMS:
                raise ValueError("roster.arms must be exactly dense, sparse")
            model_values = roster["model_seeds"]
        cadence_values = card.get("cadence_grid")
        if cadence_values is None:
            cadence_values = card["cadence_selection"]["cadence_grid"]
        prospective = card.get("prospective_datasets")
        if prospective is None:
            validation_values, test_values = card["validation_seeds"], card["test_seeds"]
        else:
            validation_values = [item["seed"] for item in prospective["validation"]]
            test_values = [item["seed"] for item in prospective["test"]]
        grid = tuple(_cadence(value) for value in cadence_values)
        models = tuple(_integer(value, name="model_seed") for value in model_values)
        validation = tuple(
            _integer(value, name="validation_seed") for value in validation_values
        )
        test = tuple(_integer(value, name="test_seed") for value in test_values)
    except KeyError as error:
        raise ValueError(f"Card lacks frozen field {error.args[0]}") from error
    if not grid or len(set(grid)) != len(grid) or DIRECT not in grid:
        raise ValueError("cadence_grid must be unique, nonempty, and contain direct")
    if len(models) != 10 or len(set(models)) != 10:
        raise ValueError("model_seeds must contain exactly ten unique seeds")
    if len(validation) != 3 or len(set(validation)) != 3:
        raise ValueError("validation_seeds must contain exactly three unique seeds")
    if len(test) != 3 or len(set(test)) != 3:
        raise ValueError("test_seeds must contain exactly three unique seeds")
    if set(validation) & set(test):
        raise ValueError("Validation and test dataset seeds must be disjoint")
    return dict(cadence_grid=grid, model_seeds=models,
                validation_seeds=validation, test_seeds=test)
def _curve(row: Mapping[str, Any], name: str, *, horizon: int,
           allow_nonfinite: bool) -> np.ndarray:
    if name not in row:
        raise ValueError(f"A forecast row lacks {name}")
    try:
        values = np.asarray(row[name], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} cannot be converted to floating point") from error
    if values.shape != (horizon,):
        raise ValueError(f"{name} must have shape ({horizon},), got {values.shape}")
    finite = np.isfinite(values)
    if np.any(values[finite] < 0.0):
        raise ValueError(f"{name} contains a negative MSE")
    if not allow_nonfinite and not finite.all():
        raise FloatingPointError(f"{name} contains a nonfinite test value")
    return values
def _prepare_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_seeds: tuple[int, ...],
    dataset_seeds: tuple[int, ...],
    cadences: tuple[Cadence, ...],
    horizon: int,
    allow_nonfinite: bool,
) -> dict[RowKey, PreparedRow]:
    expected = {
        (arm, model_seed, dataset_seed, cadence)
        for arm in ARMS
        for model_seed in model_seeds
        for dataset_seed in dataset_seeds
        for cadence in cadences
    }
    prepared: dict[RowKey, PreparedRow] = {}
    required = {"arm", "model_seed", "dataset_seed", "cadence", "horizon_steps"}
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Forecast row {index} lacks fields {sorted(missing)}")
        arm = row["arm"]
        if arm not in ARMS:
            raise ValueError(f"Forecast row {index} has unknown arm {arm!r}")
        model_seed = _integer(row["model_seed"], name="model_seed")
        dataset_seed = _integer(row["dataset_seed"], name="dataset_seed")
        cadence = _cadence(row["cadence"])
        observed_horizon = _integer(row["horizon_steps"], name="horizon_steps")
        if observed_horizon != horizon:
            raise ValueError(f"Forecast row {index} has the wrong horizon")
        key = (arm, model_seed, dataset_seed, cadence)
        if key not in expected:
            raise ValueError(f"Forecast row {index} lies outside the frozen roster: {key}")
        if key in prepared:
            raise ValueError(f"Duplicate forecast row for frozen cell {key}")
        prepared[key] = (
            _curve(
                row,
                "instantaneous_field_mse",
                horizon=horizon,
                allow_nonfinite=allow_nonfinite,
            ),
            _curve(
                row,
                "cumulative_field_mse",
                horizon=horizon,
                allow_nonfinite=allow_nonfinite,
            ),
        )
    if set(prepared) != expected:
        missing, extra = expected - set(prepared), set(prepared) - expected
        raise ValueError(
            "Forecast rows do not form the exact frozen cross: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    return prepared
def _validation_rows(rows: Sequence[Mapping[str, Any]], card: Mapping[str, Any]
                     ) -> tuple[dict[RowKey, PreparedRow], Frozen]:
    frozen = _frozen_card(card)
    prepared = _prepare_rows(
        rows,
        model_seeds=frozen["model_seeds"],
        dataset_seeds=frozen["validation_seeds"],
        cadences=frozen["cadence_grid"],
        horizon=VALIDATION_HORIZON,
        allow_nonfinite=False,
    )
    return prepared, frozen
def validation_candidate_scores(rows: Sequence[Mapping[str, Any]],
                                card: Mapping[str, Any]
                                ) -> dict[str, list[dict[str, Any]]]:
    """Score every arm/cadence recipe on the balanced 3x10 validation cross."""

    prepared, frozen = _validation_rows(rows, card)
    report: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for cadence in frozen["cadence_grid"]:
            cells = [
                prepared[(arm, model, dataset, cadence)]
                for model in frozen["model_seeds"]
                for dataset in frozen["validation_seeds"]
            ]
            per_seed = [
                np.mean(
                    [
                        prepared[(arm, model, dataset, cadence)][1][-1]
                        for dataset in frozen["validation_seeds"]
                    ]
                )
                for model in frozen["model_seeds"]
            ]
            score = float(np.mean(per_seed))
            if not np.isfinite(score):
                raise FloatingPointError("A validation score is nonfinite")
            report[arm].append({
                "cadence": cadence, "eligible": True,
                "h200_cumulative_field_mse": score,
                "nonfinite_value_count": 0, "rows": 30,
            })
    return report
def _tie_key(cadence: Cadence) -> tuple[int, int]:
    return (0, 0) if cadence == DIRECT else (1, -int(cadence))

def select_recipe_cadences(rows: Sequence[Mapping[str, Any]],
                           card: Mapping[str, Any]) -> dict[str, Cadence]:
    """Return one validation-selected cadence for each full model recipe."""

    scores, selected = validation_candidate_scores(rows, card), {}
    for arm in ARMS:
        eligible = scores[arm]
        minimum = min(float(item["h200_cumulative_field_mse"]) for item in eligible)
        tied = [
            item
            for item in eligible
            if float(item["h200_cumulative_field_mse"]) == minimum
        ]
        selected[arm] = min(tied, key=lambda item: _tie_key(item["cadence"]))["cadence"]
    return selected

def _selected_cadences(selected: Mapping[str, Any], frozen: Frozen) -> dict[str, Cadence]:
    if set(selected) != set(ARMS):
        raise ValueError("Selected cadences must contain exactly dense and sparse")
    result = {arm: _cadence(selected[arm]) for arm in ARMS}
    if any(value not in frozen["cadence_grid"] for value in result.values()):
        raise ValueError("A selected cadence is outside cadence_grid")
    return result

def _test_rows(
    rows: Sequence[Mapping[str, Any]],
    card: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> tuple[dict[RowKey, PreparedRow], Frozen, dict[str, Cadence], tuple[Cadence, ...]]:
    frozen = _frozen_card(card)
    choices = _selected_cadences(selected, frozen)
    needed = {DIRECT, choices["dense"], choices["sparse"]}
    cadences = tuple(value for value in frozen["cadence_grid"] if value in needed)
    prepared = _prepare_rows(
        rows,
        model_seeds=frozen["model_seeds"],
        dataset_seeds=frozen["test_seeds"],
        cadences=cadences,
        horizon=TEST_HORIZON,
        allow_nonfinite=False,
    )
    return prepared, frozen, choices, cadences

def validate_validation_rows(rows: Sequence[Mapping[str, Any]],
                             card: Mapping[str, Any]) -> None:
    _validation_rows(rows, card)

def validate_primary_test_rows(rows: Sequence[Mapping[str, Any]],
                               card: Mapping[str, Any]) -> None:
    """Require the complete finite sealed H200 arm-by-cadence cross."""

    frozen = _frozen_card(card)
    _prepare_rows(
        rows,
        model_seeds=frozen["model_seeds"],
        dataset_seeds=frozen["test_seeds"],
        cadences=frozen["cadence_grid"],
        horizon=VALIDATION_HORIZON,
        allow_nonfinite=False,
    )

def validate_test_rows(
    rows: Sequence[Mapping[str, Any]],
    card: Mapping[str, Any],
    selected_cadences: Mapping[str, Any],
) -> None:
    _test_rows(rows, card, selected_cadences)

def _studentized(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=-1)
    scales = values.std(axis=-1, ddof=1) / np.sqrt(values.shape[-1])
    result = np.zeros_like(means, dtype=np.float64)
    np.divide(means, scales, out=result, where=scales > 0)
    result[(scales == 0) & (means > 0)] = np.inf
    result[(scales == 0) & (means < 0)] = -np.inf
    return result

def exact_one_sided_studentized_sign_flip(
    dense_minus_sparse: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(dense_minus_sparse, dtype=np.float64)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise ValueError("Exact sign-flip inference requires ten finite paired effects")
    observed = float(_studentized(values[None, :])[0])
    integers = np.arange(2**10, dtype=np.uint16)[:, None]
    bits = (integers >> np.arange(10, dtype=np.uint16)) & 1
    signs = 2.0 * bits.astype(np.float64) - 1.0
    permuted = _studentized(signs * values[None, :])
    return {
        **json_safe_statistic(observed, name="observed_studentized_statistic"),
        "one_sided_exact_p": float(np.mean(permuted >= observed)),
        "enumerated_sign_vectors": 1024,
        "alternative": "mean_dense_minus_sparse_is_positive",
        "comparison": "T_perm >= T_observed_literal_no_tolerance",
        "null_and_assumption": "sharp paired sign-symmetry null across ten seeds",
        "enumeration_exactness_boundary": (
            "all 2^10 signs enumerated; calibration is not assumption-free"
        ),
    }

def paired_ratio_bootstrap(
    dense: Sequence[float],
    sparse: Sequence[float],
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    dense_values = np.asarray(dense, dtype=np.float64)
    sparse_values = np.asarray(sparse, dtype=np.float64)
    valid = (
        dense_values.shape == (10,)
        and sparse_values.shape == (10,)
        and np.isfinite(dense_values).all()
        and np.isfinite(sparse_values).all()
    )
    if not valid:
        raise ValueError("Paired bootstrap requires two finite ten-seed arrays")
    repetitions = _integer(replicates, name="bootstrap replicates")
    random_seed = _integer(seed, name="bootstrap seed")
    if repetitions <= 0 or random_seed < 0:
        raise ValueError("Bootstrap replicates must be positive and seed nonnegative")
    if dense_values.mean() <= 0.0:
        raise ValueError("Dense arm mean must be positive for a ratio reduction")
    indices = np.random.default_rng(random_seed).integers(0, 10, (repetitions, 10))
    dense_means = dense_values[indices].mean(axis=1)
    sparse_means = sparse_values[indices].mean(axis=1)
    if np.any(dense_means <= 0.0):
        raise ValueError("A paired bootstrap draw has a nonpositive dense mean")
    samples = 1.0 - sparse_means / dense_means
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return {
        "relative_reduction_of_arm_means": float(
            1.0 - sparse_values.mean() / dense_values.mean()
        ),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "replicates": repetitions,
        "seed": random_seed,
        "resampling_unit": "paired_model_seed_after_three_dataset_average",
    }

ENDPOINTS: tuple[tuple[str, Callable[[PreparedRow], float]], ...] = (
    ("h200_cumulative_field_mse", lambda row: float(row[1][199])),
    ("h400_cumulative_field_mse", lambda row: float(row[1][399])),
    ("h201_h400_tail_field_mse", lambda row: float(np.mean(row[0][200:400]))),
    ("h400_terminal_field_mse", lambda row: float(row[0][399])),
)

def _paired_seed_values(
    prepared: Mapping[RowKey, PreparedRow],
    frozen: Frozen,
    cadences: Mapping[str, Cadence],
    endpoint: Callable[[PreparedRow], float],
) -> dict[str, np.ndarray]:
    result: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for model in frozen["model_seeds"]:
            values = [
                endpoint(prepared[(arm, model, dataset, cadences[arm])])
                for dataset in frozen["test_seeds"]
            ]
            result[arm].append(float(np.mean(values)))
    arrays = {arm: np.asarray(values, dtype=np.float64) for arm, values in result.items()}
    if any(a.shape != (10,) or not np.isfinite(a).all() for a in arrays.values()):
        raise FloatingPointError("Endpoint aggregation is incomplete or nonfinite")
    return arrays

def _per_dataset_effects(
    prepared: Mapping[RowKey, PreparedRow],
    frozen: Frozen,
    cadences: Mapping[str, Cadence],
    endpoint: Callable[[PreparedRow], float],
) -> list[dict[str, Any]]:
    result = []
    for dataset in frozen["test_seeds"]:
        values = {
            arm: np.asarray(
                [
                    endpoint(prepared[(arm, model, dataset, cadences[arm])])
                    for model in frozen["model_seeds"]
                ]
            )
            for arm in ARMS
        }
        dense, sparse = values["dense"], values["sparse"]
        if not np.isfinite(dense).all() or not np.isfinite(sparse).all():
            raise FloatingPointError("A per-dataset endpoint is nonfinite")
        dense_mean, sparse_mean = float(dense.mean()), float(sparse.mean())
        if dense_mean <= 0.0:
            raise ValueError("A per-dataset dense mean is nonpositive")
        result.append(
            {
                "dataset_seed": int(dataset),
                "dense_mean": dense_mean,
                "sparse_mean": sparse_mean,
                "sparse_over_dense_ratio_of_arm_means": sparse_mean / dense_mean,
                "relative_reduction_of_arm_means": 1.0 - sparse_mean / dense_mean,
                "sparse_seed_wins": int(np.sum(sparse < dense)),
            }
        )
    return result

def _endpoint_summary(
    prepared: Mapping[RowKey, PreparedRow],
    frozen: Frozen,
    cadences: Mapping[str, Cadence],
    endpoint: Callable[[PreparedRow], float],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    values = _paired_seed_values(prepared, frozen, cadences, endpoint)
    dense, sparse = values["dense"], values["sparse"]
    dense_mean, sparse_mean = float(dense.mean()), float(sparse.mean())
    if dense_mean <= 0.0:
        raise ValueError("Dense arm mean is nonpositive")
    return {
        "dense_paired_seed_values": dense.tolist(),
        "sparse_paired_seed_values": sparse.tolist(),
        "dense_mean": dense_mean,
        "sparse_mean": sparse_mean,
        "sparse_over_dense_ratio_of_arm_means": sparse_mean / dense_mean,
        "relative_reduction_of_arm_means": 1.0 - sparse_mean / dense_mean,
        "sparse_seed_wins": int(np.sum(sparse < dense)),
        "exact_one_sided_studentized_sign_flip":
            exact_one_sided_studentized_sign_flip(dense - sparse),
        "paired_ratio_bootstrap": paired_ratio_bootstrap(
            dense, sparse, replicates=bootstrap_replicates, seed=bootstrap_seed
        ),
        "per_dataset_effects": _per_dataset_effects(prepared, frozen, cadences,
                                                     endpoint),
    }

def _comparison_endpoints(
    prepared: Mapping[RowKey, PreparedRow],
    frozen: Frozen,
    cadences: Mapping[str, Cadence],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    summaries = {}
    for offset, (name, endpoint) in enumerate(ENDPOINTS):
        summaries[name] = _endpoint_summary(
            prepared,
            frozen,
            cadences,
            endpoint,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + offset,
        )
        summaries[name]["inference_role"] = (
            "confirmatory_primary"
            if name == "h200_cumulative_field_mse"
            else "secondary_durability_diagnostic"
        )
    return {**{f"{arm}_cadence": cadences[arm] for arm in ARMS}, "endpoints": summaries}

def summarize_test_rows(
    rows: Sequence[Mapping[str, Any]],
    card: Mapping[str, Any],
    selected_cadences: Mapping[str, Any],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Reduce the held-out test cross using validation-frozen arm recipes."""

    prepared, frozen, selected, tested = _test_rows(rows, card, selected_cadences)
    selected_summary = _comparison_endpoints(
        prepared,
        frozen,
        selected,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    sensitivity, reported = {}, {(selected["dense"], selected["sparse"])}
    for index, cadence in enumerate(tested, start=1):
        if (cadence, cadence) in reported:
            continue
        label = DIRECT if cadence == DIRECT else f"period_{cadence}"
        sensitivity[label] = _comparison_endpoints(
            prepared,
            frozen,
            {arm: cadence for arm in ARMS},
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + 100 * index,
        )
        reported.add((cadence, cadence))
    return {
        "selected_cadences_from_validation": selected,
        "tested_cadences_equal_for_both_arms": list(tested),
        "selected_recipe_comparison": selected_summary,
        "same_cadence_sensitivity": sensitivity,
        "primary_endpoint": "h200_cumulative_field_mse",
        "aggregation_order": (
            "average_three_test_datasets_within_model_seed_then_compare_ten_"
            "paired_model_seeds"
        ),
        "selection_policy": (
            "one_recipe_level_cadence_per_arm_selected_only_on_validation;_"
            "no_per_seed_or_test_selection"
        ),
        "multiplicity_policy": (
            "only_selected_recipe_h200_cumulative_is_confirmatory;_other_"
            "endpoints_and_same_cadence_comparisons_are_diagnostics"
        ),
    }
