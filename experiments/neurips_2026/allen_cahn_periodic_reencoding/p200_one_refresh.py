"""Optional Allen--Cahn p=200 one-refresh-at-T=20 diagnostic.

Period 200 is prediction-identical to direct rollout through H200, so it is
never a validation-selection candidate.  Its only purpose is to isolate the
effect of one decoded-prediction refresh at the trained-horizon boundary on
the subsequent H201--H400 forecast.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from experiments.neurips_2026.allen_cahn_periodic_reencoding.policy_statistics import (
    _selected_vs_direct_endpoint,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.statistics import (
    ARMS,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    DIRECT,
    ENDPOINTS,
    TEST_HORIZON,
    VALIDATION_HORIZON,
    Frozen,
    PreparedRow,
    RowKey,
    _endpoint_summary,
    _frozen_card,
    _integer,
    _prepare_rows,
)


P200 = 200
PREFIX_RTOL = 1e-12
PREFIX_ATOL = 1e-14
P200_ENDPOINT_NAMES = (
    "h400_cumulative_field_mse",
    "h201_h400_tail_field_mse",
    "h400_terminal_field_mse",
)
P200_ENDPOINTS = tuple(
    (name, endpoint)
    for name, endpoint in ENDPOINTS
    if name in P200_ENDPOINT_NAMES
)


def _protocol(card: Mapping[str, Any]) -> tuple[Frozen, float]:
    frozen = _frozen_card(card)
    if P200 in frozen["cadence_grid"]:
        raise ValueError("p200 must remain outside the validation cadence grid")
    try:
        system = card["system"]
        stored_dt = float(system["stored_dt"])
        validation_horizon = _integer(
            system["validation_horizon_steps"],
            name="validation_horizon_steps",
        )
        test_horizon = _integer(
            system["test_horizon_steps"],
            name="test_horizon_steps",
        )
    except KeyError as error:
        raise ValueError(f"Card lacks p200 protocol field {error.args[0]}") from error
    if not np.isfinite(stored_dt) or stored_dt <= 0.0:
        raise ValueError("stored_dt must be finite and positive")
    if validation_horizon != VALIDATION_HORIZON or test_horizon != TEST_HORIZON:
        raise ValueError("Card horizons disagree with the p200 diagnostic")
    refresh_time = P200 * stored_dt
    if not np.isclose(refresh_time, 20.0, rtol=0.0, atol=1e-12):
        raise ValueError("p200 must be the frozen physical-time T=20 boundary")
    return frozen, refresh_time


def _prepare_exact(
    rows: Sequence[Mapping[str, Any]],
    frozen: Frozen,
    *,
    cadence: str | int,
    horizon: int,
) -> dict[RowKey, PreparedRow]:
    return _prepare_rows(
        rows,
        model_seeds=frozen["model_seeds"],
        dataset_seeds=frozen["test_seeds"],
        cadences=(cadence,),
        horizon=horizon,
        allow_nonfinite=False,
    )


def _require_direct_prefix(
    long_rows: Mapping[RowKey, PreparedRow],
    h200_direct: Mapping[RowKey, PreparedRow],
    frozen: Frozen,
    *,
    long_cadence: str | int,
    label: str,
) -> None:
    for arm in ARMS:
        for model_seed in frozen["model_seeds"]:
            for dataset_seed in frozen["test_seeds"]:
                long_row = long_rows[(arm, model_seed, dataset_seed, long_cadence)]
                short_row = h200_direct[(arm, model_seed, dataset_seed, DIRECT)]
                for curve_index, curve_name in enumerate(
                    ("instantaneous_field_mse", "cumulative_field_mse")
                ):
                    try:
                        np.testing.assert_allclose(
                            long_row[curve_index][:VALIDATION_HORIZON],
                            short_row[curve_index],
                            rtol=PREFIX_RTOL,
                            atol=PREFIX_ATOL,
                        )
                    except AssertionError as error:
                        key = (arm, model_seed, dataset_seed)
                        raise AssertionError(
                            f"{label} {curve_name} H200 prefix mismatch for {key}"
                        ) from error


def _validated_p200(
    p200_h400_rows: Sequence[Mapping[str, Any]],
    h200_direct_rows: Sequence[Mapping[str, Any]],
    card: Mapping[str, Any],
) -> tuple[dict[RowKey, PreparedRow], dict[RowKey, PreparedRow], Frozen, float]:
    frozen, refresh_time = _protocol(card)
    p200 = _prepare_exact(
        p200_h400_rows,
        frozen,
        cadence=P200,
        horizon=TEST_HORIZON,
    )
    h200_direct = _prepare_exact(
        h200_direct_rows,
        frozen,
        cadence=DIRECT,
        horizon=VALIDATION_HORIZON,
    )
    _require_direct_prefix(
        p200,
        h200_direct,
        frozen,
        long_cadence=P200,
        label="p200",
    )
    return p200, h200_direct, frozen, refresh_time


def validate_p200_h400_rows(
    p200_h400_rows: Sequence[Mapping[str, Any]],
    h200_direct_rows: Sequence[Mapping[str, Any]],
    card: Mapping[str, Any],
) -> None:
    """Require the exact finite 2x10x3 p200 cross and direct H200 prefix."""

    _validated_p200(p200_h400_rows, h200_direct_rows, card)


def _summaries(
    p200: Mapping[RowKey, PreparedRow],
    direct: Mapping[RowKey, PreparedRow],
    frozen: Frozen,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    combined = {**direct, **p200}
    fixed: dict[str, Any] = {}
    within = {arm: {"baseline_cadence": DIRECT, "candidate_cadence": P200,
                    "endpoints": {}} for arm in ARMS}
    for endpoint_index, (name, endpoint) in enumerate(P200_ENDPOINTS):
        fixed[name] = _endpoint_summary(
            p200,
            frozen,
            {arm: P200 for arm in ARMS},
            endpoint,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + endpoint_index,
        )
        fixed[name]["inference_role"] = (
            "optional_descriptive_fixed_p200_sparse_vs_dense"
        )
        for arm_index, arm in enumerate(ARMS):
            result = _selected_vs_direct_endpoint(
                combined,
                frozen,
                arm=arm,
                selected_cadence=P200,
                endpoint=endpoint,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=(
                    bootstrap_seed + 100 + 10 * arm_index + endpoint_index
                ),
            )
            result["inference_role"] = (
                "optional_descriptive_within_arm_p200_vs_direct"
            )
            within[arm]["endpoints"][name] = result
    return fixed, within


def summarize_p200_one_refresh(
    p200_h400_rows: Sequence[Mapping[str, Any]],
    direct_h400_rows: Sequence[Mapping[str, Any]],
    h200_direct_rows: Sequence[Mapping[str, Any]],
    card: Mapping[str, Any],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Summarize the optional one-refresh diagnostic after strict validation."""

    repetitions = _integer(bootstrap_replicates, name="bootstrap replicates")
    random_seed = _integer(bootstrap_seed, name="bootstrap seed")
    if repetitions <= 0 or random_seed < 0:
        raise ValueError("Bootstrap count must be positive and seed nonnegative")
    p200, h200_direct, frozen, refresh_time = _validated_p200(
        p200_h400_rows,
        h200_direct_rows,
        card,
    )
    direct = _prepare_exact(
        direct_h400_rows,
        frozen,
        cadence=DIRECT,
        horizon=TEST_HORIZON,
    )
    _require_direct_prefix(
        direct,
        h200_direct,
        frozen,
        long_cadence=DIRECT,
        label="H400 direct",
    )
    fixed, within = _summaries(
        p200,
        direct,
        frozen,
        bootstrap_replicates=repetitions,
        bootstrap_seed=random_seed,
    )
    return {
        "status": "complete",
        "diagnostic": "p200_one_refresh_at_trained_horizon",
        "cadence": P200,
        "forecast_horizon_steps": TEST_HORIZON,
        "refresh_boundary_step": P200,
        "refresh_physical_time": refresh_time,
        "refresh_count": 1,
        "encoder_calls": 2,
        "prefix_integrity": {
            "p200_matches_independent_h200_direct": True,
            "h400_direct_matches_independent_h200_direct": True,
            "curves": ["instantaneous_field_mse", "cumulative_field_mse"],
            "steps": VALIDATION_HORIZON,
            "rtol": PREFIX_RTOL,
            "atol": PREFIX_ATOL,
        },
        "fixed_p200_sparse_vs_dense": {"endpoints": fixed},
        "within_arm_p200_vs_direct": within,
        "endpoint_names": list(P200_ENDPOINT_NAMES),
        "inference_role": "optional_descriptive_one_refresh_diagnostic",
        "validation_selection_eligible": False,
        "uses_validation_or_test_outcomes_for_selection": False,
        "can_rescue_h200_primary": False,
        "can_rescue_full_grid_h400": False,
        "failure_policy": "absence_or_failure_suppresses_only_this_diagnostic",
    }


def _suppressed(reason: str, error: Exception | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "suppressed",
        "diagnostic": "p200_one_refresh_at_trained_horizon",
        "reason": reason,
        "scope": "optional_p200_diagnostic_only",
        "invalidates_h200_primary": False,
        "invalidates_fixed_or_full_grid_h400": False,
        "can_rescue_h200_primary": False,
        "can_rescue_full_grid_h400": False,
    }
    if error is not None:
        result.update(error_type=type(error).__name__, error=str(error))
    return result


def summarize_optional_p200_one_refresh(
    p200_h400_rows: Sequence[Mapping[str, Any]] | None,
    direct_h400_rows: Sequence[Mapping[str, Any]] | None,
    h200_direct_rows: Sequence[Mapping[str, Any]] | None,
    card: Mapping[str, Any],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Suppress only this optional tier when its rows are absent or invalid."""

    _protocol(card)
    repetitions = _integer(bootstrap_replicates, name="bootstrap replicates")
    random_seed = _integer(bootstrap_seed, name="bootstrap seed")
    if repetitions <= 0 or random_seed < 0:
        raise ValueError("Bootstrap count must be positive and seed nonnegative")
    if p200_h400_rows is None or direct_h400_rows is None or h200_direct_rows is None:
        return _suppressed("required_optional_rows_absent")
    try:
        return summarize_p200_one_refresh(
            p200_h400_rows,
            direct_h400_rows,
            h200_direct_rows,
            card,
            bootstrap_replicates=repetitions,
            bootstrap_seed=random_seed,
        )
    except (AssertionError, FloatingPointError, ValueError) as error:
        return _suppressed("optional_rows_failed_strict_validation", error)
