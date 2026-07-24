"""Authenticate, reduce, and check full-horizon Allen--Cahn curve evidence."""

from __future__ import annotations

import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.neurips_2026.evidence.allen_cahn_forecast_replication_curve_rendering import (
    render_full_horizon_figure,
)
from experiments.neurips_2026.evidence.allen_cahn_forecast_replication_curve_validation import (
    validate_curve_compact,
)
from experiments.neurips_2026.paths import PAPER_DATA_DIR, PAPER_EVIDENCE_DIR


PACKET_ID = "allen_cahn_new_ic_replication"
MODEL_SEEDS = tuple(range(64, 74))
DATASET_SEEDS = (1_775_404_171, 74_732_421, 293_789_188)
CURVE_SUMMARY = PAPER_DATA_DIR / f"{PACKET_ID}_curve_summary.json"
CURVE_ROWS = PAPER_DATA_DIR / f"{PACKET_ID}_curve_rows.csv"
FULL_HORIZON_FIGURE_PDF = PAPER_EVIDENCE_DIR / f"fig_{PACKET_ID}_full_horizon.pdf"
FULL_HORIZON_FIGURE_PNG = PAPER_EVIDENCE_DIR / f"fig_{PACKET_ID}_full_horizon.png"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicates(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(), object_pairs_hook=_reject_duplicates)
    _require(isinstance(payload, dict), f"Expected JSON object: {path}")
    return payload


def curve_output_paths(data_dir: Path, figure_dir: Path) -> dict[str, Path]:
    return {
        "curve_summary": data_dir / CURVE_SUMMARY.name,
        "curve_rows": data_dir / CURVE_ROWS.name,
        "full_horizon_figure_pdf": figure_dir / FULL_HORIZON_FIGURE_PDF.name,
        "full_horizon_figure_png": figure_dir / FULL_HORIZON_FIGURE_PNG.name,
    }


def curve_source_hashes() -> dict[str, str]:
    return {
        "curve_builder_sha256": _sha256(Path(__file__)),
        "curve_rendering_sha256": _sha256(
            Path(render_full_horizon_figure.__code__.co_filename)
        ),
        "curve_validation_sha256": _sha256(
            Path(validate_curve_compact.__code__.co_filename)
        ),
    }


def _validate_scientific_payload(
    payload: Mapping[str, Any],
    receipt: Mapping[str, Any],
    card: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _require(payload.get("schema_version") == 1, "Scientific curve schema drifted")
    _require(payload.get("protocol_id") == card.get("protocol_id"), "Curve protocol drifted")
    _require(payload.get("card_sha256") == receipt.get("card_sha256"), "Curve/card link drifted")
    _require(
        payload.get("source_manifest_sha256") == receipt.get("source_manifest_sha256"),
        "Curve/source link drifted",
    )
    _require(
        payload.get("checkpoint_roster_sha256") == receipt.get("checkpoint_roster_sha256"),
        "Curve/checkpoint-roster link drifted",
    )
    _require(payload.get("crossed_cells") == 60, "Curve roster is not 60 cells")
    rows = list(payload.get("rows", []))
    expected = {
        (arm, seed, dataset)
        for arm in ("dense", "sparse")
        for seed in MODEL_SEEDS
        for dataset in DATASET_SEEDS
    }
    actual = {
        (str(row["arm"]), int(row["model_seed"]), int(row["dataset_seed"]))
        for row in rows
    }
    _require(len(rows) == len(actual) == 60 and actual == expected, "Curve cells are incomplete")
    roster = {
        (str(row["arm"]), int(row["seed"])): row
        for row in receipt["checkpoint_roster"]
    }
    dataset_order = {seed: index for index, seed in enumerate(DATASET_SEEDS)}
    horizon = np.arange(1, 201, dtype=np.float64)
    for row in rows:
        key = (str(row["arm"]), int(row["model_seed"]))
        spec = roster[key]
        _require(row["checkpoint_sha256"] == spec["sha256"], "Curve checkpoint hash drifted")
        _require(int(row["checkpoint_step"]) == int(spec["checkpoint_step"]), "Curve checkpoint step drifted")
        seed = int(row["dataset_seed"])
        _require(int(row["dataset_index"]) == dataset_order[seed], "Curve dataset index drifted")
        instantaneous = np.asarray(row["instantaneous_field_mse"], dtype=np.float64)
        cumulative = np.asarray(row["cumulative_field_mse"], dtype=np.float64)
        _require(
            instantaneous.shape == cumulative.shape == (200,),
            "Curve does not contain all 200 horizons",
        )
        _require(np.isfinite(instantaneous).all() and np.isfinite(cumulative).all(), "Curve is nonfinite")
        np.testing.assert_allclose(
            cumulative,
            np.cumsum(instantaneous) / horizon,
            rtol=0,
            atol=1e-15,
        )
    return rows


def _seed_curves(
    rows: Sequence[Mapping[str, Any]],
    source_field: str,
) -> dict[str, np.ndarray]:
    lookup = {
        (str(row["arm"]), int(row["model_seed"]), int(row["dataset_seed"])): row
        for row in rows
    }
    return {
        arm: np.asarray(
            [
                np.mean(
                    [lookup[(arm, seed, dataset)][source_field] for dataset in DATASET_SEEDS],
                    axis=0,
                )
                for seed in MODEL_SEEDS
            ],
            dtype=np.float64,
        )
        for arm in ("dense", "sparse")
    }


def _pointwise_bootstrap(
    dense: np.ndarray,
    sparse: np.ndarray,
    *,
    replicates: int = 50_000,
    seed: int = 20_260_721,
    chunk_size: int = 1_000,
) -> dict[str, np.ndarray]:
    _require(dense.shape == sparse.shape == (10, 200), "Bootstrap curve shape drifted")
    dense_samples = np.empty((replicates, 200), dtype=np.float64)
    sparse_samples = np.empty_like(dense_samples)
    reduction_samples = np.empty_like(dense_samples)
    generator = np.random.default_rng(seed)
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = generator.integers(0, 10, size=(stop - start, 10))
        dense_chunk = dense[indices].mean(axis=1)
        sparse_chunk = sparse[indices].mean(axis=1)
        _require(np.all(dense_chunk > 0), "Bootstrap has a nonpositive dense mean")
        dense_samples[start:stop] = dense_chunk
        sparse_samples[start:stop] = sparse_chunk
        reduction_samples[start:stop] = 1.0 - sparse_chunk / dense_chunk
    return {
        "dense": np.quantile(dense_samples, (0.025, 0.975), axis=0),
        "sparse": np.quantile(sparse_samples, (0.025, 0.975), axis=0),
        "reduction": np.quantile(reduction_samples, (0.025, 0.975), axis=0),
    }


def _compare_decision_curves(
    decision: Mapping[str, Any],
    seed_curves: Mapping[str, np.ndarray],
    bands: Mapping[str, np.ndarray],
) -> None:
    descriptive = decision["descriptive_curves"]
    _require(descriptive["horizons"] == list(range(1, 201)), "Decision horizons drifted")
    recorded_seed_curves = descriptive["paired_model_seed_curves_after_three_dataset_average"]
    for arm in ("dense", "sparse"):
        np.testing.assert_allclose(
            seed_curves[arm],
            recorded_seed_curves[arm]["cumulative_field_mse"],
            rtol=0,
            atol=1e-15,
        )
    recorded = descriptive["pointwise_paired_seed_bootstrap"]
    _require(recorded["replicates"] == 50_000, "Pointwise replicate count drifted")
    _require(recorded["seed"] == 20_260_721, "Pointwise bootstrap seed drifted")
    _require(recorded["chunk_size"] == 1_000, "Pointwise bootstrap chunking drifted")
    names = {
        "dense": "dense_cumulative_field_mse",
        "sparse": "sparse_cumulative_field_mse",
        "reduction": "cumulative_relative_reduction",
    }
    for key, name in names.items():
        expected = np.asarray(
            [recorded[name]["lower"], recorded[name]["upper"]],
            dtype=np.float64,
        )
        np.testing.assert_allclose(bands[key], expected, rtol=0, atol=1e-15)


def _curve_rows(
    through_horizon_seed_curves: Mapping[str, np.ndarray],
    instantaneous_seed_curves: Mapping[str, np.ndarray],
    bands: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    dense = through_horizon_seed_curves["dense"].mean(axis=0)
    sparse = through_horizon_seed_curves["sparse"].mean(axis=0)
    dense_instantaneous = instantaneous_seed_curves["dense"].mean(axis=0)
    sparse_instantaneous = instantaneous_seed_curves["sparse"].mean(axis=0)
    reduction = 1.0 - sparse / dense
    return [
        {
            "horizon_step": index + 1,
            "physical_time": 0.1 * (index + 1),
            "dense_mean_through_horizon_mean_field_mse": dense[index],
            "dense_mean_instantaneous_field_mse": dense_instantaneous[index],
            "dense_pointwise_ci95_lower": bands["dense"][0, index],
            "dense_pointwise_ci95_upper": bands["dense"][1, index],
            "sparse_mean_through_horizon_mean_field_mse": sparse[index],
            "sparse_mean_instantaneous_field_mse": sparse_instantaneous[index],
            "sparse_pointwise_ci95_lower": bands["sparse"][0, index],
            "sparse_pointwise_ci95_upper": bands["sparse"][1, index],
            "relative_reduction_of_arm_means": reduction[index],
            "relative_reduction_pointwise_ci95_lower": bands["reduction"][0, index],
            "relative_reduction_pointwise_ci95_upper": bands["reduction"][1, index],
        }
        for index in range(200)
    ]


def _full_trace_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    time = np.asarray([float(row["physical_time"]) for row in rows])
    dense = np.asarray(
        [float(row["dense_mean_instantaneous_field_mse"]) for row in rows]
    )
    sparse = np.asarray(
        [float(row["sparse_mean_instantaneous_field_mse"]) for row in rows]
    )
    signed_gap = dense - sparse
    total = float(signed_gap.sum())
    after_t2 = float(signed_gap[time > 2.0].sum())
    _require(total > 0.0, "Signed full-horizon sparse gain is not positive")
    return {
        "role": "descriptive_no_simultaneous_or_curve_wide_inference",
        "instantaneous_arm_mean_definition": (
            "average three datasets within model seed, then average ten model seeds at each time"
        ),
        "evaluated_time_count": int(time.size),
        "sparse_arm_mean_lower_count": int(np.count_nonzero(signed_gap > 0.0)),
        "sparse_arm_mean_lower_at_all_evaluated_times": bool(np.all(signed_gap > 0.0)),
        "signed_gap_definition": "dense minus sparse instantaneous arm-mean field MSE",
        "signed_gap_sum_steps_1_through_200": total,
        "signed_gap_sum_after_physical_time_2": after_t2,
        "signed_gap_share_after_physical_time_2": after_t2 / total,
        "after_physical_time_2_rule": "physical_time > 2.0; steps 21--200",
    }


def _curve_summary(
    decision: Mapping[str, Any],
    receipt: Mapping[str, Any],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    primary = decision["primary"]
    pointwise_h200 = rows[-1]
    relative = np.asarray(
        [float(row["relative_reduction_of_arm_means"]) for row in rows]
    )
    full_trace = _full_trace_summary(rows)
    return {
        "schema_version": 2,
        "packet_id": PACKET_ID,
        "status": "descriptive_full_horizon_with_separate_primary_h200_inference",
        "metric_schema": {
            "through_horizon_mean_field_mse": {
                "definition": (
                    "at horizon h, cumsum(instantaneous_field_mse) over steps 1--h divided by h"
                ),
                "authenticated_source_legacy_key": "cumulative_field_mse",
                "not_an_unnormalized_cumulative_sum": True,
            },
            "instantaneous_field_mse": {
                "definition": "field-element mean squared error at exactly the named forecast time"
            },
        },
        "source": {
            "scientific_payload_sha256": receipt["scientific_payload_sha256"],
            "card_sha256": receipt["card_sha256"],
            "source_manifest_sha256": receipt["source_manifest_sha256"],
            "checkpoint_roster_sha256": receipt["checkpoint_roster_sha256"],
            "crossed_cells": 60,
        },
        "aggregation": {
            "model_seeds": list(MODEL_SEEDS),
            "datasets_per_model_seed": 3,
            "rule": "average three datasets within each model seed, then average ten paired model seeds",
            "inference_unit": "paired model seed",
            "trajectory_or_cell_pseudoreplication": False,
        },
        "curve": {
            "horizons": 200,
            "stored_dt": 0.1,
            "physical_time": 20.0,
            "pointwise_bootstrap_replicates": 50_000,
            "pointwise_bootstrap_seed": 20_260_721,
            "pointwise_interval_role": "descriptive_not_simultaneous_no_curve_wide_test_no_horizon_selection",
            "sparse_arm_mean_lower_at_all_200_horizons": bool(np.all(relative > 0)),
            "minimum_relative_reduction": float(relative.min()),
            "minimum_relative_reduction_horizon": int(relative.argmin() + 1),
            "maximum_relative_reduction": float(relative.max()),
            "maximum_relative_reduction_horizon": int(relative.argmax() + 1),
            "h200_pointwise_ci95_lower": float(
                pointwise_h200["relative_reduction_pointwise_ci95_lower"]
            ),
            "h200_pointwise_ci95_upper": float(
                pointwise_h200["relative_reduction_pointwise_ci95_upper"]
            ),
        },
        "full_trace_descriptive": full_trace,
        "primary_h200": {
            "role": (
                "prospectively_frozen_outcome_aware_primary_endpoint_for_this_new_ic_check_"
                "separate_from_curve_bands"
            ),
            "dense_mean": float(primary["dense_mean"]),
            "sparse_mean": float(primary["sparse_mean"]),
            "relative_reduction_of_arm_means": float(
                primary["bootstrap"]["relative_reduction_of_arm_means"]
            ),
            "ci95_lower": float(primary["bootstrap"]["ci95_lower"]),
            "ci95_upper": float(primary["bootstrap"]["ci95_upper"]),
            "bootstrap_replicates": int(primary["bootstrap"]["replicates"]),
            "one_sided_exact_sign_flip_p": float(
                primary["exact_sign_flip"]["one_sided_exact_p"]
            ),
            "paired_model_seed_wins": int(primary["paired_model_seed_wins"]),
        },
        "figure_title": (
            "Allen--Cahn same-checkpoint new-IC robustness check: complete 200-step horizon"
        ),
        "figure_disclosure": (
            "H200 was selected after the original four-cell gate failed; same checkpoints; "
            "no retraining or reselection."
        ),
        "caption": (
            "Complete-horizon Allen--Cahn same-checkpoint new-initial-condition replication. "
            "Lines average three datasets within each of ten paired model seeds before the arm mean; "
            "shading gives 95% paired-model-seed pointwise bootstrap intervals, which are descriptive "
            "and not simultaneous. The H200 diamond and error bar separately show the prospectively "
            "frozen, outcome-aware primary endpoint for this new-IC check: the joint sparse recipe lowers "
            "mean direct-rollout field MSE over steps "
            "1--200 by 5.86% (95% CI 2.40--9.21%; exact one-sided p=0.0088; 8/10 wins). "
            f"Descriptively, sparse instantaneous arm-mean MSE is lower at all "
            f"{full_trace['evaluated_time_count']} times and "
            f"{100.0 * float(full_trace['signed_gap_share_after_physical_time_2']):.1f}% of the signed "
            "dense-minus-sparse gap sum occurs after T=2. "
            "Checkpoints were neither retrained nor reselected; one encoding is followed by 200 global-K "
            "steps without re-encoding. The original four-cell gate failed. Curve points do not define "
            "additional tests or terminal claims."
        ),
    }


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    handle = StringIO(newline="")
    fields = list(rows[0])
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: format(value, ".17g") if isinstance(value, float) else value
                for key, value in row.items()
            }
        )
    return handle.getvalue().encode()


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def build_curve_companion(
    source_root: Path,
    decision: Mapping[str, Any],
    receipt: Mapping[str, Any],
    card: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> None:
    scientific_path = source_root / "scientific_curves.json"
    _require(scientific_path.is_file(), f"Missing scientific curves: {scientific_path}")
    _require(
        _sha256(scientific_path) == receipt["scientific_payload_sha256"],
        "Scientific curve payload hash drifted",
    )
    scientific = _read_json(scientific_path)
    source_rows = _validate_scientific_payload(scientific, receipt, card)
    through_horizon_seed_curves = _seed_curves(source_rows, "cumulative_field_mse")
    instantaneous_seed_curves = _seed_curves(source_rows, "instantaneous_field_mse")
    bands = _pointwise_bootstrap(
        through_horizon_seed_curves["dense"],
        through_horizon_seed_curves["sparse"],
    )
    _compare_decision_curves(decision, through_horizon_seed_curves, bands)
    rows = _curve_rows(through_horizon_seed_curves, instantaneous_seed_curves, bands)
    summary = _curve_summary(decision, receipt, rows)
    validate_curve_compact(summary, rows)
    paths["curve_summary"].write_bytes(_json_bytes(summary))
    paths["curve_rows"].write_bytes(_csv_bytes(rows))
    render_full_horizon_figure(
        summary,
        rows,
        paths["full_horizon_figure_pdf"],
        paths["full_horizon_figure_png"],
    )


def check_curve_companion(paths: Mapping[str, Path]) -> None:
    summary = _read_json(paths["curve_summary"])
    rows = _read_csv(paths["curve_rows"])
    validate_curve_compact(summary, rows)
    _require(
        summary["source"]["scientific_payload_sha256"]
        == "4c536871e71f47fd055db057da8c1c4a1213a0aceee9687ddb1c14dbc8963cf0",
        "Compact curve source payload drifted",
    )
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        pdf = root / FULL_HORIZON_FIGURE_PDF.name
        png = root / FULL_HORIZON_FIGURE_PNG.name
        render_full_horizon_figure(summary, rows, pdf, png)
        _require(
            pdf.read_bytes() == paths["full_horizon_figure_pdf"].read_bytes(),
            "Full-horizon PDF rendering is not deterministic",
        )
        _require(
            png.read_bytes() == paths["full_horizon_figure_png"].read_bytes(),
            "Full-horizon PNG rendering is not deterministic",
        )
