"""Build the separate forecast-optimized Allen--Cahn global-K packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping

import numpy as np
import pandas as pd

from experiments.neurips_2026.evidence.allen_cahn_global_forecast_rendering import (
    render_global_forecast,
)
from experiments.neurips_2026.evidence.allen_cahn_global_forecast_architecture import (
    AUDIT as ARCHITECTURE_AUDIT,
    validate_architecture_audit,
)
from experiments.neurips_2026.evidence.allen_cahn_global_forecast_statistics import (
    DECISION_HORIZONS,
    HORIZONS,
    SEEDS,
    summarize,
)
from experiments.neurips_2026.paths import (
    PAPER_DATA_DIR,
    PAPER_EVIDENCE_DIR,
    PAPER_TABLE_DIR,
)


PACKET_ID = "allen_cahn_global_k_forecast_optimized"
ROWS = PAPER_DATA_DIR / f"{PACKET_ID}_seed_rows.csv"
PROTOCOL = PAPER_DATA_DIR / f"{PACKET_ID}_protocol.json"
ARTIFACTS = PAPER_DATA_DIR / f"{PACKET_ID}_artifacts.csv"
STATISTICS = PAPER_DATA_DIR / f"{PACKET_ID}_statistics.json"
ARTIFACT_MANIFEST = PAPER_DATA_DIR / f"{PACKET_ID}_artifact_manifest.json"
PROVENANCE = PAPER_DATA_DIR / f"{PACKET_ID}_provenance.json"
TABLE = PAPER_TABLE_DIR / f"table_{PACKET_ID}.tex"
FIGURE_PDF = PAPER_EVIDENCE_DIR / f"fig_{PACKET_ID}.pdf"
FIGURE_PNG = PAPER_EVIDENCE_DIR / f"fig_{PACKET_ID}.png"

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_protocol(protocol: Mapping[str, object]) -> None:
    system = protocol["system"]
    if int(system["state_dim"]) != 512 or int(system["latent_dim"]) != 2048:
        raise ValueError("Allen--Cahn packet must use d_x=512 and d_z=2048")
    if int(system["latent_dim"]) < 4 * int(system["state_dim"]):
        raise ValueError("Allen--Cahn packet latent is not fourfold overcomplete")
    evaluation = protocol["evaluation"]
    if evaluation["rollout_mode"] != "no_reencode":
        raise ValueError("Forecast-optimized packet must use autonomous no-reencoding rollout")
    if evaluation["model_seeds"] != list(SEEDS):
        raise ValueError("Allen--Cahn model-seed roster drifted")
    if evaluation["reported_horizons"] != list(HORIZONS):
        raise ValueError("Allen--Cahn reporting horizons drifted")
    arms = protocol["arms"]
    sparse_contract = {
        "sparse_lista_alpha": 0.15,
        "sparse_elementwise_sparsity_weight": 0.01,
        "sparse_temporal_group_sparsity_weight": 0.0,
    }
    if any(float(arms[key]) != value for key, value in sparse_contract.items()):
        raise ValueError("Sparse forecast-optimized arm contract drifted")
    training = protocol["training"]
    expected_training = {
        "train_trajectories": 512,
        "sequence_length": 200,
        "batch_size": 8,
        "pretrain_steps": 2000,
        "forecast_training_steps": 3500,
        "learning_rate": 0.0003,
        "koopman_matrix_learning_rate": 0.000001,
        "weight_decay": 0.0,
        "prediction_weight": 1.0,
        "reconstruction_weight": 0.25,
        "latent_weight": 0.1,
        "gradient_weight": 0.05,
        "koopman_stability_weight": 0.0,
        "evaluation_horizon": 200,
        "checkpoint_evaluation_every_steps": 250,
    }
    if any(float(training[key]) != value for key, value in expected_training.items()):
        raise ValueError("Allen--Cahn common training contract drifted")
    if (
        training["optimizer"] != "Adam"
        or training["forecast_weighting"] != "uniform"
        or not training["spatial_augmentation"]
        or training["validation_partition_for_checkpointing"] != "even"
    ):
        raise ValueError("Allen--Cahn optimizer or data-budget contract drifted")
    if training["checkpoint_metric"] != "joint_endpoints" or training[
        "checkpoint_horizons"
    ] != [160, 200]:
        raise ValueError("Allen--Cahn full-horizon checkpoint objective drifted")
    gate = protocol["decision_gate"]
    if (
        int(gate["bootstrap_replicates"]) != 50_000
        or int(gate["bootstrap_seed"]) != 20_260_719
        or int(gate["max_t_swaps"]) != 1024
    ):
        raise ValueError("Allen--Cahn decision statistics contract drifted")


def load_inputs(
    rows_path: Path = ROWS,
    protocol_path: Path = PROTOCOL,
    artifacts_path: Path = ARTIFACTS,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or protocol.get("packet_id") != PACKET_ID:
        raise ValueError("Allen--Cahn forecast packet protocol is not canonical")
    validate_protocol(protocol)
    frozen = protocol["frozen_compact_evidence"]
    expected_path = str(frozen["path"])
    if rows_path.name != expected_path:
        raise ValueError(f"Expected frozen rows {expected_path}, got {rows_path.name}")
    actual_hash = sha256(rows_path)
    if actual_hash != str(frozen["sha256"]):
        raise ValueError(
            f"Frozen Allen--Cahn rows hash mismatch: expected {frozen['sha256']}, "
            f"got {actual_hash}"
        )
    rows = pd.read_csv(rows_path)
    validate_rows(rows, protocol)
    frozen_artifacts = str(frozen["artifact_roster_path"])
    if artifacts_path.name != frozen_artifacts:
        raise ValueError(
            f"Expected frozen artifact roster {frozen_artifacts}, got {artifacts_path.name}"
        )
    if sha256(artifacts_path) != str(frozen["artifact_roster_sha256"]):
        raise ValueError("Frozen Allen--Cahn artifact-roster hash mismatch")
    artifacts = pd.read_csv(artifacts_path)
    validate_artifacts(artifacts, rows, protocol)
    validate_architecture_audit(protocol, artifacts)
    return rows, protocol, artifacts


def validate_rows(rows: pd.DataFrame, protocol: Mapping[str, object]) -> None:
    required = {
        "arm",
        "source_arm",
        "seed",
        "horizon",
        "physical_time",
        "field_mse",
        "final_field_mse",
        "persistence_field_mse",
        "persistence_final_field_mse",
        "gradient_mse",
        "final_basin_consistency",
        "active_density",
        "near_zero_fraction_at_1e_minus_3",
        "mean_active_gpu_utilization_percent",
        "checkpoint_step",
        "evaluation_sha256",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Allen--Cahn rows are missing columns: {sorted(missing)}")
    expected_index = pd.MultiIndex.from_product(
        [[("dense", "dense"), ("sparse", "signed")], SEEDS, HORIZONS],
        names=["arm_source", "seed", "horizon"],
    )
    actual_index = pd.MultiIndex.from_tuples(
        [
            ((str(row.arm), str(row.source_arm)), int(row.seed), int(row.horizon))
            for row in rows.itertuples(index=False)
        ],
        names=expected_index.names,
    )
    if len(rows) != len(expected_index) or set(actual_index) != set(expected_index):
        raise ValueError("Allen--Cahn packet is not one row per arm/seed/horizon")
    if actual_index.has_duplicates:
        raise ValueError("Allen--Cahn packet contains duplicate arm/seed/horizon rows")
    numeric = [
        "physical_time",
        "field_mse",
        "final_field_mse",
        "persistence_field_mse",
        "persistence_final_field_mse",
        "gradient_mse",
        "final_basin_consistency",
        "active_density",
        "near_zero_fraction_at_1e_minus_3",
        "mean_active_gpu_utilization_percent",
    ]
    if not np.isfinite(rows[numeric].to_numpy(dtype=np.float64)).all():
        raise ValueError("Allen--Cahn packet contains a nonfinite metric")
    expected_time = rows["horizon"].to_numpy(dtype=np.float64) * float(
        protocol["system"]["stored_dt"]
    )
    np.testing.assert_allclose(rows["physical_time"], expected_time, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        rows["active_density"] + rows["near_zero_fraction_at_1e_minus_3"],
        1.0,
        rtol=0,
        atol=1e-12,
    )
    if (rows[["field_mse", "final_field_mse"]] < 0).any().any():
        raise ValueError("Forecast MSE must be nonnegative")
    if (rows[["persistence_field_mse", "persistence_final_field_mse"]] <= 0).any().any():
        raise ValueError("Persistence MSE must be positive")
    grouped = rows.groupby(["arm", "seed"], sort=True)
    for (_arm, _seed), group in grouped:
        for column in (
            "active_density",
            "near_zero_fraction_at_1e_minus_3",
            "mean_active_gpu_utilization_percent",
            "checkpoint_step",
            "evaluation_sha256",
        ):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(f"Run-level field {column} varies across horizons")
    frozen = protocol["frozen_compact_evidence"]
    if len(rows) != int(frozen["rows"]):
        raise ValueError("Frozen row count drifted")
    if rows["evaluation_sha256"].nunique() != int(frozen["unique_evaluation_artifacts"]):
        raise ValueError("Frozen evaluation-artifact count drifted")


def validate_artifacts(
    artifacts: pd.DataFrame,
    rows: pd.DataFrame,
    protocol: Mapping[str, object],
) -> None:
    required = {
        "arm",
        "seed",
        "checkpoint_step",
        "checkpoint_path",
        "checkpoint_sha256",
        "evaluation_path",
        "evaluation_sha256",
        "slurm_job_id",
        "git_commit",
    }
    if required - set(artifacts.columns):
        raise ValueError("Allen--Cahn artifact roster is missing a required field")
    expected = {(arm, seed) for arm in ("dense", "sparse") for seed in SEEDS}
    actual = {
        (str(row.arm), int(row.seed)) for row in artifacts.itertuples(index=False)
    }
    frozen = protocol["frozen_compact_evidence"]
    if len(artifacts) != int(frozen["artifact_roster_rows"]) or actual != expected:
        raise ValueError("Allen--Cahn artifact roster is incomplete")
    if artifacts.duplicated(["arm", "seed"]).any():
        raise ValueError("Allen--Cahn artifact roster has duplicate runs")
    for row in artifacts.itertuples(index=False):
        selected = rows.loc[
            (rows["arm"] == row.arm) & (rows["seed"] == row.seed)
        ]
        for column in (
            "checkpoint_step",
            "checkpoint_path",
            "evaluation_path",
            "evaluation_sha256",
        ):
            if selected[column].nunique() != 1 or str(selected.iloc[0][column]) != str(
                getattr(row, column)
            ):
                raise ValueError(f"Artifact roster disagrees with seed rows: {column}")
        for digest, length in (
            (row.checkpoint_sha256, 64),
            (row.evaluation_sha256, 64),
            (row.git_commit, 40),
        ):
            if len(str(digest)) != length:
                raise ValueError("Artifact roster contains an invalid digest")
            int(str(digest), 16)
    replacement = artifacts.loc[
        (artifacts["arm"] == "sparse") & (artifacts["seed"] == 69)
    ].iloc[0]
    if int(replacement["slurm_job_id"]) != 10157586:
        raise ValueError("Low-utilization replacement task is not preserved")


def artifact_manifest(artifacts: pd.DataFrame) -> dict[str, object]:
    records = []
    ordered = artifacts.assign(
        arm_order=artifacts["arm"].map({"dense": 0, "sparse": 1})
    ).sort_values(["arm_order", "seed"])
    for row in ordered.itertuples(index=False):
        records.append(
            {
                "arm": str(row.arm),
                "seed": int(row.seed),
                "checkpoint_step": int(row.checkpoint_step),
                "slurm_job_id": int(row.slurm_job_id),
                "git_commit": str(row.git_commit),
                "checkpoint": {
                    "path": str(row.checkpoint_path),
                    "sha256": str(row.checkpoint_sha256),
                },
                "evaluation": {
                    "path": str(row.evaluation_path),
                    "sha256": str(row.evaluation_sha256),
                },
            }
        )
    return {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "description": "Exact checkpoint and evaluator roster admitted to the final confirmation gate.",
        "runs": records,
    }


def _format_p(value: float) -> str:
    return f"{value:.3f}" if value >= 0.001 else r"$<0.001$"


def write_table(statistics: Mapping[str, object], output_path: Path) -> None:
    cells = statistics["comparison"]["cells"]
    max_t = statistics["max_t_sensitivity"]
    lines = [
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        r"Horizon & Endpoint & Dense & Sparse & Reduction & Wins & Marginal 95\% CI; max-$t$ $p$ \\",
        r"\midrule",
    ]
    for horizon in DECISION_HORIZONS:
        for metric, endpoint in (
            ("field_mse", "Mean through H"),
            ("final_field_mse", "Terminal"),
        ):
            cell = cells[f"h{horizon}_{metric}"]
            label = (
                f"H{horizon} through-horizon mean"
                if metric == "field_mse"
                else f"H{horizon} terminal"
            )
            lines.append(
                f"H{horizon} & {endpoint} & {float(cell['dense_mean']):.5f} & "
                f"{float(cell['sparse_mean']):.5f} & "
                f"{100 * float(cell['relative_reduction_of_means']):.1f}\\% & "
                f"{int(cell['sparse_seed_wins'])}/10 & "
                f"[{100 * float(cell['ci95_lower']):.1f}, "
                f"{100 * float(cell['ci95_upper']):.1f}]\\%; "
                f"{_format_p(float(max_t[label]['one_sided_max_t_fwer_adjusted_p']))} \\\\"
            )
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{7}{l}{\footnotesize Matched trainable tensor shapes and effective forward-path parameter count; sparse jointly applies soft-thresholding $0.15$ and $L_1$ weight $0.01$.} \\",
            r"\multicolumn{7}{l}{\footnotesize Internally frozen confirmation gate: failed; sealed holdout not generated or opened.} \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def build(output_dir: Path = PAPER_EVIDENCE_DIR) -> tuple[Path, ...]:
    rows, protocol, artifacts = load_inputs()
    statistics = summarize(rows, protocol, packet_id=PACKET_ID)
    stats_path = output_dir / "_data" / STATISTICS.name
    artifact_manifest_path = output_dir / "_data" / ARTIFACT_MANIFEST.name
    provenance_path = output_dir / "_data" / PROVENANCE.name
    table_path = output_dir / "_tables" / TABLE.name
    pdf_path = output_dir / FIGURE_PDF.name
    png_path = output_dir / FIGURE_PNG.name
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_manifest_path.write_text(
        json.dumps(artifact_manifest(artifacts), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_table(statistics, table_path)
    render_global_forecast(rows, statistics, pdf_path, png_path)
    provenance = {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "status": statistics["status"],
        "generated_by": "experiments.neurips_2026.evidence.allen_cahn_global_forecast",
        "inputs": {
            ROWS.name: sha256(ROWS),
            PROTOCOL.name: sha256(PROTOCOL),
            ARTIFACTS.name: sha256(ARTIFACTS),
            ARCHITECTURE_AUDIT.name: sha256(ARCHITECTURE_AUDIT),
        },
        "integrator_final": protocol["integrator_final"],
        "artifact_manifest_sha256": sha256(artifact_manifest_path),
        "outputs": {
            _relative(stats_path, output_dir): sha256(stats_path),
            _relative(artifact_manifest_path, output_dir): sha256(
                artifact_manifest_path
            ),
            _relative(table_path, output_dir): sha256(table_path),
            _relative(pdf_path, output_dir): sha256(pdf_path),
            _relative(png_path, output_dir): sha256(png_path),
        },
        "aggregation_contract": (
            "Paired model seeds 64--73; ratio-of-means reductions; 50,000 "
            "paired bootstrap resamples with seed 20260719; exact 1024-swap "
            "one-sided max-t sensitivity over four H160/H200 cells."
        ),
        "claim_boundary": (
            "Secondary evidence supports lower through-horizon mean H160/H200 "
            "MSE for the joint soft-thresholding-plus-L1 sparse treatment under "
            "matched trainable tensor shapes and effective forward-path parameter "
            "count, with a shared convolutional, decoder, and K backbone. Terminal intervals cross zero "
            "and the internally frozen four-cell gate failed."
        ),
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return (
        stats_path,
        artifact_manifest_path,
        provenance_path,
        table_path,
        pdf_path,
        png_path,
    )


def verify_active_provenance() -> None:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    if provenance.get("schema_version") != 1 or provenance.get("packet_id") != PACKET_ID:
        raise ValueError("Allen--Cahn forecast provenance is not canonical")
    for name, expected in provenance["inputs"].items():
        path = PAPER_DATA_DIR / name
        if sha256(path) != expected:
            raise ValueError(f"Allen--Cahn forecast input hash mismatch: {name}")
    for relative, expected in provenance["outputs"].items():
        path = PAPER_EVIDENCE_DIR / relative
        if sha256(path) != expected:
            raise ValueError(f"Allen--Cahn forecast output hash mismatch: {relative}")
    manifest_relative = f"_data/{ARTIFACT_MANIFEST.name}"
    if provenance["artifact_manifest_sha256"] != provenance["outputs"][
        manifest_relative
    ]:
        raise ValueError("Allen--Cahn artifact-manifest digest drifted")


def check() -> None:
    load_inputs()
    verify_active_provenance()
    with TemporaryDirectory() as temporary:
        rebuilt = Path(temporary)
        candidates = build(rebuilt)
        active = (
            STATISTICS,
            ARTIFACT_MANIFEST,
            PROVENANCE,
            TABLE,
            FIGURE_PDF,
            FIGURE_PNG,
        )
        for active_path, candidate in zip(active, candidates):
            if active_path.read_bytes() != candidate.read_bytes():
                raise ValueError(
                    f"Allen--Cahn forecast artifact is not reproducible: {active_path.name}"
                )
    print("Allen--Cahn forecast-optimized packet is byte-identical to a clean rebuild.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    if parse_args().check:
        check()
    else:
        build()
        print("Built Allen--Cahn forecast-optimized evidence packet.")


if __name__ == "__main__":
    main()
