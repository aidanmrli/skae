#!/usr/bin/env python3
"""Build the paper's staged F_abs local-map table from frozen evidence."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "docs/figures/neurips_paper_2026/_data"
TABLE_DIR = REPO_ROOT / "docs/figures/neurips_paper_2026/_tables"
DATA_PATH = DATA_DIR / "fabs_local_map_forecasting_rows.csv"
TABLE_PATH = TABLE_DIR / "table_fabs_local_k_forecasting.tex"
PROVENANCE_PATH = DATA_DIR / "local_map_forecasting_provenance.json"

SOURCE_REPOSITORY_PATH = (
    "results/staged_cstab_baseline_support_family_lista_full_20260519/"
    "wide_periodic_reeval/wide_periodic_reeval_rows.csv"
)
SOURCE_SHA256 = "b0b621e250256e577c29b8d1c9196792d80fc3d3584f098d65660dd5bed6b644"
HORIZONS = (100, 500, 1000)
PERIODS = (1, 2, 5, 10, 20, 25, 50, 100)
PATH_COLUMNS = ("staged_run_dir", "global_run_dir")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize_source(source: Path) -> int:
    with source.open(newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        if reader.fieldnames is None:
            raise ValueError(f"Source CSV has no header: {source}")
        missing = [column for column in PATH_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Source CSV is missing path columns {missing}: {source}")
        fieldnames = [name for name in reader.fieldnames if name not in PATH_COLUMNS]
        rows = list(reader)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", newline="") as destination_handle:
        writer = csv.DictWriter(
            destination_handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})
    return len(rows)


def _provenance_payload(row_count: int) -> dict[str, object]:
    return {
        "schema_version": 3,
        "generated_by": "tools/build_local_map_forecasting_tables.py",
        "machine_specific_columns_removed": list(PATH_COLUMNS),
        "training_contract": {
            "shared": {
                "architecture": "LISTA sparse Koopman autoencoder",
                "latent_dimension": 256,
                "total_training_steps": 200_000,
                "task_rows": "same system/seed architecture and training-budget recipe",
            },
            "staged_fabs": {
                "stage1_joint_steps": 100_000,
                "stage2_local_map_steps": 100_000,
                "frozen_during_stage2": "encoder, decoder, and global K",
                "local_map": "source_target_affine_learned_intercept",
                "route_fit": {
                    "support_definition": "absolute:0.001",
                    "family_jaccard_threshold": 0.4,
                    "configured_rows": 512,
                    "unique_trajectories": 256,
                    "duplication_factor": 2,
                    "construction": (
                        "two bitwise-identical copies of one 256-row "
                        "training-distribution batch"
                    ),
                    "transitions_per_trajectory": 192,
                    "states_per_trajectory": 193,
                    "seed_offset": 271_828,
                    "supports_clustered": 98_816,
                    "clustering_scope": (
                        "all 193 states per configured row, including terminal states"
                    ),
                    "map_fit_source_transitions": 98_304,
                    "unique_map_fit_source_transitions": 49_152,
                    "map_fit_scope": "first 192 source states per configured row",
                    "family_representative": "modal source-state support mask",
                    "minimum_family_transitions": 1,
                },
            },
            "global_k": {
                "joint_training_steps": 200_000,
                "local_map_stage": False,
            },
        },
        "checkpoint_selection_contract": {
            "staged_fabs": {
                "candidate_steps": {
                    "first_regular": 100_500,
                    "last_regular": 199_500,
                    "regular_interval": 500,
                    "final": 199_999,
                    "count": 200,
                },
                "starts": 32,
                "seed_offset": 12_345,
                "horizons": list(HORIZONS),
                "periods": list(PERIODS),
                "metric": (
                    "state-summed squared error; finite-prefix mean per start, "
                    "then finite-start mean"
                ),
                "aggregation": (
                    "choose the best cadence independently at H100, H500, and "
                    "H1000, then take the arithmetic mean of the three minima"
                ),
                "improvement_rule": "strict less-than; ties retain the earlier checkpoint",
            },
            "global_k": {
                "candidate_steps": "every 500 steps plus the final step",
                "starts": 16,
                "seed_offset": 999_999,
                "horizon": 200,
                "rollout": "every-step re-encoding",
                "metric": "mean final-step Euclidean state error",
                "evaluation_start_overlap": 0,
            },
            "selector_is_asymmetric": True,
        },
        "evaluation_contract": {
            "systems": 15,
            "seeds": list(range(15)),
            "paired_rows": 225,
            "horizons": list(HORIZONS),
            "periodic_reencoding_periods": list(PERIODS),
            "reported_evaluation_starts_per_pair": 100,
            "seed_offset": 12_345,
            "support_definition": "absolute:0.001",
            "family_jaccard_threshold": 0.4,
            "staged_routing_cadence": "support route is recomputed before every latent transition",
            "periodic_reencoding_role": (
                "decode-encode refreshes the latent on the selected cadence; "
                "it does not set the staged route-lookup cadence"
            ),
            "metric": (
                "state-summed squared error, averaged first over finite rollout "
                "steps 1..H within each reported start and then over starts; "
                "nonfinite steps and all-nonfinite starts are omitted"
            ),
            "cadence_reporting": (
                "each model's best periodic cadence is selected independently "
                "on the same 100 starts used to report that horizon"
            ),
            "staged_checkpoint_selector_overlap": {
                "count": 32,
                "fraction": 0.32,
                "relationship": (
                    "the selector's 32 starts are exactly the first 32 of the "
                    "100 reported evaluation starts because both use seed+12345"
                ),
            },
            "global_checkpoint_selector_overlap": {
                "count": 0,
                "relationship": "global selection uses the distinct seed+999999 starts",
            },
        },
        "comparison_contract": {
            "matched_attributes": [
                "system/seed roster",
                "architecture",
                "latent dimension",
                "total optimization-step budget",
                "reported evaluation starts",
                "reported cadence grid",
                "reported error reducer",
            ],
            "unmatched_attributes": ["checkpoint-selection protocol"],
            "interpretation": (
                "descriptive and optimistic for the staged method; not a clean "
                "held-out causal comparison until disjoint validation is rerun"
            ),
        },
        "dataset": {
            "description": "F_abs-routed local affine maps versus global-K LISTA",
            "source_repository_path": SOURCE_REPOSITORY_PATH,
            "source_sha256": SOURCE_SHA256,
            "sanitized_repository_path": str(DATA_PATH.relative_to(REPO_ROOT)),
            "sanitized_sha256": _sha256(DATA_PATH),
            "row_count": row_count,
        },
    }


def import_sanitized_source(source: Path) -> None:
    actual_source_sha = _sha256(source)
    if actual_source_sha != SOURCE_SHA256:
        raise ValueError(
            f"Unexpected SHA-256 for {source}: {actual_source_sha}; "
            f"expected {SOURCE_SHA256}"
        )
    row_count = _sanitize_source(source)
    PROVENANCE_PATH.write_text(
        json.dumps(_provenance_payload(row_count), indent=2) + "\n",
        encoding="utf-8",
    )


def verify_provenance() -> Mapping[str, object]:
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    if provenance.get("schema_version") != 3:
        raise ValueError("Local-map provenance must use schema version 3")
    selection = provenance.get("checkpoint_selection_contract", {})
    evaluation = provenance.get("evaluation_contract", {})
    if selection.get("selector_is_asymmetric") is not True:
        raise ValueError("Checkpoint-selector asymmetry is not recorded")
    overlap = evaluation.get("staged_checkpoint_selector_overlap", {})
    if overlap.get("count") != 32 or overlap.get("fraction") != 0.32:
        raise ValueError("Staged selector/evaluation overlap is not recorded")
    if evaluation.get("staged_routing_cadence") != (
        "support route is recomputed before every latent transition"
    ):
        raise ValueError("Staged every-step routing cadence is not recorded")
    record = provenance["dataset"]
    if record["source_sha256"] != SOURCE_SHA256:
        raise ValueError("Provenance source digest does not match the F_abs contract")
    actual_sanitized_sha = _sha256(DATA_PATH)
    if record["sanitized_sha256"] != actual_sanitized_sha:
        raise ValueError("Sanitized F_abs data digest does not match provenance")
    return provenance


def load_rows() -> list[dict[str, str]]:
    with DATA_PATH.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 225:
        raise ValueError(f"Expected 225 paired rows, found {len(rows)}")
    if any(row.get("status") != "ok" for row in rows):
        raise ValueError("Non-ok result found in frozen F_abs rows")
    pairs = {(row["system_key"], int(row["seed"])) for row in rows}
    if len(pairs) != len(rows):
        raise ValueError("Duplicate system/seed pair found in frozen F_abs rows")
    systems = {row["system_key"] for row in rows}
    seeds = {int(row["seed"]) for row in rows}
    if len(systems) != 15 or seeds != set(range(15)):
        raise ValueError("Frozen F_abs rows do not form the 15x15 paper matrix")
    return rows


def _values(
    rows: Iterable[Mapping[str, str]], horizon: int, kind: str
) -> list[float]:
    column = f"h{horizon}_{kind}_best_periodic_mean"
    values = [float(row[column]) for row in rows]
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"Expected finite positive values in {column}")
    return values


def _ratios(rows: Sequence[Mapping[str, str]], horizon: int) -> list[float]:
    staged = _values(rows, horizon, "staged")
    global_values = _values(rows, horizon, "global")
    return [local / baseline for local, baseline in zip(staged, global_values)]


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _geometric_mean(values: Sequence[float]) -> float:
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def render_table(rows: Sequence[Mapping[str, str]]) -> str:
    ratios = {h: _ratios(rows, h) for h in HORIZONS}
    wins = [sum(value < 1.0 for value in ratios[h]) for h in HORIZONS]
    all_wins = sum(
        all(per_horizon < 1.0 for per_horizon in per_row)
        for per_row in zip(*(ratios[h] for h in HORIZONS))
    )
    recorded = sum(row["wins_all_horizons"].lower() == "true" for row in rows)
    if all_wins != recorded:
        raise ValueError("Recorded all-horizon wins disagree with MSE columns")

    local_means = [_mean(_values(rows, h, "staged")) for h in HORIZONS]
    global_means = [_mean(_values(rows, h, "global")) for h in HORIZONS]
    median_ratios = [statistics.median(ratios[h]) for h in HORIZONS]
    geometric_ratios = [_geometric_mean(ratios[h]) for h in HORIZONS]
    return "\n".join(
        [
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Metric & H100 & H500 & H1000 & All three horizons \\",
            r"\midrule",
            "Paired local wins & "
            + " & ".join(f"${value}/225$" for value in wins)
            + f" & ${all_wins}/225$"
            + r" \\",
            r"Mean MSE, \(F_{\rm abs}^{\rm route}\)-local/global & "
            + " & ".join(
                f"${local:.3g}/{global_value:.3g}$"
                for local, global_value in zip(local_means, global_means)
            )
            + r" & -- \\",
            "Median local/global MSE ratio & "
            + " & ".join(f"${value:.3f}$" for value in median_ratios)
            + r" & -- \\",
            "Geometric local/global MSE ratio & "
            + " & ".join(f"${value:.3f}$" for value in geometric_ratios)
            + r" & -- \\",
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )


def build_table() -> str:
    verify_provenance()
    return render_table(load_rows())


def check_table(expected: str) -> None:
    actual = TABLE_PATH.read_text(encoding="utf-8") if TABLE_PATH.exists() else ""
    if actual == expected:
        return
    print(
        "".join(
            difflib.unified_diff(
                actual.splitlines(keepends=True),
                expected.splitlines(keepends=True),
                fromfile=str(TABLE_PATH),
                tofile=f"generated:{TABLE_PATH}",
            )
        ),
        end="",
    )
    raise SystemExit("F_abs local-map forecasting table is stale")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--import-source", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.import_source:
        import_sanitized_source(args.import_source)
    rendered = build_table()
    if args.check:
        check_table(rendered)
    else:
        TABLE_PATH.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
