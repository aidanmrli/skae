"""Freeze compact, authenticated row evidence for the paper's main tables."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from skae.benchmarks.controlled_alignment import alignment_protocol_metadata
from skae.benchmarks.paper_protocol import (
    CONTROLLED_ALIGNMENT_ELIGIBILITY_CRITERION,
    CONTROLLED_ALIGNMENT_EXCLUDED_OBSERVED_LABEL_COUNTS,
    CONTROLLED_ALIGNMENT_EXCLUDED_SYSTEM_KEYS,
    CONTROLLED_ALIGNMENT_OBSERVED_LABEL_COUNTS,
    CONTROLLED_ALIGNMENT_PRIMARY_SYSTEM_KEYS,
    CONTROLLED_MODEL_ROW_IDS,
    CONTROLLED_PAPER_PROTOCOL,
    DYSTS_MODEL_ROW_IDS,
    DYSTS_PAPER_PROTOCOL,
)
from skae.benchmarks.paper_statistics import IQM_CONVENTION

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "figures" / "neurips_paper_2026" / "_data"

REPAIRED_CONTROL_ROOT = "mlp_sparse_blockdiag_hardinit_basin_partition_control"
REPAIRED_DYSTS_ROOT = "sparse_mlp_bd"
CONTROL_ROOTS = CONTROLLED_MODEL_ROW_IDS
CONTROL_SYSTEMS = tuple(key.replace("claude:", "claude_") for key in CONTROLLED_PAPER_PROTOCOL.system_keys)
DYSTS_ROOTS = DYSTS_MODEL_ROW_IDS
DYSTS_SYSTEMS = DYSTS_PAPER_PROTOCOL.system_keys

@dataclass(frozen=True)
class Source:
    source_id: str
    relative_path: str
    sha256: str

CONTROL_FORECAST_SOURCES = (
    Source(
        "controlled_backfill",
        "transition_rich_table2_5model_seed15_backfill_20260428/collect_pass0/forecasting_rows.csv",
        "2f9c0f378145fb230484af65065dc592918d91e77a69e7c87f9086d1558e7815",
    ),
    Source(
        "controlled_lista_sb",
        "transition_rich_lista_sb_p256_hardinit_fairness_seed15_20260428/collect_pass0/forecasting_rows.csv",
        "67c937d8ed0b995f12f573c8eb1f669297b3ec541f2d83e7f2dc02808adda25f",
    ),
    Source(
        "controlled_lista",
        "transition_rich_lista_dense_p256_hardinit_table123_20260430/collect_pass0/forecasting_rows.csv",
        "f8ecf3bdaf60dd948eb7a8310982e160aaf666cccb4e90b0d2225f92dd4d26f2",
    ),
)
CONTROL_FORECAST_REPAIR = Source(
    "controlled_sparse_mlp_bd_repair",
    "transition_rich_sparse_mlp_bd_repaired_table1_20260506/collect_pass0/forecasting_rows.csv",
    "c4ff5cc6143ce08644d6bed2cb028819e100f3cadff49bd3305de90a6021cd38",
)
CONTROL_SUPPORT_SOURCES = (
    Source(
        "controlled_support_backfill",
        "transition_rich_table2_5model_seed15_backfill_20260428/interpretability_per_basin_deep_current_table1_pass0/interpretability_rows.csv",
        "0fb5a5a870428cbf566e0c4ad0d0febb4eaedc0e284bf0ad71511f2d6545c92f",
    ),
    Source(
        "controlled_support_lista_sb",
        "transition_rich_lista_sb_p256_hardinit_fairness_seed15_20260428/interpretability_per_basin_deep_current_table1_pass0/interpretability_rows.csv",
        "a9e1696aec64b09d2ada4d00399f2b7b8e0787ab26d04d7f3dcf88ae23af0d99",
    ),
    Source(
        "controlled_support_lista",
        "transition_rich_lista_dense_p256_hardinit_table123_20260430/interpretability_per_basin_deep_current_table1_pass0/interpretability_rows.csv",
        "f0884c0570fea1168818ab0525ca0e40742635fa0224be1bc9d10d75abc26a70",
    ),
)
CONTROL_SUPPORT_REPAIR = Source(
    "controlled_support_sparse_mlp_bd_repair",
    "transition_rich_sparse_mlp_bd_repaired_table1_20260506/interpretability_per_basin_deep_pass0/interpretability_rows.csv",
    "56a4f1578ff0661f246a1c0d09397906381e4ed83d4a94c41a08e6ff20daed6c",
)
DYSTS_SOURCE = Source(
    "dysts_dt30",
    "dysts_dt30_basinblock_p256_seq10_100k_20260430/long_horizon_eval/collect/forecasting_rows.csv",
    "e17086dc5d918ea0067ed6b131e4ac8e7a975d3d1232fa2a397a93c36c95b3ea",
)
DYSTS_REPAIR = Source(
    "dysts_dt30_sparse_mlp_bd_repair",
    "dysts_dt30_sparse_mlp_bd_repaired_20260506/long_horizon_eval/collect/forecasting_rows.csv",
    "17770e5d8d1696d45adb4f4b1a82adf4135a4e705c90728667d375e4e257d7a6",
)
ALL_SOURCES = (
    *CONTROL_FORECAST_SOURCES,
    CONTROL_FORECAST_REPAIR,
    *CONTROL_SUPPORT_SOURCES,
    CONTROL_SUPPORT_REPAIR,
    DYSTS_SOURCE,
    DYSTS_REPAIR,
)

CONTROL_FORECAST_COLUMNS = (
    "root_label",
    "system_name",
    "seed",
    "h100_best_periodic_mean",
    "h500_best_periodic_mean",
    "h1000_best_periodic_mean",
)
CONTROL_SUPPORT_COLUMNS = (
    "root_label",
    "system_name",
    "seed",
    "support_scheme",
    "subset",
    "num_states",
    "observed_label_count",
    "family_jaccard_threshold",
    "family_h_basin_given_family",
    "family_unique_count",
)
CONTROL_SUPPORT_SOURCE_COLUMNS = tuple(column for column in CONTROL_SUPPORT_COLUMNS if column != "observed_label_count")
EVIDENCE_SCHEMA_VERSION = 4
CONTROL_SUPPORT_SCHEMA_NOTE = (
    "Retains only row identity, the frozen absolute-support protocol, sample "
    "count, observed evaluation-label count, basin-given-family entropy, and "
    "observed family count. Superseded "
    "wrong-support-freeze diagnostics are intentionally excluded. num_states "
    "counts the tie-inclusive margin >= q75 slice, which can exceed 25% of a "
    "label's states."
)
SUPPORT_FAMILY_FIT_POPULATION = "all generated evaluation-trajectory states"
SUPPORT_SCORE_POPULATION = "per-observed-label center-margin >= empirical q75 subset (tie-inclusive)"
ALIGNMENT_PRIMARY_SYSTEMS = tuple(key.replace("claude:", "claude_") for key in CONTROLLED_ALIGNMENT_PRIMARY_SYSTEM_KEYS)
ALIGNMENT_EXCLUDED_SYSTEMS = tuple(key.replace("claude:", "claude_") for key in CONTROLLED_ALIGNMENT_EXCLUDED_SYSTEM_KEYS)
ALIGNMENT_OBSERVED_LABEL_COUNTS = {
    key.replace("claude:", "claude_"): value
    for key, value in CONTROLLED_ALIGNMENT_OBSERVED_LABEL_COUNTS.items()
}
AGGREGATION_METADATA = {
    "within_system_seed_summary": IQM_CONVENTION,
    "complete_cell_retained_seed_count": 9,
    "cross_system_summary": "arithmetic_mean",
    "controlled_forecasting_system_count": len(CONTROL_SYSTEMS),
    "controlled_alignment_primary_system_count": len(ALIGNMENT_PRIMARY_SYSTEMS),
    "controlled_alignment_all_system_sensitivity_count": len(CONTROL_SYSTEMS),
}
ALIGNMENT_FILTER_METADATA = {
    "support_family_fit_population": SUPPORT_FAMILY_FIT_POPULATION,
    "support_score_population": SUPPORT_SCORE_POPULATION,
    "support_alignment_protocol": alignment_protocol_metadata(),
    "support_alignment_primary_systems": list(ALIGNMENT_PRIMARY_SYSTEMS),
    "support_alignment_excluded_systems": list(ALIGNMENT_EXCLUDED_SYSTEMS),
    "support_alignment_eligibility_criterion": CONTROLLED_ALIGNMENT_ELIGIBILITY_CRITERION,
    "support_alignment_observed_label_counts": ALIGNMENT_OBSERVED_LABEL_COUNTS,
    "support_alignment_excluded_observed_label_counts": {
        key.replace("claude:", "claude_"): value
        for key, value in CONTROLLED_ALIGNMENT_EXCLUDED_OBSERVED_LABEL_COUNTS.items()
    },
    "support_alignment_raw_rows_retained": True,
    "support_alignment_sensitivity_artifact": "../_tables/controlled_support_alignment_sensitivity.csv",
}
DYSTS_COLUMNS = (
    "root_label",
    "system_key",
    "seed",
    "status",
    *(
        f"h{h}_{suffix}"
        for h in (100, 500, 1000, 1500, 2000, 3000, 4000, 5000)
        for suffix in (
            "best_periodic_mean",
            "best_periodic_mode",
            "best_periodic_full_finite_fraction",
            "best_periodic_finite_step_fraction",
            "best_periodic_num_full_horizon_finite",
            "best_periodic_median_finite_prefix_length",
            "best_periodic_min_finite_prefix_length",
        )
    ),
)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_path(results_root: Path, source: Source) -> Path:
    path = results_root / source.relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing source {source.source_id}: {path}")
    actual = sha256_file(path)
    if actual != source.sha256:
        raise ValueError(
            f"SHA256 mismatch for {source.source_id}: expected {source.sha256}, got {actual}"
        )
    return path


def read_source(results_root: Path, source: Source, columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.read_csv(
        verified_path(results_root, source),
        usecols=list(columns),
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )


def validate_grid(
    frame: pd.DataFrame,
    *,
    roots: tuple[str, ...],
    systems: tuple[str, ...],
    system_column: str,
    label: str,
) -> None:
    keys = ["root_label", system_column, "seed"]
    duplicates = frame.duplicated(keys, keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, keys].head().to_dict("records")
        raise ValueError(f"Duplicate {label} rows: {examples}")
    expected_rows = len(roots) * len(systems) * 15
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} {label} rows, found {len(frame)}")
    expected_roots = set(roots)
    expected_systems = set(systems)
    if set(frame["root_label"]) != expected_roots:
        raise ValueError(f"Unexpected {label} roots: {sorted(set(frame['root_label']))}")
    if set(frame[system_column]) != expected_systems:
        raise ValueError(f"Unexpected {label} systems: {sorted(set(frame[system_column]))}")
    seeds = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    if set(seeds) != set(range(15)):
        raise ValueError(f"{label} does not contain exactly seeds 0--14")
    frame["seed"] = seeds


def freeze_control_forecasting(results_root: Path) -> pd.DataFrame:
    frames = [read_source(results_root, source, CONTROL_FORECAST_COLUMNS) for source in CONTROL_FORECAST_SOURCES]
    base = pd.concat(frames, ignore_index=True, sort=False)
    base = base[base["root_label"] != REPAIRED_CONTROL_ROOT]
    repair = read_source(results_root, CONTROL_FORECAST_REPAIR, CONTROL_FORECAST_COLUMNS)
    frame = pd.concat([base, repair], ignore_index=True, sort=False)
    frame = frame[
        frame["root_label"].isin(CONTROL_ROOTS) & frame["system_name"].isin(CONTROL_SYSTEMS)
    ].copy()
    for horizon in (100, 500, 1000):
        metric = f"h{horizon}_best_periodic_mean"
        frame[f"h{horizon}_best_periodic_is_finite"] = np.isfinite(
            pd.to_numeric(frame[metric], errors="coerce")
        )
    validate_grid(
        frame,
        roots=CONTROL_ROOTS,
        systems=CONTROL_SYSTEMS,
        system_column="system_name",
        label="controlled forecasting",
    )
    return frame.reset_index(drop=True)


def freeze_control_support(results_root: Path) -> pd.DataFrame:
    frames = [
        read_source(results_root, source, CONTROL_SUPPORT_SOURCE_COLUMNS)
        for source in CONTROL_SUPPORT_SOURCES
    ]
    base = pd.concat(frames, ignore_index=True, sort=False)
    base = base[base["root_label"] != REPAIRED_CONTROL_ROOT]
    repair = read_source(
        results_root, CONTROL_SUPPORT_REPAIR, CONTROL_SUPPORT_SOURCE_COLUMNS
    )
    frame = pd.concat([base, repair], ignore_index=True, sort=False)
    threshold = pd.to_numeric(frame["family_jaccard_threshold"], errors="coerce")
    frame = frame[
        frame["root_label"].isin(CONTROL_ROOTS)
        & frame["system_name"].isin(CONTROL_SYSTEMS)
        & (frame["support_scheme"] == "absolute:0.001")
        & (frame["subset"] == "deep")
        & (threshold == 0.5)
    ].copy()
    frame["observed_label_count"] = frame["system_name"].map(
        ALIGNMENT_OBSERVED_LABEL_COUNTS
    )
    if frame["observed_label_count"].isna().any():
        raise ValueError("Missing frozen observed-label count for controlled support row")
    validate_grid(
        frame,
        roots=CONTROL_ROOTS,
        systems=CONTROL_SYSTEMS,
        system_column="system_name",
        label="controlled support",
    )
    return frame.loc[:, CONTROL_SUPPORT_COLUMNS].reset_index(drop=True)


def freeze_dysts_forecasting(results_root: Path) -> pd.DataFrame:
    base = read_source(results_root, DYSTS_SOURCE, DYSTS_COLUMNS)
    base = base[base["root_label"] != REPAIRED_DYSTS_ROOT]
    repair = read_source(results_root, DYSTS_REPAIR, DYSTS_COLUMNS)
    frame = pd.concat([base, repair], ignore_index=True, sort=False)
    frame = frame[
        frame["root_label"].isin(DYSTS_ROOTS) & frame["system_key"].isin(DYSTS_SYSTEMS)
    ].copy()
    validate_grid(
        frame,
        roots=DYSTS_ROOTS,
        systems=DYSTS_SYSTEMS,
        system_column="system_key",
        label="Dysts forecasting",
    )
    return frame.reset_index(drop=True)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(
        buffer,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    return buffer.getvalue().encode("utf-8")


def make_outputs(results_root: Path) -> dict[str, bytes]:
    frames = {
        "controlled_forecasting_rows.csv": freeze_control_forecasting(results_root),
        "controlled_support_rows.csv": freeze_control_support(results_root),
        "dysts_forecasting_rows.csv": freeze_dysts_forecasting(results_root),
    }
    outputs = {name: csv_bytes(frame) for name, frame in frames.items()}
    source_records = [
        {
            "source_id": source.source_id,
            "relative_to_results_root": source.relative_path,
            "sha256": source.sha256,
        }
        for source in ALL_SOURCES
    ]
    provenance = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "description": "Frozen row-level evidence for the paper's headline forecasting and support tables.",
        "schema_notes": {
            "controlled_support_rows.csv": CONTROL_SUPPORT_SCHEMA_NOTE,
        },
        "aggregation": AGGREGATION_METADATA,
        "sources": source_records,
        "replacement_policy": {
            REPAIRED_CONTROL_ROOT: CONTROL_FORECAST_REPAIR.source_id,
            f"support:{REPAIRED_CONTROL_ROOT}": CONTROL_SUPPORT_REPAIR.source_id,
            REPAIRED_DYSTS_ROOT: DYSTS_REPAIR.source_id,
        },
        "filters": {
            "controlled_roots": list(CONTROL_ROOTS),
            "controlled_systems": list(CONTROL_SYSTEMS),
            "controlled_seeds": list(range(15)),
            "support_scheme": "absolute:0.001",
            "support_subset": "deep",
            "family_jaccard_threshold": 0.5,
            **ALIGNMENT_FILTER_METADATA,
            "dysts_roots": list(DYSTS_ROOTS),
            "dysts_systems": list(DYSTS_SYSTEMS),
            "dysts_seeds": list(range(15)),
        },
        "outputs": {
            name: {
                "rows": int(len(frames[name])),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "columns": list(frames[name].columns),
            }
            for name, payload in outputs.items()
        },
    }
    outputs["main_paper_evidence_provenance.json"] = (
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return outputs


def compact_existing_support_outputs(output_dir: Path) -> dict[str, bytes]:
    """Authenticate and migrate the locally frozen compact evidence packet."""

    provenance_path = output_dir / "main_paper_evidence_provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(provenance_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    output_specs = provenance.get("outputs", {})
    for name, spec in output_specs.items():
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        expected = spec.get("sha256") if isinstance(spec, dict) else None
        if actual != expected:
            raise ValueError(
                f"Frozen evidence hash mismatch for {name}: "
                f"expected {expected}, got {actual}"
            )

    support_path = output_dir / "controlled_support_rows.csv"
    support = pd.read_csv(
        support_path,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    missing = [column for column in CONTROL_SUPPORT_SOURCE_COLUMNS if column not in support]
    if missing:
        raise ValueError(f"Frozen support packet is missing columns: {missing}")
    support["observed_label_count"] = support["system_name"].map(
        ALIGNMENT_OBSERVED_LABEL_COUNTS
    )
    if support["observed_label_count"].isna().any():
        raise ValueError("Missing frozen observed-label count for controlled support row")
    support = support.loc[:, CONTROL_SUPPORT_COLUMNS]
    support_payload = csv_bytes(support)

    provenance["schema_version"] = EVIDENCE_SCHEMA_VERSION
    provenance["schema_notes"] = {
        "controlled_support_rows.csv": CONTROL_SUPPORT_SCHEMA_NOTE,
    }
    provenance["aggregation"] = AGGREGATION_METADATA
    provenance.setdefault("filters", {}).update(ALIGNMENT_FILTER_METADATA)
    provenance["outputs"]["controlled_support_rows.csv"] = {
        "rows": int(len(support)),
        "bytes": len(support_payload),
        "sha256": hashlib.sha256(support_payload).hexdigest(),
        "columns": list(support.columns),
    }
    provenance_payload = (
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return {
        "controlled_support_rows.csv": support_payload,
        "main_paper_evidence_provenance.json": provenance_payload,
    }


def write_or_check(outputs: dict[str, bytes], output_dir: Path, *, check: bool) -> None:
    if check:
        mismatches = []
        for name, expected in outputs.items():
            path = output_dir / name
            if not path.is_file() or path.read_bytes() != expected:
                mismatches.append(name)
        if mismatches:
            raise SystemExit(f"Frozen evidence is stale: {', '.join(mismatches)}")
        print(f"Frozen evidence is current in {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        (output_dir / name).write_bytes(payload)
        print(f"wrote {output_dir / name} ({len(payload):,} bytes)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(os.environ.get("SKAE_RESULTS_ROOT", ROOT / "results")),
        help="Directory containing the pinned collector result directories.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check", action="store_true", help="Fail if tracked outputs differ.")
    parser.add_argument(
        "--compact-existing-support",
        action="store_true",
        help=(
            "Authenticate the existing frozen packet and migrate/check only "
            "its compact active support schema; no external results root needed."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.compact_existing_support:
        outputs = compact_existing_support_outputs(output_dir)
    else:
        outputs = make_outputs(args.results_root.resolve())
    write_or_check(outputs, output_dir, check=args.check)


if __name__ == "__main__":
    main()
