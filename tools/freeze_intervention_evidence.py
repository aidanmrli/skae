"""Freeze portable evidence for the paper's coordinate-intervention case study."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT / "results" / "support_coordinate_interventions_20260506"
DEFAULT_OUTPUT_DIR = (
    ROOT / "docs" / "figures" / "neurips_paper_2026" / "_data" / "interventions"
)
DROP_RUN = "gated_local_linear_lista_seed0_n100"
RANDOM_RUN = "gated_local_linear_lista_seed0_n100_random"
NUM_INITIAL_POINTS = 100


@dataclass(frozen=True)
class Source:
    source_id: str
    run: str
    filename: str
    sha256: str


SOURCES = (
    Source("drop_initial_points", DROP_RUN, "initial_points.csv", "3bad4023ab53224cbd1f3f983af6d07ba7b90a9d7bb32c1e7ef3e5eab92c96d6"),
    Source("drop_horizon_metrics", DROP_RUN, "intervention_horizon_metrics.csv", "17953e480acd842e7eceef0ea9ee31b0294f7a95d6d768382fc3eb2e856f52b9"),
    Source("drop_point_metrics", DROP_RUN, "intervention_point_metrics.csv", "3b546122dd38b5014b59d58ec837f67695f842fac4ce1941a6452338bbadec11"),
    Source("drop_summary", DROP_RUN, "run_summary.json", "8d30ad47973504ff9cea38cfa855f7ca1a0bd3212eb4e41d80c3e48c7e0acb78"),
    Source("random_initial_points", RANDOM_RUN, "initial_points.csv", "ce10ac97aa70c7d272fda30b73528089c685641476ce4ab804021921045e9a90"),
    Source("random_horizon_metrics", RANDOM_RUN, "intervention_horizon_metrics.csv", "0ab8e25d06f82f4a92b9480a0a0861d5bfabd0060bbc27fa9eded2a040e8fe2e"),
    Source("random_point_metrics", RANDOM_RUN, "intervention_point_metrics.csv", "9c335102079c9c0949df25b080d3cd5c9354e814e2f2286b1697da0f47bd09f8"),
    Source("random_support_moves", RANDOM_RUN, "random_support_moves.csv", "a7ee159a144dcc4bd4405cf85c72d4516ba230b687352496a99af0db32035636"),
    Source("random_summary", RANDOM_RUN, "run_summary.json", "7c6b894a34f0bf0436483f80ea307df4ede88afb622dd5bb124e03cb382391b9"),
)
SOURCE_BY_ID = {source.source_id: source for source in SOURCES}

HORIZON_COLUMNS = (
    "condition",
    "intervention_type",
    "drop_count",
    "random_repeat",
    "horizon",
    "num_initial_points",
    "mse_at_h_mean",
    "mse_at_h_median",
    "cumulative_mse_sum_mean",
    "cumulative_mse_sum_median",
    "cumulative_mse_mean_mean",
    "basin_mismatch_vs_true_future_fraction",
    "basin_mismatch_vs_initial_fraction",
    "any_basin_mismatch_vs_true_future_fraction",
)
POINT_COLUMNS = (
    "condition",
    "intervention_type",
    "drop_count",
    "random_repeat",
    "point_id",
    "horizon",
    "mse_at_h",
    "cumulative_mse_sum",
    "cumulative_mse_mean",
    "initial_basin",
    "true_basin_at_h",
    "pred_basin_at_h",
    "basin_mismatch_vs_true_future",
    "basin_mismatch_vs_initial",
)
INITIAL_COLUMNS = (
    "point_id",
    "trajectory_index",
    "time_index",
    "initial_basin",
    "support_size",
    "top_indices",
    "top_abs_values",
    "x0",
)
MOVE_COLUMNS = ("random_repeat", "point_id", "source_indices", "destination_indices")
OUTPUT_SPECS = (
    ("drop_horizon_metrics.csv.gz", "drop_horizon_metrics", HORIZON_COLUMNS, 121),
    ("drop_point_metrics.csv.gz", "drop_point_metrics", POINT_COLUMNS, 12_100),
    ("drop_initial_points.csv.gz", "drop_initial_points", INITIAL_COLUMNS, 100),
    ("random_horizon_metrics.csv.gz", "random_horizon_metrics", HORIZON_COLUMNS, 231),
    ("random_point_metrics.csv.gz", "random_point_metrics", POINT_COLUMNS, 23_100),
    ("random_initial_points.csv.gz", "random_initial_points", INITIAL_COLUMNS, 100),
    ("random_support_moves.csv.gz", "random_support_moves", MOVE_COLUMNS, 2_000),
)
PROTOCOL_FIELDS = (
    "root_label",
    "system_key",
    "system_name",
    "seed",
    "support_definition",
    "depth_slice_mode",
    "label_source",
    "eval_seed",
    "num_candidate_trajectories",
    "trajectory_length",
    "max_horizon",
    "mean_initial_support_size",
    "min_initial_support_size",
    "max_initial_support_size",
    "horizons",
    "conditions",
    "worst_intervention_condition",
)
ACTIVE_ARTIFACTS = (
    "docs/figures/neurips_paper_2026/_tables/table_support_coordinate_interventions_h21.tex",
    "docs/figures/neurips_paper_2026/fig_support_coordinate_dropping_accumulated_mse.pdf",
    "docs/figures/neurips_paper_2026/fig_support_coordinate_random_shuffle_accumulated_mse.pdf",
    "docs/figures/neurips_paper_2026/fig_support_coordinate_trajectories_random_support_19.pdf",
)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def verified_source(source_root: Path, source_id: str) -> Path:
    source = SOURCE_BY_ID[source_id]
    path = source_root / source.run / source.filename
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_bytes(path.read_bytes())
    if actual != source.sha256:
        raise ValueError(f"SHA256 mismatch for {source_id}: {actual}")
    return path


def sanitized_csv(source_root: Path, source_id: str, columns: tuple[str, ...]) -> tuple[bytes, int]:
    path = verified_source(source_root, source_id)
    output = io.StringIO(newline="")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(columns).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns in {source_id}: {sorted(missing)}")
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        rows = 0
        for row in reader:
            writer.writerow({column: row[column] for column in columns})
            rows += 1
    raw = output.getvalue().encode("utf-8")
    compressed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, compresslevel=9, mtime=0) as handle:
        handle.write(raw)
    return compressed.getvalue(), rows


def protocol_record(source_root: Path, source_id: str) -> dict[str, object]:
    summary = json.loads(verified_source(source_root, source_id).read_text(encoding="utf-8"))
    record = {field: summary[field] for field in PROTOCOL_FIELDS}
    record["num_initial_points"] = NUM_INITIAL_POINTS
    return record


def start_selection_record(source_root: Path) -> dict[str, object]:
    columns = ("point_id", "trajectory_index", "time_index", "initial_basin")

    def records(source_id: str) -> list[tuple[str, ...]]:
        with verified_source(source_root, source_id).open(
            newline="", encoding="utf-8"
        ) as handle:
            return [tuple(row[column] for column in columns) for row in csv.DictReader(handle)]

    drop = records("drop_initial_points")
    random = records("random_initial_points")
    if drop != random:
        raise ValueError("Drop and random-support artifacts do not use identical starts")
    basin_counts = Counter(record[-1] for record in drop)
    if basin_counts != {"0": 34, "1": 33, "2": 33}:
        raise ValueError(f"Unexpected intervention start balance: {basin_counts}")
    return {
        "candidate_subset": "tie-inclusive per-native-label center-margin >= q75",
        "stability_requirement": (
            "native label remains equal to the initial label through max_horizon=21"
        ),
        "balance_rule": "equal allocation over sorted observed native labels; remainder to earliest label",
        "realized_native_label_counts": dict(sorted(basin_counts.items())),
        "coordinate_drop_and_random_support_starts_identical": True,
    }


def build_outputs(source_root: Path) -> dict[str, bytes]:
    outputs: dict[str, bytes] = {}
    output_records: dict[str, object] = {}
    for output_name, source_id, columns, expected_rows in OUTPUT_SPECS:
        content, rows = sanitized_csv(source_root, source_id, columns)
        if rows != expected_rows:
            raise ValueError(f"Expected {expected_rows} rows from {source_id}, found {rows}")
        outputs[output_name] = content
        output_records[output_name] = {
            "rows": rows,
            "bytes": len(content),
            "sha256": sha256_bytes(content),
            "columns": list(columns),
            "compression": "gzip level 9, mtime 0",
        }
    source_records = [
        {
            "source_id": source.source_id,
            "relative_to_source_root": f"{source.run}/{source.filename}",
            "sha256": source.sha256,
        }
        for source in SOURCES
    ]
    artifact_records = {}
    for relative in ACTIVE_ARTIFACTS:
        path = ROOT / relative
        if path.is_file():
            artifact_records[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_bytes(path.read_bytes()),
            }
    provenance = {
        "schema_version": 2,
        "description": "Sanitized evidence for the one-checkpoint coordinate-intervention case study.",
        "protocol": {
            "coordinate_dropping": protocol_record(source_root, "drop_summary"),
            "random_support": protocol_record(source_root, "random_summary"),
            "start_selection": start_selection_record(source_root),
            "random_support_summary_unit": (
                "pooled dependent point-shuffle outcomes; "
                "100 starts x 20 shuffles = 2000 per horizon"
            ),
        },
        "sources": source_records,
        "outputs": output_records,
        "active_artifacts_at_freeze": artifact_records,
        "limitations": {
            "scope": "one system, one LISTA checkpoint, seed 0",
            "trajectory_figure": "requires the external checkpoint and is pinned but not regenerated from compact rows",
        },
    }
    outputs["provenance.json"] = (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode()
    return outputs


def write_or_check(outputs: dict[str, bytes], output_dir: Path, *, check: bool) -> None:
    if check:
        stale = [
            name
            for name, content in outputs.items()
            if not (output_dir / name).is_file() or (output_dir / name).read_bytes() != content
        ]
        if stale:
            raise SystemExit(f"Frozen intervention evidence is stale: {', '.join(stale)}")
        print("Frozen intervention evidence is current.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (output_dir / name).write_bytes(content)
        print(f"wrote {output_dir / name} ({len(content):,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(os.environ.get("SKAE_INTERVENTION_SOURCE_ROOT", DEFAULT_SOURCE_ROOT)),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    write_or_check(
        build_outputs(args.source_root.resolve()),
        args.output_dir.resolve(),
        check=args.check,
    )


if __name__ == "__main__":
    main()
