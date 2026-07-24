"""Build the authenticated Allen--Cahn periodic-reencoding paper packet."""

from __future__ import annotations

import argparse
import csv
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from experiments.neurips_2026.evidence.allen_cahn_periodic_reencoding_contract import (
    PACKET_ID,
    TABLE_ROW_IDS,
    authenticate_summary,
    read_json,
    sha256_path,
    verify_frozen_sources,
)
from experiments.neurips_2026.evidence.allen_cahn_periodic_reencoding_reduction import (
    build_compact,
    validate_compact,
)
from experiments.neurips_2026.evidence.allen_cahn_periodic_reencoding_rendering import (
    comparison_table_bytes,
    render_periodic_figure,
)
from experiments.neurips_2026.paths import (
    PAPER_DATA_DIR,
    PAPER_EVIDENCE_DIR,
    PAPER_TABLE_DIR,
)


DATA_SUBDIR = "allen_cahn_periodic_reencoding_confirmation"
SUMMARY_NAME = "summary.json"
ROWS_NAME = "comparison_rows.csv"
MANIFEST_NAME = "evidence_manifest.json"
TABLE_NAME = "table_allen_cahn_periodic_reencoding_confirmation.tex"
FIGURE_PDF_NAME = "fig_allen_cahn_periodic_reencoding_confirmation.pdf"
FIGURE_PNG_NAME = "fig_allen_cahn_periodic_reencoding_confirmation.png"
ROW_FIELDS = (
    "row_id", "family", "endpoint", "contrast", "baseline_policy",
    "comparison_policy", "baseline_mean_mse", "comparison_mean_mse",
    "relative_reduction", "ci95_lower", "ci95_upper", "one_sided_p",
    "wins_out_of_10", "inference_role", "status",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(ROW_FIELDS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        encoded = {}
        for field in ROW_FIELDS:
            value = row[field]
            if value is None:
                encoded[field] = ""
            elif isinstance(value, float):
                encoded[field] = format(value, ".17g")
            else:
                encoded[field] = value
        writer.writerow(encoded)
    return handle.getvalue().encode()


def _paths(data_dir: Path, figure_dir: Path, table_dir: Path) -> dict[str, Path]:
    packet_dir = data_dir / DATA_SUBDIR
    return {
        "summary": packet_dir / SUMMARY_NAME,
        "rows": packet_dir / ROWS_NAME,
        "manifest": packet_dir / MANIFEST_NAME,
        "figure_pdf": figure_dir / FIGURE_PDF_NAME,
        "figure_png": figure_dir / FIGURE_PNG_NAME,
        "table": table_dir / TABLE_NAME,
    }


def _builder_source_hashes() -> dict[str, str]:
    return {
        "builder_sha256": sha256_path(Path(__file__)),
        "contract_sha256": sha256_path(
            Path(authenticate_summary.__code__.co_filename)
        ),
        "reduction_sha256": sha256_path(Path(build_compact.__code__.co_filename)),
        "rendering_sha256": sha256_path(
            Path(render_periodic_figure.__code__.co_filename)
        ),
    }


def build(
    source_root: Path,
    expected_summary_receipt_sha256: str,
    data_dir: Path = PAPER_DATA_DIR,
    figure_dir: Path = PAPER_EVIDENCE_DIR,
    table_dir: Path = PAPER_TABLE_DIR,
) -> dict[str, Path]:
    paths = _paths(data_dir, figure_dir, table_dir)
    existing = [str(path) for path in paths.values() if path.exists()]
    _require(not existing, f"Refusing to overwrite evidence artifacts: {existing}")
    compact, rows = build_compact(
        authenticate_summary(source_root, expected_summary_receipt_sha256)
    )
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_bytes(_json_bytes(compact))
    paths["rows"].write_bytes(_csv_bytes(rows))
    paths["table"].write_bytes(comparison_table_bytes(rows))
    render_periodic_figure(compact, paths["figure_pdf"], paths["figure_png"])
    manifest = {
        "schema_version": 1, "packet_id": PACKET_ID,
        "fixed_panel_ids": compact["fixed_display_contract"]["panel_ids"],
        "fixed_table_row_ids": list(TABLE_ROW_IDS),
        "source_authentication": compact["source_authentication"],
        "builder_sources": _builder_source_hashes(),
        "outputs": {
            key: {"path": path.name, "sha256": sha256_path(path)}
            for key, path in paths.items() if key != "manifest"
        },
        "portable_check": "uv run skae-paper build allen-cahn-periodic-reencoding --check",
    }
    paths["manifest"].write_bytes(_json_bytes(manifest))
    check_packet(data_dir, figure_dir, table_dir)
    return paths


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows = []
    numeric = {"baseline_mean_mse", "comparison_mean_mse", "relative_reduction", "ci95_lower", "ci95_upper", "one_sided_p"}
    for source in raw:
        row: dict[str, Any] = dict(source)
        for key in numeric:
            row[key] = None if source[key] == "" else float(source[key])
        row["wins_out_of_10"] = None if source["wins_out_of_10"] == "" else int(source["wins_out_of_10"])
        rows.append(row)
    return rows


def check_packet(
    data_dir: Path = PAPER_DATA_DIR,
    figure_dir: Path = PAPER_EVIDENCE_DIR,
    table_dir: Path = PAPER_TABLE_DIR,
) -> None:
    verify_frozen_sources()
    paths = _paths(data_dir, figure_dir, table_dir)
    manifest = read_json(paths["manifest"])
    _require(manifest.get("schema_version") == 1, "Evidence manifest version drifted")
    _require(manifest.get("packet_id") == PACKET_ID, "Evidence manifest ID drifted")
    _require(
        manifest.get("fixed_panel_ids")
        == [
            "validation_risk", "heldout_curves", "paired_ratios",
            "accuracy_refresh_frontier",
        ],
        "Manifest panel roster drifted",
    )
    _require(tuple(manifest["fixed_table_row_ids"]) == TABLE_ROW_IDS, "Manifest table roster drifted")
    expected_outputs = set(paths) - {"manifest"}
    _require(set(manifest["outputs"]) == expected_outputs, "Manifest output roster drifted")
    for key, record in manifest["outputs"].items():
        _require(set(record) == {"path", "sha256"}, f"Manifest output schema drifted: {key}")
        _require(paths[key].name == record["path"], f"Manifest path drifted: {key}")
        _require(sha256_path(paths[key]) == record["sha256"], f"Output hash drifted: {key}")
    _require(manifest["builder_sources"] == _builder_source_hashes(), "Evidence builder source drifted")
    summary, rows = read_json(paths["summary"]), _read_rows(paths["rows"])
    _require(
        manifest["source_authentication"] == summary["source_authentication"],
        "Manifest/summary source authentication drifted",
    )
    validate_compact(summary, rows)
    _require(paths["table"].read_bytes() == comparison_table_bytes(rows), "LaTeX table is not deterministic")
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        pdf, png = root / FIGURE_PDF_NAME, root / FIGURE_PNG_NAME
        render_periodic_figure(summary, pdf, png)
        _require(pdf.read_bytes() == paths["figure_pdf"].read_bytes(), "PDF is not deterministic")
        _require(png.read_bytes() == paths["figure_png"].read_bytes(), "PNG is not deterministic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--expected-summary-receipt-sha256")
    parser.add_argument("--output-data-dir", type=Path, default=PAPER_DATA_DIR)
    parser.add_argument("--output-figure-dir", type=Path, default=PAPER_EVIDENCE_DIR)
    parser.add_argument("--output-table-dir", type=Path, default=PAPER_TABLE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        _require(args.source_root is None and args.expected_summary_receipt_sha256 is None, "Portable --check accepts no source arguments")
        check_packet(args.output_data_dir, args.output_figure_dir, args.output_table_dir)
        result = {"command": "check", "status": "ok"}
    else:
        _require(args.source_root is not None, "--source-root is required when building")
        _require(args.expected_summary_receipt_sha256 is not None, "--expected-summary-receipt-sha256 is required")
        paths = build(
            args.source_root, args.expected_summary_receipt_sha256,
            args.output_data_dir, args.output_figure_dir, args.output_table_dir,
        )
        result = {"command": "build", "status": "ok", "outputs": {key: str(path) for key, path in paths.items()}}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
