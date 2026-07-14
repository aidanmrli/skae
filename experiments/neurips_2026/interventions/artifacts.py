"""Build the active coordinate-intervention curves and H=21 table."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path

import numpy as np

from experiments.neurips_2026.interventions.evaluate import (
    _configure_paper_style,
    _plot_drop_absolute_curves_with_bands,
    _plot_random_absolute_band,
)
from experiments.neurips_2026.interventions.protocol import (
    validate_intervention_protocol_record,
)
from experiments.neurips_2026.paths import PAPER_DATA_DIR, PAPER_EVIDENCE_DIR, REPO_ROOT


ROOT = REPO_ROOT
DATA_DIR = PAPER_DATA_DIR / "interventions"
FIGURE_DIR = PAPER_EVIDENCE_DIR
TABLE_DIR = FIGURE_DIR / "_tables"
PROVENANCE = DATA_DIR / "provenance.json"
TABLE_NAME = "table_support_coordinate_interventions_h21.tex"
DROP_FIGURE = "fig_support_coordinate_dropping_accumulated_mse.pdf"
RANDOM_FIGURE = "fig_support_coordinate_random_shuffle_accumulated_mse.pdf"
TRAJECTORY_FIGURE = "fig_support_coordinate_trajectories_random_support_19.pdf"
DROP_CREATION_DATE = b"D:20260506204244-04'00'"
RANDOM_CREATION_DATE = b"D:20260506225614-04'00'"


def read_rows(filename: str) -> list[dict[str, str]]:
    with gzip.open(DATA_DIR / filename, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_frozen_inputs() -> dict[str, object]:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    for filename, record in provenance["outputs"].items():
        path = DATA_DIR / filename
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != record["sha256"]:
            raise ValueError(f"Frozen input hash mismatch for {filename}")
    for condition in ("coordinate_dropping", "random_support"):
        validate_intervention_protocol_record(provenance["protocol"][condition])
    return provenance


def fmt_num(value: float, *, sig: int = 3) -> str:
    if not math.isfinite(value):
        return "--"
    if value == 0.0:
        return "0"
    absolute = abs(value)
    if absolute >= 1000.0 or absolute < 1e-3:
        exponent = int(math.floor(math.log10(absolute)))
        mantissa = value / (10**exponent)
        return rf"{mantissa:.{sig - 1}f}{{\times}}10^{{{exponent}}}"
    decimals = max(0, sig - 1 - int(math.floor(math.log10(absolute))))
    return f"{value:.{decimals}f}"


def summary(values: list[float]) -> tuple[float, float, float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    q25, q75 = np.percentile(array, [25, 75])
    return (
        float(np.mean(array)),
        float(np.std(array, ddof=1)),
        float(np.median(array)),
        float(q25),
        float(q75),
    )


def table_bytes(drop_rows: list[dict[str, str]], random_rows: list[dict[str, str]]) -> bytes:
    conditions = (
        ("Standard", "baseline"),
        ("Drop top-1", "drop_top_1"),
        ("Drop top-2", "drop_top_2"),
        ("Drop top-3", "drop_top_3"),
        ("Drop top-5", "drop_top_5"),
        ("Drop top-10", "drop_top_10"),
    )
    rows = [
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"Rollout & Mean \(\pm\) SD & Median [IQR] \\",
        r"\midrule",
    ]
    for display, condition in conditions:
        values = [
            float(row["cumulative_mse_sum"])
            for row in drop_rows
            if row["condition"] == condition and int(row["horizon"]) == 21
        ]
        mean, std, median, q25, q75 = summary(values)
        if condition == "baseline":
            formatted = tuple(f"{value:.4f}" for value in (mean, std, median, q25, q75))
        else:
            formatted = tuple(fmt_num(value) for value in (mean, std, median, q25, q75))
        rows.append(
            f"{display} & ${formatted[0]}{{\\pm}}{formatted[1]}$ & "
            f"${formatted[2]}\\,[{formatted[3]},{formatted[4]}]$ \\\\"
        )
    random_values = [
        float(row["cumulative_mse_sum"])
        for row in random_rows
        if row["condition"].startswith("random_support_") and int(row["horizon"]) == 21
    ]
    mean, std, median, q25, q75 = summary(random_values)
    rows.append(
        f"Random support & ${fmt_num(mean)}{{\\pm}}{fmt_num(std)}$ & "
        f"${fmt_num(median)}\\,[{fmt_num(q25)},{fmt_num(q75)}]$ \\\\"
    )
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    return ("\n".join(rows) + "\n").encode("utf-8")


def pin_pdf_creation_date(path: Path, creation_date: bytes) -> None:
    content = path.read_bytes()
    matches = list(re.finditer(rb"D:\d{14}[+-]\d{2}'\d{2}'", content))
    if len(matches) != 1 or len(matches[0].group()) != len(creation_date):
        raise ValueError(f"Unexpected PDF creation-date metadata in {path}")
    path.write_bytes(content[: matches[0].start()] + creation_date + content[matches[0].end() :])


def build(output_figure_dir: Path, output_table_dir: Path) -> None:
    drop_points = read_rows("drop_point_metrics.csv.gz")
    random_horizons = read_rows("random_horizon_metrics.csv.gz")
    random_points = read_rows("random_point_metrics.csv.gz")
    output_figure_dir.mkdir(parents=True, exist_ok=True)
    output_table_dir.mkdir(parents=True, exist_ok=True)
    _configure_paper_style()
    _plot_drop_absolute_curves_with_bands(
        drop_points,
        output_dir=output_figure_dir,
        plot_formats=("pdf",),
        point_metric="cumulative_mse_sum",
        ylabel="Mean accumulated MSE",
        filename_stem=DROP_FIGURE.removesuffix(".pdf"),
        title="Coordinate dropping",
    )
    pin_pdf_creation_date(output_figure_dir / DROP_FIGURE, DROP_CREATION_DATE)
    _plot_random_absolute_band(
        random_horizons,
        random_points,
        output_dir=output_figure_dir,
        plot_formats=("pdf",),
        horizon_metric="cumulative_mse_sum_mean",
        point_metric="cumulative_mse_sum",
        ylabel="Accumulated MSE",
        filename_stem=RANDOM_FIGURE.removesuffix(".pdf"),
        title="Random support shuffle",
    )
    pin_pdf_creation_date(output_figure_dir / RANDOM_FIGURE, RANDOM_CREATION_DATE)
    (output_table_dir / TABLE_NAME).write_bytes(table_bytes(drop_points, random_points))


def check(provenance: dict[str, object]) -> None:
    with tempfile.TemporaryDirectory(prefix="skae-intervention-check-") as temp:
        root = Path(temp)
        build(root / "figures", root / "tables")
        comparisons = (
            (root / "figures" / DROP_FIGURE, FIGURE_DIR / DROP_FIGURE),
            (root / "figures" / RANDOM_FIGURE, FIGURE_DIR / RANDOM_FIGURE),
            (root / "tables" / TABLE_NAME, TABLE_DIR / TABLE_NAME),
        )
        stale = [target.name for generated, target in comparisons if generated.read_bytes() != target.read_bytes()]
    trajectory_path = FIGURE_DIR / TRAJECTORY_FIGURE
    relative = trajectory_path.relative_to(ROOT).as_posix()
    pinned = provenance["active_artifacts_at_freeze"][relative]["sha256"]
    if hashlib.sha256(trajectory_path.read_bytes()).hexdigest() != pinned:
        stale.append(TRAJECTORY_FIGURE)
    if stale:
        raise SystemExit(f"Intervention artifacts are stale: {', '.join(stale)}")
    print("Intervention table and three active PDFs match frozen evidence.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    provenance = verify_frozen_inputs()
    if args.check:
        check(provenance)
    else:
        build(FIGURE_DIR, TABLE_DIR)


if __name__ == "__main__":
    main()
