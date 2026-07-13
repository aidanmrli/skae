#!/usr/bin/env python3
"""Regenerate paper-style support-coordinate intervention figures from CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from evaluate_support_coordinate_interventions import (
    _configure_paper_style,
    _plot_drop_absolute_curves_with_bands,
    _plot_random_absolute_band,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result_dir",
        required=True,
        help="Directory containing intervention_horizon_metrics.csv and intervention_point_metrics.csv.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for regenerated figures. Defaults to --result_dir.",
    )
    parser.add_argument("--plot_format", default="pdf,png")
    parser.add_argument("--filename_prefix", default="fig_support_coordinate_")
    parser.add_argument("--only", choices=["all", "drop", "random"], default="all")
    return parser.parse_args()


def _read_csv(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = _parse_args()
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir) if args.output_dir else result_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_formats = [item.strip().lstrip(".") for item in args.plot_format.split(",") if item.strip()]

    horizon_rows = _read_csv(result_dir / "intervention_horizon_metrics.csv")
    point_rows = _read_csv(result_dir / "intervention_point_metrics.csv")

    _configure_paper_style()
    prefix = args.filename_prefix

    if args.only in {"all", "drop"}:
        _plot_drop_absolute_curves_with_bands(
            point_rows,
            output_dir=output_dir,
            plot_formats=plot_formats,
            point_metric="cumulative_mse_sum",
            ylabel="Mean accumulated MSE",
            filename_stem=f"{prefix}dropping_accumulated_mse",
            title="Coordinate dropping",
        )
        _plot_drop_absolute_curves_with_bands(
            point_rows,
            output_dir=output_dir,
            plot_formats=plot_formats,
            point_metric="mse_at_h",
            ylabel="Mean MSE at horizon",
            filename_stem=f"{prefix}dropping_horizon_mse",
            title="Coordinate dropping",
        )
    if args.only in {"all", "random"}:
        _plot_random_absolute_band(
            horizon_rows,
            point_rows,
            output_dir=output_dir,
            plot_formats=plot_formats,
            horizon_metric="cumulative_mse_sum_mean",
            point_metric="cumulative_mse_sum",
            ylabel="Accumulated MSE",
            filename_stem=f"{prefix}random_shuffle_accumulated_mse",
            title="Random support shuffle",
        )
        _plot_random_absolute_band(
            horizon_rows,
            point_rows,
            output_dir=output_dir,
            plot_formats=plot_formats,
            horizon_metric="mse_at_h_mean",
            point_metric="mse_at_h",
            ylabel="MSE at horizon",
            filename_stem=f"{prefix}random_shuffle_horizon_mse",
            title="Random support shuffle",
        )

    print(f"Wrote paper-style support-coordinate figures to {output_dir}")


if __name__ == "__main__":
    main()
