"""Plot the Dysts cross-system median horizon-MSE curves for the class report.

This figure combines:
- short-horizon Dysts benchmark rows for the sparse MLP and dense LISTA roots
- long-horizon Dysts reevaluation rows for the sparse MLP, zero-sparsity MLP,
  and dense LISTA roots
- a local markdown fallback for the zero-sparsity MLP at H1000, because that
  root is not present in the short-horizon CSV packet currently checked in

The plotted statistic is the cross-system median of the per-system seed IQM of
the best-periodic MSE.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
SHORT_CSV = REPO_ROOT / "results/paper_followup_recipes_200k_mlp_blockdiag_dysts_20260415/collect/forecasting_rows.csv"
LONG_CSV = REPO_ROOT / "results/dysts_long_horizon_eval_20260414/collect/forecasting_rows.csv"
ZERO_SHORT_FALLBACK = REPO_ROOT / "docs/DYSTS_SYSTEM_PERFORMANCE_20260331.md"
DEFAULT_OUTPUT = REPO_ROOT / "docs/figures/class_project/dysts_horizon_mse_curve.png"

PLOT_HORIZONS = (100, 500, 1000, 5000, 10000, 20000, 30000)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    root_label: str
    display_name: str
    color: str
    marker: str
    annotation_offset: tuple[int, int]


MODEL_SPECS = (
    ModelSpec(
        key="sparse_mlp",
        root_label="generic_sparse_ns200k_best",
        display_name="Sparse MLP ($\\ell_1$)",
        color="#0072B2",
        marker="o",
        annotation_offset=(-18, 10),
    ),
    ModelSpec(
        key="zero_mlp",
        root_label="generic_sparse_sc0_ns200k_best",
        display_name="Zero-sparsity MLP",
        color="#E69F00",
        marker="s",
        annotation_offset=(18, -14),
    ),
    ModelSpec(
        key="dense_lista",
        root_label="lista_dense_promoted_stage4",
        display_name="LISTA",
        color="#009E73",
        marker="D",
        annotation_offset=(0, 13),
    ),
)

MODEL_BY_ROOT = {spec.root_label: spec for spec in MODEL_SPECS}
MODEL_BY_KEY = {spec.key: spec for spec in MODEL_SPECS}


def interquartile_mean(values: list[float]) -> float:
    """Compute the empirical IQM used in the report tables."""

    if not values:
        raise ValueError("IQM requires at least one value")

    arr = np.sort(np.asarray(values, dtype=float))
    n = float(arr.size)
    lower = 0.25
    upper = 0.75
    total = 0.0

    for idx, value in enumerate(arr):
        seg_lo = idx / n
        seg_hi = (idx + 1.0) / n
        overlap = max(0.0, min(seg_hi, upper) - max(seg_lo, lower))
        if overlap > 0.0:
            total += float(value) * overlap

    return total / (upper - lower)


def _parse_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw or raw.upper() == "N/A":
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def _collect_cross_system_medians(
    csv_path: Path,
    horizons: tuple[int, ...],
) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, str]]]:
    grouped: DefaultDict[tuple[str, int, str], list[float]] = defaultdict(list)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            root_label = row.get("root_label", "")
            model_spec = MODEL_BY_ROOT.get(root_label)
            if model_spec is None:
                continue
            system_key = row.get("system_key", "")
            if not system_key.startswith("dysts:"):
                continue
            for horizon in horizons:
                raw = row.get(f"h{horizon}_best_periodic_mse") or row.get(f"h{horizon}_best_periodic_mean")
                value = _parse_float(raw)
                if value is None:
                    continue
                grouped[(model_spec.key, horizon, system_key)].append(value)

    medians: dict[str, dict[int, float]] = {spec.key: {} for spec in MODEL_SPECS}
    sources: dict[str, dict[int, str]] = {spec.key: {} for spec in MODEL_SPECS}
    per_system_iqms: DefaultDict[tuple[str, int], list[float]] = defaultdict(list)

    for (model_key, horizon, _system_key), seed_values in grouped.items():
        per_system_iqms[(model_key, horizon)].append(interquartile_mean(seed_values))

    for (model_key, horizon), system_iqms in per_system_iqms.items():
        medians[model_key][horizon] = float(np.median(np.asarray(system_iqms, dtype=float)))
        sources[model_key][horizon] = str(csv_path.relative_to(REPO_ROOT))

    return medians, sources


def _load_zero_sparse_h1000_fallback(note_path: Path) -> float:
    in_section = False
    values: list[float] = []

    for raw_line in note_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "## Per-system IQM at H1000":
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if not in_section or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or cells[0] in {"System", "---"}:
            continue
        if set(cells[0]) == {"-"}:
            continue
        if len(cells) < 4:
            continue
        value = _parse_float(cells[3])
        if value is not None:
            values.append(value)

    if not values:
        raise RuntimeError(f"Could not recover zero-sparsity H1000 values from {note_path}")
    return float(np.median(np.asarray(values, dtype=float)))


def load_series() -> tuple[dict[str, dict[int, float]], dict[str, dict[int, str]]]:
    short_medians, short_sources = _collect_cross_system_medians(SHORT_CSV, (100, 500, 1000))
    long_medians, long_sources = _collect_cross_system_medians(LONG_CSV, (5000, 10000, 20000, 30000))

    series: dict[str, dict[int, float]] = {spec.key: {} for spec in MODEL_SPECS}
    sources: dict[str, dict[int, str]] = {spec.key: {} for spec in MODEL_SPECS}

    for model_key, horizon_map in short_medians.items():
        series[model_key].update(horizon_map)
        sources[model_key].update(short_sources[model_key])
    for model_key, horizon_map in long_medians.items():
        series[model_key].update(horizon_map)
        sources[model_key].update(long_sources[model_key])

    if 1000 not in series["zero_mlp"]:
        series["zero_mlp"][1000] = _load_zero_sparse_h1000_fallback(ZERO_SHORT_FALLBACK)
        sources["zero_mlp"][1000] = str(ZERO_SHORT_FALLBACK.relative_to(REPO_ROOT))

    return series, sources


def _format_value(value: float) -> str:
    if value < 0.01:
        return f"{value:.2e}"
    if value < 1.0:
        return f"{value:.3f}"
    return f"{value:.2f}"


def save_plot(series: dict[str, dict[int, float]], output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path.with_suffix(".pdf")

    x_positions = {horizon: idx for idx, horizon in enumerate(PLOT_HORIZONS)}

    fig, ax = plt.subplots(figsize=(10.6, 5.8), constrained_layout=True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", which="major", linestyle=":", linewidth=0.9, alpha=0.4)
    ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.6, alpha=0.2)

    ax.set_yscale("log")
    ax.set_xlabel("Forecast Horizon $H$")
    ax.set_ylabel("Cross-system median seed-IQM MSE")
    ax.set_xticks(range(len(PLOT_HORIZONS)))
    ax.set_xticklabels([str(horizon) for horizon in PLOT_HORIZONS])

    for spec in MODEL_SPECS:
        horizon_values = [(h, series[spec.key][h]) for h in PLOT_HORIZONS if h in series[spec.key]]
        if not horizon_values:
            continue

        xs = [x_positions[horizon] for horizon, _value in horizon_values]
        ys = [value for _horizon, value in horizon_values]

        ax.plot(
            xs,
            ys,
            color=spec.color,
            marker=spec.marker,
            markersize=6.5,
            linewidth=2.4,
            solid_capstyle="round",
            label=spec.display_name,
            zorder=3,
        )

        for idx, (horizon, value) in enumerate(horizon_values):
            dx, dy = spec.annotation_offset
            if idx % 2 == 1:
                dy += 4 if dy >= 0 else -4
            if horizon == 30000:
                dx = max(dx - 6, -8)
            ax.annotate(
                _format_value(value),
                xy=(x_positions[horizon], value),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8.5,
                color=spec.color,
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": spec.color,
                    "linewidth": 0.9,
                    "alpha": 0.95,
                },
                zorder=4,
            )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=2.4,
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def save_metadata(
    series: dict[str, dict[int, float]],
    sources: dict[str, dict[int, str]],
    output_path: Path,
) -> None:
    json_path = output_path.with_suffix(".json")
    payload = {
        "figure": str(output_path.relative_to(REPO_ROOT)),
        "statistic": "cross-system median of per-system seed IQM best-periodic MSE",
        "horizons": list(PLOT_HORIZONS),
        "models": {
            MODEL_BY_KEY[model_key].display_name: {
                str(horizon): {
                    "value": value,
                    "source": sources[model_key][horizon],
                }
                for horizon, value in sorted(horizon_map.items())
            }
            for model_key, horizon_map in series.items()
            if horizon_map
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path. A PDF and JSON sibling will also be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    series, sources = load_series()
    save_plot(series, args.output)
    save_metadata(series, sources, args.output)

    for spec in MODEL_SPECS:
        values = ", ".join(
            f"H{horizon}={series[spec.key][horizon]:.6g}"
            for horizon in PLOT_HORIZONS
            if horizon in series[spec.key]
        )
        print(f"{spec.display_name}: {values}")
    print(f"Saved figure to {args.output}")
    print(f"Saved PDF to {args.output.with_suffix('.pdf')}")
    print(f"Saved metadata to {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
