#!/usr/bin/env python3
"""Build an appendix training-dynamics diagnostic figure from saved logs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/mila/l/lia/skae")
FIG_DIR = ROOT / "docs" / "figures" / "neurips_paper_2026"
TABLE_DIR = FIG_DIR / "_tables"

SYSTEM_KEY = "gated_local_linear"
DYSTS_SYSTEM_KEYS = [
    "dysts_Chua",
    "dysts_Dadras",
    "dysts_DequanLi",
    "dysts_Hadley",
    "dysts_LuChenCheng",
    "dysts_QiChen",
    "dysts_Sakarya",
    "dysts_SanUmSrisuchinwong",
    "dysts_ShimizuMorioka",
    "dysts_WangSun",
]
SEED = 0
METRIC_NAMES = {
    "train/loss",
    "eval/final_error",
    "train/spectral_radius",
    "train/sparsity_ratio",
}


@dataclass(frozen=True)
class ModelTraceSpec:
    label: str
    root: Path
    color: str


MODEL_SPECS = [
    ModelTraceSpec(
        "LISTA",
        Path(
            "/network/scratch/l/lia/skae/"
            "transition_rich_lista_dense_p256_hardinit_table123_20260430/"
            "transition_rich_basin_partition/"
            "lista_dense_signsplit_p256_hardinit_basin_partition"
        ),
        "#785EF0",
    ),
    ModelTraceSpec(
        "LISTA-BD",
        Path(
            "/network/scratch/l/lia/skae/"
            "transition_rich_basin_partition_final_seed10_20260409/"
            "transition_rich_basin_partition/"
            "lista_blockdiag_signsplit_hardinit_basin_partition"
        ),
        "#0072B2",
    ),
    ModelTraceSpec(
        "LISTA-SB",
        Path(
            "/network/scratch/l/lia/skae/"
            "transition_rich_lista_sb_p256_hardinit_fairness_seed15_20260428/"
            "transition_rich_basin_partition/"
            "lista_dense_softblock_signsplit_p256_hardinit_basin_partition"
        ),
        "#56B4E9",
    ),
    ModelTraceSpec(
        "Sparse MLP-BD",
        Path(
            "/network/scratch/l/lia/skae/"
            "transition_rich_hardinit_mlp_controls_seed10_20260416/"
            "transition_rich_basin_partition/"
            "mlp_sparse_blockdiag_hardinit_basin_partition_control"
        ),
        "#44AA99",
    ),
    ModelTraceSpec(
        "Sparse MLP",
        Path(
            "/network/scratch/l/lia/skae/"
            "transition_rich_hardinit_mlp_controls_seed10_20260416/"
            "transition_rich_basin_partition/"
            "mlp_sparse_hardinit_basin_partition_control"
        ),
        "#009E73",
    ),
    ModelTraceSpec(
        "Dense MLP",
        Path(
            "/network/scratch/l/lia/skae/"
            "transition_rich_hardinit_mlp_controls_seed10_20260416/"
            "transition_rich_basin_partition/"
            "mlp_zero_sparse_hardinit_basin_partition_control"
        ),
        "#D55E00",
    ),
]

DYSTS_MODEL_SPECS = [
    ModelTraceSpec(
        "LISTA",
        Path(
            "/network/scratch/l/lia/skae/"
            "dysts_dt30_basinblock_p256_seq10_100k_20260430/"
            "dysts_dt30_basinblock_p256_seq10_100k/lista"
        ),
        "#785EF0",
    ),
    ModelTraceSpec(
        "LISTA-BD",
        Path(
            "/network/scratch/l/lia/skae/"
            "dysts_dt30_basinblock_p256_seq10_100k_20260430/"
            "dysts_dt30_basinblock_p256_seq10_100k/lista_bd"
        ),
        "#0072B2",
    ),
    ModelTraceSpec(
        "LISTA-SB",
        Path(
            "/network/scratch/l/lia/skae/"
            "dysts_dt30_basinblock_p256_seq10_100k_20260430/"
            "dysts_dt30_basinblock_p256_seq10_100k/lista_sb"
        ),
        "#56B4E9",
    ),
    ModelTraceSpec(
        "Sparse MLP-BD",
        Path(
            "/network/scratch/l/lia/skae/"
            "dysts_dt30_basinblock_p256_seq10_100k_20260430/"
            "dysts_dt30_basinblock_p256_seq10_100k/sparse_mlp_bd"
        ),
        "#44AA99",
    ),
    ModelTraceSpec(
        "Sparse MLP",
        Path(
            "/network/scratch/l/lia/skae/"
            "dysts_dt30_basinblock_p256_seq10_100k_20260430/"
            "dysts_dt30_basinblock_p256_seq10_100k/sparse_mlp"
        ),
        "#009E73",
    ),
    ModelTraceSpec(
        "Dense MLP",
        Path(
            "/network/scratch/l/lia/skae/"
            "dysts_dt30_basinblock_p256_seq10_100k_20260430/"
            "dysts_dt30_basinblock_p256_seq10_100k/dense_mlp_tanh"
        ),
        "#D55E00",
    ),
]


def find_metrics_file(spec: ModelTraceSpec, system_key: str = SYSTEM_KEY) -> Path:
    candidates = sorted((spec.root / system_key).glob(f"dt_*/seed_{SEED}/*/metrics_history.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No metrics_history.jsonl found for {spec.label} under {spec.root / system_key}"
        )
    return candidates[-1]


def _is_relevant_line(line: str) -> bool:
    return any(f'"name": "{name}"' in line for name in METRIC_NAMES)


def load_metric_trace(path: Path, sample_every: int) -> dict[str, list[tuple[int, float]]]:
    traces: dict[str, list[tuple[int, float]]] = {name: [] for name in METRIC_NAMES}
    last_seen: dict[str, tuple[int, float]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not _is_relevant_line(line):
                continue
            entry = json.loads(line)
            name = str(entry["name"])
            value = entry["value"]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                continue
            step = int(entry["step"])
            value_f = float(value)
            last_seen[name] = (step, value_f)
            if name.startswith("eval/") or step % sample_every == 0:
                traces[name].append((step, value_f))

    for name, pair in last_seen.items():
        if not traces[name] or traces[name][-1] != pair:
            traces[name].append(pair)

    return traces


def write_long_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    fieldnames = ["model", "metric", "step", "value", "metrics_file"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_dysts_long_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    fieldnames = ["model", "system", "metric", "step", "value", "metrics_file"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def array_for(trace: list[tuple[int, float]]) -> tuple[np.ndarray, np.ndarray]:
    if not trace:
        return np.array([], dtype=float), np.array([], dtype=float)
    arr = np.asarray(trace, dtype=float)
    return arr[:, 0], arr[:, 1]


def metric_min(trace: list[tuple[int, float]]) -> tuple[int | None, float | None]:
    finite = [(step, value) for step, value in trace if math.isfinite(value)]
    if not finite:
        return None, None
    return min(finite, key=lambda pair: pair[1])


def make_figure(
    all_traces: dict[str, dict[str, list[tuple[int, float]]]],
    output_base: Path,
    model_specs: list[ModelTraceSpec] | None = None,
) -> None:
    specs = model_specs if model_specs is not None else MODEL_SPECS
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 250,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    colors = {spec.label: spec.color for spec in specs}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9), sharex=True)
    panels = [
        ("eval/final_error", "Validation final error", "Validation final error", "log"),
        ("train/loss", "Training objective", "Training loss", "log"),
        ("train/spectral_radius", r"Spectral radius of $K$", r"$\rho(K)$", "linear"),
        ("train/sparsity_ratio", "Sparsity-ratio diagnostic", "Sparsity ratio", "linear"),
    ]

    for ax, (metric, title, ylabel, scale) in zip(axes.flat, panels, strict=True):
        for spec in specs:
            trace = all_traces[spec.label].get(metric, [])
            steps, values = array_for(trace)
            if steps.size == 0:
                continue
            ax.plot(
                steps / 1000.0,
                values,
                color=colors[spec.label],
                lw=1.35,
                alpha=0.9,
                label=spec.label,
            )
            if metric == "eval/final_error":
                best_step, best_value = metric_min(trace)
                if best_step is not None and best_value is not None:
                    ax.scatter(
                        [best_step / 1000.0],
                        [best_value],
                        color=colors[spec.label],
                        s=18,
                        zorder=5,
                        edgecolor="white",
                        linewidth=0.4,
                    )
        if metric == "train/spectral_radius":
            ax.axhline(1.0, color="#555555", lw=0.8, ls="--", alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_yscale(scale)
        ax.grid(True, alpha=0.22, lw=0.5)

    for ax in axes[-1, :]:
        ax.set_xlabel("Optimization step (thousands)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.015),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def build_controlled_trace() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    all_traces: dict[str, dict[str, list[tuple[int, float]]]] = {}
    summary: dict[str, object] = {
        "system_key": SYSTEM_KEY,
        "seed": SEED,
        "sample_every": 500,
        "models": {},
        "notes": (
            "Dots in the validation panel mark the minimum eval/final_error "
            "within each plotted trace, matching the validation checkpoint rule."
        ),
    }
    csv_rows: list[dict[str, object]] = []

    for spec in MODEL_SPECS:
        metrics_file = find_metrics_file(spec)
        traces = load_metric_trace(metrics_file, sample_every=500)
        all_traces[spec.label] = traces
        best_step, best_value = metric_min(traces["eval/final_error"])
        final_eval = traces["eval/final_error"][-1] if traces["eval/final_error"] else (None, None)
        final_rho = (
            traces["train/spectral_radius"][-1]
            if traces["train/spectral_radius"]
            else (None, None)
        )
        summary["models"][spec.label] = {
            "metrics_file": str(metrics_file),
            "best_validation_step": best_step,
            "best_validation_final_error": best_value,
            "final_validation_step": final_eval[0],
            "final_validation_final_error": final_eval[1],
            "final_spectral_radius_step": final_rho[0],
            "final_spectral_radius": final_rho[1],
        }
        for metric_name, trace in traces.items():
            for step, value in trace:
                csv_rows.append(
                    {
                        "model": spec.label,
                        "metric": metric_name,
                        "step": step,
                        "value": value,
                        "metrics_file": str(metrics_file),
                    }
                )

    output_base = FIG_DIR / "appfig_training_dynamics_gated_local_seed0"
    make_figure(all_traces, output_base, MODEL_SPECS)
    write_long_csv(csv_rows, TABLE_DIR / "training_dynamics_gated_local_seed0.csv")
    (TABLE_DIR / "training_dynamics_gated_local_seed0_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_base.with_suffix('.pdf')}")
    print(f"Wrote {output_base.with_suffix('.png')}")
    print(f"Wrote {TABLE_DIR / 'training_dynamics_gated_local_seed0.csv'}")
    print(f"Wrote {TABLE_DIR / 'training_dynamics_gated_local_seed0_summary.json'}")


def _median_trace(step_values: dict[int, list[float]]) -> list[tuple[int, float]]:
    trace: list[tuple[int, float]] = []
    for step in sorted(step_values):
        values = np.asarray(step_values[step], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            trace.append((step, float(np.median(values))))
    return trace


def build_dysts_trace() -> None:
    aggregate: dict[str, dict[str, dict[int, list[float]]]] = {
        spec.label: {name: {} for name in METRIC_NAMES} for spec in DYSTS_MODEL_SPECS
    }
    all_traces: dict[str, dict[str, list[tuple[int, float]]]] = {
        spec.label: {name: [] for name in METRIC_NAMES} for spec in DYSTS_MODEL_SPECS
    }
    summary: dict[str, object] = {
        "systems": DYSTS_SYSTEM_KEYS,
        "seed": SEED,
        "sample_every": 500,
        "aggregation": "median over retained Dysts systems at each step; non-finite values omitted",
        "models": {},
    }
    csv_rows: list[dict[str, object]] = []

    for spec in DYSTS_MODEL_SPECS:
        source_files: dict[str, str] = {}
        for system_key in DYSTS_SYSTEM_KEYS:
            metrics_file = find_metrics_file(spec, system_key)
            source_files[system_key] = str(metrics_file)
            traces = load_metric_trace(metrics_file, sample_every=500)
            for metric_name, trace in traces.items():
                for step, value in trace:
                    aggregate[spec.label][metric_name].setdefault(step, []).append(value)
                    csv_rows.append(
                        {
                            "model": spec.label,
                            "system": system_key,
                            "metric": metric_name,
                            "step": step,
                            "value": value,
                            "metrics_file": str(metrics_file),
                        }
                    )

        for metric_name, step_values in aggregate[spec.label].items():
            all_traces[spec.label][metric_name] = _median_trace(step_values)
        best_step, best_value = metric_min(all_traces[spec.label]["eval/final_error"])
        final_eval = (
            all_traces[spec.label]["eval/final_error"][-1]
            if all_traces[spec.label]["eval/final_error"]
            else (None, None)
        )
        final_rho = (
            all_traces[spec.label]["train/spectral_radius"][-1]
            if all_traces[spec.label]["train/spectral_radius"]
            else (None, None)
        )
        summary["models"][spec.label] = {
            "source_files": source_files,
            "best_median_validation_step": best_step,
            "best_median_validation_final_error": best_value,
            "final_median_validation_step": final_eval[0],
            "final_median_validation_final_error": final_eval[1],
            "final_median_spectral_radius_step": final_rho[0],
            "final_median_spectral_radius": final_rho[1],
        }

    output_base = FIG_DIR / "appfig_training_dynamics_dysts_dt30_seed0"
    make_figure(all_traces, output_base, DYSTS_MODEL_SPECS)
    write_dysts_long_csv(csv_rows, TABLE_DIR / "training_dynamics_dysts_dt30_seed0.csv")
    (TABLE_DIR / "training_dynamics_dysts_dt30_seed0_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_base.with_suffix('.pdf')}")
    print(f"Wrote {output_base.with_suffix('.png')}")
    print(f"Wrote {TABLE_DIR / 'training_dynamics_dysts_dt30_seed0.csv'}")
    print(f"Wrote {TABLE_DIR / 'training_dynamics_dysts_dt30_seed0_summary.json'}")


def main() -> None:
    build_controlled_trace()
    build_dysts_trace()


if __name__ == "__main__":
    main()
