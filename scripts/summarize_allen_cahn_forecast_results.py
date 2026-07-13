#!/usr/bin/env python
"""Summarize Allen-Cahn spatialized reaction-diffusion pilot evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


FORECAST_METRICS = (
    "field_mse",
    "final_field_mse",
    "gradient_mse",
    "fourier_low_mse",
    "fourier_mid_mse",
    "fourier_high_mse",
    "pixel_basin_accuracy",
    "pixel_basin_mean_iou",
    "final_basin_consistency",
    "majority_fraction_mae",
)

MODEL_LABELS = {
    "conv_dense": "Dense KAE",
    "conv_lista": "LISTA KAE",
    "conv_sparse_mlp": "Sparse-MLP KAE",
}


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: object) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _summary_stats(values: list[float]) -> dict[str, float | int]:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return {"n": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "n": len(clean),
        "mean": mean(clean),
        "std": stdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
    }


def collect_rows(task_tsv: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in _csv_rows(task_tsv):
        eval_path = Path(task["eval_path"])
        run_dir = Path(task["run_dir"])
        if not eval_path.exists():
            rows.append(
                {
                    "task_id": task["task_id"],
                    "model_variant": task["model_variant"],
                    "seed": task["seed"],
                    "status": "missing_evaluation",
                    "eval_path": str(eval_path),
                }
            )
            continue

        evaluation = _read_json(eval_path)
        training_summary = {}
        training_summary_path = run_dir / "training_summary.json"
        if training_summary_path.exists():
            training_summary = _read_json(training_summary_path)
        model_config = {}
        model_config_path = run_dir / "model_config.json"
        if model_config_path.exists():
            model_config = _read_json(model_config_path)

        forecast_modes = evaluation.get("forecast_modes")
        if not isinstance(forecast_modes, dict):
            forecast_modes = {"no_reencode": evaluation.get("forecast", {})}

        for rollout_mode, horizon_metrics in forecast_modes.items():
            if not isinstance(horizon_metrics, dict):
                continue
            for horizon_text, metrics in horizon_metrics.items():
                if not isinstance(metrics, dict):
                    continue
                row: dict[str, object] = {
                "task_id": int(task["task_id"]),
                "source_system": task["source_system"],
                "grid_size": int(task["grid_size"]),
                "state_dim": int(task["state_dim"]),
                "latent_dim": int(task["target_size"]),
                "latent_state_ratio": float(task["latent_state_ratio"]),
                "model_variant": task["model_variant"],
                "model_label": MODEL_LABELS.get(task["model_variant"], task["model_variant"]),
                "seed": int(task["seed"]),
                "horizon": int(horizon_text),
                "effective_horizon": int(metrics.get("effective_horizon", horizon_text)),
                "rollout_mode": str(rollout_mode),
                "reencode_period": "" if metrics.get("reencode_period") is None else int(metrics.get("reencode_period")),
                "conv_activation": task["conv_activation"],
                "encoder_kind": model_config.get("encoder_kind", ""),
                "sparsity_coeff": _safe_float(task.get("sparsity_coeff", 0.0)),
                "support_threshold": _safe_float(task.get("support_threshold", 0.0)),
                "k_stability_weight": 0.0,
                "best_val_mse": _safe_float(training_summary.get("best_val_mse")),
                "checkpoint_step": evaluation.get("checkpoint_step", ""),
                "checkpoint": evaluation.get("checkpoint", ""),
                "eval_path": str(eval_path),
                "status": training_summary.get("status", "completed"),
                "device": evaluation.get("device", ""),
                }
                for metric_name in FORECAST_METRICS:
                    row[metric_name] = _safe_float(metrics.get(metric_name))
                rows.append(row)
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "missing_evaluation":
            continue
        grouped[(str(row["model_variant"]), str(row.get("rollout_mode", "no_reencode")), int(row["horizon"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for (model_variant, rollout_mode, horizon), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        base = {
            "model_variant": model_variant,
            "model_label": MODEL_LABELS.get(model_variant, model_variant),
            "rollout_mode": rollout_mode,
            "horizon": horizon,
            "n_seeds": len({int(row["seed"]) for row in group}),
        }
        for metric_name in FORECAST_METRICS:
            stats = _summary_stats([_safe_float(row.get(metric_name)) for row in group])
            base[f"{metric_name}_mean"] = stats["mean"]
            base[f"{metric_name}_std"] = stats["std"]
            base[f"{metric_name}_min"] = stats["min"]
            base[f"{metric_name}_max"] = stats["max"]
        best_val = _summary_stats([_safe_float(row.get("best_val_mse")) for row in group])
        base["best_val_mse_mean"] = best_val["mean"]
        base["best_val_mse_std"] = best_val["std"]
        summary_rows.append(base)
    return summary_rows


def paired_dense_differences(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {
        (str(row["model_variant"]), str(row.get("rollout_mode", "no_reencode")), int(row["seed"]), int(row["horizon"])): row
        for row in rows
        if row.get("status") != "missing_evaluation"
    }
    diffs: list[dict[str, object]] = []
    model_variants = sorted({str(row["model_variant"]) for row in rows if row.get("status") != "missing_evaluation"})
    rollout_modes = sorted({str(row.get("rollout_mode", "no_reencode")) for row in rows if row.get("status") != "missing_evaluation"})
    seeds = sorted({int(row["seed"]) for row in rows if row.get("status") != "missing_evaluation"})
    horizons = sorted({int(row["horizon"]) for row in rows if row.get("status") != "missing_evaluation"})
    for model_variant in model_variants:
        if model_variant == "conv_dense":
            continue
        for rollout_mode in rollout_modes:
            for horizon in horizons:
                paired = []
                for seed in seeds:
                    model_row = by_key.get((model_variant, rollout_mode, seed, horizon))
                    dense_row = by_key.get(("conv_dense", rollout_mode, seed, horizon))
                    if model_row is None or dense_row is None:
                        continue
                    paired.append(
                        _safe_float(model_row.get("field_mse"))
                        - _safe_float(dense_row.get("field_mse"))
                    )
                stats = _summary_stats(paired)
                diffs.append(
                    {
                        "model_variant": model_variant,
                        "model_label": MODEL_LABELS.get(model_variant, model_variant),
                        "baseline": "conv_dense",
                        "rollout_mode": rollout_mode,
                        "horizon": horizon,
                        "n_pairs": stats["n"],
                        "field_mse_diff_mean": stats["mean"],
                        "field_mse_diff_std": stats["std"],
                        "field_mse_diff_min": stats["min"],
                        "field_mse_diff_max": stats["max"],
                        "interpretation": "negative favors model over dense",
                    }
                )
    return diffs


def write_markdown(
    path: Path,
    rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    paired_rows: list[dict[str, object]],
    queue: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    horizons = sorted({int(row["horizon"]) for row in rows if row.get("status") != "missing_evaluation"})
    models = sorted({str(row["model_variant"]) for row in rows if row.get("status") != "missing_evaluation"})
    run_count = len({(row["model_variant"], row["seed"]) for row in rows if row.get("status") != "missing_evaluation"})
    mode_count = len({str(row.get("rollout_mode", "no_reencode")) for row in rows if row.get("status") != "missing_evaluation"})
    lines = [
        "# Allen-Cahn multistable PDE pilot results",
        "",
        f"Experiment tag: `{queue.get('experiment_tag', '')}`.",
        f"Completed model/seed runs with evaluations: {run_count}.",
        f"Models: {', '.join(MODEL_LABELS.get(model, model) for model in models)}.",
        f"Horizons: {', '.join(str(horizon) for horizon in horizons)}.",
        f"Rollout modes: {mode_count}.",
        "",
        "Protocol checks from the launch manifest:",
        "",
        f"- Source system: `{queue.get('systems_csv', '')}`.",
        f"- Dense and convolutional activations: `{queue.get('conv_activation', '')}`.",
        f"- Minimum latent/state ratio: `{queue.get('min_latent_state_ratio', '')}`.",
        f"- Koopman stability regularization weight: `{queue.get('k_stability_weight', '')}`.",
        f"- Dataset preflight enabled: `{queue.get('preflight_validate_datasets', '')}`.",
        "",
        "## Forecast field MSE",
        "",
        "| Model | Mode | Horizon | Mean | Std | Seeds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(summary_rows, key=lambda item: (str(item["rollout_mode"]), int(item["horizon"]), str(item["model_variant"]))):
        lines.append(
            "| {model} | {mode} | {horizon} | {mean:.6f} | {std:.6f} | {n} |".format(
                model=row["model_label"],
                mode=row["rollout_mode"],
                horizon=int(row["horizon"]),
                mean=_safe_float(row["field_mse_mean"]),
                std=_safe_float(row["field_mse_std"]),
                n=int(row["n_seeds"]),
            )
        )

    lines.extend(
        [
            "",
            "## Paired field MSE differences vs dense",
            "",
            "Negative values favor the sparse/lifted model over the dense KAE at the same seed and horizon.",
            "",
            "| Model | Mode | Horizon | Mean diff | Std diff | Pairs |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(paired_rows, key=lambda item: (str(item["model_variant"]), str(item["rollout_mode"]), int(item["horizon"]))):
        lines.append(
            "| {model} | {mode} | {horizon} | {mean:.6f} | {std:.6f} | {n} |".format(
                model=row["model_label"],
                mode=row["rollout_mode"],
                horizon=int(row["horizon"]),
                mean=_safe_float(row["field_mse_diff_mean"]),
                std=_safe_float(row["field_mse_diff_std"]),
                n=int(row["n_pairs"]),
            )
        )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "This is a pilot grid-16 run with three seeds and 3000 updates per run. It checks the requested setup and gives a useful direction, but it is not yet a final statistical claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plot(path_base: Path, summary_rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        path_base.with_suffix(".plot_skipped.txt").write_text(str(exc), encoding="utf-8")
        return

    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    plot_rows = [row for row in summary_rows if str(row.get("rollout_mode", "no_reencode")) == "no_reencode"]
    if not plot_rows:
        plot_rows = summary_rows
    for model_variant in sorted({str(row["model_variant"]) for row in plot_rows}):
        model_rows = sorted(
            [row for row in plot_rows if row["model_variant"] == model_variant],
            key=lambda row: int(row["horizon"]),
        )
        horizons = [int(row["horizon"]) for row in model_rows]
        means = [_safe_float(row["field_mse_mean"]) for row in model_rows]
        stds = [_safe_float(row["field_mse_std"]) for row in model_rows]
        ax.errorbar(
            horizons,
            means,
            yerr=stds,
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=MODEL_LABELS.get(model_variant, model_variant),
        )
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Field MSE")
    ax.set_title("Allen-Cahn multistable PDE pilot")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=200)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/allen_cahn_multistable_pde_pilot_20260628"),
    )
    parser.add_argument("--task-tsv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    task_tsv = args.task_tsv or results_dir / "spatialized_rd_tasks.tsv"
    queue_path = results_dir / "queue.json"
    queue = _read_json(queue_path) if queue_path.exists() else {}

    rows = collect_rows(task_tsv)
    summary_rows = summarize(rows)
    paired_rows = paired_dense_differences(rows)

    per_seed_fields = [
        "task_id",
        "source_system",
        "grid_size",
        "state_dim",
        "latent_dim",
        "latent_state_ratio",
        "model_variant",
        "model_label",
        "seed",
        "horizon",
        "effective_horizon",
        "rollout_mode",
        "reencode_period",
        "conv_activation",
        "encoder_kind",
        "sparsity_coeff",
        "support_threshold",
        "k_stability_weight",
        "best_val_mse",
        "checkpoint_step",
        "status",
        "device",
        *FORECAST_METRICS,
        "checkpoint",
        "eval_path",
    ]
    summary_fields = [
        "model_variant",
        "model_label",
        "rollout_mode",
        "horizon",
        "n_seeds",
        "best_val_mse_mean",
        "best_val_mse_std",
    ]
    for metric_name in FORECAST_METRICS:
        summary_fields.extend(
            [
                f"{metric_name}_mean",
                f"{metric_name}_std",
                f"{metric_name}_min",
                f"{metric_name}_max",
            ]
        )
    paired_fields = [
        "model_variant",
        "model_label",
        "baseline",
        "rollout_mode",
        "horizon",
        "n_pairs",
        "field_mse_diff_mean",
        "field_mse_diff_std",
        "field_mse_diff_min",
        "field_mse_diff_max",
        "interpretation",
    ]

    _write_csv(results_dir / "forecast_horizon_per_seed.csv", rows, per_seed_fields)
    _write_csv(results_dir / "forecast_horizon_summary.csv", summary_rows, summary_fields)
    _write_csv(results_dir / "forecast_paired_vs_dense.csv", paired_rows, paired_fields)
    write_markdown(results_dir / "forecast_results.md", rows, summary_rows, paired_rows, queue)
    write_plot(results_dir / "figures" / "forecast_horizon_field_mse", summary_rows)

    print(f"Wrote {results_dir / 'forecast_horizon_per_seed.csv'}")
    print(f"Wrote {results_dir / 'forecast_horizon_summary.csv'}")
    print(f"Wrote {results_dir / 'forecast_paired_vs_dense.csv'}")
    print(f"Wrote {results_dir / 'forecast_results.md'}")


if __name__ == "__main__":
    main()
