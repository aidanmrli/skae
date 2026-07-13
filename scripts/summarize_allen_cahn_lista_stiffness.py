#!/usr/bin/env python
"""Summarize the Allen-Cahn LISTA shrink/depth diagnostic sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


FORECAST_METRICS = (
    "field_mse",
    "final_field_mse",
    "gradient_mse",
    "pixel_basin_accuracy",
    "pixel_basin_mean_iou",
    "final_basin_consistency",
    "majority_fraction_mae",
)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _csv_rows(path: Path, *, delimiter: str = "\t") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _safe_float(value: object) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _safe_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _stats(values: Iterable[object]) -> dict[str, float | int]:
    clean = [_safe_float(value) for value in values]
    clean = [value for value in clean if math.isfinite(value)]
    if not clean:
        return {"n": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "n": len(clean),
        "mean": mean(clean),
        "std": stdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
    }


def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collect_forecast_rows(task_tsv: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in _csv_rows(task_tsv):
        eval_path = Path(task["eval_path"])
        run_dir = Path(task["run_dir"])
        training_summary_path = run_dir / "training_summary.json"
        model_config_path = run_dir / "model_config.json"
        training_summary = _read_json(training_summary_path) if training_summary_path.exists() else {}
        model_config = _read_json(model_config_path) if model_config_path.exists() else {}
        base = {
            "task_id": _safe_int(task.get("task_id")),
            "source_system": task.get("source_system", ""),
            "seed": _safe_int(task.get("seed")),
            "model_variant": task.get("model_variant", ""),
            "setting_slug": task.get("setting_slug", ""),
            "grid_size": _safe_int(task.get("grid_size")),
            "state_dim": _safe_int(task.get("state_dim")),
            "latent_dim": _safe_int(task.get("target_size")),
            "latent_state_ratio": _safe_float(task.get("latent_state_ratio")),
            "conv_activation": task.get("conv_activation", ""),
            "lista_num_loops": _safe_int(task.get("lista_num_loops")),
            "lista_alpha": _safe_float(task.get("lista_alpha")),
            "sparsity_coeff": _safe_float(task.get("sparsity_coeff")),
            "support_threshold": _safe_float(task.get("support_threshold")),
            "best_val_mse": _safe_float(training_summary.get("best_val_mse")),
            "status": training_summary.get("status", "missing_training_summary"),
            "encoder_kind": model_config.get("encoder_kind", ""),
            "checkpoint": training_summary.get("checkpoint", ""),
            "eval_path": str(eval_path),
            "run_dir": str(run_dir),
        }
        if not eval_path.exists():
            rows.append({**base, "status": "missing_evaluation"})
            continue
        evaluation = _read_json(eval_path)
        forecast_modes = evaluation.get("forecast_modes")
        if not isinstance(forecast_modes, dict):
            forecast_modes = {"no_reencode": evaluation.get("forecast", {})}
        for rollout_mode, horizon_metrics in forecast_modes.items():
            if not isinstance(horizon_metrics, dict):
                continue
            for horizon_text, metrics in horizon_metrics.items():
                if not isinstance(metrics, dict):
                    continue
                row = {
                    **base,
                    "status": training_summary.get("status", "completed"),
                    "horizon": _safe_int(horizon_text),
                    "rollout_mode": str(rollout_mode),
                    "reencode_period": "" if metrics.get("reencode_period") is None else _safe_int(metrics.get("reencode_period")),
                    "checkpoint_step": evaluation.get("checkpoint_step", ""),
                    "device": evaluation.get("device", ""),
                }
                for metric_name in FORECAST_METRICS:
                    row[metric_name] = _safe_float(metrics.get(metric_name))
                rows.append(row)
    return rows


def collect_history_rows(task_tsv: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in _csv_rows(task_tsv):
        history_path = Path(task["run_dir"]) / "metrics_history.jsonl"
        for record in _iter_jsonl(history_path):
            rows.append(
                {
                    "task_id": _safe_int(task.get("task_id")),
                    "seed": _safe_int(task.get("seed")),
                    "setting_slug": task.get("setting_slug", ""),
                    "lista_num_loops": _safe_int(task.get("lista_num_loops")),
                    "lista_alpha": _safe_float(task.get("lista_alpha")),
                    "step": _safe_int(record.get("step")),
                    "eval_horizon": _safe_int(record.get("eval_horizon")),
                    "val_mse": _safe_float(record.get("val_mse")),
                    "val_final_mse": _safe_float(record.get("val_final_mse")),
                    "prediction_loss": _safe_float(record.get("prediction_loss")),
                    "reconstruction_loss": _safe_float(record.get("reconstruction_loss")),
                    "latent_loss": _safe_float(record.get("latent_loss")),
                    "sparsity_ratio_1e-4": _safe_float(record.get("sparsity_ratio_1e-4")),
                    "sparsity_loss": _safe_float(record.get("sparsity_loss")),
                    "k_fro_sqrt": _safe_float(record.get("k_fro_sqrt")),
                }
            )
    return rows


def summarize_forecasts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, float, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "missing_evaluation" or row.get("horizon") is None:
            continue
        groups[
            (
                int(row["lista_num_loops"]),
                float(row["lista_alpha"]),
                str(row["rollout_mode"]),
                int(row["horizon"]),
            )
        ].append(row)
    summary_rows: list[dict[str, object]] = []
    for (loops, alpha, mode, horizon), group in sorted(groups.items()):
        out: dict[str, object] = {
            "lista_num_loops": loops,
            "lista_alpha": alpha,
            "rollout_mode": mode,
            "horizon": horizon,
            "n_seeds": len({int(row["seed"]) for row in group if row.get("seed") is not None}),
        }
        for metric_name in ("best_val_mse", *FORECAST_METRICS):
            stats = _stats(row.get(metric_name) for row in group)
            out[f"{metric_name}_mean"] = stats["mean"]
            out[f"{metric_name}_std"] = stats["std"]
            out[f"{metric_name}_min"] = stats["min"]
            out[f"{metric_name}_max"] = stats["max"]
        summary_rows.append(out)
    return summary_rows


def summarize_history(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, float, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("step") is None:
            continue
        groups[(int(row["lista_num_loops"]), float(row["lista_alpha"]), int(row["step"]))].append(row)
    summary_rows: list[dict[str, object]] = []
    for (loops, alpha, step), group in sorted(groups.items()):
        out: dict[str, object] = {
            "lista_num_loops": loops,
            "lista_alpha": alpha,
            "step": step,
            "n_seeds": len({int(row["seed"]) for row in group if row.get("seed") is not None}),
        }
        for metric_name in (
            "val_mse",
            "val_final_mse",
            "prediction_loss",
            "reconstruction_loss",
            "sparsity_ratio_1e-4",
            "sparsity_loss",
            "k_fro_sqrt",
        ):
            stats = _stats(row.get(metric_name) for row in group)
            out[f"{metric_name}_mean"] = stats["mean"]
            out[f"{metric_name}_std"] = stats["std"]
        summary_rows.append(out)
    return summary_rows


def load_reference(reference_dir: Path | None) -> list[dict[str, str]]:
    if reference_dir is None:
        return []
    path = reference_dir / "forecast_horizon_summary.csv"
    if not path.exists():
        return []
    return _csv_rows(path, delimiter=",")


def write_plots(results_dir: Path, summary_rows: list[dict[str, object]], history_rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        (results_dir / "figures" / "plot_skipped.txt").write_text(str(exc), encoding="utf-8")
        return
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for mode in ("no_reencode", "periodic_20"):
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        for loops in sorted({int(row["lista_num_loops"]) for row in summary_rows}):
            plot_rows = sorted(
                [
                    row
                    for row in summary_rows
                    if int(row["lista_num_loops"]) == loops
                    and str(row["rollout_mode"]) == mode
                    and int(row["horizon"]) == 200
                ],
                key=lambda row: float(row["lista_alpha"]),
            )
            if not plot_rows:
                continue
            alphas = [float(row["lista_alpha"]) for row in plot_rows]
            means = [_safe_float(row["field_mse_mean"]) for row in plot_rows]
            stds = [_safe_float(row["field_mse_std"]) for row in plot_rows]
            ax.errorbar(alphas, means, yerr=stds, marker="o", capsize=3, label=f"{loops} loop(s)")
        ax.set_xscale("symlog", linthresh=1e-5)
        ax.set_xlabel("LISTA shrink threshold alpha")
        ax.set_ylabel("H200 field MSE")
        ax.set_title(f"Allen-Cahn LISTA stiffness sweep ({mode})")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / f"lista_stiffness_h200_{mode}.png", dpi=200)
        fig.savefig(fig_dir / f"lista_stiffness_h200_{mode}.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for loops, alpha in sorted({(int(row["lista_num_loops"]), float(row["lista_alpha"])) for row in history_rows}):
        plot_rows = sorted(
            [row for row in history_rows if int(row["lista_num_loops"]) == loops and float(row["lista_alpha"]) == alpha],
            key=lambda row: int(row["step"]),
        )
        if not plot_rows:
            continue
        label = f"L{loops}, a={alpha:g}"
        ax.plot(
            [int(row["step"]) for row in plot_rows],
            [_safe_float(row["val_mse_mean"]) for row in plot_rows],
            linewidth=1.2,
            alpha=0.8,
            label=label,
        )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation rollout MSE")
    ax.set_title("LISTA validation convergence")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "lista_stiffness_validation_curves.png", dpi=200)
    fig.savefig(fig_dir / "lista_stiffness_validation_curves.pdf")
    plt.close(fig)


def write_markdown(results_dir: Path, summary_rows: list[dict[str, object]], reference_rows: list[dict[str, str]]) -> None:
    modes = ["no_reencode", "periodic_20"]
    lines = [
        "# Allen-Cahn LISTA stiffness sweep",
        "",
        "This report is generated from completed evaluations only. It does not modify the paper draft.",
        "",
        "## Best LISTA settings",
        "",
        "| Mode | Horizon | Loops | Alpha | Field MSE mean | Std | Seeds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in modes:
        for horizon in (50, 100, 200):
            candidates = [
                row
                for row in summary_rows
                if str(row.get("rollout_mode")) == mode
                and int(row.get("horizon", -1)) == horizon
                and int(row.get("n_seeds", 0)) > 0
            ]
            if not candidates:
                continue
            best = min(candidates, key=lambda row: _safe_float(row.get("field_mse_mean")))
            lines.append(
                "| {mode} | {horizon} | {loops} | {alpha:g} | {mean:.6f} | {std:.6f} | {n} |".format(
                    mode=mode,
                    horizon=horizon,
                    loops=int(best["lista_num_loops"]),
                    alpha=float(best["lista_alpha"]),
                    mean=_safe_float(best["field_mse_mean"]),
                    std=_safe_float(best["field_mse_std"]),
                    n=int(best["n_seeds"]),
                )
            )
    if reference_rows:
        lines.extend(
            [
                "",
                "## Existing 50k reference",
                "",
                "| Model | Mode | Horizon | Field MSE mean | Std |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in reference_rows:
            if row.get("rollout_mode") not in {"no_reencode", "periodic_20"}:
                continue
            if row.get("horizon") not in {"50", "100", "200"}:
                continue
            lines.append(
                "| {model} | {mode} | {horizon} | {mean:.6f} | {std:.6f} |".format(
                    model=row.get("model_label", row.get("model_variant", "")),
                    mode=row.get("rollout_mode", ""),
                    horizon=int(row.get("horizon", 0)),
                    mean=_safe_float(row.get("field_mse_mean")),
                    std=_safe_float(row.get("field_mse_std")),
                )
            )
    lines.extend(
        [
            "",
            "## Framing note",
            "",
            "A LISTA-as-faster-convergence or LISTA-as-regularization framing is supported only if the validation curves show an early-step advantage that persists under matched budgets or if the tuned shrink setting improves long-horizon forecasts. If the best tuned LISTA remains behind Sparse-MLP at 50k, the safer interpretation is that shrinkage is a useful regularizer in some training regimes but the current Allen-Cahn evidence favors sparse-code mechanisms that are less restrictive than unfolded soft-thresholding.",
            "",
        ]
    )
    (results_dir / "lista_stiffness_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--task-tsv", type=Path, default=None)
    parser.add_argument("--reference-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    task_tsv = args.task_tsv or results_dir / "spatialized_rd_tasks.tsv"
    forecast_rows = collect_forecast_rows(task_tsv)
    history_rows = collect_history_rows(task_tsv)
    forecast_summary = summarize_forecasts(forecast_rows)
    history_summary = summarize_history(history_rows)
    reference_rows = load_reference(args.reference_dir)

    per_seed_fields = [
        "task_id",
        "source_system",
        "seed",
        "model_variant",
        "setting_slug",
        "grid_size",
        "state_dim",
        "latent_dim",
        "latent_state_ratio",
        "conv_activation",
        "encoder_kind",
        "lista_num_loops",
        "lista_alpha",
        "sparsity_coeff",
        "support_threshold",
        "best_val_mse",
        "status",
        "horizon",
        "rollout_mode",
        "reencode_period",
        "checkpoint_step",
        "device",
        *FORECAST_METRICS,
        "checkpoint",
        "eval_path",
        "run_dir",
    ]
    summary_fields = [
        "lista_num_loops",
        "lista_alpha",
        "rollout_mode",
        "horizon",
        "n_seeds",
        "best_val_mse_mean",
        "best_val_mse_std",
        "best_val_mse_min",
        "best_val_mse_max",
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
    history_fields = [
        "lista_num_loops",
        "lista_alpha",
        "step",
        "n_seeds",
        "val_mse_mean",
        "val_mse_std",
        "val_final_mse_mean",
        "val_final_mse_std",
        "prediction_loss_mean",
        "prediction_loss_std",
        "reconstruction_loss_mean",
        "reconstruction_loss_std",
        "sparsity_ratio_1e-4_mean",
        "sparsity_ratio_1e-4_std",
        "sparsity_loss_mean",
        "sparsity_loss_std",
        "k_fro_sqrt_mean",
        "k_fro_sqrt_std",
    ]
    _write_csv(results_dir / "lista_stiffness_per_seed.csv", forecast_rows, per_seed_fields)
    _write_csv(results_dir / "lista_stiffness_summary.csv", forecast_summary, summary_fields)
    _write_csv(results_dir / "lista_stiffness_history.csv", history_summary, history_fields)
    write_plots(results_dir, forecast_summary, history_summary)
    write_markdown(results_dir, forecast_summary, reference_rows)
    print(f"Wrote {results_dir / 'lista_stiffness_summary.csv'}")
    print(f"Wrote {results_dir / 'lista_stiffness_history.csv'}")
    print(f"Wrote {results_dir / 'lista_stiffness_report.md'}")


if __name__ == "__main__":
    main()
