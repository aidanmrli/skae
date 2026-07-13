"""Analyze Lorenz-96 sparsity-mechanism hyperparameter sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILY_ORDER = ["dense_mlp_tanh", "sparse_mlp_relu", "lista_relu", "lista_shrink"]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_resamples: int) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    boot = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        boot[i] = rng.choice(values, size=values.size, replace=True).mean()
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    headers = [str(col) for col in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def parse_label_metadata(raw: pd.DataFrame) -> pd.DataFrame:
    paths = sorted(str(path) for path in raw["configuration_path"].dropna().unique() if str(path))
    rows: list[dict[str, object]] = []
    for path in paths:
        payload_path = Path(path)
        if not payload_path.exists():
            continue
        payload = json.loads(payload_path.read_text())
        variant: Dict[str, object] = dict(payload.get("model_variant", {}))
        rows.append(
            {
                "model": payload.get("label", ""),
                "family": variant.get("family", payload.get("label", "")),
                "seed": int(payload_path.parent.name.split("_seed", 1)[1].split("_", 1)[0]) if "_seed" in payload_path.parent.name else -1,
                "latent_dim": int(payload.get("latent_dim", 0)),
                "sparsity_coefficient": float(payload.get("sparsity_coeff", 0.0)),
                "lista_alpha": float(variant.get("lista_alpha", np.nan)),
                "lista_final_op": variant.get("lista_final_op", ""),
                "model_preset": payload.get("model_preset", ""),
                "best_val_nrmse_training_horizon": float(payload.get("best_val_nrmse", np.nan)),
                "training_steps": len(payload.get("history", [])),
                "training_time_seconds": float(payload.get("training_time_seconds", np.nan)),
                "parameter_count": int(payload.get("parameter_count", 0)),
            }
        )
    return pd.DataFrame(rows)


def add_metadata(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if meta.empty:
        df = df.copy()
        df["family"] = df["model"]
        df["lista_alpha"] = np.nan
        return df
    keys = ["model", "seed"]
    cols = keys + ["family", "lista_alpha", "lista_final_op", "model_preset", "training_steps", "training_time_seconds", "parameter_count"]
    merged = df.merge(meta[cols].drop_duplicates(keys), on=keys, how="left")
    merged["family"] = merged["family"].fillna(merged["model"])
    return merged


def active_density_table(raw: pd.DataFrame, meta: pd.DataFrame, split: str = "val") -> pd.DataFrame:
    raw = raw.copy()
    raw["metric_name"] = raw["metric_name"].replace(
        {
            "latent_active_active_coords_abs_1e-3": "latent_active_coords_abs_1e-3",
            "latent_active_active_coords_abs_1e-4": "latent_active_coords_abs_1e-4",
            "latent_active_active_coords_abs_1e-2": "latent_active_coords_abs_1e-2",
            "latent_active_active_coords_rel_1e-3max": "latent_active_coords_rel_1e-3max",
        }
    )
    sub = raw[
        (raw["split"] == split)
        & (raw["trajectory_identifier"] == "model")
        & (raw["metric_name"].isin(["latent_active_density_abs_1e-3", "latent_active_coords_abs_1e-3", "latent_mean_abs"]))
    ].copy()
    sub = add_metadata(sub, meta)
    pivot = sub.pivot_table(index=["model", "family", "seed"], columns="metric_name", values="metric_value").reset_index()
    grouped = (
        pivot.groupby(["model", "family"])[["latent_active_density_abs_1e-3", "latent_active_coords_abs_1e-3", "latent_mean_abs"]]
        .mean()
        .reset_index()
    )
    return grouped


def nrmse_summary(raw: pd.DataFrame, meta: pd.DataFrame, split: str, n_resamples: int) -> pd.DataFrame:
    sub = raw[(raw["split"] == split) & (raw["metric_name"] == "nrmse")].copy()
    sub = add_metadata(sub, meta)
    rng = np.random.default_rng(0 if split == "val" else 1)
    rows: list[dict[str, object]] = []
    for (model, family, horizon), part in sub.groupby(["model", "family", "horizon"]):
        values = part["metric_value"].to_numpy(dtype=np.float64)
        ci_low, ci_high = bootstrap_ci(values, rng, n_resamples)
        rows.append(
            {
                "split": split,
                "model": model,
                "family": family,
                "horizon": int(horizon),
                "mean_nrmse": float(values.mean()),
                "std_nrmse": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def select_by_validation(val_summary: pd.DataFrame, density: pd.DataFrame) -> pd.DataFrame:
    h100 = val_summary[val_summary["horizon"] == 100].copy()
    h100 = h100.merge(density, on=["model", "family"], how="left")
    rows: list[dict[str, object]] = []
    for family, part in h100[h100["family"].isin(["sparse_mlp_relu", "lista_relu", "lista_shrink"])].groupby("family"):
        best = float(part["mean_nrmse"].min())
        retained = part[part["mean_nrmse"] <= 1.05 * best].copy()
        retained = retained.sort_values(["latent_active_density_abs_1e-3", "mean_nrmse", "model"])
        selected = retained.iloc[0].to_dict()
        selected["family_best_val_h100"] = best
        selected["selection_rule"] = "within_5pct_best_val_h100_choose_lowest_val_active_density_abs_1e-3"
        rows.append(selected)
    return pd.DataFrame(rows)


def paired_vs_dense(raw: pd.DataFrame, selected: pd.DataFrame, n_resamples: int) -> pd.DataFrame:
    sub = raw[(raw["split"] == "test") & (raw["metric_name"] == "nrmse")].copy()
    dense = sub[sub["model"] == "dense_mlp_tanh"]
    rng = np.random.default_rng(2)
    rows: list[dict[str, object]] = []
    for _, selected_row in selected.iterrows():
        model = str(selected_row["model"])
        for horizon in [50, 100]:
            a = sub[(sub["model"] == model) & (sub["horizon"] == horizon)]
            b = dense[dense["horizon"] == horizon]
            merged = a.merge(b, on=["seed", "trajectory_identifier", "condition", "horizon"], suffixes=("_model", "_dense"))
            diff = (merged["metric_value_model"] - merged["metric_value_dense"]).to_numpy(dtype=np.float64)
            ci_low, ci_high = bootstrap_ci(diff, rng, n_resamples)
            rows.append(
                {
                    "family": selected_row["family"],
                    "model": model,
                    "horizon": int(horizon),
                    "model_mean_nrmse": float(merged["metric_value_model"].mean()),
                    "dense_mean_nrmse": float(merged["metric_value_dense"].mean()),
                    "diff_vs_dense": float(diff.mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_pairs": int(diff.size),
                }
            )
    return pd.DataFrame(rows)


def plot_pareto(test_h100: pd.DataFrame, density: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = test_h100.merge(density, on=["model", "family"], how="left")
    colors = {
        "dense_mlp_tanh": "#4C78A8",
        "sparse_mlp_relu": "#F58518",
        "lista_relu": "#54A24B",
        "lista_shrink": "#B279A2",
    }
    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    for family in FAMILY_ORDER:
        part = data[data["family"] == family]
        if part.empty:
            continue
        ax.scatter(
            part["latent_active_density_abs_1e-3"],
            part["mean_nrmse"],
            label=family,
            s=55,
            color=colors.get(family),
            alpha=0.85,
        )
    ax.set_xlabel("validation latent active density at |z| > 1e-3")
    ax.set_ylabel("test H100 NRMSE")
    ax.set_title("Lorenz-96 sparsity mechanism Pareto view")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"lorenz96_sparsity_mechanism_pareto.{ext}", dpi=220)
    plt.close(fig)


def write_report(
    *,
    root: Path,
    selected: pd.DataFrame,
    selected_test: pd.DataFrame,
    paired: pd.DataFrame,
    pareto: pd.DataFrame,
    meta: pd.DataFrame,
) -> None:
    report = root / "reports" / "sparsity_mechanism_sweep_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Lorenz-96 Sparsity Mechanism Sweep",
        "",
        "Condition: D=128, dz=512, noise 0.05, train window 20, dense K, no K regularization. Training budget and patience are read from each resolved configuration.",
        "",
        "## Validation-Selected Rows",
        "",
        markdown_table(selected[["family", "model", "mean_nrmse", "latent_active_density_abs_1e-3", "latent_active_coords_abs_1e-3", "selection_rule"]]),
        "",
        "## Selected Test NRMSE",
        "",
        markdown_table(selected_test.pivot_table(index=["family", "model"], columns="horizon", values="mean_nrmse").reset_index()),
        "",
        "## Selected Paired Differences vs Dense MLP",
        "",
        markdown_table(paired),
        "",
        "## Test H100 Pareto Table",
        "",
        markdown_table(pareto.sort_values(["family", "latent_active_density_abs_1e-3", "mean_nrmse"])),
        "",
        "## Training Summary",
        "",
        markdown_table(meta.groupby(["family", "model"])[["training_steps", "training_time_seconds", "parameter_count"]].mean().reset_index()),
    ]
    report.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/lorenz96_sparsity_mechanism_sweep_20260624")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()

    root = Path(args.root)
    results_dir = root / "results"
    figures_dir = root / "reports" / "figures"
    raw_path = results_dir / "raw_metrics.parquet"
    if not raw_path.exists():
        raw_path = results_dir / "raw_metrics.partial.parquet"
    raw = pd.read_parquet(raw_path)
    meta = parse_label_metadata(raw)
    val_summary = nrmse_summary(raw, meta, "val", args.bootstrap_resamples)
    test_summary = nrmse_summary(raw, meta, "test", args.bootstrap_resamples)
    density = active_density_table(raw, meta, "val")
    selected = select_by_validation(val_summary, density)
    paired = paired_vs_dense(raw, selected, args.bootstrap_resamples)
    selected_test = test_summary[test_summary["model"].isin(selected["model"])].copy()
    h100_pareto = test_summary[test_summary["horizon"] == 100].merge(density, on=["model", "family"], how="left")

    val_summary.to_csv(results_dir / "validation_nrmse_by_model_horizon.csv", index=False)
    test_summary.to_csv(results_dir / "test_nrmse_by_model_horizon.csv", index=False)
    density.to_csv(results_dir / "validation_latent_density_by_model.csv", index=False)
    selected.to_csv(results_dir / "validation_selected_by_family.csv", index=False)
    selected_test.to_csv(results_dir / "validation_selected_test_nrmse.csv", index=False)
    paired.to_csv(results_dir / "selected_paired_vs_dense.csv", index=False)
    h100_pareto.to_csv(results_dir / "test_h100_pareto.csv", index=False)
    meta.to_csv(results_dir / "model_config_summary.csv", index=False)
    plot_pareto(test_summary[test_summary["horizon"] == 100], density, figures_dir)
    write_report(root=root, selected=selected, selected_test=selected_test, paired=paired, pareto=h100_pareto, meta=meta)

    print("Validation-selected rows:")
    print(selected[["family", "model", "mean_nrmse", "latent_active_density_abs_1e-3", "latent_active_coords_abs_1e-3"]].to_string(index=False))
    print("\nSelected paired vs dense:")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
