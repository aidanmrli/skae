"""Summarize the Lorenz-96 latent-sparsity training-horizon sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_resamples: int) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        means[i] = rng.choice(values, size=values.size, replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def condition_train_horizon(condition: str) -> int:
    marker = "_Htrain"
    if marker not in condition:
        return -1
    return int(condition.split(marker, 1)[1].split("_", 1)[0])


def paired_sparse_dense(df: pd.DataFrame, n_resamples: int) -> pd.DataFrame:
    nrmse = df[
        (df["metric_name"] == "nrmse")
        & (df["split"] == "test")
        & (df["model"].isin(["dense_kae", "skae_latent_l1"]))
    ].copy()
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(0)
    group_cols = ["condition", "horizon", "sparsity_coefficient"]
    for (condition, horizon, sparsity), sparse in nrmse[nrmse["model"] == "skae_latent_l1"].groupby(group_cols):
        dense = nrmse[
            (nrmse["condition"] == condition)
            & (nrmse["horizon"] == horizon)
            & (nrmse["sparsity_coefficient"] == 0.0)
            & (nrmse["model"] == "dense_kae")
        ]
        merged = sparse.merge(
            dense,
            on=["condition", "horizon", "seed", "trajectory_identifier"],
            suffixes=("_sparse", "_dense"),
        )
        if merged.empty:
            continue
        diff = (merged["metric_value_sparse"] - merged["metric_value_dense"]).to_numpy(dtype=np.float64)
        ci_low, ci_high = bootstrap_ci(diff, rng, n_resamples)
        dense_mean = float(merged["metric_value_dense"].mean())
        rows.append(
            {
                "condition": condition,
                "train_horizon": condition_train_horizon(str(condition)),
                "rollout_horizon": int(horizon),
                "sparsity_coefficient": float(sparsity),
                "n_pairs": int(diff.size),
                "sparse_mean_nrmse": float(merged["metric_value_sparse"].mean()),
                "dense_mean_nrmse": dense_mean,
                "diff_sparse_minus_dense": float(diff.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "pct_change_vs_dense": float(100.0 * diff.mean() / dense_mean) if dense_mean else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["train_horizon", "rollout_horizon", "diff_sparse_minus_dense"])


def spectral_summary(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[
        (df["trajectory_identifier"] == "model")
        & (df["metric_name"].isin(["spectral_radius", "k_l1", "effective_density_1e3"]))
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["train_horizon"] = sub["condition"].map(condition_train_horizon)
    grouped = (
        sub.groupby(["train_horizon", "model", "sparsity_coefficient", "metric_name"])["metric_value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return grouped.sort_values(["train_horizon", "metric_name", "model", "sparsity_coefficient"])


def validation_selection(df: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    config_paths = sorted(
        str(path)
        for path in df.loc[df["configuration_path"].notna(), "configuration_path"].unique()
        if str(path)
    )
    rows: list[dict[str, object]] = []
    for path in config_paths:
        path_obj = Path(path)
        if not path_obj.exists():
            continue
        payload = json.loads(path_obj.read_text())
        train_cfg = payload.get("training", {})
        if payload.get("label") not in {"dense_kae", "skae_latent_l1"}:
            continue
        run_parts = path_obj.parts
        condition = next((part for part in run_parts if "_Htrain" in part), "")
        rows.append(
            {
                "condition": condition,
                "train_horizon": condition_train_horizon(condition),
                "model": payload.get("label"),
                "sparsity_coefficient": float(payload.get("sparsity_coeff", 0.0)),
                "sparsity_target": payload.get("sparsity_target", ""),
                "best_val_nrmse": float(payload.get("best_val_nrmse", float("nan"))),
                "training_time_seconds": float(payload.get("training_time_seconds", float("nan"))),
                "trainable_parameter_count": int(payload.get("trainable_parameter_count", 0)),
                "train_horizon_config": int(train_cfg.get("train_horizon", -1)),
                "resolved_config": str(path_obj),
            }
        )
    configs = pd.DataFrame(rows)
    if configs.empty:
        return configs
    sparse_mean = (
        configs[configs["model"] == "skae_latent_l1"]
        .groupby(["train_horizon", "sparsity_coefficient"], as_index=False)["best_val_nrmse"]
        .mean()
        .rename(columns={"best_val_nrmse": "mean_val_nrmse"})
    )
    selected = sparse_mean.sort_values(["train_horizon", "mean_val_nrmse", "sparsity_coefficient"]).groupby("train_horizon", as_index=False).head(1)
    out_rows: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        train_horizon = int(row["train_horizon"])
        sparsity = float(row["sparsity_coefficient"])
        for rollout_horizon in [25, 50, 100]:
            match = paired[
                (paired["train_horizon"] == train_horizon)
                & (paired["sparsity_coefficient"] == sparsity)
                & (paired["rollout_horizon"] == rollout_horizon)
            ]
            if match.empty:
                continue
            record = match.iloc[0].to_dict()
            record["selection_rule"] = "lowest_mean_validation_nrmse_across_sparse_coefficients"
            record["selected_mean_val_nrmse"] = float(row["mean_val_nrmse"])
            out_rows.append(record)
    return pd.DataFrame(out_rows)


def plot_long_horizon_deltas(paired: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_df = paired[paired["rollout_horizon"].isin([25, 50, 100])].copy()
    if plot_df.empty:
        return
    plot_df["lambda_label"] = plot_df["sparsity_coefficient"].map(lambda v: f"{v:g}")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True, constrained_layout=True)
    horizons = [25, 50, 100]
    palette = {10: "#4C78A8", 20: "#F58518", 40: "#54A24B"}
    for ax, rollout_horizon in zip(axes, horizons):
        hdf = plot_df[plot_df["rollout_horizon"] == rollout_horizon]
        for train_horizon, tdf in hdf.groupby("train_horizon"):
            tdf = tdf.sort_values("sparsity_coefficient")
            yerr = np.vstack(
                [
                    tdf["diff_sparse_minus_dense"].to_numpy() - tdf["ci_low"].to_numpy(),
                    tdf["ci_high"].to_numpy() - tdf["diff_sparse_minus_dense"].to_numpy(),
                ]
            )
            ax.errorbar(
                tdf["lambda_label"],
                tdf["diff_sparse_minus_dense"],
                yerr=yerr,
                marker="o",
                linewidth=1.8,
                capsize=3,
                label=f"train horizon {train_horizon}",
                color=palette.get(int(train_horizon), None),
            )
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_title(f"Rollout horizon {rollout_horizon}")
        ax.set_xlabel("latent L1 coefficient")
        ax.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("Paired NRMSE difference\n(sparse - dense)")
    axes[-1].legend(frameon=False, fontsize=9)
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"lorenz96_latent_sparse_horizon_deltas.{ext}", dpi=220)
    plt.close(fig)


def write_topline(paired: pd.DataFrame, out_path: Path) -> None:
    selected = paired[paired["rollout_horizon"].isin([25, 50, 100])].copy()
    best = (
        selected.sort_values("diff_sparse_minus_dense")
        .groupby(["train_horizon", "rollout_horizon"], as_index=False)
        .head(1)
        .sort_values(["train_horizon", "rollout_horizon"])
    )
    best.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/lorenz96_latent_sparse_horizon_20260624")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()

    root = Path(args.root)
    df = pd.read_parquet(root / "results" / "raw_metrics.parquet")
    results_dir = root / "results"
    figures_dir = root / "reports" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)

    paired = paired_sparse_dense(df, args.bootstrap_resamples)
    paired.to_csv(results_dir / "paired_sparse_minus_dense_nrmse.csv", index=False)
    write_topline(paired, results_dir / "best_sparse_delta_by_horizon.csv")

    spectra = spectral_summary(df)
    spectra.to_csv(results_dir / "spectral_summary.csv", index=False)

    selected = validation_selection(df, paired)
    selected.to_csv(results_dir / "validation_selected_sparse_test_nrmse.csv", index=False)

    plot_long_horizon_deltas(paired, figures_dir)

    print("Wrote", results_dir / "paired_sparse_minus_dense_nrmse.csv")
    print("Wrote", results_dir / "best_sparse_delta_by_horizon.csv")
    print("Wrote", results_dir / "spectral_summary.csv")
    print("Wrote", results_dir / "validation_selected_sparse_test_nrmse.csv")
    print("Wrote", figures_dir / "lorenz96_latent_sparse_horizon_deltas.pdf")
    print(paired[paired["rollout_horizon"].isin([25, 50, 100])].to_string(index=False))


if __name__ == "__main__":
    main()
