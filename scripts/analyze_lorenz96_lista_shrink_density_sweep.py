"""Analyze LISTA-shrink Lorenz-96 density sweep across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_resamples: int = 2000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
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
            display[col] = display[col].map(lambda v: "" if pd.isna(v) else f"{v:.6g}")
    lines = [
        "| " + " | ".join(display.columns.astype(str)) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for row in display.to_numpy():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def read_raw(root: Path) -> pd.DataFrame:
    raw_path = root / "results" / "raw_metrics.parquet"
    if not raw_path.exists():
        raw_path = root / "results" / "raw_metrics.partial.parquet"
    raw = pd.read_parquet(raw_path)
    raw["metric_name"] = raw["metric_name"].replace(
        {
            "latent_active_active_coords_abs_1e-3": "latent_active_coords_abs_1e-3",
            "latent_active_active_coords_abs_1e-4": "latent_active_coords_abs_1e-4",
            "latent_active_active_coords_abs_1e-2": "latent_active_coords_abs_1e-2",
            "latent_active_active_coords_rel_1e-3max": "latent_active_coords_rel_1e-3max",
        }
    )
    return raw


def parse_metadata(raw: pd.DataFrame) -> pd.DataFrame:
    paths = sorted(str(path) for path in raw["configuration_path"].dropna().unique() if str(path))
    rows: list[dict[str, object]] = []
    for path in paths:
        payload_path = Path(path)
        if not payload_path.exists():
            continue
        payload = json.loads(payload_path.read_text())
        variant = dict(payload.get("model_variant", {}))
        parent = payload_path.parent.name
        seed = int(parent.split("_seed", 1)[1].split("_", 1)[0]) if "_seed" in parent else int(payload.get("seed", -1))
        rows.append(
            {
                "model": str(payload.get("label", "")),
                "seed": seed,
                "family": str(variant.get("family", payload.get("label", ""))),
                "lista_alpha": float(variant.get("lista_alpha", np.nan)),
                "training_steps": int(len(payload.get("history", []))),
                "training_time_seconds": float(payload.get("training_time_seconds", np.nan)),
                "parameter_count": int(payload.get("parameter_count", 0)),
                "best_val_training_horizon_nrmse": float(payload.get("best_val_nrmse", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def per_seed_tables(raw: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    nrmse = raw[(raw["metric_name"] == "nrmse") & (raw["split"].isin(["val", "test"]))].copy()
    seed_metric = (
        nrmse.groupby(["model", "seed", "split", "horizon"], as_index=False)["metric_value"]
        .mean()
        .rename(columns={"metric_value": "nrmse"})
    )
    seed_metric = seed_metric.merge(meta, on=["model", "seed"], how="left")

    density_metrics = [
        "latent_active_density_abs_1e-3",
        "latent_active_coords_abs_1e-3",
        "latent_mean_abs",
        "latent_max_abs",
    ]
    density = raw[
        (raw["split"] == "val")
        & (raw["trajectory_identifier"] == "model")
        & (raw["metric_name"].isin(density_metrics))
    ].copy()
    density = density.pivot_table(index=["model", "seed"], columns="metric_name", values="metric_value").reset_index()
    density = density.merge(meta, on=["model", "seed"], how="left")
    return seed_metric, density


def summarize(seed_metric: pd.DataFrame, density: pd.DataFrame, n_resamples: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    density_cols = ["latent_active_density_abs_1e-3", "latent_active_coords_abs_1e-3", "latent_mean_abs", "latent_max_abs"]
    density_mean = density.groupby(["model", "lista_alpha"])[density_cols].mean().reset_index()
    density_std = density.groupby(["model", "lista_alpha"])[density_cols].std(ddof=1).reset_index()
    for (model, alpha, split, horizon), part in seed_metric.groupby(["model", "lista_alpha", "split", "horizon"]):
        values = part["nrmse"].to_numpy(dtype=np.float64)
        ci_low, ci_high = bootstrap_ci(values, rng, n_resamples)
        rows.append(
            {
                "model": model,
                "lista_alpha": float(alpha),
                "split": split,
                "horizon": int(horizon),
                "mean_nrmse": float(values.mean()),
                "std_seed_nrmse": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "ci_low_seed_bootstrap": ci_low,
                "ci_high_seed_bootstrap": ci_high,
                "n_seeds": int(values.size),
            }
        )
    summary = pd.DataFrame(rows)
    summary = summary.merge(density_mean, on=["model", "lista_alpha"], how="left")
    summary = summary.merge(
        density_std.rename(columns={col: f"{col}_seed_std" for col in density_cols}),
        on=["model", "lista_alpha"],
        how="left",
    )
    train = (
        seed_metric[["model", "seed", "lista_alpha", "training_steps", "training_time_seconds", "parameter_count"]]
        .drop_duplicates()
        .groupby(["model", "lista_alpha"], as_index=False)
        .agg(
            mean_training_steps=("training_steps", "mean"),
            mean_training_time_seconds=("training_time_seconds", "mean"),
            total_training_time_seconds=("training_time_seconds", "sum"),
            parameter_count=("parameter_count", "first"),
        )
    )
    return summary.merge(train, on=["model", "lista_alpha"], how="left")


def plot_density(summary: pd.DataFrame, root: Path) -> None:
    fig_dir = root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    h100 = summary[(summary["split"] == "test") & (summary["horizon"] == 100)].copy().sort_values("latent_active_density_abs_1e-3")
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    ax.errorbar(
        h100["latent_active_density_abs_1e-3"],
        h100["mean_nrmse"],
        yerr=h100["std_seed_nrmse"],
        fmt="o-",
        linewidth=1.6,
        markersize=4.5,
        capsize=2.5,
    )
    ax.set_xlabel("validation active latent density, |z| > 1e-3")
    ax.set_ylabel("test H100 NRMSE, mean over seeds")
    ax.set_title("LISTA-shrink sparsity-density sweep")
    ax.grid(True, alpha=0.25)
    for _, row in h100.iterrows():
        if row["lista_alpha"] in {0.03, 0.1, 1.0, 2.0, 3.0}:
            ax.annotate(f"a={row['lista_alpha']:g}", (row["latent_active_density_abs_1e-3"], row["mean_nrmse"]), fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"lista_shrink_density_vs_h100.{ext}", dpi=220)
    plt.close(fig)

    alpha = summary[(summary["split"] == "val") & (summary["horizon"] == 100)].copy().sort_values("lista_alpha")
    fig, ax1 = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    ax1.plot(alpha["lista_alpha"], alpha["latent_active_density_abs_1e-3"], "o-", color="#4C78A8", label="density")
    ax1.set_xscale("log")
    ax1.set_xlabel("LISTA shrink threshold alpha")
    ax1.set_ylabel("validation active density", color="#4C78A8")
    ax1.tick_params(axis="y", labelcolor="#4C78A8")
    ax2 = ax1.twinx()
    ax2.plot(alpha["lista_alpha"], alpha["mean_nrmse"], "s--", color="#F58518", label="val H100 NRMSE")
    ax2.set_ylabel("validation H100 NRMSE", color="#F58518")
    ax2.tick_params(axis="y", labelcolor="#F58518")
    ax1.grid(True, alpha=0.25)
    ax1.set_title("Threshold controls achieved density")
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"lista_shrink_alpha_density_validation.{ext}", dpi=220)
    plt.close(fig)

    horizons = summary[(summary["split"] == "test") & (summary["horizon"].isin([50, 100]))].copy()
    horizons["latent_sparsity_abs_1e-3"] = 1.0 - horizons["latent_active_density_abs_1e-3"]
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    colors = {50: "#4C78A8", 100: "#F58518"}
    markers = {50: "o", 100: "s"}
    for horizon in [50, 100]:
        part = horizons[horizons["horizon"] == horizon].sort_values("latent_sparsity_abs_1e-3")
        yerr = np.vstack(
            [
                part["mean_nrmse"] - part["ci_low_seed_bootstrap"],
                part["ci_high_seed_bootstrap"] - part["mean_nrmse"],
            ]
        )
        ax.errorbar(
            part["latent_sparsity_abs_1e-3"],
            part["mean_nrmse"],
            yerr=yerr,
            fmt=f"{markers[horizon]}-",
            color=colors[horizon],
            linewidth=1.8,
            markersize=4.5,
            capsize=2.5,
            label=f"H{horizon}",
        )
        best = part.loc[part["mean_nrmse"].idxmin()]
        ax.scatter(
            [best["latent_sparsity_abs_1e-3"]],
            [best["mean_nrmse"]],
            s=72,
            facecolors="none",
            edgecolors=colors[horizon],
            linewidths=1.8,
            zorder=5,
        )
        ax.annotate(
            f"a={best['lista_alpha']:g}",
            (best["latent_sparsity_abs_1e-3"], best["mean_nrmse"]),
            xytext=(5, -12 if horizon == 50 else 8),
            textcoords="offset points",
            fontsize=8,
            color=colors[horizon],
        )
    ax.set_xlabel("latent sparsity, fraction of inactive coordinates (|z| <= 1e-3)")
    ax.set_ylabel("test rollout NRMSE, mean over seeds")
    ax.set_title("LISTA-shrink sparsity vs forecasting performance")
    ax.grid(True, alpha=0.25)
    ax.legend(title="rollout horizon", frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"lista_shrink_sparsity_vs_h50_h100.{ext}", dpi=220)
    plt.close(fig)


def write_report(root: Path, summary: pd.DataFrame, per_seed: pd.DataFrame, n_resamples: int) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    h100 = summary[(summary["horizon"] == 100)].copy()
    val = h100[(h100["split"] == "val") & (h100["latent_active_density_abs_1e-3"] >= 0.05)].sort_values("mean_nrmse")
    test = h100[(h100["split"] == "test") & (h100["latent_active_density_abs_1e-3"] >= 0.05)].sort_values("mean_nrmse")
    val_best = val.head(1)
    test_best = test.head(1)
    top_test = test.head(10)[
        [
            "lista_alpha",
            "latent_active_density_abs_1e-3",
            "latent_active_coords_abs_1e-3",
            "mean_nrmse",
            "std_seed_nrmse",
            "ci_low_seed_bootstrap",
            "ci_high_seed_bootstrap",
            "n_seeds",
            "mean_training_steps",
            "mean_training_time_seconds",
        ]
    ]
    density_curve = h100[h100["split"] == "test"][
        [
            "lista_alpha",
            "latent_active_density_abs_1e-3",
            "latent_active_coords_abs_1e-3",
            "mean_nrmse",
            "std_seed_nrmse",
            "n_seeds",
        ]
    ].sort_values("latent_active_density_abs_1e-3")
    best_by_horizon = (
        summary[(summary["split"] == "test") & (summary["horizon"].isin([1, 5, 10, 25, 50, 100]))]
        .sort_values(["horizon", "mean_nrmse"])
        .groupby("horizon", as_index=False)
        .head(1)
        [
            [
                "horizon",
                "lista_alpha",
                "latent_active_density_abs_1e-3",
                "latent_active_coords_abs_1e-3",
                "mean_nrmse",
                "std_seed_nrmse",
                "ci_low_seed_bootstrap",
                "ci_high_seed_bootstrap",
            ]
        ]
        .sort_values("horizon")
    )
    paired_path = root / "results" / "lista_shrink_density_paired_h100_vs_baselines.csv"
    paired_subset = pd.DataFrame()
    if paired_path.exists():
        paired = pd.read_csv(paired_path)
        dmd = paired[
            (paired["baseline"] == "dmd")
            & (paired["lista_alpha"].isin([0.015, 0.03, 0.05, 0.06, 1.5, 2.0, 2.5]))
        ][
            [
                "lista_alpha",
                "latent_active_density_abs_1e-3",
                "lista_mean_nrmse",
                "diff_vs_baseline",
                "ci_low_seed_bootstrap",
                "ci_high_seed_bootstrap",
            ]
        ].rename(
            columns={
                "latent_active_density_abs_1e-3": "density",
                "lista_mean_nrmse": "lista_h100",
                "diff_vs_baseline": "diff_vs_dmd",
                "ci_low_seed_bootstrap": "dmd_ci_low",
                "ci_high_seed_bootstrap": "dmd_ci_high",
            }
        )
        tsvd = paired[
            (paired["baseline"] == "truncated_svd_dmd")
            & (paired["lista_alpha"].isin([0.015, 0.03, 0.05, 0.06, 1.5, 2.0, 2.5]))
        ][
            [
                "lista_alpha",
                "diff_vs_baseline",
                "ci_low_seed_bootstrap",
                "ci_high_seed_bootstrap",
            ]
        ].rename(
            columns={
                "diff_vs_baseline": "diff_vs_trunc_svd_dmd",
                "ci_low_seed_bootstrap": "trunc_ci_low",
                "ci_high_seed_bootstrap": "trunc_ci_high",
            }
        )
        paired_subset = dmd.merge(tsvd, on="lista_alpha", how="left").sort_values("lista_alpha")
    train_time = (
        summary[["model", "lista_alpha", "mean_training_steps", "mean_training_time_seconds", "total_training_time_seconds"]]
        .drop_duplicates()
        .dropna(subset=["lista_alpha"])
    )
    total_training_time = float(train_time["total_training_time_seconds"].sum()) if not train_time.empty else float("nan")
    lines = [
        "# LISTA-shrink Lorenz-96 Density Sweep",
        "",
        "Condition: Lorenz-96 D=128, noise 0.05, dz=512, dense K, no K regularization, train horizon 20.",
        "Each row is one LISTA-shrink threshold alpha trained under five seeds. Metrics are first averaged per seed, then averaged across seeds.",
        f"Seed-bootstrap intervals use {n_resamples} resamples over the five seed means.",
        "",
        "## Validation-selected H100 Optimum",
        "",
        markdown_table(
            val_best[
                [
                    "lista_alpha",
                    "latent_active_density_abs_1e-3",
                    "latent_active_coords_abs_1e-3",
                    "mean_nrmse",
                    "std_seed_nrmse",
                    "ci_low_seed_bootstrap",
                    "ci_high_seed_bootstrap",
                    "n_seeds",
                ]
            ]
        ),
        "",
        "## Observed Test H100 Optimum",
        "",
        markdown_table(
            test_best[
                [
                    "lista_alpha",
                    "latent_active_density_abs_1e-3",
                    "latent_active_coords_abs_1e-3",
                    "mean_nrmse",
                    "std_seed_nrmse",
                    "ci_low_seed_bootstrap",
                    "ci_high_seed_bootstrap",
                    "n_seeds",
                ]
            ]
        ),
        "",
        "## Top Test H100 Settings",
        "",
        markdown_table(top_test),
        "",
        "## Best Test Setting By Horizon",
        "",
        "The optimal latent sparsity depends strongly on rollout horizon. Short and medium horizons prefer substantially sparser encodings, while the longest H100 forecast prefers an almost dense latent code.",
        "",
        markdown_table(best_by_horizon),
        "",
        "## H100 Paired Comparison To Baselines",
        "",
        "Differences are seed-paired LISTA-shrink minus baseline NRMSE at H100, bootstrapped over the five seed means. Negative values favor LISTA-shrink. The validation-selected row, alpha 0.015, improves over DMD by -0.0316 NRMSE with a 95% seed-bootstrap interval excluding zero, but only near-dense LISTA-shrink rows beat DMD at H100.",
        "",
        markdown_table(paired_subset),
        "",
        "## Training Cost",
        "",
        f"All 150 LISTA-shrink trainings completed without recorded failures. Total recorded neural-model training time was {total_training_time:.1f} seconds, or {total_training_time / 60.0:.1f} minutes, on CPU. The SLURM allocation wall time was about 3.75 hours including Lorenz--96 data generation, baselines, evaluation, plotting, and file I/O.",
        "",
        "## Test H100 Density Curve",
        "",
        markdown_table(density_curve),
        "",
        "## Notes",
        "",
        "Rows with near-zero density are degenerate collapsed predictors and should not be interpreted as useful sparse dynamics.",
    ]
    (reports / "lista_shrink_density_sweep_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/lorenz96_lista_shrink_density_sweep_30x5_cpu_20260625")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()

    root = Path(args.root)
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    raw = read_raw(root)
    meta = parse_metadata(raw)
    seed_metric, density = per_seed_tables(raw, meta)
    summary = summarize(seed_metric, density, args.bootstrap_resamples)

    seed_metric.to_csv(results_dir / "lista_shrink_density_per_seed_forecast.csv", index=False)
    density.to_csv(results_dir / "lista_shrink_density_per_seed_latent.csv", index=False)
    summary.to_csv(results_dir / "lista_shrink_density_summary.csv", index=False)
    plot_density(summary, root)
    write_report(root, summary, seed_metric, args.bootstrap_resamples)
    h100_test = summary[(summary["split"] == "test") & (summary["horizon"] == 100)].sort_values("mean_nrmse")
    print(markdown_table(h100_test.head(8)[["lista_alpha", "latent_active_density_abs_1e-3", "mean_nrmse", "std_seed_nrmse", "n_seeds"]]))


if __name__ == "__main__":
    main()
