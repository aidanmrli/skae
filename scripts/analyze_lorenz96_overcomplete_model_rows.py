"""Summarize overcomplete Lorenz-96 model-row benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "dmd",
    "truncated_svd_dmd",
    "dense_mlp_kae",
    "sparse_mlp_l1_0.01",
    "sparse_mlp_l1_0.1",
    "lista_relu",
    "lista_shrink",
    "lista_sign_split",
    "hyperlista",
    "persistence",
]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_resamples: int) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        samples[i] = rng.choice(values, size=values.size, replace=True).mean()
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def summarize_nrmse(df: pd.DataFrame, n_resamples: int) -> pd.DataFrame:
    sub = df[(df["metric_name"] == "nrmse") & (df["split"] == "test")].copy()
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for (model, horizon, latent_dim, sparsity), part in sub.groupby(
        ["model", "horizon", "latent_dim", "sparsity_coefficient"]
    ):
        values = part["metric_value"].to_numpy(dtype=np.float64)
        ci_low, ci_high = bootstrap_ci(values, rng, n_resamples)
        rows.append(
            {
                "model": model,
                "horizon": int(horizon),
                "latent_dim": int(latent_dim),
                "sparsity_coefficient": float(sparsity),
                "mean_nrmse": float(values.mean()),
                "std_nrmse": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": int(values.size),
            }
        )
    out = pd.DataFrame(rows)
    out["model_order"] = out["model"].map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(999)
    return out.sort_values(["horizon", "mean_nrmse", "model_order"]).drop(columns=["model_order"])


def paired_vs_dense(df: pd.DataFrame, n_resamples: int) -> pd.DataFrame:
    sub = df[(df["metric_name"] == "nrmse") & (df["split"] == "test")].copy()
    dense = sub[sub["model"] == "dense_mlp_kae"]
    rng = np.random.default_rng(1)
    rows: list[dict[str, object]] = []
    for (model, horizon), part in sub[sub["model"] != "dense_mlp_kae"].groupby(["model", "horizon"]):
        merged = part.merge(
            dense[dense["horizon"] == horizon],
            on=["seed", "trajectory_identifier", "horizon", "condition"],
            suffixes=("_model", "_dense"),
        )
        if merged.empty:
            continue
        diff = (merged["metric_value_model"] - merged["metric_value_dense"]).to_numpy(dtype=np.float64)
        ci_low, ci_high = bootstrap_ci(diff, rng, n_resamples)
        dense_mean = float(merged["metric_value_dense"].mean())
        rows.append(
            {
                "model": model,
                "horizon": int(horizon),
                "model_mean_nrmse": float(merged["metric_value_model"].mean()),
                "dense_mean_nrmse": dense_mean,
                "diff_vs_dense": float(diff.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "pct_change_vs_dense": float(100.0 * diff.mean() / dense_mean) if dense_mean else float("nan"),
                "n_pairs": int(diff.size),
            }
        )
    out = pd.DataFrame(rows)
    out["model_order"] = out["model"].map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(999)
    return out.sort_values(["horizon", "model_order"]).drop(columns=["model_order"])


def config_summary(df: pd.DataFrame) -> pd.DataFrame:
    paths = sorted(str(p) for p in df["configuration_path"].dropna().unique() if str(p))
    rows: list[dict[str, object]] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        payload = json.loads(p.read_text())
        if "parameter_count" not in payload:
            continue
        rows.append(
            {
                "model": payload.get("label", ""),
                "seed": int(p.parent.name.split("_seed", 1)[1].split("_", 1)[0]) if "_seed" in p.parent.name else -1,
                "latent_dim": int(payload.get("latent_dim", 0)),
                "sparsity_coefficient": float(payload.get("sparsity_coeff", 0.0)),
                "sparsity_target": payload.get("sparsity_target", ""),
                "parameter_count": int(payload.get("parameter_count", 0)),
                "trainable_parameter_count": int(payload.get("trainable_parameter_count", 0)),
                "best_val_nrmse": float(payload.get("best_val_nrmse", float("nan"))),
                "training_time_seconds": float(payload.get("training_time_seconds", float("nan"))),
                "model_preset": payload.get("model_preset", ""),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "seed"]) if rows else pd.DataFrame()


def diagnostics_summary(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[
        (df["trajectory_identifier"] == "model")
        & (df["metric_name"].isin(["spectral_radius", "effective_density_1e3", "k_l1", "inference_time_seconds"]))
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.groupby(["model", "latent_dim", "sparsity_coefficient", "metric_name"])["metric_value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["metric_name", "model"])
    )


def plot_nrmse(summary: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    pivot = summary.pivot_table(index="horizon", columns="model", values="mean_nrmse")
    for suffix, ylim in [("", None), ("_zoom", (0.15, 1.55))]:
        fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
        for model in MODEL_ORDER:
            if model not in pivot.columns:
                continue
            ax.plot(pivot.index, pivot[model], marker="o", linewidth=1.8, label=model)
        ax.set_xlabel("rollout horizon")
        ax.set_ylabel("test NRMSE")
        ax.set_title("Lorenz-96 D=128, dz=512 overcomplete model rows")
        if ylim is not None:
            ax.set_ylim(*ylim)
            ax.set_title("Lorenz-96 D=128, dz=512 overcomplete model rows (zoom)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        for ext in ("pdf", "png"):
            fig.savefig(figures_dir / f"lorenz96_overcomplete_model_rows_nrmse{suffix}.{ext}", dpi=220)
        plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda v: "" if pd.isna(v) else f"{v:.6g}")
    headers = [str(col) for col in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_markdown(summary: pd.DataFrame, paired: pd.DataFrame, configs: pd.DataFrame, diagnostics: pd.DataFrame, out_path: Path) -> None:
    h100 = summary[summary["horizon"] == 100].sort_values("mean_nrmse")
    h50 = summary[summary["horizon"] == 50].sort_values("mean_nrmse")
    paired_h100 = paired[paired["horizon"] == 100]
    wide = summary.pivot_table(index="model", columns="horizon", values="mean_nrmse")
    wide = wide.reindex([model for model in MODEL_ORDER if model in wide.index]).reset_index()
    wide.columns = ["model"] + [f"H{int(col)}" for col in wide.columns[1:]]
    lines = [
        "# Lorenz-96 D128 Overcomplete Model Rows",
        "",
        "Condition: D=128, F=8, full observation, observation noise 0.05, 64 train / 16 validation / 16 test trajectories, seeds 0/1/2, training window 20, latent dimension 512 for neural rows.",
        "",
        "No neural row regularizes K. Dense K is used throughout; sparsity is induced by latent L1 and/or the encoder nonlinearity.",
        "",
        "## Mean Test NRMSE Across All Horizons",
        "",
        markdown_table(wide),
        "",
        "## H100 Ranking",
        "",
        markdown_table(h100[["model", "latent_dim", "sparsity_coefficient", "mean_nrmse", "std_nrmse", "ci_low", "ci_high", "n"]]),
        "",
        "## H50 Ranking",
        "",
        markdown_table(h50[["model", "latent_dim", "sparsity_coefficient", "mean_nrmse", "std_nrmse", "ci_low", "ci_high", "n"]]),
        "",
        "## H100 Paired Differences vs Dense MLP KAE",
        "",
        markdown_table(paired_h100[["model", "model_mean_nrmse", "dense_mean_nrmse", "diff_vs_dense", "ci_low", "ci_high", "pct_change_vs_dense", "n_pairs"]]),
        "",
        "## Parameter and Validation Summary",
        "",
        markdown_table(configs.groupby("model")[["parameter_count", "trainable_parameter_count", "best_val_nrmse", "training_time_seconds"]].mean().reset_index()) if not configs.empty else "_No config summary available._",
        "",
        "## Diagnostics",
        "",
        markdown_table(diagnostics) if not diagnostics.empty else "_No diagnostics available._",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/lorenz96_overcomplete_model_rows_20260624")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()

    root = Path(args.root)
    results_dir = root / "results"
    figures_dir = root / "reports" / "figures"
    reports_dir = root / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(results_dir / "raw_metrics.parquet")
    summary = summarize_nrmse(raw, args.bootstrap_resamples)
    paired = paired_vs_dense(raw, args.bootstrap_resamples)
    configs = config_summary(raw)
    diagnostics = diagnostics_summary(raw)
    wide = summary.pivot_table(index="model", columns="horizon", values="mean_nrmse")
    wide = wide.reindex([model for model in MODEL_ORDER if model in wide.index]).reset_index()
    wide.columns = ["model"] + [f"H{int(col)}" for col in wide.columns[1:]]

    summary.to_csv(results_dir / "nrmse_by_model_horizon.csv", index=False)
    wide.to_csv(results_dir / "nrmse_wide_by_model.csv", index=False)
    paired.to_csv(results_dir / "paired_vs_dense_mlp_kae.csv", index=False)
    configs.to_csv(results_dir / "model_config_summary.csv", index=False)
    diagnostics.to_csv(results_dir / "diagnostics_summary.csv", index=False)
    plot_nrmse(summary, figures_dir)
    write_markdown(summary, paired, configs, diagnostics, reports_dir / "overcomplete_model_rows_report.md")

    print("H100 ranking:")
    print(summary[summary["horizon"] == 100].sort_values("mean_nrmse").to_string(index=False))
    print("\nH100 paired vs dense:")
    print(paired[paired["horizon"] == 100].to_string(index=False))


if __name__ == "__main__":
    main()
