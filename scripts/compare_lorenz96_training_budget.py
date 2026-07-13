"""Compare Lorenz-96 sparsity mechanism results across training budgets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(root / "results" / name)


def _horizon_table(root: Path, label: str) -> pd.DataFrame:
    df = _read(root, "test_nrmse_by_model_horizon.csv")
    keep = df[df["horizon"].isin([50, 100])].copy()
    return keep[["family", "model", "horizon", "mean_nrmse"]].rename(columns={"mean_nrmse": f"{label}_test_nrmse"})


def _val_h100(root: Path, label: str) -> pd.DataFrame:
    df = _read(root, "validation_nrmse_by_model_horizon.csv")
    keep = df[df["horizon"] == 100].copy()
    return keep[["family", "model", "mean_nrmse"]].rename(columns={"mean_nrmse": f"{label}_val_h100_nrmse"})


def _density(root: Path, label: str) -> pd.DataFrame:
    df = _read(root, "validation_latent_density_by_model.csv")
    cols = ["family", "model", "latent_active_density_abs_1e-3", "latent_active_coords_abs_1e-3", "latent_mean_abs"]
    out = df[cols].copy()
    return out.rename(
        columns={
            "latent_active_density_abs_1e-3": f"{label}_val_density_1e3",
            "latent_active_coords_abs_1e-3": f"{label}_val_active_coords_1e3",
            "latent_mean_abs": f"{label}_val_latent_mean_abs",
        }
    )


def _training(root: Path, label: str) -> pd.DataFrame:
    df = _read(root, "model_config_summary.csv")
    cols = ["family", "model", "training_steps", "training_time_seconds", "parameter_count"]
    out = df[cols].copy()
    return out.rename(
        columns={
            "training_steps": f"{label}_training_steps",
            "training_time_seconds": f"{label}_training_time_seconds",
            "parameter_count": f"{label}_parameter_count",
        }
    )


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--short-root", required=True)
    parser.add_argument("--long-root", required=True)
    parser.add_argument("--short-label", default="updates300")
    parser.add_argument("--long-label", default="updates1000")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--noncollapsed-density", type=float, default=0.05)
    args = parser.parse_args()

    short_root = Path(args.short_root)
    long_root = Path(args.long_root)
    out_root = Path(args.out_root) if args.out_root else long_root
    results_dir = out_root / "results"
    reports_dir = out_root / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    short = args.short_label
    long = args.long_label
    merged = _horizon_table(short_root, short).merge(_horizon_table(long_root, long), on=["family", "model", "horizon"], how="outer")
    merged = merged.merge(_val_h100(short_root, short), on=["family", "model"], how="left")
    merged = merged.merge(_val_h100(long_root, long), on=["family", "model"], how="left")
    merged = merged.merge(_density(short_root, short), on=["family", "model"], how="left")
    merged = merged.merge(_density(long_root, long), on=["family", "model"], how="left")
    merged = merged.merge(_training(short_root, short), on=["family", "model"], how="left")
    merged = merged.merge(_training(long_root, long), on=["family", "model"], how="left")
    merged["test_nrmse_delta_long_minus_short"] = merged[f"{long}_test_nrmse"] - merged[f"{short}_test_nrmse"]
    merged["training_time_ratio_long_over_short"] = merged[f"{long}_training_time_seconds"] / merged[f"{short}_training_time_seconds"]
    merged.to_csv(results_dir / "training_budget_comparison.csv", index=False)

    h100 = merged[merged["horizon"] == 100].copy()
    h100_noncollapsed = h100[h100[f"{long}_val_density_1e3"].fillna(1.0) >= args.noncollapsed_density].copy()
    best_noncollapsed = (
        h100_noncollapsed[h100_noncollapsed["family"].isin(["dense_mlp_tanh", "sparse_mlp_relu", "lista_relu", "lista_shrink"])]
        .sort_values(["family", f"{long}_test_nrmse"])
        .groupby("family", as_index=False)
        .first()
    )
    best_noncollapsed.to_csv(results_dir / "best_noncollapsed_h100_by_family.csv", index=False)

    key_cols = [
        "family",
        "model",
        "horizon",
        f"{short}_test_nrmse",
        f"{long}_test_nrmse",
        "test_nrmse_delta_long_minus_short",
        f"{short}_training_steps",
        f"{long}_training_steps",
        f"{short}_training_time_seconds",
        f"{long}_training_time_seconds",
        "training_time_ratio_long_over_short",
        f"{long}_val_density_1e3",
    ]
    report = [
        "# Lorenz-96 Training-Budget Comparison",
        "",
        f"Short run: `{short_root}`.",
        f"Long run: `{long_root}`.",
        "",
        "Negative deltas mean the longer run improved test NRMSE. The noncollapsed selection requires validation active latent density "
        f"at `|z| > 1e-3` of at least {args.noncollapsed_density:g}, preventing all-zero LISTA rows from being selected as useful sparse models.",
        "",
        "## H50/H100 Comparison",
        "",
        markdown_table(merged[key_cols].sort_values(["family", "model", "horizon"])),
        "",
        "## Best Noncollapsed H100 Row By Family After Longer Training",
        "",
        markdown_table(
            best_noncollapsed[
                [
                    "family",
                    "model",
                    f"{long}_test_nrmse",
                    f"{short}_test_nrmse",
                    "test_nrmse_delta_long_minus_short",
                    f"{long}_val_density_1e3",
                    f"{long}_training_steps",
                    f"{long}_training_time_seconds",
                ]
            ]
        ),
        "",
    ]
    (reports_dir / "training_budget_comparison.md").write_text("\n".join(report))
    print(markdown_table(best_noncollapsed[["family", "model", f"{long}_test_nrmse", "test_nrmse_delta_long_minus_short", f"{long}_val_density_1e3"]]))


if __name__ == "__main__":
    main()
