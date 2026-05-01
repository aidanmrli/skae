"""Summarize partition-control self-routed forecasting results.

This script uses the same robust Table 2 conventions as
``tools/per_system_paired_tests.py``: finite routed/global ratios are
summarized by per-system IQM and then cross-system IQM, while the bracketed
K/N counts come from censored within-system seed-paired Wilcoxon/Holm tests.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tools.per_system_paired_tests import (
    ROUTING_EXPECTED_SEEDS,
    cell_summary,
    per_system_censored_routing_wilcoxon,
    routing_censor_cap,
    system_iqm_summary,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROWS_CSV = (
    REPO_ROOT
    / "results"
    / "transition_rich_table2_controls_20260430"
    / "self_routed_controls"
    / "self_routed_forecasting_rows.csv"
)
DEFAULT_TABLE_DIR = REPO_ROOT / "docs" / "figures" / "neurips_paper_2026" / "_tables"

MODEL_DISPLAY = {
    "lista_dense_softblock_signsplit_p64_hardinit_basin_partition": "LISTA-SB (p64)",
    "lista_dense_softblock_signsplit_p256_hardinit_basin_partition": "LISTA-SB",
    "lista_blockdiag_signsplit_hardinit_basin_partition": "LISTA-BD",
    "mlp_sparse_blockdiag_hardinit_basin_partition_control": "Sparse MLP, BD",
    "mlp_sparse_hardinit_basin_partition_control": "Sparse MLP",
    "mlp_zero_sparse_hardinit_basin_partition_control": "Dense MLP",
}

SELECTOR_DISPLAY = {
    "family_local_centered": "Support family",
    "oracle_basin_local_centered": "Basin labels",
    "latent_kmeans_local_centered": "Latent clusters",
    "random_count_matched_local_centered": "Random matched",
}

HORIZONS = [100, 1000]
SLICES = {"all": "all", "deep": "deep"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", type=Path, default=DEFAULT_ROWS_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument(
        "--support_definition",
        default="topk:8",
        help="Support definition to summarize.",
    )
    parser.add_argument(
        "--csv_name",
        default="table2_partition_controls_h100_h1000_censored_seed15_summary.csv",
    )
    parser.add_argument(
        "--json_name",
        default="table2_partition_controls_h100_h1000_censored_seed15_summary.json",
    )
    parser.add_argument(
        "--tex_name",
        default="table2_partition_controls_compact.tex",
    )
    return parser.parse_args()


def _finite_ratio_subframe(sub: pd.DataFrame, ratio_col: str) -> tuple[pd.Series, pd.DataFrame]:
    finite_ratios = pd.to_numeric(sub[ratio_col], errors="coerce")
    finite_ratio_mask = np.isfinite(finite_ratios) & (finite_ratios > 0.0)
    finite_sub = sub.loc[finite_ratio_mask].copy()
    finite_sub["_finite_ratio"] = finite_ratios.loc[finite_ratio_mask].to_numpy()
    return finite_ratios.loc[finite_ratio_mask], finite_sub


def _format_latex_number(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    value = float(value)
    abs_value = abs(value)
    if abs_value == 0.0:
        return "0"
    if 1e-3 <= abs_value < 1e3:
        return f"{value:.3g}"
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"{float(mantissa):.2g}{{\times}}10^{{{int(exponent)}}}"


def _format_cell(value: float, k: int, n: int) -> str:
    return rf"${_format_latex_number(value)}\,[{k}/{n}]$"


def _write_compact_tex(summary: pd.DataFrame, path: Path) -> None:
    models = [display for display in MODEL_DISPLAY.values() if display in set(summary["model"])]
    selectors = [
        display for display in SELECTOR_DISPLAY.values() if display in set(summary["selector"])
    ]
    lookup = {
        (row.model, row.selector, row.subset, int(row.horizon)): row
        for row in summary.itertuples(index=False)
    }

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        (
            r"\caption{Partition-control local forecasting for the top-$8$ support-family comparison. "
            r"Entries are routed/global MSE-ratio IQMs with within-system Wilcoxon/Holm "
            r"counts in brackets; lower is better. The LISTA-SB row in this auxiliary "
            r"control table uses the p64 diagnostic artifact, while the main routing "
            r"table uses the matched $d_z{=}256$ LISTA-SB row.}"
        ),
        r"\label{tab:self_routing_partition_controls}",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{@{}ll cccc@{}}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{$H100$} & \multicolumn{2}{c}{$H1000$} \\",
        r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
        r"Model & Selector & All & Deep & All & Deep \\",
        r"\midrule",
    ]

    for model_idx, model in enumerate(models):
        if model_idx:
            lines.append(r"\addlinespace")
        for selector_idx, selector in enumerate(selectors):
            row_label = model if selector_idx == 0 else ""
            cells = []
            for horizon in HORIZONS:
                for subset in ("all", "deep"):
                    row = lookup.get((model, selector, subset, horizon))
                    if row is None:
                        cells.append("--")
                    else:
                        cells.append(_format_cell(row.finite_ratio_iqm, int(row.K), int(row.N)))
            lines.append(
                f"{row_label} & {selector} & "
                + " & ".join(cells)
                + r" \\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
        "",
    ])
    path.write_text("\n".join(lines))


def summarize(rows_csv: Path, output_dir: Path, support_definition: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(rows_csv, low_memory=False)
    df = df[df["support_definition"] == support_definition].copy()
    df = df[df["rollout_mode"].isin(["global_k", *SELECTOR_DISPLAY.keys()])].copy()
    df = df[df["root_label"].isin(MODEL_DISPLAY)].copy()
    expected_systems = sorted(df["system_key"].dropna().unique().tolist())

    global_cols = ["root_label", "system_key", "seed", "depth_stratum", "support_definition"]
    global_cols.extend(f"h{horizon}_mean" for horizon in HORIZONS)
    global_df = df[df["rollout_mode"] == "global_k"][global_cols].rename(
        columns={f"h{horizon}_mean": f"global_h{horizon}_mean" for horizon in HORIZONS}
    )
    routed = df[df["rollout_mode"].isin(SELECTOR_DISPLAY)].merge(
        global_df,
        on=["root_label", "system_key", "seed", "depth_stratum", "support_definition"],
        how="left",
    )

    full = {}
    rows = []
    for horizon in HORIZONS:
        cap = routing_censor_cap(routed, horizon)
        full[f"H{horizon}"] = {}
        ratio_col = f"h{horizon}_over_global"
        for root_label, model in MODEL_DISPLAY.items():
            model_df = routed[routed["root_label"] == root_label]
            if model_df.empty:
                continue
            full[f"H{horizon}"][model] = {}
            for subset, depth_value in SLICES.items():
                full[f"H{horizon}"][model][subset] = {}
                for mode, selector in SELECTOR_DISPLAY.items():
                    sub = model_df[
                        (model_df["depth_stratum"] == depth_value)
                        & (model_df["rollout_mode"] == mode)
                    ]
                    res = per_system_censored_routing_wilcoxon(
                        sub,
                        horizon=horizon,
                        expected_systems=expected_systems,
                        expected_seeds=ROUTING_EXPECTED_SEEDS,
                        cap=cap,
                    )
                    finite_ratios, finite_sub = _finite_ratio_subframe(sub, ratio_col)
                    res["cell"] = cell_summary(finite_ratios)
                    res["system_cell"] = system_iqm_summary(
                        finite_sub,
                        "_finite_ratio",
                        positive_only=True,
                    )
                    full[f"H{horizon}"][model][subset][selector] = res
                    rows.append({
                        "horizon": int(horizon),
                        "model": model,
                        "root_label": root_label,
                        "subset": subset,
                        "depth_stratum": depth_value,
                        "selector": selector,
                        "rollout_mode": mode,
                        "finite_ratio_iqm": res["system_cell"]["iqm"],
                        "finite_ratio_global_iqm": res["cell"]["iqm"],
                        "finite_ratio_n": res["cell"]["n"],
                        "finite_ratio_n_systems": res["system_cell"]["n_systems"],
                        "censored_log10_iqm_ratio": res["censored_log10_cell"]["ratio_from_iqm_delta"],
                        "K": res["K"],
                        "N": res["N"],
                        "sign_test_iqm_in_direction": res["sign_test_iqm"]["n_in_direction"],
                        "sign_test_iqm_total": res["sign_test_iqm"]["n_total"],
                        "sign_test_iqm_p": res["sign_test_iqm"]["p_value"],
                        **{
                            f"censor_{key}": value
                            for key, value in res["censor_class_counts"].items()
                        },
                    })

    return pd.DataFrame(rows), full


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, full = summarize(args.rows_csv, args.output_dir, args.support_definition)
    csv_path = args.output_dir / args.csv_name
    json_path = args.output_dir / args.json_name
    tex_path = args.output_dir / args.tex_name

    summary.to_csv(csv_path, index=False)
    with json_path.open("w") as f:
        json.dump(full, f, indent=2)
    _write_compact_tex(summary, tex_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
