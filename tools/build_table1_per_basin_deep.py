"""Build the current Table 1 with per-basin deep-slice support diagnostics.

Forecasting columns remain all-held-out rollout metrics from the current Table 1
source packets. Support columns use the current-roster
interpretability_per_basin_deep_current_table1_pass0 outputs.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path("/home/mila/l/lia/skae")
TBL_DIR = ROOT / "docs" / "figures" / "neurips_paper_2026" / "_tables"
TBL_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDED_SYSTEMS = {
    "multiwell_strong_transition",
    "claude_checkerboard_potential",
    "claude:checkerboard_potential",
}

FC_CSVS = [
    ROOT
    / "results"
    / "transition_rich_table2_5model_seed15_backfill_20260428"
    / "collect_pass0"
    / "forecasting_rows.csv",
    ROOT
    / "results"
    / "transition_rich_lista_sb_p256_hardinit_fairness_seed15_20260428"
    / "collect_pass0"
    / "forecasting_rows.csv",
    ROOT
    / "results"
    / "transition_rich_lista_dense_p256_hardinit_table123_20260430"
    / "collect_pass0"
    / "forecasting_rows.csv",
]

PER_BASIN_INTERP_CSVS = [
    ROOT
    / "results"
    / "transition_rich_table2_5model_seed15_backfill_20260428"
    / "interpretability_per_basin_deep_current_table1_pass0"
    / "interpretability_rows.csv",
    ROOT
    / "results"
    / "transition_rich_lista_sb_p256_hardinit_fairness_seed15_20260428"
    / "interpretability_per_basin_deep_current_table1_pass0"
    / "interpretability_rows.csv",
    ROOT
    / "results"
    / "transition_rich_lista_dense_p256_hardinit_table123_20260430"
    / "interpretability_per_basin_deep_current_table1_pass0"
    / "interpretability_rows.csv",
]

ROOTS = {
    "lista_dense_signsplit_p256_hardinit_basin_partition": "LISTA",
    "lista_blockdiag_signsplit_hardinit_basin_partition": "LISTA-BD",
    "lista_dense_softblock_signsplit_p256_hardinit_basin_partition": "LISTA-SB",
    "mlp_sparse_blockdiag_hardinit_basin_partition_control": "Sparse MLP, BD",
    "mlp_sparse_hardinit_basin_partition_control": "Sparse MLP",
    "mlp_zero_sparse_hardinit_basin_partition_control": "Dense MLP",
}
BASELINE = "mlp_zero_sparse_hardinit_basin_partition_control"
HORIZONS = (100, 1000)
ALPHA = 0.05


def iqm(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if arr.size < 4:
        return float(np.mean(arr))
    lo, hi = np.percentile(arr, [25, 75])
    keep = arr[(arr >= lo) & (arr <= hi)]
    if keep.size == 0:
        return float(np.median(arr))
    return float(np.mean(keep))


def mean_finite(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def holm(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx])]
    adjusted = np.empty(order.size, dtype=float)
    running = 0.0
    m = order.size
    for rank, original_idx in enumerate(order):
        val = min(1.0, (m - rank) * p[original_idx])
        running = max(running, val)
        adjusted[rank] = running
    for original_idx, val in zip(order, adjusted):
        out[original_idx] = val
    return out


def paired_system_counts(
    df: pd.DataFrame,
    *,
    metric_col: str,
    root_label: str,
    alternative: str,
    log_values: bool = False,
) -> tuple[int, int]:
    base = df[df["root_label"] == BASELINE].pivot_table(
        index="system_name", columns="seed", values=metric_col, aggfunc="first"
    )
    cand = df[df["root_label"] == root_label].pivot_table(
        index="system_name", columns="seed", values=metric_col, aggfunc="first"
    )
    p_vals: list[float] = []
    for system_name in cand.index.intersection(base.index):
        c = cand.loc[system_name].to_numpy(dtype=float)
        b = base.loc[system_name].to_numpy(dtype=float)
        mask = np.isfinite(c) & np.isfinite(b)
        if log_values:
            mask &= (c > 0.0) & (b > 0.0)
        c = c[mask]
        b = b[mask]
        if c.size < 4:
            p_vals.append(float("nan"))
            continue
        if log_values:
            c = np.log10(c)
            b = np.log10(b)
        try:
            p_vals.append(float(stats.wilcoxon(c, b, alternative=alternative, zero_method="wilcox").pvalue))
        except ValueError:
            p_vals.append(float("nan"))
    p_arr = np.asarray(p_vals, dtype=float)
    p_holm = holm(p_arr)
    valid = np.isfinite(p_holm)
    return int(np.sum(p_holm[valid] < ALPHA)), int(np.sum(valid))


def fmt_num(value: float, sig: int = 3) -> str:
    if not np.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 1000.0 or abs_value < 1e-3:
        exponent = int(math.floor(math.log10(abs_value)))
        mantissa = value / (10**exponent)
        return rf"{mantissa:.{sig - 1}f}{{\times}}10^{{{exponent}}}"
    if abs_value < 0.1:
        return f"{value:.3f}"
    if abs_value < 10:
        return f"{value:.3g}"
    if abs_value < 100:
        return f"{value:.3g}"
    return f"{value:.0f}"


def metric_cell(
    row: dict[str, float | str],
    key: str,
    best_indices: set[int],
    row_idx: int,
    *,
    counts: tuple[int, int] | None,
    is_ratio: bool = False,
    sig: int = 3,
) -> str:
    value = float(row[key])
    if not np.isfinite(value):
        return "--"
    body = fmt_num(value, sig=sig)
    if row_idx in best_indices and row["root"] != BASELINE:
        body = rf"\mathbf{{{body}}}"
    cell = f"${body}$"
    if row["root"] == BASELINE:
        return cell + r"\,\emph{[baseline]}"
    if counts is not None:
        cell += rf"\,[{counts[0]}/{counts[1]}]"
    return cell


fc = pd.concat([pd.read_csv(path, low_memory=False) for path in FC_CSVS], ignore_index=True)
fc = fc[fc["root_label"].isin(ROOTS)].copy()

itp = pd.concat([pd.read_csv(path, low_memory=False) for path in PER_BASIN_INTERP_CSVS], ignore_index=True)
itp = itp[itp["root_label"].isin(ROOTS)].copy()

for column in ("system_name", "system_key", "train_env_name"):
    if column in fc:
        fc = fc[~fc[column].isin(EXCLUDED_SYSTEMS)].copy()
    if column in itp:
        itp = itp[~itp[column].isin(EXCLUDED_SYSTEMS)].copy()

itp_deep = itp[(itp["support_scheme"] == "absolute:0.001") & (itp["subset"] == "deep")].copy()

rows: list[dict[str, float | str]] = []
for root_label, label in ROOTS.items():
    row: dict[str, float | str] = {"root": root_label, "label": label}
    sub_fc = fc[fc["root_label"] == root_label]
    for h in HORIZONS:
        col = f"h{h}_best_periodic_mean"
        per_system = sub_fc.groupby("system_name")[col].apply(iqm)
        row[f"H{h}"] = mean_finite(per_system)
    sub_itp = itp_deep[itp_deep["root_label"] == root_label]
    metric_sources = {
        "HBgivenF": "family_h_basin_given_family",
        "FreezeWrongH1": "support_freeze_wrong_over_base_h1",
        "FreezeWrongH20": "support_freeze_wrong_over_base_h20",
        "FamilyUniqueCount": "family_unique_count",
    }
    for key, col in metric_sources.items():
        if key == "FamilyUniqueCount":
            per_system = sub_itp.groupby("system_name")[col].mean()
        else:
            per_system = sub_itp.groupby("system_name")[col].apply(iqm)
        row[key] = mean_finite(per_system)
    rows.append(row)

summary = pd.DataFrame(rows)
summary.to_csv(TBL_DIR / "table1_fixed17_alignment_per_basin_deep.csv", index=False)

forecast_counts: dict[tuple[str, int], tuple[int, int]] = {}
for root_label in ROOTS:
    if root_label == BASELINE:
        continue
    for h in HORIZONS:
        forecast_counts[(root_label, h)] = paired_system_counts(
            fc,
            metric_col=f"h{h}_best_periodic_mean",
            root_label=root_label,
            alternative="less",
            log_values=True,
        )

support_counts: dict[tuple[str, str], tuple[int, int]] = {}
for root_label in ROOTS:
    if root_label == BASELINE:
        continue
    support_counts[(root_label, "HBgivenF")] = paired_system_counts(
        itp_deep,
        metric_col="family_h_basin_given_family",
        root_label=root_label,
        alternative="less",
    )
    support_counts[(root_label, "FreezeWrongH1")] = paired_system_counts(
        itp_deep,
        metric_col="support_freeze_wrong_over_base_h1",
        root_label=root_label,
        alternative="greater",
    )
    support_counts[(root_label, "FreezeWrongH20")] = paired_system_counts(
        itp_deep,
        metric_col="support_freeze_wrong_over_base_h20",
        root_label=root_label,
        alternative="greater",
    )

best_h100 = {int(summary["H100"].astype(float).idxmin())}
best_h1000 = {int(summary["H1000"].astype(float).idxmin())}
nonbase = summary[summary["root"] != BASELINE]
best_hbf = set(nonbase.index[nonbase["HBgivenF"].astype(float) == nonbase["HBgivenF"].astype(float).min()])
best_fw1 = set(
    nonbase.index[
        nonbase["FreezeWrongH1"].astype(float) == nonbase["FreezeWrongH1"].astype(float).max()
    ]
)
best_fw20 = set(
    nonbase.index[
        nonbase["FreezeWrongH20"].astype(float) == nonbase["FreezeWrongH20"].astype(float).max()
    ]
)

lines = [
    r"\begin{tabular}{@{}l rr rrr r@{}}",
    r"\toprule",
    r"& \multicolumn{2}{c}{Forecasting (all held-out)} & \multicolumn{4}{c}{Support diagnostics (per-basin deep slice)} \\",
    r"\cmidrule(lr){2-3}\cmidrule(l){4-7}",
    r"Model & H100\,$\downarrow$ & H1000\,$\downarrow$ & $H(B\!\mid\!F_{\rm abs})$\,$\downarrow$ & wr.\,$h{=}1$\,$\uparrow$ & wr.\,$h{=}20$\,$\uparrow$ & $\overline{|F_{\rm abs}|}$ \\",
    r"\midrule",
]

for idx, row in summary.iterrows():
    root_label = str(row["root"])
    is_baseline = root_label == BASELINE
    forecast_cells = []
    for h, best_set in ((100, best_h100), (1000, best_h1000)):
        value = float(row[f"H{h}"])
        body = fmt_num(value)
        if idx in best_set and not is_baseline:
            body = rf"\mathbf{{{body}}}"
        cell = f"${body}$"
        if is_baseline:
            cell += r"\,\emph{[baseline]}"
        else:
            count = forecast_counts[(root_label, h)]
            cell += rf"\,[{count[0]}/{count[1]}]"
        forecast_cells.append(cell)

    hbf = metric_cell(
        row,
        "HBgivenF",
        best_hbf,
        idx,
        counts=None if is_baseline else support_counts[(root_label, "HBgivenF")],
        sig=3,
    )
    fw1 = metric_cell(
        row,
        "FreezeWrongH1",
        best_fw1,
        idx,
        counts=None if is_baseline else support_counts[(root_label, "FreezeWrongH1")],
        is_ratio=True,
    )
    fw20 = metric_cell(
        row,
        "FreezeWrongH20",
        best_fw20,
        idx,
        counts=None if is_baseline else support_counts[(root_label, "FreezeWrongH20")],
        is_ratio=True,
    )
    family_count = float(row["FamilyUniqueCount"])
    family_cell = f"${family_count:.1f}$" if np.isfinite(family_count) else "--"
    if is_baseline:
        family_cell += r"\,\emph{[baseline]}"
    lines.append(
        f"{row['label']} & {' & '.join(forecast_cells)} & {hbf} & {fw1} & {fw20} & {family_cell} \\\\"
    )

lines.extend([r"\bottomrule", r"\end{tabular}"])
table_text = "\n".join(lines)
(TBL_DIR / "table1_fixed17_alignment_per_basin_deep.tex").write_text(table_text)
(TBL_DIR / "table1_fixed17_alignment.tex").write_text(table_text)

metadata = {
    "forecasting_csvs": [str(path) for path in FC_CSVS],
    "per_basin_interpretability_csvs": [str(path) for path in PER_BASIN_INTERP_CSVS],
    "excluded_systems": sorted(EXCLUDED_SYSTEMS),
    "retained_system_count": int(fc["system_name"].nunique()),
    "support_slice": "per_basin_deep",
}
(TBL_DIR / "table1_fixed17_alignment_per_basin_deep_metadata.json").write_text(
    pd.Series(metadata).to_json(indent=2)
)

print(table_text)
