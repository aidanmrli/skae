"""Rendering helpers for the paper's compact forecasting/support tables."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE_CONTROL_ROOT = "mlp_zero_sparse_hardinit_basin_partition_control"
BASELINE_DYSTS_ROOT = "dense_mlp_tanh"
CONTROL_HORIZONS = (100, 500, 1000)
DYSTS_HORIZONS = (100, 2000, 4000)
ALPHA = 0.05
# The primary alignment roster excludes the single-label Triple-well Duffing
# proxy; the remaining manifest counts sum to 60 over 14 systems.
PRIMARY_ALIGNMENT_MEAN_MANIFEST_COUNT = 60.0 / 14.0
STANDALONE_METHODS = (
    "dmd",
    "edmd_poly",
    "rbf_dictionary_edmd",
    "kmeans_hard",
    "gmm_hard",
    "gmm_soft",
)


def fmt_num(value: float, *, sig: int = 3) -> str:
    if not np.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 1000.0 or abs_value < 1e-3:
        exponent = int(math.floor(math.log10(abs_value)))
        mantissa = value / (10**exponent)
        return rf"{mantissa:.{sig - 1}f}{{\times}}10^{{{exponent}}}"
    decimals = max(0, sig - 1 - int(math.floor(math.log10(abs_value))))
    return f"{value:.{decimals}f}"


def math_cell(
    value: float,
    *,
    bold: bool = False,
    star: bool = False,
    baseline: bool = False,
    baseline_suffix: bool = True,
    sig: int = 3,
) -> str:
    body = fmt_num(value, sig=sig)
    if bold:
        body = rf"\mathbf{{{body}}}"
    cell = rf"${{{body}}}^{{\ast}}$" if star else rf"${body}$"
    if baseline and baseline_suffix:
        cell += r"\,\emph{[baseline]}"
    return cell


def is_full_star(
    significance: dict[tuple[str, str], tuple[int, int]], root: str, key: str
) -> bool:
    count = significance.get((root, key))
    return bool(count and count[1] > 0 and count[0] == count[1])


def load_dysts_tables(table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = table_dir / "dysts_dt30_iqm_summary.csv"
    tests_path = table_dir / "dysts_dt30_aggregate_tests_vs_dense.csv"
    for path in (summary_path, tests_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    return pd.read_csv(summary_path), pd.read_csv(tests_path)


def standalone_baseline_rows(table_dir: Path) -> list[str]:
    control_path = table_dir / "paper_baseline_multibasin_summary.csv"
    dysts_path = table_dir / "paper_baseline_dysts_summary.csv"
    for path in (control_path, dysts_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    controlled = pd.read_csv(control_path).set_index(["method", "horizon"])
    dysts = pd.read_csv(dysts_path).set_index(["method", "horizon"])

    def cell(value: float) -> str:
        return rf"${fmt_num(value, sig=2 if abs(value) >= 1000.0 else 3)}$"

    rows = []
    for method in STANDALONE_METHODS:
        display = str(controlled.loc[(method, CONTROL_HORIZONS[0]), "display"])
        values = [
            *(float(controlled.loc[(method, h), "cross_system_seed_iqm_mean"]) for h in CONTROL_HORIZONS),
            *(float(dysts.loc[(method, h), "cross_system_seed_iqm_mean"]) for h in DYSTS_HORIZONS),
        ]
        rows.append(f"{display} & {' & '.join(cell(value) for value in values)}" + r" \\")
    return rows


def dysts_value(summary: pd.DataFrame, root: str, horizon: int) -> float:
    row = summary[(summary["root_label"] == root) & (summary["horizon"] == horizon)]
    return float("nan") if row.empty else float(row.iloc[0]["cross_system_mean"])


def dysts_star(tests: pd.DataFrame, root: str, horizon: int) -> bool:
    if root == BASELINE_DYSTS_ROOT:
        return False
    row = tests[(tests["root_label"] == root) & (tests["horizon"] == horizon)]
    return bool(not row.empty and float(row.iloc[0]["p_system_sign_holm_all"]) < ALPHA)


def best_indices(
    summary: pd.DataFrame,
    columns: list[str],
    *,
    max_columns: set[str] | None = None,
) -> dict[str, set[int]]:
    max_columns = max_columns or set()
    nonbase = summary[summary["root"] != BASELINE_CONTROL_ROOT]
    out: dict[str, set[int]] = {}
    for col in columns:
        values = nonbase[col].astype(float)
        target = values.max() if col in max_columns else values.min()
        out[col] = set(nonbase.index[np.isclose(values, target, rtol=1e-12, atol=1e-15)])
    return out


def write_tables(
    control: pd.DataFrame,
    significance: dict[tuple[str, str], tuple[int, int]],
    *,
    output_dir: Path,
    table_dir: Path,
) -> None:
    dysts_summary, dysts_tests = load_dysts_tables(table_dir)
    best = best_indices(
        control,
        [*(f"H{h}" for h in CONTROL_HORIZONS), "HBgivenF"],
    )
    nonbase = control[control["root"] != BASELINE_CONTROL_ROOT].copy()
    family_distance = (
        nonbase["FamilyUniqueCount"].astype(float)
        - PRIMARY_ALIGNMENT_MEAN_MANIFEST_COUNT
    ).abs()
    best_family = set(nonbase.index[np.isclose(family_distance, family_distance.min())])
    dysts_best = {}
    for horizon in DYSTS_HORIZONS:
        values = {
            str(row["dysts_root"]): dysts_value(dysts_summary, str(row["dysts_root"]), horizon)
            for _, row in nonbase.iterrows()
        }
        minimum = min(values.values())
        dysts_best[horizon] = {root for root, value in values.items() if np.isclose(value, minimum)}

    def control_cell(row: pd.Series, horizon: int) -> str:
        root = str(row["root"])
        return math_cell(
            float(row[f"H{horizon}"]),
            bold=row.name in best[f"H{horizon}"] and root != BASELINE_CONTROL_ROOT,
            star=is_full_star(significance, root, f"H{horizon}"),
            baseline=root == BASELINE_CONTROL_ROOT,
            baseline_suffix=False,
        )

    def dysts_cell(row: pd.Series, horizon: int) -> str:
        root = str(row["dysts_root"])
        return math_cell(
            dysts_value(dysts_summary, root, horizon),
            bold=root in dysts_best[horizon] and root != BASELINE_DYSTS_ROOT,
            star=dysts_star(dysts_tests, root, horizon),
            baseline=root == BASELINE_DYSTS_ROOT,
            baseline_suffix=False,
        )

    table1 = [
        r"\begin{tabular}[t]{@{}l rrr rrr@{}}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Multibasin, 15 systems} & \multicolumn{3}{c}{Dysts \(dt{\times}30\), 10 systems} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r"Model & H100 & H500 & H1000 & H100 & H2000 & H4000 \\",
        r"\midrule",
    ]
    support_short = [
        r"\begin{tabular}[t]{@{}l rr@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Tie-inclusive high-margin slice, 14 systems} \\",
        r"\cmidrule(l){2-3}",
        r"Model & $H(B\!\mid\!F_{\rm abs})\,[\mathrm{nats}]\downarrow$ & $\overline{|F_{\rm abs}^{\rm obs}|}$ \\",
        r"\midrule",
    ]
    for _, row in control.iterrows():
        root = str(row["root"])
        is_baseline = root == BASELINE_CONTROL_ROOT
        if is_baseline:
            table1.append(r"\midrule")
            support_short.append(r"\midrule")
        control_cells = [control_cell(row, h) for h in CONTROL_HORIZONS]
        dysts_cells = [dysts_cell(row, h) for h in DYSTS_HORIZONS]
        latex_label = str(row["latex_label"])
        table1.append(f"{latex_label} & {' & '.join(control_cells + dysts_cells)}" + r" \\")
        hbf = math_cell(
            float(row["HBgivenF"]),
            bold=row.name in best["HBgivenF"] and not is_baseline,
            star=is_full_star(significance, root, "HBgivenF"),
            baseline=is_baseline,
            baseline_suffix=False,
        )
        family_body = f"{float(row['FamilyUniqueCount']):.1f}"
        if row.name in best_family and not is_baseline:
            family_body = rf"\mathbf{{{family_body}}}"
        family = rf"${family_body}$"
        if is_baseline:
            hbf += r"\,\emph{[baseline]}"
            family += r"\,\emph{[baseline]}"
        label = str(row["label"])
        support_short.append(f"{label} & {hbf} & {family}" + r" \\")

    table1.append(r"\midrule")
    table1.extend(standalone_baseline_rows(table_dir))
    tables = {
        "table1_forecasting_multibasin_dysts.tex": table1,
        "table2_support_alignment.tex": support_short,
    }
    for name, lines in tables.items():
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        (output_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
