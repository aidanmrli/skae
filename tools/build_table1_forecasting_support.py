"""Build the compact paper-facing forecasting/support tables.

This builder keeps the existing Table 1 roster but replaces the stale
controlled Sparse MLP-BD packet with the repaired block-diagonal GenericKM
artifacts under transition_rich_sparse_mlp_bd_repaired_table1_20260506.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path("/home/mila/l/lia/skae")
TABLE_DIR = ROOT / "docs" / "figures" / "neurips_paper_2026" / "_tables"

REPAIRED_CONTROL_ROOT = "mlp_sparse_blockdiag_hardinit_basin_partition_control"
BASELINE_CONTROL_ROOT = "mlp_zero_sparse_hardinit_basin_partition_control"
BASELINE_DYSTS_ROOT = "dense_mlp_tanh"

CONTROL_FORECAST_CSVS = [
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
REPAIRED_CONTROL_FORECAST_CSV = (
    ROOT
    / "results"
    / "transition_rich_sparse_mlp_bd_repaired_table1_20260506"
    / "collect_pass0"
    / "forecasting_rows.csv"
)

CONTROL_INTERP_CSVS = [
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
REPAIRED_CONTROL_INTERP_CSV = (
    ROOT
    / "results"
    / "transition_rich_sparse_mlp_bd_repaired_table1_20260506"
    / "interpretability_per_basin_deep_pass0"
    / "interpretability_rows.csv"
)

DYSTS_SUMMARY_CSV = TABLE_DIR / "dysts_dt30_iqm_summary.csv"
DYSTS_TESTS_CSV = TABLE_DIR / "dysts_dt30_aggregate_tests_vs_dense.csv"

CONTROL_HORIZONS = (100, 500, 1000)
DYSTS_HORIZONS = (100, 2000, 4000)
ALPHA = 0.05
RETAINED_MEAN_BASIN_COUNT = 4.20

EXCLUDED_SYSTEMS = {
    "multiwell_strong_transition",
    "claude_checkerboard_potential",
    "claude:checkerboard_potential",
}

CONTROL_ROOTS = {
    "lista_dense_signsplit_p256_hardinit_basin_partition": {
        "label": "LISTA",
        "dysts_root": "lista",
    },
    "lista_blockdiag_signsplit_hardinit_basin_partition": {
        "label": "LISTA-BD",
        "dysts_root": "lista_bd",
    },
    "lista_dense_softblock_signsplit_p256_hardinit_basin_partition": {
        "label": "LISTA-SB",
        "dysts_root": "lista_sb",
    },
    REPAIRED_CONTROL_ROOT: {
        "label": "Sparse MLP, BD",
        "dysts_root": "sparse_mlp_bd",
    },
    "mlp_sparse_hardinit_basin_partition_control": {
        "label": r"Sparse MLP \citep{fathi2024course}",
        "plain_label": "Sparse MLP",
        "dysts_root": "sparse_mlp",
    },
    BASELINE_CONTROL_ROOT: {
        "label": r"Dense MLP \citep{lusch_deep_2018} \emph{[baseline]}",
        "plain_label": "Dense MLP",
        "dysts_root": BASELINE_DYSTS_ROOT,
    },
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


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


def finite_mean(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def holm(p_values: list[float] | np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx])]
    running = 0.0
    m = order.size
    for rank, original_idx in enumerate(order):
        value = min(1.0, (m - rank) * p[original_idx])
        running = max(running, value)
        out[original_idx] = running
    return out


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


def fmt_family_count(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.1f}"


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
    if star:
        cell = rf"${{{body}}}^{{\ast}}$"
    else:
        cell = rf"${body}$"
    if baseline and baseline_suffix:
        cell += r"\,\emph{[baseline]}"
    return cell


def load_control_forecasting() -> pd.DataFrame:
    frames = []
    for path in CONTROL_FORECAST_CSVS:
        require_file(path)
        frame = pd.read_csv(path, low_memory=False)
        frame = frame[frame["root_label"] != REPAIRED_CONTROL_ROOT].copy()
        frames.append(frame)
    require_file(REPAIRED_CONTROL_FORECAST_CSV)
    frames.append(pd.read_csv(REPAIRED_CONTROL_FORECAST_CSV, low_memory=False))
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df[df["root_label"].isin(CONTROL_ROOTS)].copy()
    for col in ("system_name", "system_key", "train_env_name"):
        if col in df:
            df = df[~df[col].isin(EXCLUDED_SYSTEMS)].copy()
    return df


def load_control_interpretability() -> pd.DataFrame:
    frames = []
    for path in CONTROL_INTERP_CSVS:
        require_file(path)
        frame = pd.read_csv(path, low_memory=False)
        frame = frame[frame["root_label"] != REPAIRED_CONTROL_ROOT].copy()
        frames.append(frame)
    require_file(REPAIRED_CONTROL_INTERP_CSV)
    frames.append(pd.read_csv(REPAIRED_CONTROL_INTERP_CSV, low_memory=False))
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df[df["root_label"].isin(CONTROL_ROOTS)].copy()
    for col in ("system_name", "system_key", "train_env_name"):
        if col in df:
            df = df[~df[col].isin(EXCLUDED_SYSTEMS)].copy()
    return df[
        (df["support_scheme"] == "absolute:0.001")
        & (df["subset"] == "deep")
        & (pd.to_numeric(df["family_jaccard_threshold"], errors="coerce") == 0.5)
    ].copy()


def paired_system_counts(
    df: pd.DataFrame,
    *,
    root_label: str,
    metric_col: str,
    alternative: str,
    log_values: bool = False,
) -> tuple[int, int]:
    base = df[df["root_label"] == BASELINE_CONTROL_ROOT].pivot_table(
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
            p_vals.append(
                float(stats.wilcoxon(c, b, alternative=alternative, zero_method="wilcox").pvalue)
            )
        except ValueError:
            p_vals.append(float("nan"))
    p_holm = holm(p_vals)
    valid = np.isfinite(p_holm)
    return int(np.sum(p_holm[valid] < ALPHA)), int(np.sum(valid))


def build_control_summary(fc: pd.DataFrame, itp: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for root_label, meta in CONTROL_ROOTS.items():
        row: dict[str, float | str] = {
            "root": root_label,
            "label": meta.get("plain_label", meta["label"]),
            "latex_label": meta["label"],
            "dysts_root": meta["dysts_root"],
        }
        sub_fc = fc[fc["root_label"] == root_label]
        for h in CONTROL_HORIZONS:
            col = f"h{h}_best_periodic_mean"
            row[f"H{h}"] = finite_mean(sub_fc.groupby("system_name")[col].apply(iqm))
        sub_itp = itp[itp["root_label"] == root_label]
        metric_map = {
            "HBgivenF": "family_h_basin_given_family",
            "FreezeWrongH1": "support_freeze_wrong_over_base_h1",
            "FreezeWrongH20": "support_freeze_wrong_over_base_h20",
        }
        for key, col in metric_map.items():
            row[key] = finite_mean(sub_itp.groupby("system_name")[col].apply(iqm))
        row["FamilyUniqueCount"] = finite_mean(
            sub_itp.groupby("system_name")["family_unique_count"].mean()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_control_significance(fc: pd.DataFrame, itp: pd.DataFrame) -> dict[tuple[str, str], tuple[int, int]]:
    sig: dict[tuple[str, str], tuple[int, int]] = {}
    for root_label in CONTROL_ROOTS:
        if root_label == BASELINE_CONTROL_ROOT:
            continue
        for h in CONTROL_HORIZONS:
            sig[(root_label, f"H{h}")] = paired_system_counts(
                fc,
                root_label=root_label,
                metric_col=f"h{h}_best_periodic_mean",
                alternative="less",
                log_values=True,
            )
        sig[(root_label, "HBgivenF")] = paired_system_counts(
            itp,
            root_label=root_label,
            metric_col="family_h_basin_given_family",
            alternative="less",
        )
        sig[(root_label, "FreezeWrongH1")] = paired_system_counts(
            itp,
            root_label=root_label,
            metric_col="support_freeze_wrong_over_base_h1",
            alternative="greater",
        )
        sig[(root_label, "FreezeWrongH20")] = paired_system_counts(
            itp,
            root_label=root_label,
            metric_col="support_freeze_wrong_over_base_h20",
            alternative="greater",
        )
    return sig


def is_full_star(sig: dict[tuple[str, str], tuple[int, int]], root: str, key: str) -> bool:
    count = sig.get((root, key))
    return bool(count and count[1] > 0 and count[0] == count[1])


def load_dysts_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    require_file(DYSTS_SUMMARY_CSV)
    require_file(DYSTS_TESTS_CSV)
    summary = pd.read_csv(DYSTS_SUMMARY_CSV)
    tests = pd.read_csv(DYSTS_TESTS_CSV)
    return summary, tests


def dysts_value(summary: pd.DataFrame, root: str, horizon: int) -> float:
    row = summary[(summary["root_label"] == root) & (summary["horizon"] == horizon)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["cross_system_mean"])


def dysts_star(tests: pd.DataFrame, root: str, horizon: int) -> bool:
    if root == BASELINE_DYSTS_ROOT:
        return False
    row = tests[(tests["root_label"] == root) & (tests["horizon"] == horizon)]
    if row.empty:
        return False
    return float(row.iloc[0]["p_system_sign_holm_all"]) < ALPHA


def best_indices(summary: pd.DataFrame, columns: list[str], *, max_columns: set[str] | None = None) -> dict[str, set[int]]:
    max_columns = max_columns or set()
    nonbase = summary[summary["root"] != BASELINE_CONTROL_ROOT]
    out: dict[str, set[int]] = {}
    for col in columns:
        values = nonbase[col].astype(float)
        target = values.max() if col in max_columns else values.min()
        out[col] = set(nonbase.index[np.isclose(values, target, rtol=1e-12, atol=1e-15)])
    return out


def write_tables(control: pd.DataFrame, sig: dict[tuple[str, str], tuple[int, int]]) -> None:
    dysts_summary, dysts_tests = load_dysts_tables()

    # Best markers.
    best = best_indices(
        control,
        [*(f"H{h}" for h in CONTROL_HORIZONS), "HBgivenF", "FreezeWrongH1", "FreezeWrongH20"],
        max_columns={"FreezeWrongH1", "FreezeWrongH20"},
    )
    nonbase = control[control["root"] != BASELINE_CONTROL_ROOT].copy()
    family_distance = (nonbase["FamilyUniqueCount"].astype(float) - RETAINED_MEAN_BASIN_COUNT).abs()
    best_family = set(nonbase.index[np.isclose(family_distance, family_distance.min())])

    dysts_best: dict[int, set[str]] = {}
    for horizon in DYSTS_HORIZONS:
        values = {
            str(row["dysts_root"]): dysts_value(dysts_summary, str(row["dysts_root"]), horizon)
            for _, row in nonbase.iterrows()
        }
        min_value = min(values.values())
        dysts_best[horizon] = {root for root, value in values.items() if np.isclose(value, min_value)}

    def control_fc_cell(row: pd.Series, horizon: int) -> str:
        root = str(row["root"])
        return math_cell(
            float(row[f"H{horizon}"]),
            bold=row.name in best[f"H{horizon}"] and root != BASELINE_CONTROL_ROOT,
            star=is_full_star(sig, root, f"H{horizon}"),
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

    table1_lines = [
        r"\begin{tabular}[t]{@{}l rrr rrr@{}}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Multibasin, 15 systems} & \multicolumn{3}{c}{Dysts \(dt{\times}30\), 10 systems} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r"Model & H100 & H500 & H1000 & H100 & H2000 & H4000 \\",
        r"\midrule",
    ]
    table1_support_lines = [
        r"\begin{tabular}{@{}l rrr rrr@{\hspace{0.9em}}rr@{}}",
        r"\toprule",
        r"& \multicolumn{6}{c}{Forecasting MSE\(\downarrow\)} & \multicolumn{2}{c}{Support diagnostics on multibasin systems} \\",
        r"\cmidrule(lr){2-7}\cmidrule(l){8-9}",
        r"& \multicolumn{3}{c}{Multibasin, 15 systems} & \multicolumn{3}{c}{Dysts \(dt{\times}30\), 10 systems} & \multicolumn{2}{c}{Per-basin deep slice} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-9}",
        r"Model & H100 & H500 & H1000 & H100 & H2000 & H4000 & $H(B\!\mid\!F_{\rm abs})\downarrow$ & $\overline{|F_{\rm abs}|}$ \\",
        r"\midrule",
    ]
    support_lines = [
        r"\begin{tabular}{@{}l rrrr@{}}",
        r"\toprule",
        r"Model & $H(B\!\mid\!F_{\rm abs})\downarrow$ & $\overline{|F_{\rm abs}|}$ & wrong supp.\ H1\(\uparrow\) & wrong supp.\ H20\(\uparrow\) \\",
        r"\midrule",
    ]
    support_short_lines = [
        r"\begin{tabular}[t]{@{}l rr@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Per-basin deep slice} \\",
        r"\cmidrule(l){2-3}",
        r"Model & $H(B\!\mid\!F_{\rm abs})\downarrow$ & $\overline{|F_{\rm abs}|}$ \\",
        r"\midrule",
    ]
    fixed_alignment_lines = [
        r"\begin{tabular}{@{}l rr rrr r@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Forecasting (all held-out)} & \multicolumn{4}{c}{Support diagnostics (per-basin deep slice)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(l){4-7}",
        r"Model & H100\,$\downarrow$ & H1000\,$\downarrow$ & $H(B\!\mid\!F_{\rm abs})$\,$\downarrow$ & $\overline{|F_{\rm abs}|}$ & wr.\,$h{=}1$\,$\uparrow$ & wr.\,$h{=}20$\,$\uparrow$ \\",
        r"\midrule",
    ]

    for _, row in control.iterrows():
        root = str(row["root"])
        is_baseline = root == BASELINE_CONTROL_ROOT
        label = str(row["latex_label"])
        if is_baseline:
            table1_lines.append(r"\midrule")
            table1_support_lines.append(r"\midrule")
            support_short_lines.append(r"\midrule")

        control_cells = [control_fc_cell(row, h) for h in CONTROL_HORIZONS]
        dysts_cells = [dysts_cell(row, h) for h in DYSTS_HORIZONS]
        table1_lines.append(f"{label} & {' & '.join(control_cells + dysts_cells)} \\\\")

        hbf_cell = math_cell(
            float(row["HBgivenF"]),
            bold=row.name in best["HBgivenF"] and not is_baseline,
            star=is_full_star(sig, root, "HBgivenF"),
            baseline=is_baseline,
            baseline_suffix=False,
        )
        family_body = fmt_family_count(float(row["FamilyUniqueCount"]))
        if row.name in best_family and not is_baseline:
            family_body = rf"\mathbf{{{family_body}}}"
        family_cell = rf"${family_body}$"
        fw1_cell = math_cell(
            float(row["FreezeWrongH1"]),
            bold=row.name in best["FreezeWrongH1"] and not is_baseline,
            star=is_full_star(sig, root, "FreezeWrongH1"),
            baseline=is_baseline,
        )
        fw20_cell = math_cell(
            float(row["FreezeWrongH20"]),
            bold=row.name in best["FreezeWrongH20"] and not is_baseline,
            star=is_full_star(sig, root, "FreezeWrongH20"),
            baseline=is_baseline,
        )
        if is_baseline:
            hbf_cell += r"\,\emph{[baseline]}"
            family_cell += r"\,\emph{[baseline]}"
        table1_support_lines.append(
            f"{label} & {' & '.join(control_cells + dysts_cells)} & {hbf_cell} & {family_cell} \\\\"
        )
        support_label = str(row["label"])
        support_lines.append(
            f"{support_label} & {hbf_cell} & {family_cell} & {fw1_cell} & {fw20_cell} \\\\"
        )
        support_short_lines.append(f"{support_label} & {hbf_cell} & {family_cell} \\\\")
        fixed_alignment_lines.append(
            f"{support_label} & {control_fc_cell(row, 100)} & {control_fc_cell(row, 1000)} & "
            f"{hbf_cell} & {family_cell} & {fw1_cell} & {fw20_cell} \\\\"
        )

    for lines in (table1_lines, table1_support_lines, support_lines, support_short_lines, fixed_alignment_lines):
        lines.extend([r"\bottomrule", r"\end{tabular}"])

    (TABLE_DIR / "table1_forecasting_multibasin_dysts.tex").write_text(
        "\n".join(table1_lines) + "\n", encoding="utf-8"
    )
    (TABLE_DIR / "table1_forecasting_support_diagnostics.tex").write_text(
        "\n".join(table1_support_lines) + "\n", encoding="utf-8"
    )
    (TABLE_DIR / "table2_support_diagnostics_per_basin_deep.tex").write_text(
        "\n".join(support_lines) + "\n", encoding="utf-8"
    )
    (TABLE_DIR / "table2_support_diagnostics_per_basin_deep_no_wrong_support.tex").write_text(
        "\n".join(support_short_lines) + "\n", encoding="utf-8"
    )
    (TABLE_DIR / "table1_fixed17_alignment.tex").write_text(
        "\n".join(fixed_alignment_lines) + "\n", encoding="utf-8"
    )
    (TABLE_DIR / "table1_fixed17_alignment_per_basin_deep.tex").write_text(
        "\n".join(fixed_alignment_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    fc = load_control_forecasting()
    itp = load_control_interpretability()
    control = build_control_summary(fc, itp)
    sig = build_control_significance(fc, itp)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    control.to_csv(TABLE_DIR / "table1_fixed17_alignment_per_basin_deep.csv", index=False)
    write_tables(control, sig)

    metadata = {
        "control_forecasting_csvs": [str(path) for path in CONTROL_FORECAST_CSVS],
        "control_forecasting_repaired_override": str(REPAIRED_CONTROL_FORECAST_CSV),
        "control_interpretability_csvs": [str(path) for path in CONTROL_INTERP_CSVS],
        "control_interpretability_repaired_override": str(REPAIRED_CONTROL_INTERP_CSV),
        "repaired_control_root": REPAIRED_CONTROL_ROOT,
        "dysts_summary_csv": str(DYSTS_SUMMARY_CSV),
        "dysts_tests_csv": str(DYSTS_TESTS_CSV),
        "excluded_systems": sorted(EXCLUDED_SYSTEMS),
        "retained_control_system_count": int(fc["system_name"].nunique()),
        "retained_dysts_system_count": 10,
        "support_slice": "per_basin_deep",
        "support_scheme": "absolute:0.001",
        "family_jaccard_threshold": 0.5,
        "control_significance_full_star_rule": "within-system Wilcoxon/Holm; compact star iff all eligible systems pass",
        "dysts_significance_rule": "exact system-sign/Holm from dysts_dt30_aggregate_tests_vs_dense.csv",
    }
    (TABLE_DIR / "table1_fixed17_alignment_per_basin_deep_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    repaired = control[control["root"] == REPAIRED_CONTROL_ROOT].iloc[0]
    print("Rebuilt compact Table 1 and support-diagnostic tables.")
    print(
        "Repaired Sparse MLP-BD controlled values: "
        f"H100={repaired['H100']:.6g}, "
        f"H500={repaired['H500']:.6g}, "
        f"H1000={repaired['H1000']:.6g}, "
        f"H(B|F)={repaired['HBgivenF']:.6g}, "
        f"|F|={repaired['FamilyUniqueCount']:.6g}, "
        f"wrong_h1={repaired['FreezeWrongH1']:.6g}, "
        f"wrong_h20={repaired['FreezeWrongH20']:.6g}"
    )
    print(f"Rows in control summary: {len(control)}")


if __name__ == "__main__":
    main()
