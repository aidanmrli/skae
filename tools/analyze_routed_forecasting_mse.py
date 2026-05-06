#!/usr/bin/env python3
"""Aggregate routed forecasting MSE against current paper-table baselines."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MULTIBASIN_SYSTEMS = [
    "claude:arrested_spiral",
    "claude:cal_asymmetric_3",
    "claude:cal_hexagon_6",
    "claude:cal_high_cross_3",
    "claude:cal_octagon_8",
    "claude:cal_pentagon_5",
    "claude:cal_square_4",
    "claude:duffing_triple_well",
    "claude:snic_multi",
    "claude:transition_routes_4",
    "claude:var_depth_gradient_4",
    "claude:var_diamond_4",
    "claude:var_l_shape_5",
    "gated_local_linear",
    "gated_transfer_linear",
]

DYSTS_SYSTEMS = [
    "dysts:Chua",
    "dysts:Dadras",
    "dysts:DequanLi",
    "dysts:Hadley",
    "dysts:LuChenCheng",
    "dysts:QiChen",
    "dysts:Sakarya",
    "dysts:SanUmSrisuchinwong",
    "dysts:ShimizuMorioka",
    "dysts:WangSun",
]

MULTIBASIN_ROOTS = {
    "lista_dense_signsplit_p256_hardinit_basin_partition": "LISTA",
    "lista_blockdiag_signsplit_hardinit_basin_partition": "LISTA-BD",
    "lista_dense_softblock_signsplit_p256_hardinit_basin_partition": "LISTA-SB",
}

DYSTS_ROOTS = {
    "lista": "LISTA",
    "lista_bd": "LISTA-BD",
    "lista_sb": "LISTA-SB",
}

MULTIBASIN_HORIZONS = [100, 500, 1000]
DYSTS_HORIZONS = [100, 500, 1000, 1500, 2000, 3000, 4000, 5000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--multibasin_routed_csv",
        default="results/routed_forecasting_mse_20260505/multibasin/self_routed_forecasting_rows.csv",
    )
    parser.add_argument(
        "--dysts_routed_csv",
        default="results/routed_forecasting_mse_20260505/dysts/self_routed_forecasting_rows.csv",
    )
    parser.add_argument(
        "--multibasin_baseline_csvs",
        default=",".join(
            [
                "results/transition_rich_lista_dense_p256_hardinit_table123_20260430/collect_pass0/forecasting_rows.csv",
                "results/transition_rich_table2_5model_seed15_backfill_20260428/collect_pass0/forecasting_rows.csv",
                "results/transition_rich_lista_sb_p256_hardinit_fairness_seed15_20260428/collect_pass0/forecasting_rows.csv",
            ]
        ),
    )
    parser.add_argument(
        "--dysts_baseline_csv",
        default="results/dysts_dt30_basinblock_p256_seq10_100k_20260430/long_horizon_eval/collect/forecasting_rows.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="results/routed_forecasting_mse_20260505/aggregation",
    )
    parser.add_argument("--support_definition", default="topk:8")
    parser.add_argument("--rollout_mode", default="family_local_centered")
    parser.add_argument("--depth_stratum", default="all")
    parser.add_argument("--datasets", default="multibasin,dysts", help="comma-separated subset from {multibasin,dysts}")
    return parser.parse_args()


def iqm(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if arr.size < 4:
        return float(np.mean(arr))
    lo, hi = np.percentile(arr, [25, 75])
    selected = arr[(arr >= lo) & (arr <= hi)]
    if selected.size == 0:
        return float(np.median(arr))
    return float(np.mean(selected))


def mean_finite(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def finite_count(values: Iterable[float]) -> int:
    arr = np.asarray(list(values), dtype=float)
    return int(np.isfinite(arr).sum())


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def load_multibasin_baseline(raw_csvs: str) -> pd.DataFrame:
    frames = [load_csv(item) for item in raw_csvs.split(",") if item]
    return pd.concat(frames, ignore_index=True)


def prepare_routed_df(
    df: pd.DataFrame,
    *,
    support_definition: str,
    rollout_mode: str,
    depth_stratum: str,
) -> pd.DataFrame:
    sub = df.copy()
    if "reencode_period" not in sub.columns:
        sub["reencode_period"] = 0
    sub["reencode_period"] = pd.to_numeric(sub["reencode_period"], errors="coerce").fillna(0).astype(int)
    if "route_freeze_mode" not in sub.columns:
        sub["route_freeze_mode"] = "none"
    if "local_map_source" not in sub.columns:
        sub["local_map_source"] = "posthoc_ridge"
    sub = sub[sub["support_definition"].astype(str) == support_definition]
    sub = sub[sub["rollout_mode"].astype(str) == rollout_mode]
    sub = sub[sub["depth_stratum"].astype(str) == depth_stratum]
    return sub


def system_seed_iqms(
    df: pd.DataFrame,
    *,
    value_col: str,
    systems: list[str],
    roots: dict[str, str],
    extra_group_cols: list[str] | None = None,
) -> pd.DataFrame:
    sub = df[df["root_label"].isin(roots)].copy()
    sub = sub[sub["system_key"].isin(systems)]
    extra_group_cols = extra_group_cols or []
    rows = []
    group_cols = ["root_label", "system_key", *extra_group_cols]
    for group_key, grp in sub.groupby(group_cols, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_values = dict(zip(group_cols, group_key))
        root = str(group_values["root_label"])
        system = str(group_values["system_key"])
        vals = pd.to_numeric(grp[value_col], errors="coerce")
        row = {
            "root_label": root,
            "display": roots[root],
            "system_key": system,
            "system_iqm": iqm(vals),
            "finite_seed_values": finite_count(vals),
            "total_rows": int(grp.shape[0]),
        }
        for col in extra_group_cols:
            row[col] = group_values[col]
        rows.append(row)
    return pd.DataFrame(rows)


def route_diagnostics(
    routed_df: pd.DataFrame,
    *,
    systems: list[str],
    roots: dict[str, str],
    group_cols: list[str],
) -> pd.DataFrame:
    sub = routed_df[routed_df["root_label"].isin(roots)].copy()
    sub = sub[sub["system_key"].isin(systems)]
    rows = []
    full_group_cols = ["root_label", *group_cols]
    for group_key, grp in sub.groupby(full_group_cols, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_values = dict(zip(full_group_cols, group_key))
        row = {
            "root_label": group_values["root_label"],
            "route_coverage_mean": mean_finite(pd.to_numeric(grp["route_coverage_fraction"], errors="coerce")),
            "fallback_fraction_mean": mean_finite(pd.to_numeric(grp["fallback_fraction"], errors="coerce")),
            "route_switch_rate_mean": mean_finite(pd.to_numeric(grp["route_switch_rate"], errors="coerce")),
            "fit_family_class_count_fit_mean": mean_finite(
                pd.to_numeric(grp["fit_family_class_count_fit"], errors="coerce")
            ),
            "fit_family_class_count_total_mean": mean_finite(
                pd.to_numeric(grp["fit_family_class_count_total"], errors="coerce")
            ),
        }
        for col in group_cols:
            row[col] = group_values[col]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_dataset(
    *,
    dataset: str,
    routed_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    systems: list[str],
    roots: dict[str, str],
    horizons: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    per_system_rows = []
    group_cols = [
        col
        for col in ["reencode_period", "route_freeze_mode", "local_map_source"]
        if col in routed_df.columns
    ]
    diagnostics = route_diagnostics(routed_df, systems=systems, roots=roots, group_cols=group_cols)

    for horizon in horizons:
        routed_col = f"h{horizon}_mean"
        base_cols = {
            "best_periodic": f"h{horizon}_best_periodic_mean",
            "no_reencode": f"h{horizon}_no_reencode_mean",
            "every_step": f"h{horizon}_every_step_mean",
        }
        routed_sys = system_seed_iqms(
            routed_df,
            value_col=routed_col,
            systems=systems,
            roots=roots,
            extra_group_cols=group_cols,
        ).rename(
            columns={
                "system_iqm": "routed_system_iqm",
                "finite_seed_values": "routed_finite_seed_values",
                "total_rows": "routed_total_rows",
            }
        )
        base_sys = {}
        for mode, col in base_cols.items():
            base_sys[mode] = system_seed_iqms(
                baseline_df,
                value_col=col,
                systems=systems,
                roots=roots,
            ).rename(
                columns={
                    "system_iqm": f"{mode}_system_iqm",
                    "finite_seed_values": f"{mode}_finite_seed_values",
                    "total_rows": f"{mode}_total_rows",
                }
            )

        merged = routed_sys
        for mode, frame in base_sys.items():
            keep_cols = [
                "root_label",
                "system_key",
                f"{mode}_system_iqm",
                f"{mode}_finite_seed_values",
                f"{mode}_total_rows",
            ]
            merged = merged.merge(frame[keep_cols], on=["root_label", "system_key"], how="outer")
        merged["dataset"] = dataset
        merged["horizon"] = horizon
        merged["reencode_period"] = pd.to_numeric(merged["reencode_period"], errors="coerce").fillna(0).astype(int)
        for mode in base_cols:
            merged[f"log10_routed_over_{mode}"] = np.log10(
                merged["routed_system_iqm"].astype(float) / merged[f"{mode}_system_iqm"].astype(float)
            )
            merged[f"routed_better_than_{mode}"] = (
                merged["routed_system_iqm"].astype(float) < merged[f"{mode}_system_iqm"].astype(float)
            )
        per_system_rows.append(merged)

        for root, display in roots.items():
            root_merged = merged[merged["root_label"] == root].copy()
            if root_merged.empty:
                continue
            grouped = root_merged.groupby(group_cols, dropna=False, sort=True) if group_cols else [((), root_merged)]
            for group_key, sub in grouped:
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)
                group_values = dict(zip(group_cols, group_key))
                if sub.empty:
                    continue
                diag = diagnostics[diagnostics["root_label"] == root].copy()
                for col, value in group_values.items():
                    diag = diag[diag[col].astype(str) == str(value)]
                diag_row = diag.iloc[0].to_dict() if not diag.empty else {}
                routed_vals = sub["routed_system_iqm"].to_numpy(dtype=float)
                best_vals = sub["best_periodic_system_iqm"].to_numpy(dtype=float)
                no_re_vals = sub["no_reencode_system_iqm"].to_numpy(dtype=float)
                every_vals = sub["every_step_system_iqm"].to_numpy(dtype=float)
                summary = {
                    "dataset": dataset,
                    "root_label": root,
                    "display": display,
                    "reencode_period": int(group_values.get("reencode_period", 0)),
                    "route_freeze_mode": str(group_values.get("route_freeze_mode", "none")),
                    "local_map_source": str(group_values.get("local_map_source", "posthoc_ridge")),
                    "horizon": horizon,
                    "n_systems": int(np.isfinite(routed_vals).sum()),
                    "routed_iqm_mean": mean_finite(routed_vals),
                    "best_periodic_iqm_mean": mean_finite(best_vals),
                    "no_reencode_iqm_mean": mean_finite(no_re_vals),
                    "every_step_iqm_mean": mean_finite(every_vals),
                    "ratio_routed_over_best_periodic": mean_finite(routed_vals) / mean_finite(best_vals),
                    "ratio_routed_over_no_reencode": mean_finite(routed_vals) / mean_finite(no_re_vals),
                    "ratio_routed_over_every_step": mean_finite(routed_vals) / mean_finite(every_vals),
                    "mean_log10_routed_over_best_periodic": mean_finite(sub["log10_routed_over_best_periodic"]),
                    "mean_log10_routed_over_no_reencode": mean_finite(sub["log10_routed_over_no_reencode"]),
                    "mean_log10_routed_over_every_step": mean_finite(sub["log10_routed_over_every_step"]),
                    "systems_routed_better_than_best_periodic": int(sub["routed_better_than_best_periodic"].sum()),
                    "systems_routed_better_than_no_reencode": int(sub["routed_better_than_no_reencode"].sum()),
                    "systems_routed_better_than_every_step": int(sub["routed_better_than_every_step"].sum()),
                    "routed_finite_seed_values": int(sub["routed_finite_seed_values"].sum()),
                    "best_periodic_finite_seed_values": int(sub["best_periodic_finite_seed_values"].sum()),
                    **{
                        k: v
                        for k, v in diag_row.items()
                        if k not in {"root_label", "reencode_period", "route_freeze_mode", "local_map_source"}
                    },
                }
                summaries.append(summary)
    return pd.DataFrame(summaries), pd.concat(per_system_rows, ignore_index=True)


def fmt(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.3g}"
    return f"{value:.4g}"


def write_readout(summary: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Routed Forecasting MSE Aggregation",
        "",
        "Point estimates are seed IQMs within each system, then arithmetic means across systems.",
        "`routed` is `F_top8` `family_local_centered`; `best-periodic` is the current paper-table forecasting baseline.",
        "",
    ]
    for dataset in ["multibasin", "dysts"]:
        lines.extend([f"## {dataset}", ""])
        sub = summary[summary["dataset"] == dataset].copy()
        setting_cols = [
            col
            for col in ["local_map_source", "route_freeze_mode", "reencode_period"]
            if col in sub.columns
        ]
        settings = sub.groupby(setting_cols, dropna=False, sort=True) if setting_cols else [((), sub)]
        for setting_key, setting_sub in settings:
            if not isinstance(setting_key, tuple):
                setting_key = (setting_key,)
            setting = dict(zip(setting_cols, setting_key))
            period = int(setting.get("reencode_period", 0))
            lines.append(f"### Re-encode every {int(period)} steps" if int(period) > 0 else "### No re-encoding")
            source = str(setting.get("local_map_source", "posthoc_ridge"))
            freeze_mode = str(setting.get("route_freeze_mode", "none"))
            if source != "posthoc_ridge" or freeze_mode != "none":
                lines.append("")
                lines.append(f"`local_map_source={source}`, `route_freeze_mode={freeze_mode}`")
            lines.append("")
            period_sub = setting_sub.copy()
            horizons = sorted(period_sub["horizon"].unique())
            for horizon in horizons:
                lines.append(f"#### H{int(horizon)}")
                lines.append("")
                lines.append(
                    "| Model | routed | best-periodic | routed/best | wins vs best | no-reencode | routed/no-re | coverage | fallback |"
                )
                lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
                for _, row in period_sub[period_sub["horizon"] == horizon].sort_values("display").iterrows():
                    wins = f"{int(row['systems_routed_better_than_best_periodic'])}/{int(row['n_systems'])}"
                    lines.append(
                        "| "
                        + str(row["display"])
                        + " | "
                        + " | ".join(
                            [
                                fmt(row["routed_iqm_mean"]),
                                fmt(row["best_periodic_iqm_mean"]),
                                fmt(row["ratio_routed_over_best_periodic"]),
                                wins,
                                fmt(row["no_reencode_iqm_mean"]),
                                fmt(row["ratio_routed_over_no_reencode"]),
                                fmt(row["route_coverage_mean"]),
                                fmt(row["fallback_fraction_mean"]),
                            ]
                        )
                        + " |"
                    )
                lines.append("")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {item.strip() for item in args.datasets.split(",") if item.strip()}

    summary_frames = []
    per_system_frames = []
    if "multibasin" in datasets:
        mb_routed = prepare_routed_df(
            load_csv(args.multibasin_routed_csv),
            support_definition=args.support_definition,
            rollout_mode=args.rollout_mode,
            depth_stratum=args.depth_stratum,
        )
        mb_base = load_multibasin_baseline(args.multibasin_baseline_csvs)
        mb_summary, mb_per_system = summarize_dataset(
            dataset="multibasin",
            routed_df=mb_routed,
            baseline_df=mb_base,
            systems=MULTIBASIN_SYSTEMS,
            roots=MULTIBASIN_ROOTS,
            horizons=MULTIBASIN_HORIZONS,
        )
        summary_frames.append(mb_summary)
        per_system_frames.append(mb_per_system)
    if "dysts" in datasets:
        dy_routed = prepare_routed_df(
            load_csv(args.dysts_routed_csv),
            support_definition=args.support_definition,
            rollout_mode=args.rollout_mode,
            depth_stratum=args.depth_stratum,
        )
        dy_base = load_csv(args.dysts_baseline_csv)
        dy_summary, dy_per_system = summarize_dataset(
            dataset="dysts",
            routed_df=dy_routed,
            baseline_df=dy_base,
            systems=DYSTS_SYSTEMS,
            roots=DYSTS_ROOTS,
            horizons=DYSTS_HORIZONS,
        )
        summary_frames.append(dy_summary)
        per_system_frames.append(dy_per_system)

    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    per_system = pd.concat(per_system_frames, ignore_index=True) if per_system_frames else pd.DataFrame()

    summary_path = output_dir / "routed_forecasting_iqm_summary.csv"
    per_system_path = output_dir / "routed_forecasting_per_system_iqm.csv"
    readout_path = output_dir / "routed_forecasting_iqm_readout.md"
    summary.to_csv(summary_path, index=False)
    per_system.to_csv(per_system_path, index=False)
    write_readout(summary, readout_path)

    print(f"Wrote {summary_path}")
    print(f"Wrote {per_system_path}")
    print(f"Wrote {readout_path}")


if __name__ == "__main__":
    main()
