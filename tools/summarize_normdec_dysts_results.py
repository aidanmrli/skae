#!/usr/bin/env python3
"""Summarize normalized-decoder Dysts KAE replacement/ablation packets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

import pandas as pd


RETAINED_SYSTEMS = [
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

DISPLAY_BY_ROOT = {
    "lista": "LISTA",
    "lista_bd": "LISTA-BD",
    "lista_sb": "LISTA-SB",
    "sparse_mlp_bd": "Sparse MLP-BD",
    "sparse_mlp": "Sparse MLP",
    "dense_mlp_tanh": "Dense MLP",
}

ROOT_ORDER = [
    "lista",
    "lista_bd",
    "lista_sb",
    "sparse_mlp_bd",
    "sparse_mlp",
    "dense_mlp_tanh",
]


def iqm(values: list[float]) -> float:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return float("nan")
    if len(finite) < 4:
        return mean(finite)
    q25 = quantile(finite, 0.25)
    q75 = quantile(finite, 0.75)
    kept = [v for v in finite if q25 <= v <= q75]
    return mean(kept or finite)


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def base_root_label(label: str) -> str:
    for suffix in ("_normdec_rollout", "_normdec_encoded"):
        if label.endswith(suffix):
            return label[: -len(suffix)]
    return label


def load_current_summary(path: Path) -> dict[tuple[str, int], float]:
    if not path.exists():
        return {}
    out: dict[tuple[str, int], float] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            root = row.get("root_label", "")
            horizon = int(row.get("horizon", "0"))
            try:
                out[(root, horizon)] = float(row.get("cross_system_mean", "nan"))
            except ValueError:
                continue
    return out


def summarize_packet(
    csv_path: Path,
    packet: str,
    horizons: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[(df["status"] == "complete") & df["system_key"].isin(RETAINED_SYSTEMS)].copy()
    df["root_base"] = df["root_label"].map(base_root_label)
    df = df[df["root_base"].isin(DISPLAY_BY_ROOT)].copy()
    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)

    per_system_rows = []
    summary_rows = []
    for root in ROOT_ORDER:
        root_df = df[df["root_base"] == root]
        for horizon in horizons:
            col = f"h{horizon}_best_periodic_mean"
            for system, system_df in root_df.groupby("system_key", sort=True):
                values = [
                    float(value)
                    for value in pd.to_numeric(system_df[col], errors="coerce").tolist()
                    if math.isfinite(float(value)) and float(value) > 0.0
                ]
                if not values:
                    continue
                per_system_rows.append(
                    {
                        "packet": packet,
                        "root_label": root,
                        "display": DISPLAY_BY_ROOT[root],
                        "system_key": system,
                        "horizon": horizon,
                        "n_seeds": len(values),
                        "mse_iqm": iqm(values),
                        "mse_median": median(values),
                    }
                )

    per_system = pd.DataFrame(per_system_rows)
    dense = per_system[per_system["root_label"] == "dense_mlp_tanh"][
        ["system_key", "horizon", "mse_iqm"]
    ].rename(columns={"mse_iqm": "dense_mse_iqm"})

    for root in ROOT_ORDER:
        for horizon in horizons:
            sub = per_system[(per_system["root_label"] == root) & (per_system["horizon"] == horizon)]
            values = [float(v) for v in sub["mse_iqm"].tolist() if math.isfinite(float(v))]
            merged = sub.merge(dense[dense["horizon"] == horizon], on=["system_key", "horizon"], how="inner")
            ratio_values = [
                float(row["mse_iqm"]) / float(row["dense_mse_iqm"])
                for _, row in merged.iterrows()
                if math.isfinite(float(row["dense_mse_iqm"])) and float(row["dense_mse_iqm"]) > 0.0
            ]
            summary_rows.append(
                {
                    "packet": packet,
                    "root_label": root,
                    "display": DISPLAY_BY_ROOT[root],
                    "horizon": horizon,
                    "n_systems": len(values),
                    "min_seeds_per_system": int(sub["n_seeds"].min()) if not sub.empty else 0,
                    "cross_system_mean": mean(values) if values else float("nan"),
                    "system_median": median(values) if values else float("nan"),
                    "systems_better_than_dense": sum(1 for value in ratio_values if value < 1.0),
                    "systems_ratio_n": len(ratio_values),
                    "mean_ratio_to_dense": mean(ratio_values) if ratio_values else float("nan"),
                }
            )

    return per_system, pd.DataFrame(summary_rows)


def read_missing(summary_json: Path) -> list[dict[str, object]]:
    if not summary_json.exists():
        return []
    payload = json.loads(summary_json.read_text())
    rows = []
    for row in payload.get("missing", []):
        system = row.get("system_key")
        if system in {"dysts:LorenzCoupled", "dysts:MultiChua"}:
            continue
        rows.append(
            {
                "root_label": base_root_label(str(row.get("root_label", ""))),
                "system_key": system,
                "seed": row.get("seed"),
                "reason": row.get("reason"),
            }
        )
    return rows


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if abs(value) >= 1000 or abs(value) < 1e-3:
        return f"{value:.3g}"
    return f"{value:.4g}"


def write_markdown(
    path: Path,
    summary: pd.DataFrame,
    current: dict[tuple[str, int], float],
    missing: dict[str, list[dict[str, object]]],
    table_horizons: list[int],
) -> None:
    lines = [
        "# Normalized-Decoder Dysts KAE Summary",
        "",
        "Estimator: for each retained Dysts system, compute seed IQM of best-periodic MSE; then report the arithmetic mean across systems.",
        "",
        "## Paper-Horizon Means",
        "",
    ]
    for packet in ["normdec_rollout", "normdec_encoded"]:
        lines.extend([f"### {packet}", "", "| Model | H100 | H2000 | H4000 | H5000 | vs current H4000 |", "|---|---:|---:|---:|---:|---:|"])
        packet_df = summary[summary["packet"] == packet]
        for root in ROOT_ORDER:
            row_cells = []
            for horizon in [100, 2000, 4000, 5000]:
                match = packet_df[(packet_df["root_label"] == root) & (packet_df["horizon"] == horizon)]
                row_cells.append(fmt(float(match["cross_system_mean"].iloc[0])) if not match.empty else "nan")
            current_h4000 = current.get((root, 4000), float("nan"))
            new_h4000 = float(
                packet_df[(packet_df["root_label"] == root) & (packet_df["horizon"] == 4000)][
                    "cross_system_mean"
                ].iloc[0]
            )
            delta = (new_h4000 / current_h4000 - 1.0) if math.isfinite(current_h4000) and current_h4000 > 0 else float("nan")
            lines.append(f"| {DISPLAY_BY_ROOT[root]} | {row_cells[0]} | {row_cells[1]} | {row_cells[2]} | {row_cells[3]} | {delta:+.1%} |")
        lines.append("")

    lines.extend(["## Encoded vs Rollout", "", "| Model | H100 | H2000 | H4000 | H5000 |", "|---|---:|---:|---:|---:|"])
    for root in ROOT_ORDER:
        cells = []
        for horizon in [100, 2000, 4000, 5000]:
            rollout = summary[
                (summary["packet"] == "normdec_rollout")
                & (summary["root_label"] == root)
                & (summary["horizon"] == horizon)
            ]["cross_system_mean"].iloc[0]
            encoded = summary[
                (summary["packet"] == "normdec_encoded")
                & (summary["root_label"] == root)
                & (summary["horizon"] == horizon)
            ]["cross_system_mean"].iloc[0]
            cells.append(f"{encoded / rollout:.3f}x")
        lines.append(f"| {DISPLAY_BY_ROOT[root]} | {' | '.join(cells)} |")

    lines.extend(["", "## Coverage Notes", ""])
    for packet, rows in missing.items():
        lines.append(f"- `{packet}` has {len(rows)} missing retained-system seed runs.")
        for row in rows:
            lines.append(
                f"  - {DISPLAY_BY_ROOT.get(str(row['root_label']), row['root_label'])}: "
                f"{row['system_key']} seed {row['seed']} ({row['reason']})"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-csv", type=Path, required=True)
    parser.add_argument("--encoded-csv", type=Path, required=True)
    parser.add_argument("--rollout-task-summary-json", type=Path, required=True)
    parser.add_argument("--encoded-task-summary-json", type=Path, required=True)
    parser.add_argument("--current-summary-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=[100, 500, 1000, 1500, 2000, 3000, 4000, 5000])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_rollout, sum_rollout = summarize_packet(args.rollout_csv, "normdec_rollout", args.horizons)
    per_encoded, sum_encoded = summarize_packet(args.encoded_csv, "normdec_encoded", args.horizons)
    per_system = pd.concat([per_rollout, per_encoded], ignore_index=True)
    summary = pd.concat([sum_rollout, sum_encoded], ignore_index=True)

    per_system.to_csv(args.out_dir / "per_system_iqm.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)

    current = load_current_summary(args.current_summary_csv)
    missing = {
        "normdec_rollout": read_missing(args.rollout_task_summary_json),
        "normdec_encoded": read_missing(args.encoded_task_summary_json),
    }
    (args.out_dir / "coverage_missing.json").write_text(json.dumps(missing, indent=2) + "\n")
    write_markdown(args.out_dir / "summary.md", summary, current, missing, args.horizons)
    print((args.out_dir / "summary.md").resolve())


if __name__ == "__main__":
    main()
