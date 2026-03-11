#!/usr/bin/env python3
"""Compare in-time (every-step) prediction quality across models.

Reads the collected forecasting CSVs and builds per-system seed-median
tables for every-step MSE at H100, H500, H1000, plus cross-model
win-count and ratio summaries.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median


def load_csv(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_system_medians(rows: list[dict], horizons: list[int]):
    """Group by (root_label, system_key) → seed-median every-step values."""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["root_label"], r["system_key"])
        for h in horizons:
            val = safe_float(r.get(f"h{h}_every_step_mean"))
            if val is not None:
                grouped[key][h].append(val)

    result = {}
    for (root, sys_key), hdata in grouped.items():
        result[(root, sys_key)] = {h: median(vals) if vals else None for h, vals in hdata.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_paths", nargs="+", help="Forecasting CSV files to compare")
    parser.add_argument("--anchor", default="generic_sparse", help="Anchor model for ratios")
    parser.add_argument("--horizons", default="100,500,1000", help="Comma-separated horizons")
    parser.add_argument("--output", "-o", default=None, help="Output markdown file")
    args = parser.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]

    # Load all CSVs
    all_rows = []
    for p in args.csv_paths:
        all_rows.extend(load_csv(p))

    # Get unique models and systems
    models = sorted(set(r["root_label"] for r in all_rows))
    systems = sorted(set(r["system_key"] for r in all_rows))

    medians = build_system_medians(all_rows, horizons)

    lines = []
    lines.append("# In-Time Prediction (Every-Step MSE) Comparison\n")

    # --- Cross-system summary table ---
    lines.append("## Cross-System Summary (seed-median, then system-median)\n")
    for h in horizons:
        lines.append(f"### H{h}\n")
        header = "| Model | System-Median Every-Step | Good Systems (< 1.0) | Good Systems (< 10.0) |"
        sep = "|---|---:|---:|---:|"
        lines.append(header)
        lines.append(sep)
        for model in models:
            sys_vals = []
            for sys in systems:
                v = medians.get((model, sys), {}).get(h)
                if v is not None:
                    sys_vals.append(v)
            if sys_vals:
                med = median(sys_vals)
                good_1 = sum(1 for v in sys_vals if v < 1.0)
                good_10 = sum(1 for v in sys_vals if v < 10.0)
                lines.append(f"| {model} | {med:.4e} | {good_1}/{len(sys_vals)} | {good_10}/{len(sys_vals)} |")
            else:
                lines.append(f"| {model} | — | — | — |")
        lines.append("")

    # --- Per-system tables with pairwise wins ---
    for h in horizons:
        lines.append(f"## Per-System Seed-Median Every-Step MSE — H{h}\n")

        # Build header
        header_parts = ["| System"]
        for m in models:
            header_parts.append(m)
        if args.anchor in models:
            for m in models:
                if m != args.anchor:
                    header_parts.append(f"ratio ({m}/{args.anchor})")
        header_parts.append("|")
        header = " | ".join(header_parts)
        sep = "|" + "|".join(["---"] + ["---:"] * (len(header_parts) - 2)) + "|"
        lines.append(header)
        lines.append(sep)

        win_counts = defaultdict(int)
        ratio_lists = defaultdict(list)

        for sys in systems:
            parts = [f"| {sys}"]
            anchor_val = medians.get((args.anchor, sys), {}).get(h)

            for m in models:
                v = medians.get((m, sys), {}).get(h)
                if v is not None:
                    parts.append(f"{v:.4e}")
                else:
                    parts.append("—")

            if args.anchor in models and anchor_val is not None:
                for m in models:
                    if m == args.anchor:
                        continue
                    v = medians.get((m, sys), {}).get(h)
                    if v is not None and anchor_val > 0:
                        ratio = v / anchor_val
                        ratio_lists[m].append(ratio)
                        parts.append(f"{ratio:.3f}")
                        if v < anchor_val:
                            win_counts[m] += 1
                    else:
                        parts.append("—")

            parts.append("|")
            lines.append(" | ".join(parts))

        lines.append("")

        # Win count summary
        if args.anchor in models:
            lines.append(f"### Win Counts vs `{args.anchor}` — H{h}\n")
            n_sys = len(systems)
            header = "| Model | Wins | Losses | Median Ratio |"
            sep = "|---|---:|---:|---:|"
            lines.append(header)
            lines.append(sep)
            for m in models:
                if m == args.anchor:
                    continue
                w = win_counts[m]
                rats = ratio_lists[m]
                med_r = median(rats) if rats else float("nan")
                lines.append(f"| {m} | {w}/{n_sys} | {n_sys - w}/{n_sys} | {med_r:.4f} |")
            lines.append("")

    out = "\n".join(lines)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out)
        print(f"Written to {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
