"""Build the controlled-benchmark per-system appendix tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from experiments.neurips_2026.protocol import (
    CONTROLLED_MODEL_DISPLAY_NAMES,
    CONTROLLED_PAPER_PROTOCOL,
    controlled_system_display_name,
)
from experiments.neurips_2026.paths import PAPER_DATA_DIR, PAPER_TABLE_DIR


DATA_DIR = PAPER_DATA_DIR
TABLE_DIR = PAPER_TABLE_DIR
FORECAST_CSV = DATA_DIR / "controlled_forecasting_rows.csv"
SUPPORT_CSV = DATA_DIR / "controlled_support_rows.csv"
PROVENANCE_JSON = DATA_DIR / "main_paper_evidence_provenance.json"
BASELINE = "mlp_zero_sparse_hardinit_basin_partition_control"
ROOTS = dict(CONTROLLED_MODEL_DISPLAY_NAMES)
CANDIDATES = tuple(root for root in ROOTS if root != BASELINE)
HORIZONS = (100, 500, 1000)
ALPHA = 0.05
SYSTEM_ORDER = tuple(
    key.replace("claude:", "claude_") for key in CONTROLLED_PAPER_PROTOCOL.system_keys
)
OUTPUTS = {
    100: "table_persystem_h100.tex",
    500: "table_persystem_h500.tex",
    1000: "table_persystem_h1000.tex",
    "HBgivenF": "table_persystem_HBgivenF.tex",
}


def verify_frozen_inputs() -> None:
    provenance = json.loads(PROVENANCE_JSON.read_text(encoding="utf-8"))
    for path in (FORECAST_CSV, SUPPORT_CSV):
        expected = provenance["outputs"][path.name]["sha256"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen input hash mismatch for {path.name}")


def holm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(values.size)
    running = 0.0
    for rank, index in enumerate(order):
        adjusted[index] = max(running, min(1.0, (values.size - rank) * values[index]))
        running = adjusted[index]
    return adjusted


def paired_records(
    frame: pd.DataFrame,
    *,
    metric: str,
    log_values: bool,
    alternative: str,
) -> pd.DataFrame:
    baseline = frame[frame["root_label"] == BASELINE].pivot_table(
        index="system_name", columns="seed", values=metric, aggfunc="first"
    )
    records: list[dict[str, object]] = []
    for root in CANDIDATES:
        candidate = frame[frame["root_label"] == root].pivot_table(
            index="system_name", columns="seed", values=metric, aggfunc="first"
        )
        systems = candidate.index.intersection(baseline.index)
        raw_p: list[float] = []
        effects: list[tuple[float, float]] = []
        for system in systems:
            cand = candidate.loc[system].to_numpy(dtype=float)
            base = baseline.loc[system].to_numpy(dtype=float)
            mask = np.isfinite(cand) & np.isfinite(base)
            if log_values:
                mask &= (cand > 0.0) & (base > 0.0)
            cand, base = cand[mask], base[mask]
            if cand.size < 4:
                raw_p.append(float("nan"))
                effects.append((float("nan"), float("nan")))
                continue
            if log_values:
                cand, base = np.log10(cand), np.log10(base)
            paired = cand - base
            if np.all(paired == 0.0):
                # The signed-rank statistic is undefined when every paired
                # difference is zero. Keep the effect but do not invent a p-value.
                p_value = float("nan")
            else:
                try:
                    p_value = float(
                        stats.wilcoxon(
                            cand,
                            base,
                            alternative=alternative,
                            zero_method="wilcox",
                        ).pvalue
                    )
                except ValueError:
                    p_value = float("nan")
            raw_p.append(p_value)
            effects.append((float(np.mean(paired)), float(np.median(paired))))
        raw = np.asarray(raw_p)
        adjusted = np.full_like(raw, np.nan)
        valid = np.isfinite(raw)
        if valid.any():
            adjusted[valid] = holm(raw[valid])
        for index, system in enumerate(systems):
            records.append(
                {
                    "root_label": root,
                    "system": system,
                    "p_holm": adjusted[index],
                    "mean_diff": effects[index][0],
                    "median_diff": effects[index][1],
                }
            )
    return pd.DataFrame(records)


def fmt_p(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return r"$<\!10^{-3}$" if value < 1e-3 else f"${value:.3f}$"


def pretty_system(value: str) -> str:
    return controlled_system_display_name(value)


def table_header() -> list[str]:
    column_spec = "@{}l " + " ".join(["rr"] * len(CANDIDATES)) + "@{}"
    top = "System & " + " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{ROOTS[root]}}}" for root in CANDIDATES
    ) + r" \\"
    rules = " ".join(
        f"\\cmidrule(lr){{{2 + 2 * index}-{3 + 2 * index}}}"
        for index in range(len(CANDIDATES))
    )
    sub = " & " + " & ".join([r"$\Delta$ & $p_{\rm H}$"] * len(CANDIDATES)) + r" \\"
    return [f"\\begin{{tabular}}{{{column_spec}}}", r"\toprule", top, rules, sub, r"\midrule"]


def render_table(records: pd.DataFrame, *, order: list[str], use_mean: bool) -> bytes:
    effect_column = "mean_diff" if use_mean else "median_diff"
    lines = table_header()
    for system in order:
        cells = [pretty_system(system)]
        for root in CANDIDATES:
            row = records[(records["root_label"] == root) & (records["system"] == system)]
            if row.empty:
                cells.extend(["--", "--"])
                continue
            effect = float(row.iloc[0][effect_column])
            p_value = float(row.iloc[0]["p_holm"])
            value = f"{effect:+.2f}"
            cells.append(rf"$\mathbf{{{value}}}$" if np.isfinite(p_value) and p_value < ALPHA else f"${value}$")
            cells.append(fmt_p(p_value))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    summary = [r"\emph{Significant/eligible}"]
    for root in CANDIDATES:
        subset = records[records["root_label"] == root]
        count = int((subset["p_holm"] < ALPHA).sum())
        total = int(subset["p_holm"].notna().sum())
        summary.append(rf"\multicolumn{{2}}{{c}}{{$K{{=}}{count}\,/\,N{{=}}{total}$}}")
    lines.append(" & ".join(summary) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines).encode("utf-8")


def build_outputs() -> dict[str, bytes]:
    forecast = pd.read_csv(FORECAST_CSV, low_memory=False)
    support = pd.read_csv(SUPPORT_CSV, low_memory=False)
    forecast_records = {
        horizon: paired_records(
            forecast,
            metric=f"h{horizon}_best_periodic_mean",
            log_values=True,
            alternative="less",
        )
        for horizon in HORIZONS
    }
    order = list(SYSTEM_ORDER)
    outputs = {
        OUTPUTS[horizon]: render_table(
            forecast_records[horizon], order=order, use_mean=True
        )
        for horizon in HORIZONS
    }
    support_records = paired_records(
        support,
        metric="family_h_basin_given_family",
        log_values=False,
        alternative="less",
    )
    outputs[OUTPUTS["HBgivenF"]] = render_table(
        support_records, order=order, use_mean=True
    )
    return outputs


def write_or_check(outputs: dict[str, bytes], *, check: bool) -> None:
    if check:
        stale = [
            name
            for name, content in outputs.items()
            if not (TABLE_DIR / name).is_file() or (TABLE_DIR / name).read_bytes() != content
        ]
        if stale:
            raise SystemExit(f"Per-system tables are stale: {', '.join(stale)}")
        print("Per-system tables are byte-identical to a clean rebuild.")
        return
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (TABLE_DIR / name).write_bytes(content)
        print(f"wrote {TABLE_DIR / name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    verify_frozen_inputs()
    write_or_check(build_outputs(), check=args.check)


if __name__ == "__main__":
    main()
