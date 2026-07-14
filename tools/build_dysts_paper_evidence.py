#!/usr/bin/env python3
"""Build Dysts summaries and displays from the frozen paper-evidence packet."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from skae.benchmarks.paper_protocol import DYSTS_PAPER_PROTOCOL
from skae.benchmarks.paper_statistics import (
    interquartile_mean as iqm,
    rowwise_interquartile_mean as _row_iqm,
)
from tools.dysts_paper_rendering import render_dysts_figure


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "figures" / "neurips_paper_2026"
TABLE_DIR = FIG_DIR / "_tables"
DEFAULT_INPUT = FIG_DIR / "_data" / "dysts_forecasting_rows.csv"
DEFAULT_PROVENANCE = FIG_DIR / "_data" / "main_paper_evidence_provenance.json"
HORIZONS = (100, 500, 1000, 1500, 2000, 3000, 4000, 5000)
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20_260_501
BASELINE_ROOT = "dense_mlp_tanh"
METHODS = OrderedDict(
    [
        ("lista", ("LISTA", "#7B3294", "-")),
        ("lista_bd", ("LISTA-BD", "#0072B2", "-")),
        ("lista_sb", ("LISTA-SB", "#56B4E9", "-")),
        ("sparse_mlp", ("Sparse MLP", "#009E73", "--")),
        ("sparse_mlp_bd", ("Sparse MLP-BD", "#44AA99", "--")),
        ("dense_mlp_tanh", ("Dense MLP", "#D55E00", "--")),
    ]
)
OUTPUT_PATHS = (
    TABLE_DIR / "dysts_dt30_iqm_summary.csv",
    TABLE_DIR / "dysts_dt30_aggregate_tests_vs_dense.csv",
    TABLE_DIR / "dysts_dt30_iqm_over_iqm_summary.csv",
    TABLE_DIR / "table_dysts_dt30_iqm_over_iqm.tex",
    TABLE_DIR / "table_dysts_dt30_ratio_to_dense.tex",
    FIG_DIR / "fig_dysts_dt30_iqm_over_iqm_horizon.pdf",
)


def verify_provenance(input_path: Path, provenance_path: Path) -> None:
    """Require the input bytes to match their named frozen-output entry."""
    if input_path.resolve() != DEFAULT_INPUT.resolve():
        if provenance_path.resolve() == DEFAULT_PROVENANCE.resolve():
            raise ValueError("A custom --input requires a corresponding --provenance")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        spec = provenance["outputs"][input_path.name]
        expected = spec["sha256"]
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Missing valid provenance for {input_path.name} in {provenance_path}"
        ) from error
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"Invalid SHA-256 provenance for {input_path.name}")
    actual = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            f"Frozen evidence hash mismatch for {input_path.name}: "
            f"expected {expected}, got {actual}"
        )


def _triplet(draws: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
        float(np.std(draws, ddof=1)),
    )


def _seed_bootstrap(
    values_by_system: list[np.ndarray], rng: np.random.Generator, n_reps: int
) -> tuple[float, float, float]:
    draws = []
    for values in values_by_system:
        indices = rng.integers(0, values.size, size=(n_reps, values.size))
        draws.append(_row_iqm(values[indices]))
    return _triplet(np.mean(np.column_stack(draws), axis=1))


def _system_bootstrap(
    system_values: np.ndarray, rng: np.random.Generator, n_reps: int
) -> tuple[float, float, float]:
    indices = rng.integers(
        0, system_values.size, size=(n_reps, system_values.size)
    )
    return _triplet(np.mean(system_values[indices], axis=1))


def _log_seed_bootstrap(
    values_by_system: list[np.ndarray], rng: np.random.Generator, n_reps: int
) -> tuple[float, float, float]:
    draws = []
    for values in values_by_system:
        logged = np.log10(values)
        indices = rng.integers(0, logged.size, size=(n_reps, logged.size))
        draws.append(_row_iqm(logged[indices]))
    joined = np.mean(np.column_stack(draws), axis=1)
    low, high, spread = _triplet(joined)
    return 10.0**low, 10.0**high, spread


def _relative_seed_bootstrap(
    values_by_system: list[np.ndarray],
    center: float,
    rng: np.random.Generator,
    n_reps: int,
) -> tuple[float, float, float]:
    relative_draws = []
    for values in values_by_system:
        system_center = iqm(values)
        indices = rng.integers(0, values.size, size=(n_reps, values.size))
        draws = np.clip(_row_iqm(values[indices]), np.finfo(float).tiny, None)
        relative_draws.append(np.log10(draws) - math.log10(system_center))
    joined = np.mean(np.column_stack(relative_draws), axis=1)
    low, high, spread = _triplet(joined)
    return center * 10.0**low, center * 10.0**high, spread


def _holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [float("nan")] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def load_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path, low_memory=False)
    required = {"root_label", "system_key", "seed", "status"}
    required |= {f"h{h}_best_periodic_mean" for h in HORIZONS}
    required |= {f"h{h}_best_periodic_full_finite_fraction" for h in HORIZONS}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise RuntimeError(f"{path} is missing columns: {missing}")
    rows = rows[rows["status"] == "complete"].copy()
    expected_methods = set(METHODS)
    expected_systems = set(DYSTS_PAPER_PROTOCOL.system_keys)
    if set(rows["root_label"].unique()) != expected_methods:
        raise RuntimeError("Frozen rows do not contain exactly the six paper methods")
    if set(rows["system_key"].unique()) != expected_systems:
        raise RuntimeError("Frozen rows do not contain exactly the ten retained systems")
    rows["seed"] = pd.to_numeric(rows["seed"], errors="raise").astype(int)
    if set(rows["seed"].unique()) != set(DYSTS_PAPER_PROTOCOL.seeds):
        raise RuntimeError("Frozen rows do not contain exactly the fifteen paper seeds")
    keys = ["root_label", "system_key", "seed"]
    if rows.duplicated(keys).any() or len(rows) != 900:
        raise RuntimeError("Frozen rows must have one row per method/system/seed")
    return rows


def summarize_rows(
    rows: pd.DataFrame, *, bootstrap_reps: int = BOOTSTRAP_REPS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_system_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    seed_rng = np.random.default_rng(BOOTSTRAP_SEED)
    system_rng = np.random.default_rng(BOOTSTRAP_SEED + 104_729)
    log_rng = np.random.default_rng(BOOTSTRAP_SEED + 130_363)
    relative_rng = np.random.default_rng(BOOTSTRAP_SEED + 155_921)
    for root_label, (display, _, _) in METHODS.items():
        root_rows = rows[rows["root_label"] == root_label]
        for horizon in HORIZONS:
            mse_column = f"h{horizon}_best_periodic_mean"
            coverage_column = f"h{horizon}_best_periodic_full_finite_fraction"
            values_by_system: list[np.ndarray] = []
            horizon_records = []
            for system, group in root_rows.groupby("system_key", sort=True):
                values = pd.to_numeric(group[mse_column], errors="coerce").to_numpy()
                values = values[np.isfinite(values) & (values > 0.0)]
                coverage = pd.to_numeric(group[coverage_column], errors="coerce").to_numpy()
                coverage = coverage[np.isfinite(coverage)]
                if values.size == 0:
                    continue
                values_by_system.append(values)
                record = {
                    "root_label": root_label,
                    "display": display,
                    "system_key": system,
                    "horizon": horizon,
                    "mse_iqm": iqm(values),
                    "mse_log10_iqm": iqm(np.log10(values)),
                    "full_finite_iqm": iqm(coverage),
                }
                per_system_records.append(record)
                horizon_records.append(record)
            system_iqms = np.asarray([r["mse_iqm"] for r in horizon_records])
            system_log_iqms = np.asarray([r["mse_log10_iqm"] for r in horizon_records])
            coverages = np.asarray([r["full_finite_iqm"] for r in horizon_records])
            seed_ci = _seed_bootstrap(values_by_system, seed_rng, bootstrap_reps)
            system_ci = _system_bootstrap(system_iqms, system_rng, bootstrap_reps)
            log_ci = _log_seed_bootstrap(values_by_system, log_rng, bootstrap_reps)
            system_mean = float(np.mean(system_iqms))
            relative_ci = _relative_seed_bootstrap(
                values_by_system, system_mean, relative_rng, bootstrap_reps
            )
            log_mean = float(np.mean(system_log_iqms))
            summary_records.append(
                {
                    "root_label": root_label,
                    "display": display,
                    "horizon": horizon,
                    "n_systems": len(system_iqms),
                    "cross_system_mean": system_mean,
                    "cross_system_iqm": system_mean,
                    "cross_system_iqm_legacy": iqm(system_iqms),
                    "cross_system_log10_iqm_mean": log_mean,
                    "cross_system_log_iqm_geomean": 10.0**log_mean,
                    "system_q25": float(np.percentile(system_iqms, 25)),
                    "system_q75": float(np.percentile(system_iqms, 75)),
                    "system_median": float(np.median(system_iqms)),
                    "seed_bootstrap_ci95_low": seed_ci[0],
                    "seed_bootstrap_ci95_high": seed_ci[1],
                    "seed_bootstrap_se": seed_ci[2],
                    "log_relative_seed_bootstrap_ci95_low": relative_ci[0],
                    "log_relative_seed_bootstrap_ci95_high": relative_ci[1],
                    "log_relative_seed_bootstrap_se_log10": relative_ci[2],
                    "system_bootstrap_ci95_low": system_ci[0],
                    "system_bootstrap_ci95_high": system_ci[1],
                    "system_bootstrap_se": system_ci[2],
                    "log_seed_bootstrap_ci95_low": log_ci[0],
                    "log_seed_bootstrap_ci95_high": log_ci[1],
                    "log_seed_bootstrap_se_log10": log_ci[2],
                    "seed_bootstrap_reps": bootstrap_reps,
                    "full_finite_mean": float(np.mean(coverages)),
                    "full_finite_iqm": float(np.mean(coverages)),
                }
            )
    return pd.DataFrame(per_system_records), pd.DataFrame(summary_records)


def aggregate_tests(per_system: pd.DataFrame) -> pd.DataFrame:
    dense = per_system[per_system["root_label"] == BASELINE_ROOT][
        ["system_key", "horizon", "mse_iqm"]
    ].rename(columns={"mse_iqm": "dense_system_iqm"})
    records = []
    for root_label, (display, _, _) in METHODS.items():
        if root_label == BASELINE_ROOT:
            continue
        ratios = per_system[per_system["root_label"] == root_label].merge(
            dense, on=["system_key", "horizon"], how="inner"
        )
        ratios["ratio"] = ratios["mse_iqm"] / ratios["dense_system_iqm"]
        for horizon, group in ratios.groupby("horizon", sort=True):
            values = group["ratio"].to_numpy(dtype=float)
            log_values = np.log10(values)
            n_better = int(np.sum(log_values < 0.0))
            records.append(
                {
                    "root_label": root_label,
                    "display": display,
                    "horizon": int(horizon),
                    "n_systems": len(values),
                    "systems_with_ratio_lt_1": n_better,
                    "ratio_iqm": float(np.mean(values)),
                    "ratio_median": float(np.median(values)),
                    "ratio_mean": float(np.mean(values)),
                    "ratio_sd_systems": float(np.std(values, ddof=1)),
                    "ratio_q25": float(np.percentile(values, 25)),
                    "ratio_q75": float(np.percentile(values, 75)),
                    "log10_ratio_iqm": float(np.mean(log_values)),
                    "log10_ratio_median": float(np.median(log_values)),
                    "log10_ratio_mean": float(np.mean(log_values)),
                    "log10_ratio_sd_systems": float(np.std(log_values, ddof=1)),
                    "p_system_wilcoxon_raw": float(
                        stats.wilcoxon(
                            log_values, alternative="less", zero_method="wilcox"
                        ).pvalue
                    ),
                    "p_system_sign_raw": float(
                        stats.binomtest(
                            n_better, len(values), p=0.5, alternative="greater"
                        ).pvalue
                    ),
                    "p_system_ttest_raw": float(
                        stats.ttest_1samp(
                            log_values, popmean=0.0, alternative="less"
                        ).pvalue
                    ),
                }
            )
    result = pd.DataFrame(records)
    corrections = ("p_system_wilcoxon", "p_system_sign", "p_system_ttest")
    for prefix in corrections:
        raw = f"{prefix}_raw"
        result[f"{prefix}_holm_all"] = _holm(result[raw].tolist())
    for prefix in corrections:
        raw = f"{prefix}_raw"
        result[f"{prefix}_holm_by_horizon"] = float("nan")
        for indices in result.groupby("horizon").groups.values():
            indices = list(indices)
            adjusted = _holm([float(result.at[index, raw]) for index in indices])
            for index, value in zip(indices, adjusted):
                result.at[index, f"{prefix}_holm_by_horizon"] = value
    return result


def _tex_number(value: float) -> str:
    absolute = abs(value)
    if absolute == 0.0:
        return "0"
    if absolute >= 1000.0 or absolute < 1e-3:
        exponent = math.floor(math.log10(absolute))
        return rf"{value / 10.0**exponent:.2f}{{\times}}10^{{{exponent}}}"
    decimals = max(2 - math.floor(math.log10(absolute)), 0)
    return f"{value:.{decimals}f}"


def robust_summary(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary[
        [
            "root_label",
            "display",
            "horizon",
            "n_systems",
            "cross_system_iqm_legacy",
            "cross_system_mean",
            "system_median",
            "system_q25",
            "system_q75",
        ]
    ].rename(
        columns={"cross_system_iqm_legacy": "iqm_over_system_seed_iqms"}
    )
    return result


def render_table(robust: pd.DataFrame) -> bytes:
    best = {
        horizon: robust[robust["horizon"] == horizon]
        .sort_values("iqm_over_system_seed_iqms")
        .iloc[0]["root_label"]
        for horizon in HORIZONS
    }
    lines = [
        r"\begin{tabular}{@{}l " + " ".join(["r"] * len(HORIZONS)) + r"@{}}",
        r"\toprule",
        "Model & " + " & ".join(f"H{h}" for h in HORIZONS) + r" \\",
        r"\midrule",
    ]
    for root_label, (display, _, _) in METHODS.items():
        cells = []
        for horizon in HORIZONS:
            row = robust[
                (robust["root_label"] == root_label)
                & (robust["horizon"] == horizon)
            ].iloc[0]
            value = _tex_number(float(row["iqm_over_system_seed_iqms"]))
            if root_label == best[horizon]:
                value = rf"\mathbf{{{value}}}"
            cells.append(rf"${value}$")
        lines.append(f"{display} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return ("\n".join(lines) + "\n").encode()


def render_ratio_table(tests: pd.DataFrame) -> bytes:
    lookup = {
        (row["root_label"], int(row["horizon"])): float(row["ratio_mean"])
        for row in tests.to_dict(orient="records")
    }
    lines = [
        r"\begin{tabular}{@{}l r r r r r r r r@{}}",
        r"\toprule",
        "Model & " + " & ".join(f"H{h}" for h in HORIZONS) + r" \\",
        r"\midrule",
    ]
    for root_label, (display, _, _) in METHODS.items():
        values = [
            1.0 if root_label == BASELINE_ROOT else lookup[(root_label, horizon)]
            for horizon in HORIZONS
        ]
        cells = " & ".join(rf"\({value:.3f}\)" for value in values)
        lines.append(f"{display} & {cells}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return ("\n".join(lines) + "\n").encode()


def build_outputs(
    input_path: Path = DEFAULT_INPUT,
    provenance_path: Path = DEFAULT_PROVENANCE,
    *,
    bootstrap_reps: int = BOOTSTRAP_REPS,
) -> dict[Path, bytes]:
    verify_provenance(input_path, provenance_path)
    rows = load_rows(input_path)
    per_system, summary = summarize_rows(rows, bootstrap_reps=bootstrap_reps)
    tests = aggregate_tests(per_system)
    summary_bytes = summary.to_csv(index=False, lineterminator="\n").encode()
    tests_bytes = tests.to_csv(index=False, lineterminator="\n").encode()
    # Preserve the appendix builder's historical two-stage CSV rounding.
    round_tripped = pd.read_csv(io.BytesIO(summary_bytes))
    robust = robust_summary(round_tripped)
    robust_bytes = robust.to_csv(index=False, lineterminator="\n").encode()
    contents = (
        summary_bytes,
        tests_bytes,
        robust_bytes,
        render_table(robust),
        render_ratio_table(tests),
        render_dysts_figure(robust, METHODS, HORIZONS),
    )
    return dict(zip(OUTPUT_PATHS, contents))


def write_or_check(outputs: dict[Path, bytes], *, check: bool) -> None:
    stale = [
        path
        for path, content in outputs.items()
        if not path.is_file() or path.read_bytes() != content
    ]
    if check:
        if stale:
            names = ", ".join(map(str, stale))
            raise RuntimeError(f"Dysts paper artifacts are stale: {names}")
        print(f"Verified {len(outputs)} Dysts paper artifacts")
        return
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_outputs(args.input, args.provenance)
    write_or_check(outputs, check=args.check)


if __name__ == "__main__":
    main()
