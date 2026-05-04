"""
Per-system significance checks for paper Tables 2, 3, and 4.

Table 2 uses the benchmark system as the independent inferential unit for the
main paper display: seed-level routed/global log-ratios are summarized within
each system, then the system effects are tested against zero with an exact
one-sided sign-flip test. The older within-system seed-paired Wilcoxon/Holm
counts are still produced as diagnostics in the JSON/CSV artifacts. Tables 1
diagnostics and Table 3 keep their within-system paired reproducibility counts.
For Table 3, the within-system paired unit is the controlled transfer instance
(refreshed-support vs previous-support), with all completed seeds contributing
transfer instances.

For Tables 2 and 3, the paper-facing LISTA-SB row uses the matched-dimension
``d_z=256`` artifacts. Table 3 additionally includes any completed matched
plain-LISTA and MLP-control refresh packets when present.

Outputs JSON to docs/figures/neurips_paper_2026/_tables/per_system_paired_tests.json
so the paper-side tables and prose can be updated by hand.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parent.parent
ROUTING_CSVS = [
    REPO_ROOT
    / "results"
    / "transition_rich_table2_5model_seed15_backfill_20260428"
    / "self_routed_forecasting"
    / "self_routed_forecasting_rows.csv",
    REPO_ROOT
    / "results"
    / "transition_rich_lista_sb_p256_hardinit_fairness_seed15_20260428"
    / "self_routed_forecasting"
    / "self_routed_forecasting_rows.csv",
    REPO_ROOT
    / "results"
    / "transition_rich_lista_dense_p256_hardinit_table123_20260430"
    / "self_routed_forecasting"
    / "self_routed_forecasting_rows.csv",
]
ROUTING_HORIZONS = [100, 1000]
ROUTING_EXPECTED_SEEDS = list(range(15))
ROUTING_MAIN_ROUTE = "family_local_centered"
FIXED_BENCHMARK_EXCLUDED_SYSTEMS = {
    "multiwell_strong_transition",
    "claude_checkerboard_potential",
    "claude:checkerboard_potential",
}
REFRESH_CSVS = [
    REPO_ROOT
    / "results"
    / "periodic_support_refresh_fixed17_seed0to14_20260429"
    / "merged"
    / "periodic_support_refresh_rows.csv",
    REPO_ROOT
    / "results"
    / "periodic_support_refresh_lista_sb_p256_seed0to14_20260430"
    / "merged"
    / "periodic_support_refresh_rows.csv",
    REPO_ROOT
    / "results"
    / "transition_rich_lista_dense_p256_hardinit_table123_20260430"
    / "periodic_support_refresh"
    / "merged"
    / "periodic_support_refresh_rows.csv",
    REPO_ROOT
    / "results"
    / "periodic_support_refresh_mlp_controls_seed0to14_20260430"
    / "merged"
    / "periodic_support_refresh_rows.csv",
]
DYSTS_CSVS = [
    REPO_ROOT
    / "results"
    / "dysts_long_horizon_eval_seq10_h60k_seeds0to14_20260428"
    / "collect"
    / "forecasting_rows.csv",
]
TBL_DIR = REPO_ROOT / "docs" / "figures" / "neurips_paper_2026" / "_tables"
OUT_JSON = TBL_DIR / "per_system_paired_tests.json"

ROUTING_MODELS = {
    "lista_dense_signsplit_p256_hardinit_basin_partition": "LISTA",
    "lista_dense_softblock_signsplit_p256_hardinit_basin_partition": "LISTA-SB",
    "lista_blockdiag_signsplit_hardinit_basin_partition": "LISTA-BD",
    "mlp_sparse_blockdiag_hardinit_basin_partition_control": "Sparse MLP, BD",
    "mlp_sparse_hardinit_basin_partition_control": "Sparse MLP",
    "mlp_zero_sparse_hardinit_basin_partition_control": "Dense MLP",
}

REFRESH_MODELS = {
    "lista_dense_signsplit_p256_hardinit_basin_partition": "LISTA",
    "lista_dense_softblock_signsplit_p256_hardinit_basin_partition": "LISTA-SB",
    "lista_blockdiag_signsplit_hardinit_basin_partition": "LISTA-BD",
    "mlp_sparse_blockdiag_hardinit_basin_partition_control": "Sparse MLP, BD",
    "mlp_sparse_hardinit_basin_partition_control": "Sparse MLP",
    "mlp_zero_sparse_hardinit_basin_partition_control": "Dense MLP",
}

DYSTS_BASELINE = "generic_sparse_sc0_ns200k_best"
DYSTS_DISPLAY = {
    "lista_dense_promoted_stage4": "LISTA-D",
    "lista_blockdiag_ns200k_denseopt_sc6em3": "LISTA-BD",
    "generic_sparse_ns200k_best": "Sparse MLP",
    "generic_sparse_blockdiag_ns200k_sc6em3": "Sparse MLP, BD",
    "generic_sparse_sc0_ns200k_best": "Dense MLP",
}

PALETTE = {
    "LISTA": "#785EF0",
    "LISTA-SB": "#D55E00",
    "LISTA-BD": "#0072B2",
    "Sparse MLP, BD": "#56B4E9",
    "Sparse MLP": "#009E73",
    "Dense MLP": "#000000",
}


def filter_fixed_benchmark_systems(df: pd.DataFrame) -> pd.DataFrame:
    """Drop systems removed from the 15-system paper benchmark."""
    filtered = df.copy()
    for column in ("system_key", "system_name", "train_env_name"):
        if column in filtered:
            filtered = filtered[~filtered[column].isin(FIXED_BENCHMARK_EXCLUDED_SYSTEMS)]
    return filtered.copy()

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def iqm(values) -> float:
    """Interquartile mean: mean of values within [25th, 75th] percentile.

    Falls back to plain mean when n<4 (IQR is degenerate).
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    if len(arr) < 4:
        return float(np.mean(arr))
    lo = np.percentile(arr, 25)
    hi = np.percentile(arr, 75)
    mask = (arr >= lo) & (arr <= hi)
    if not mask.any():
        return float(np.mean(arr))
    return float(np.mean(arr[mask]))


def mean_finite(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.mean(arr))


def cell_summary(values) -> dict:
    """Cross-rollout summary of a cell: IQM, median, IQR endpoints, n."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"n": 0, "iqm": float("nan"), "median": float("nan"),
                "q25": float("nan"), "q75": float("nan")}
    return {
        "n": int(len(arr)),
        "iqm": iqm(arr),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def system_iqm_summary(
    df: pd.DataFrame,
    value_col: str,
    system_col: str = "system_key",
    positive_only: bool = False,
) -> dict:
    """Summarize by per-system IQM, then arithmetic mean across systems."""
    rows = []
    for system, grp in df.groupby(system_col, sort=True):
        values = pd.to_numeric(grp[value_col], errors="coerce")
        values = values[np.isfinite(values)]
        if positive_only:
            values = values[values > 0.0]
        if len(values) == 0:
            continue
        rows.append({"system": system, "iqm": iqm(values), "n": int(len(values))})
    arr = np.asarray([row["iqm"] for row in rows], dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "n_systems": 0,
            "mean": float("nan"),
            "iqm": float("nan"),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "std": float("nan"),
            "per_system": rows,
        }
    mean_value = mean_finite(arr)
    return {
        "n_systems": int(len(arr)),
        "mean": mean_value,
        "iqm": mean_value,
        "system_iqm_iqm": iqm(arr),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "per_system": rows,
    }


def holm_corrected(p_values: list[float]) -> list[float]:
    """Return Holm-corrected p-values (preserves input order)."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adj = [0.0] * n
    running_max = 0.0
    for rank, i in enumerate(order):
        raw = p_values[i]
        scaled = (n - rank) * raw
        running_max = max(running_max, scaled)
        adj[i] = min(running_max, 1.0)
    return adj


def exact_signflip_mean_test(values, alternative: str = "less") -> dict:
    """Exact one-sample sign-flip test on paired system effects.

    The benchmark systems are the independent units. Under the paired null, the
    sign of each system effect is exchangeable; the observed magnitudes are
    retained and all sign assignments are enumerated. The p-value and reported
    p-value uses the mean log-effect across systems. The returned dictionary
    also includes the IQM log-effect used as the robust table point estimate.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "n": 0,
            "iqm_delta": float("nan"),
            "mean_delta": float("nan"),
            "ratio_from_iqm_delta": float("nan"),
            "ratio_from_mean_delta": float("nan"),
            "p_raw": float("nan"),
            "n_in_direction": 0,
        }

    observed_iqm = iqm(arr)
    observed_mean = float(np.mean(arr))
    magnitudes = np.abs(arr)
    n = len(magnitudes)
    total = 1 << n
    bits = ((np.arange(total, dtype=np.uint32)[:, None] >> np.arange(n, dtype=np.uint32)) & 1)
    signs = np.where(bits == 1, -1.0, 1.0)
    stats_mean = (signs * magnitudes).mean(axis=1)
    if alternative == "less":
        extreme = int(np.sum(stats_mean <= observed_mean + 1e-15))
    elif alternative == "greater":
        extreme = int(np.sum(stats_mean >= observed_mean - 1e-15))
    else:
        extreme = int(np.sum(np.abs(stats_mean) >= abs(observed_mean) - 1e-15))

    if alternative == "less":
        n_in_direction = int(np.sum(arr < 0.0))
    elif alternative == "greater":
        n_in_direction = int(np.sum(arr > 0.0))
    else:
        n_in_direction = int(np.sum(arr != 0.0))

    return {
        "n": int(n),
        "iqm_delta": float(observed_iqm),
        "mean_delta": float(observed_mean),
        "ratio_from_iqm_delta": float(10.0 ** observed_iqm),
        "ratio_from_mean_delta": float(10.0 ** observed_mean),
        "p_raw": float(extreme / total),
        "n_in_direction": n_in_direction,
    }


def bootstrap_iqm_ci(values, *, n_boot: int = 10000, seed: int = 0) -> dict:
    """Cluster bootstrap CI for the system-level mean effect."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0 or n_boot <= 0:
        return {
            "log10_low": float("nan"),
            "log10_high": float("nan"),
            "ratio_low": float("nan"),
            "ratio_high": float("nan"),
        }
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        draws[i] = mean_finite(sample)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "log10_low": float(lo),
        "log10_high": float(hi),
        "ratio_low": float(10.0 ** lo),
        "ratio_high": float(10.0 ** hi),
    }


def is_finite_nonnegative(value) -> bool:
    return pd.notna(value) and math.isfinite(float(value)) and float(value) >= 0.0


def is_finite_positive(value) -> bool:
    return pd.notna(value) and math.isfinite(float(value)) and float(value) > 0.0


def per_system_paired_wilcoxon(
    df: pd.DataFrame,
    system_col: str,
    seed_col: str,
    delta_col: str,
    alternative: str = "less",
    expected_systems: list[str] | None = None,
) -> dict:
    """For each system, run a one-sided paired Wilcoxon on per-seed deltas.

    If `expected_systems` is given, the denominator N is forced to that count
    (systems with no valid data still appear with passes=False) so that
    different (model, slice, route) cells share the same denominator.

    Also computes a system-level sign test as a power-conservative summary of
    whether the per-system IQM-delta is < 0 (or > 0 for alternative="greater").
    """
    df = df.copy()
    rows = []
    seen = set()
    for system, grp in df.groupby(system_col, sort=True):
        seen.add(system)
        deltas = grp[delta_col].dropna().to_numpy()
        if len(deltas) < 2 or np.allclose(deltas, deltas[0]):
            rows.append({
                "system": system,
                "n": int(len(deltas)),
                "median_delta": float(np.median(deltas)) if len(deltas) else float("nan"),
                "iqm_delta": iqm(deltas) if len(deltas) else float("nan"),
                "p_raw": float("nan"),
                "p_holm": float("nan"),
                "passes": False,
                "reason": "too_few_or_constant",
            })
            continue
        try:
            res = stats.wilcoxon(deltas, alternative=alternative, zero_method="wilcox")
            p_raw = float(res.pvalue)
        except ValueError:
            p_raw = float("nan")
        rows.append({
            "system": system,
            "n": int(len(deltas)),
            "median_delta": float(np.median(deltas)),
            "iqm_delta": iqm(deltas),
            "p_raw": p_raw,
            "p_holm": float("nan"),
            "passes": False,
            "reason": "",
        })

    if expected_systems:
        for system in expected_systems:
            if system not in seen:
                rows.append({
                    "system": system,
                    "n": 0,
                    "median_delta": float("nan"),
                    "iqm_delta": float("nan"),
                    "p_raw": float("nan"),
                    "p_holm": float("nan"),
                    "passes": False,
                    "reason": "no_data",
                })
        rows.sort(key=lambda r: r["system"])

    raw_ps = [r["p_raw"] for r in rows]
    valid_idx = [i for i, p in enumerate(raw_ps) if not math.isnan(p)]
    if valid_idx:
        valid_ps = [raw_ps[i] for i in valid_idx]
        adj_ps = holm_corrected(valid_ps)
        for j, i in enumerate(valid_idx):
            rows[i]["p_holm"] = float(adj_ps[j])
            rows[i]["passes"] = bool(adj_ps[j] < 0.05)
    k = sum(1 for r in rows if r["passes"])

    # System-level sign test on per-system IQM-deltas: how many of the
    # per-system IQM-deltas point in the claim direction? Always test the
    # "more than half" alternative (greater).
    iqm_deltas = [r["iqm_delta"] for r in rows if not math.isnan(r["iqm_delta"])]
    if iqm_deltas:
        if alternative == "less":
            n_in_direction = sum(1 for m in iqm_deltas if m < 0)
        elif alternative == "greater":
            n_in_direction = sum(1 for m in iqm_deltas if m > 0)
        else:
            n_in_direction = sum(1 for m in iqm_deltas if m != 0)
        n_total = len(iqm_deltas)
        try:
            sign_p = float(stats.binomtest(n_in_direction, n_total, p=0.5, alternative="greater").pvalue)
        except Exception:
            sign_p = float("nan")
    else:
        n_in_direction = 0
        n_total = 0
        sign_p = float("nan")

    return {
        "K": int(k),
        "N": int(len(rows)),
        "sign_test_iqm": {
            "n_in_direction": int(n_in_direction),
            "n_total": int(n_total),
            "p_value": sign_p,
        },
        "per_system": rows,
    }


# ----------------------------------------------------------------------------
# TABLE 2: routing
# ----------------------------------------------------------------------------
def routing_censor_cap(df: pd.DataFrame, horizon: int) -> float:
    """Choose a finite log-scale sentinel beyond observed finite log-ratios."""
    route_col = f"h{horizon}_mean"
    global_col = f"global_h{horizon}_mean"
    finite_deltas = []
    for _, row in df.iterrows():
        route_mse = row[route_col]
        global_mse = row[global_col]
        if is_finite_positive(route_mse) and is_finite_positive(global_mse):
            finite_deltas.append(float(np.log10(route_mse) - np.log10(global_mse)))
    if not finite_deltas:
        return 12.0
    return float(math.ceil(max(12.0, max(abs(delta) for delta in finite_deltas))) + 1.0)


def censored_routing_delta(route_mse, global_mse, cap: float) -> tuple[float, str]:
    """Return log10(route/global), censoring invalid H-step comparisons.

    Finite catastrophic values are kept. A finite routed value with an invalid
    same-model global value is a censored routed win; the reverse is a censored
    loss. If both sides are invalid or a seed row is absent, the seed is neutral.
    """
    route_ok = is_finite_nonnegative(route_mse)
    global_ok = is_finite_nonnegative(global_mse)
    if route_ok and global_ok:
        route_val = float(route_mse)
        global_val = float(global_mse)
        if route_val > 0.0 and global_val > 0.0:
            delta = float(np.log10(route_val) - np.log10(global_val))
            return float(np.clip(delta, -cap, cap)), "finite_finite"
        if route_val == 0.0 and global_val > 0.0:
            return -cap, "zero_routed_good"
        if route_val > 0.0 and global_val == 0.0:
            return cap, "zero_global_bad_denominator"
        return 0.0, "both_zero_neutral"
    if route_ok and not global_ok:
        return -cap, "routed_finite_global_invalid_win"
    if not route_ok and global_ok:
        return cap, "routed_invalid_global_finite_loss"
    return 0.0, "both_invalid_neutral"


def per_system_censored_routing_wilcoxon(
    df: pd.DataFrame,
    horizon: int,
    expected_systems: list[str],
    expected_seeds: list[int],
    cap: float,
) -> dict:
    route_col = f"h{horizon}_mean"
    global_col = f"global_h{horizon}_mean"
    rows = []
    raw_ps = []
    all_deltas = []
    class_counts: dict[str, int] = defaultdict(int)

    for system in expected_systems:
        system_deltas = []
        seed_rows = []
        sys_df = df[df["system_key"] == system]
        for seed in expected_seeds:
            seed_df = sys_df[sys_df["seed"] == seed]
            if seed_df.empty:
                delta, reason = 0.0, "missing_row_neutral"
                route_mse = float("nan")
                global_mse = float("nan")
            else:
                row = seed_df.iloc[-1]
                route_mse = row[route_col]
                global_mse = row[global_col]
                delta, reason = censored_routing_delta(route_mse, global_mse, cap)
            class_counts[reason] += 1
            system_deltas.append(delta)
            all_deltas.append(delta)
            seed_rows.append({
                "seed": int(seed),
                "delta_log10": float(delta),
                "reason": reason,
                "route_mse": float(route_mse) if pd.notna(route_mse) else float("nan"),
                "global_mse": float(global_mse) if pd.notna(global_mse) else float("nan"),
            })

        deltas = np.asarray(system_deltas, dtype=float)
        if np.allclose(deltas, 0.0):
            p_raw = 1.0
            reason = "all_neutral"
        else:
            p_raw = float(stats.wilcoxon(deltas, alternative="less", zero_method="wilcox").pvalue)
            reason = ""
        raw_ps.append(p_raw)
        rows.append({
            "system": system,
            "n": int(len(deltas)),
            "median_delta": float(np.median(deltas)),
            "iqm_delta": iqm(deltas),
            "p_raw": p_raw,
            "p_holm": float("nan"),
            "passes": False,
            "reason": reason,
            "seed_rows": seed_rows,
        })

    adj_ps = holm_corrected(raw_ps)
    for row, p_holm in zip(rows, adj_ps):
        row["p_holm"] = float(p_holm)
        row["passes"] = bool(p_holm < 0.05)
    k = sum(1 for row in rows if row["passes"])

    iqm_deltas = [row["iqm_delta"] for row in rows if not math.isnan(row["iqm_delta"])]
    n_in_direction = sum(1 for delta in iqm_deltas if delta < 0.0)
    n_total = len(iqm_deltas)
    sign_p = (
        float(stats.binomtest(n_in_direction, n_total, p=0.5, alternative="greater").pvalue)
        if n_total
        else float("nan")
    )

    return {
        "K": int(k),
        "N": int(len(rows)),
        "censor_cap_log10": float(cap),
        "censor_class_counts": dict(class_counts),
        "censored_log10_cell": {
            "iqm_delta": iqm(all_deltas),
            "ratio_from_iqm_delta": float(10.0 ** iqm(all_deltas)),
        },
        "sign_test_iqm": {
            "n_in_direction": int(n_in_direction),
            "n_total": int(n_total),
            "p_value": sign_p,
        },
        "per_system": rows,
    }


def per_system_finite_routing_signflip(
    df: pd.DataFrame,
    horizon: int,
    expected_systems: list[str],
    n_boot_ci: int = 0,
) -> dict:
    """System-level Table 2 inference for mutually finite routed/global pairs."""
    route_col = f"h{horizon}_mean"
    global_col = f"global_h{horizon}_mean"
    rows = []
    system_deltas = []

    for system in expected_systems:
        grp = df[df["system_key"] == system]
        deltas = []
        for _, row in grp.iterrows():
            route_mse = row[route_col]
            global_mse = row[global_col]
            if is_finite_positive(route_mse) and is_finite_positive(global_mse):
                deltas.append(float(np.log10(route_mse) - np.log10(global_mse)))
        if deltas:
            system_delta = iqm(deltas)
            system_deltas.append(system_delta)
            rows.append({
                "system": system,
                "n_pairs": int(len(deltas)),
                "iqm_delta": float(system_delta),
                "ratio_from_iqm_delta": float(10.0 ** system_delta),
                "median_delta": float(np.median(deltas)),
            })
        else:
            rows.append({
                "system": system,
                "n_pairs": 0,
                "iqm_delta": float("nan"),
                "ratio_from_iqm_delta": float("nan"),
                "median_delta": float("nan"),
            })

    test = exact_signflip_mean_test(system_deltas, alternative="less")
    ci = bootstrap_iqm_ci(system_deltas, n_boot=n_boot_ci)
    return {
        **test,
        "p_holm": float("nan"),
        "ci95": ci,
        "per_system": rows,
    }


ROUTE_DISPLAY = {
    "support_gated_k": "Gated",
    "support_local_centered": "Support-local",
    "family_local_centered": "Family-local",
}


def tex_number(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    value = float(value)
    abs_value = abs(value)
    if abs_value == 0.0:
        return "0"
    if abs_value >= 1000.0 or abs_value < 1e-3:
        mantissa, exponent = f"{value:.2e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        return rf"{mantissa}\times10^{{{int(exponent)}}}"
    if abs_value < 0.1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.3g}"


def p_stars(p_value: float) -> str:
    if p_value is None or not math.isfinite(float(p_value)):
        return ""
    p_value = float(p_value)
    if p_value < 0.01:
        return r"^{\ast\ast}"
    if p_value < 0.05:
        return r"^{\ast}"
    return ""


def write_routing_main_table(summary_df: pd.DataFrame) -> None:
    """Write the main Table-2 fragment for support-family local routing."""
    if summary_df.empty:
        return
    main = summary_df[
        (summary_df["subset"] == "all")
        & (summary_df["route"] == ROUTING_MAIN_ROUTE)
    ].copy()
    rows_by_cell = {
        (str(row["horizon"]), str(row["model"])): row
        for row in main.to_dict(orient="records")
    }
    horizons = ["H100", "H1000"]

    best: dict[str, float] = {}
    for horizon in horizons:
        vals = [
            float(rows_by_cell[(horizon, display)]["finite_system_ratio_iqm"])
            for display in ROUTING_MODELS.values()
            if (horizon, display) in rows_by_cell
            and math.isfinite(float(rows_by_cell[(horizon, display)]["finite_system_ratio_iqm"]))
        ]
        best[horizon] = min(vals) if vals else float("nan")

    lines = [
        r"\begin{tabular}{@{}l cc cc@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{$H100$} & \multicolumn{2}{c}{$H1000$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
        r"Model & \(F_{\rm top8}\) ratio & system wins & \(F_{\rm top8}\) ratio & system wins \\",
        r"\midrule",
    ]
    for display in ROUTING_MODELS.values():
        cells = []
        has_data = False
        for horizon in horizons:
            row = rows_by_cell.get((horizon, display))
            if row is None:
                cells.extend(["--", "--"])
                continue
            has_data = True
            value = float(row["finite_system_ratio_iqm"])
            body = tex_number(value)
            if math.isfinite(value) and math.isclose(value, best[horizon], rel_tol=1e-12, abs_tol=1e-12):
                body = rf"\mathbf{{{body}}}"
            stars = p_stars(float(row["finite_system_p_holm"]))
            cells.append(rf"${{{body}}}{stars}$")
            cells.append(rf"${int(row['censored_system_n_in_direction'])}/{int(row['censored_system_n'])}$")
        if has_data:
            lines.append(f"{display} & {' & '.join(cells)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TBL_DIR / "table2_self_routing_h100_h1000.tex").write_text("\n".join(lines))


def plot_routing_summary(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return
    routes = ["support_gated_k", "support_local_centered", "family_local_centered"]
    labels = [ROUTE_DISPLAY[route] for route in routes]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharex=True)
    for ax, subset, title in zip(axes, ["all", "deep"], ["All states", "Deep states"]):
        sub = summary_df[(summary_df["horizon"] == "H1000") & (summary_df["subset"] == subset)]
        y_base = np.arange(len(ROUTING_MODELS))
        height = 0.22
        for offset, route, route_label in zip([-height, 0.0, height], routes, labels):
            vals = []
            colors = []
            for display in ROUTING_MODELS.values():
                row = sub[(sub["model"] == display) & (sub["route"] == route)]
                if row.empty:
                    vals.append(float("nan"))
                else:
                    vals.append(float(row.iloc[0]["finite_ratio_iqm"]))
                colors.append(PALETTE[display])
            clipped = [min(v, 2.0) if math.isfinite(v) else np.nan for v in vals]
            ax.barh(y_base + offset, clipped, height=height, color=colors, alpha=0.78, label=route_label)
            for y, raw, shown in zip(y_base + offset, vals, clipped):
                if not math.isfinite(raw):
                    continue
                text = f">{tex_number(raw)}" if raw > 2.0 else tex_number(raw)
                ax.text(min(shown + 0.03, 2.05), y, text, va="center", fontsize=6)
        ax.axvline(1.0, color="black", ls="--", lw=0.7)
        ax.set_yticks(y_base)
        ax.set_yticklabels(list(ROUTING_MODELS.values()))
        ax.set_xlim(0, 2.25)
        ax.set_xlabel("Routed/global MSE ratio at H1000")
        ax.set_title(title)
        ax.grid(axis="x", lw=0.3, alpha=0.35)
        ax.invert_yaxis()
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(TBL_DIR.parent / "fig_routing_ratios.pdf", bbox_inches="tight")
    fig.savefig(TBL_DIR.parent / "fig_routing_ratios.png", bbox_inches="tight")
    plt.close(fig)


def analyze_routing() -> dict:
    frames = [pd.read_csv(path, low_memory=False) for path in ROUTING_CSVS if path.exists()]
    if not frames:
        raise FileNotFoundError("No routing CSVs found")
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df[df["support_definition"] == "topk:8"].copy()
    df = df[df["root_label"].isin(ROUTING_MODELS)]
    df = filter_fixed_benchmark_systems(df)

    expected_systems = sorted(df["system_key"].dropna().unique().tolist())
    routes = ["support_gated_k", "support_local_centered", "family_local_centered"]
    # In the self-routed evaluator, q1 is the boundary quartile and q4 is the
    # deepest quartile by basin-depth margin.
    slices = {"all": "all", "deep": "q4"}

    global_cols = ["root_label", "system_key", "seed", "depth_stratum", "support_definition"]
    for horizon in ROUTING_HORIZONS:
        global_cols.append(f"h{horizon}_mean")
    global_df = df[df["rollout_mode"] == "global_k"][global_cols].rename(
        columns={f"h{horizon}_mean": f"global_h{horizon}_mean" for horizon in ROUTING_HORIZONS}
    )
    routed = df[df["rollout_mode"].isin(routes)].merge(
        global_df,
        on=["root_label", "system_key", "seed", "depth_stratum", "support_definition"],
        how="left",
    )

    out = {}
    summary_rows = []
    for horizon in ROUTING_HORIZONS:
        horizon_key = f"H{horizon}"
        ratio_col = f"h{horizon}_over_global"
        cap = routing_censor_cap(routed, horizon)
        out[horizon_key] = {}
        for root_label, display in ROUTING_MODELS.items():
            out[horizon_key][display] = {}
            for slice_name, depth_value in slices.items():
                out[horizon_key][display][slice_name] = {}
                for route in routes:
                    sub = routed[
                        (routed["root_label"] == root_label)
                        & (routed["depth_stratum"] == depth_value)
                        & (routed["rollout_mode"] == route)
                    ]
                    res = per_system_censored_routing_wilcoxon(
                        sub,
                        horizon=horizon,
                        expected_systems=expected_systems,
                        expected_seeds=ROUTING_EXPECTED_SEEDS,
                        cap=cap,
                    )
                    finite_system = per_system_finite_routing_signflip(
                        sub,
                        horizon=horizon,
                        expected_systems=expected_systems,
                    )
                    finite_ratios = pd.to_numeric(sub[ratio_col], errors="coerce")
                    finite_ratio_mask = np.isfinite(finite_ratios) & (finite_ratios > 0.0)
                    finite_ratios = finite_ratios[finite_ratio_mask]
                    finite_sub = sub.loc[finite_ratio_mask].copy()
                    finite_sub["_finite_ratio"] = finite_ratios.to_numpy()
                    res["cell"] = cell_summary(finite_ratios)
                    res["system_cell"] = system_iqm_summary(
                        finite_sub,
                        "_finite_ratio",
                        positive_only=True,
                    )
                    res["finite_system_signflip"] = finite_system
                    res["horizon"] = horizon
                    out[horizon_key][display][slice_name][route] = res
                    summary_rows.append({
                        "horizon": horizon_key,
                        "model": display,
                        "subset": slice_name,
                        "route": route,
                        "finite_ratio_mean": res["system_cell"]["mean"],
                        "finite_ratio_iqm": res["system_cell"]["mean"],
                        "finite_ratio_global_iqm": res["cell"]["iqm"],
                        "finite_ratio_n": res["cell"]["n"],
                        "finite_ratio_n_systems": res["system_cell"]["n_systems"],
                        "finite_system_log10_mean": finite_system["mean_delta"],
                        "finite_system_log10_iqm": finite_system["iqm_delta"],
                        "finite_system_ratio_mean": finite_system["ratio_from_mean_delta"],
                        "finite_system_ratio_iqm": finite_system["ratio_from_mean_delta"],
                        "finite_system_n": finite_system["n"],
                        "finite_system_n_in_direction": finite_system["n_in_direction"],
                        "finite_system_p_raw": finite_system["p_raw"],
                        "finite_system_ci95_low": finite_system["ci95"]["ratio_low"],
                        "finite_system_ci95_high": finite_system["ci95"]["ratio_high"],
                        "finite_system_p_holm": float("nan"),
                        "censored_log10_iqm_ratio": res["censored_log10_cell"]["ratio_from_iqm_delta"],
                        "censored_system_n_in_direction": res["sign_test_iqm"]["n_in_direction"],
                        "censored_system_n": res["sign_test_iqm"]["n_total"],
                        "censored_system_p_raw": res["sign_test_iqm"]["p_value"],
                        "K": res["K"],
                        "N": res["N"],
                        **{f"censor_{key}": value for key, value in res["censor_class_counts"].items()},
                    })

    summary_df = pd.DataFrame(summary_rows)
    main_mask = (
        (summary_df["subset"] == "all")
        & (summary_df["route"] == ROUTING_MAIN_ROUTE)
        & np.isfinite(pd.to_numeric(summary_df["finite_system_p_raw"], errors="coerce"))
    )
    main_indices = list(summary_df.index[main_mask])
    if main_indices:
        adjusted = holm_corrected(
            [float(summary_df.loc[idx, "finite_system_p_raw"]) for idx in main_indices]
        )
        for idx, p_holm in zip(main_indices, adjusted):
            summary_df.loc[idx, "finite_system_p_holm"] = p_holm
            horizon = str(summary_df.loc[idx, "horizon"])
            model = str(summary_df.loc[idx, "model"])
            route = str(summary_df.loc[idx, "route"])
            out[horizon][model]["all"][route]["finite_system_signflip"]["p_holm"] = float(p_holm)
    summary_df.to_csv(TBL_DIR / "table2_routing_h100_h1000_censored_seed15_summary.csv", index=False)
    write_routing_main_table(summary_df)
    plot_routing_summary(summary_df)
    return out


# ----------------------------------------------------------------------------
# TABLE 3: refresh
# ----------------------------------------------------------------------------


def _tex_number(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    value = float(value)
    abs_value = abs(value)
    if abs_value == 0.0:
        return "0"
    if abs_value >= 1000.0 or abs_value < 1e-3:
        mantissa, exponent = f"{value:.2e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        exponent_int = int(exponent)
        return rf"{mantissa}\times10^{{{exponent_int}}}"
    if abs_value < 0.1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.3g}"


def write_refresh_period_table(summary_df: pd.DataFrame) -> None:
    """Write a compact Table-3 tabular grouped by refresh period.

    The route-target cells are descriptive IQMs. The MSE-ratio cells mirror
    Table 2 style: point estimate followed by the per-system Wilcoxon/Holm
    count. IQRs remain in the CSV for appendix/provenance use.
    """
    if summary_df.empty:
        return

    rows_by_model_period = {
        (str(row["model"]), int(row["period"])): row
        for row in summary_df.to_dict(orient="records")
        if int(row.get("n_rows", 0) or 0) > 0
    }

    lines = [
        r"\begin{tabular}{@{}l cc cc@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Period $1$} & \multicolumn{2}{c}{Period $10$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
        r"Model & Target-route & MSE ratio & Target-route & MSE ratio \\",
        r"\midrule",
    ]

    for display in REFRESH_MODELS.values():
        cells = []
        has_data = False
        for period in (1, 10):
            row = rows_by_model_period.get((display, period))
            if row is None:
                cells.extend(["--", "--"])
                continue
            has_data = True
            route = row.get("route_target_iqm", float("nan"))
            ratio = row.get("mse_ratio_iqm", float("nan"))
            k = int(row.get("K", 0) or 0)
            n = int(row.get("N", 0) or 0)
            cells.append(rf"${_tex_number(route)}$")
            cells.append(rf"${_tex_number(ratio)}\,[{k}/{n}]$")
        if has_data:
            lines.append(f"{display} & {' & '.join(cells)} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TBL_DIR / "table3_support_refresh_period_grouped.tex").write_text("\n".join(lines))


def plot_refresh_summary(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for display in REFRESH_MODELS.values():
        sub = summary_df[(summary_df["model"] == display) & (summary_df["n_rows"] > 0)].sort_values("period")
        if sub.empty:
            continue
        color = PALETTE.get(display, "#666666")
        x = sub["period"].to_numpy(dtype=float)
        axes[0].plot(x, sub["route_target_iqm"], "-o", color=color, label=display, ms=4)
        axes[1].plot(x, sub["mse_ratio_iqm"], "-o", color=color, label=display, ms=4)
        axes[1].fill_between(
            x,
            sub["mse_ratio_q25_systems"].to_numpy(dtype=float),
            sub["mse_ratio_q75_systems"].to_numpy(dtype=float),
            color=color,
            alpha=0.12,
            lw=0,
        )
    axes[0].set_xlabel("Re-encode period (steps)")
    axes[0].set_ylabel("Target-route fraction (IQM)")
    axes[0].set_xscale("log")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, lw=0.35, alpha=0.35)
    axes[1].set_xlabel("Re-encode period (steps)")
    axes[1].set_ylabel("Refreshed/stale MSE ratio (IQM)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", lw=0.35, alpha=0.35)
    axes[1].legend(frameon=False, fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(TBL_DIR.parent / "fig_periodic_support_refresh.pdf", bbox_inches="tight")
    fig.savefig(TBL_DIR.parent / "fig_periodic_support_refresh.png", bbox_inches="tight")
    plt.close(fig)


def analyze_refresh() -> dict:
    frames = [pd.read_csv(path, low_memory=False) for path in REFRESH_CSVS if path.exists()]
    if not frames:
        raise FileNotFoundError("No refresh CSVs found")
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df[df["root_label"].isin(REFRESH_MODELS)].copy()
    df = filter_fixed_benchmark_systems(df)
    df = df[df["status"] == "ok"]
    df = df[df["support_definition"] == "topk:8"]
    df = df[df["object_kind"] == "support"]  # exact-support refresh; family is reported separately
    if "transfer_success" in df:
        df = df[df["transfer_success"] == True]  # noqa: E712
    if "start_mode" in df:
        df = df[df["start_mode"] == "post_start"]
    if "rollout_mode" in df:
        df = df[df["rollout_mode"] == "current_support_gated_periodic"]

    ratio_col = "refreshed_gated_mse_vs_frozen_source_gated_ratio"
    df = df.dropna(subset=[ratio_col]).copy()
    df = df[df[ratio_col] > 0]
    df["log10_ratio"] = np.log10(df[ratio_col].astype(float))

    expected_systems = sorted(df["system_key"].dropna().unique().tolist())

    out = {}
    summary_rows = []
    periods = [1.0, 10.0]
    for root_label, display in REFRESH_MODELS.items():
        out[display] = {}
        for period in periods:
            sub = df[(df["root_label"] == root_label) & (df["reencode_period"] == period)]
            res = per_system_paired_wilcoxon(
                sub,
                system_col="system_key",
                seed_col="seed",
                delta_col="log10_ratio",
                alternative="less",
                expected_systems=expected_systems,
            )
            res["cell"] = cell_summary(sub[ratio_col].astype(float))
            res["system_cell"] = system_iqm_summary(sub, ratio_col, positive_only=True)
            res["route_target_cell"] = system_iqm_summary(sub, "route_target_fraction")
            res["fallback_cell"] = system_iqm_summary(sub, "route_fallback_fraction")
            out[display][f"period_{int(period)}"] = res
            summary_rows.append({
                "model": display,
                "period": int(period),
                "route_target_iqm": res["route_target_cell"]["iqm"],
                "route_target_mean": res["route_target_cell"]["mean"],
                "route_target_std_systems": res["route_target_cell"]["std"],
                "fallback_iqm": res["fallback_cell"]["mean"],
                "fallback_mean": res["fallback_cell"]["mean"],
                "mse_ratio_iqm": res["system_cell"]["mean"],
                "mse_ratio_mean": res["system_cell"]["mean"],
                "mse_ratio_q25_systems": res["system_cell"]["q25"],
                "mse_ratio_q75_systems": res["system_cell"]["q75"],
                "mse_ratio_global_iqm": res["cell"]["iqm"],
                "n_systems": res["system_cell"]["n_systems"],
                "n_rows": res["cell"]["n"],
                "K": res["K"],
                "N": res["N"],
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TBL_DIR / "table3_support_refresh_matched_summary.csv", index=False)
    write_refresh_period_table(summary_df)
    plot_refresh_summary(summary_df)
    return out


# ----------------------------------------------------------------------------
# TABLE 4: Dysts long horizon
# ----------------------------------------------------------------------------
def analyze_dysts() -> dict:
    frames = [pd.read_csv(path, low_memory=False) for path in DYSTS_CSVS if path.exists()]
    if not frames:
        raise FileNotFoundError("No Dysts CSVs found")
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df[df["root_label"].isin(DYSTS_DISPLAY)].copy()

    horizons = [5000, 10000, 20000, 30000, 40000, 50000, 60000]
    horizon_cols = {h: f"h{h}_best_periodic_mean" for h in horizons}
    needed = list(horizon_cols.values())

    df = df.dropna(subset=needed, how="any")

    # For each horizon, baseline = Dense MLP rows (root=DYSTS_BASELINE).
    # Build per-(system, seed) candidate vs baseline log10-MSE difference.
    out = {}
    base = df[df["root_label"] == DYSTS_BASELINE]
    base_keys = base.set_index(["system_key", "seed"])

    for root_label, display in DYSTS_DISPLAY.items():
        if root_label == DYSTS_BASELINE:
            continue
        cand = df[df["root_label"] == root_label]
        cand_keys = cand.set_index(["system_key", "seed"])
        common_idx = cand_keys.index.intersection(base_keys.index)
        out[display] = {}
        for h, col in horizon_cols.items():
            cand_vals = cand_keys.loc[common_idx, col].astype(float)
            base_vals = base_keys.loc[common_idx, col].astype(float)
            # Use mean of duplicate (system, seed) entries if any
            cand_vals = cand_vals.groupby(level=[0, 1]).mean()
            base_vals = base_vals.groupby(level=[0, 1]).mean()
            common_idx_clean = cand_vals.index.intersection(base_vals.index)
            cand_vals = cand_vals.loc[common_idx_clean]
            base_vals = base_vals.loc[common_idx_clean]
            valid = (cand_vals > 0) & (base_vals > 0)
            cand_vals = cand_vals[valid]
            base_vals = base_vals[valid]
            paired = pd.DataFrame({
                "system_key": [idx[0] for idx in cand_vals.index],
                "seed": [idx[1] for idx in cand_vals.index],
                "log10_delta": np.log10(cand_vals.values) - np.log10(base_vals.values),
            })
            expected_systems = sorted(paired["system_key"].dropna().unique().tolist())
            res = per_system_paired_wilcoxon(
                paired,
                system_col="system_key",
                seed_col="seed",
                delta_col="log10_delta",
                alternative="less",
                expected_systems=expected_systems,
            )
            out[display][f"H{h}"] = res
    return out


def main():
    print("Reading routing rows from:")
    for path in ROUTING_CSVS:
        print(f"  - {path}")
    routing = analyze_routing()
    print("Reading refresh rows from:")
    for path in REFRESH_CSVS:
        print(f"  - {path}")
    refresh = analyze_refresh()
    print("Reading Dysts rows from:")
    for path in DYSTS_CSVS:
        print(f"  - {path}")
    dysts = analyze_dysts()

    summary = {
        "routing": routing,
        "refresh": refresh,
        "dysts": dysts,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {OUT_JSON}")

    def fmt(res):
        s = res["sign_test_iqm"]
        cell = res.get("cell")
        system_cell = res.get("system_cell")
        finite_system = res.get("finite_system_signflip")
        cell_str = ""
        if finite_system:
            p_holm = finite_system.get("p_holm", float("nan"))
            cell_str = (
                f" finite-system mean ratio={finite_system['ratio_from_mean_delta']:.3g} "
                f"(n={finite_system['n']}, p={finite_system['p_raw']:.2e}, "
                f"pHolm={p_holm:.2e})"
            )
        elif system_cell:
            cell_str = f" sys_mean={system_cell['mean']:.3g} (systems={system_cell['n_systems']})"
        elif cell:
            cell_str = f" iqm={cell['iqm']:.3g} (n={cell['n']})"
        return (f"K/N = {res['K']}/{res['N']}  "
                f"sign-iqm {s['n_in_direction']}/{s['n_total']} (p={s['p_value']:.2e})"
                f"{cell_str}")

    # Print human-readable summary
    print("\n=== TABLE 2 (Routing) — main display uses system-level finite log-ratio sign-flip/Holm; K/N is diagnostic ===")
    for horizon, models in routing.items():
        print(f"  {horizon}")
        for model, slices in models.items():
            for slice_name, routes in slices.items():
                for route, res in routes.items():
                    print(f"    {model:14s}  {slice_name:5s}  {route:25s}  {fmt(res)}")

    print("\n=== TABLE 3 (Refresh) - cell IQM of refreshed/previous-support ratio ===")
    for model, periods in refresh.items():
        for period_label, res in periods.items():
            print(f"  {model:10s}  {period_label:10s}  {fmt(res)}")

    print("\n=== TABLE 4 (Dysts vs Dense MLP) — system-level tests on per-system seed-IQMs ===")
    for model, horizons in dysts.items():
        for h_label, res in horizons.items():
            print(f"  {model:30s}  {h_label:6s}  {fmt(res)}")


if __name__ == "__main__":
    main()
