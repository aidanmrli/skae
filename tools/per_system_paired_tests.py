"""
Per-system paired Wilcoxon + Holm correction for paper Tables 2 (routing),
3 (support refresh), and 4 (Dysts long-horizon forecasting).

Mirrors the framework already used for Table 1: within each system, use the
N seeds as paired (candidate, baseline) values; run a one-sided Wilcoxon
signed-rank test; Holm-correct across the systems within each
(candidate, slice) cell; report K/N where K is the number of systems passing
Holm at alpha=0.05.

Outputs JSON to docs/figures/neurips_paper_2026/_tables/per_system_paired_tests.json
so the paper-side tables and prose can be updated by hand.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parent.parent
ROUTING_CSV = REPO_ROOT / "results" / "transition_rich_self_routed_forecasting_20260420" / "self_routed_forecasting_rows.csv"
REFRESH_CSV = REPO_ROOT / "results" / "periodic_support_refresh_fixed17_seed0_20260425" / "merged" / "periodic_support_refresh_rows.csv"
DYSTS_PRIMARY_CSV = REPO_ROOT / "results" / "dysts_long_horizon_eval_20260414" / "collect" / "forecasting_rows.csv"
DYSTS_MLPBD_CSV = REPO_ROOT / "results" / "dysts_long_horizon_eval_mlp_blockdiag_20260415" / "collect" / "forecasting_rows.csv"
OUT_JSON = REPO_ROOT / "docs" / "figures" / "neurips_paper_2026" / "_tables" / "per_system_paired_tests.json"

ROUTING_MODELS = {
    "lista_dense_softblock_signsplit_p64_hardinit_basin_partition": "LISTA-SB",
    "lista_blockdiag_signsplit_hardinit_basin_partition": "LISTA-BD",
    "mlp_zero_sparse_basin_partition_control": "Dense MLP",
}

REFRESH_MODELS = {
    "lista_dense_softblock_signsplit_p64_hardinit_basin_partition": "LISTA-SB",
    "lista_blockdiag_signsplit_hardinit_basin_partition": "LISTA-BD",
}

DYSTS_BASELINE = "generic_sparse_sc0_ns200k_best"
DYSTS_DISPLAY = {
    "lista_dense_promoted_stage4": "LISTA-D",
    "lista_blockdiag_ns200k_denseopt_sc3em3": "LISTA-BD (low sp)",
    "lista_blockdiag_ns200k_denseopt_sc6em3": "LISTA-BD (high sp)",
    "generic_sparse_ns200k_best": "Sparse MLP",
    "generic_sparse_blockdiag_ns200k_sc3em3": "Sparse MLP, BD (low sp)",
    "generic_sparse_blockdiag_ns200k_sc6em3": "Sparse MLP, BD (high sp)",
    "generic_sparse_sc0_ns200k_best": "Dense MLP",
}


def iqm(values) -> float:
    """Interquartile mean: mean of values within [25th, 75th] percentile.

    Falls back to plain mean when n<4 (IQR is degenerate).
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
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


def cell_summary(values) -> dict:
    """Cross-rollout summary of a cell: IQM, median, IQR endpoints, n."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
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
def analyze_routing() -> dict:
    df = pd.read_csv(ROUTING_CSV)
    df = df[df["support_definition"] == "topk:8"].copy()
    df = df[df["root_label"].isin(ROUTING_MODELS)]

    df["log10_ratio"] = np.log10(df["h1000_over_global"].astype(float))

    expected_systems = sorted(df["system_key"].dropna().unique().tolist())

    out = {}
    routes = ["support_gated_k", "support_local_centered", "family_local_centered"]
    slices = {"all": "all", "deep": "q1"}

    for root_label, display in ROUTING_MODELS.items():
        out[display] = {}
        for slice_name, depth_value in slices.items():
            out[display][slice_name] = {}
            for route in routes:
                sub = df[
                    (df["root_label"] == root_label)
                    & (df["depth_stratum"] == depth_value)
                    & (df["rollout_mode"] == route)
                ].dropna(subset=["log10_ratio"])
                res = per_system_paired_wilcoxon(
                    sub,
                    system_col="system_key",
                    seed_col="seed",
                    delta_col="log10_ratio",
                    alternative="less",
                    expected_systems=expected_systems,
                )
                res["cell"] = cell_summary(sub["h1000_over_global"].astype(float))
                out[display][slice_name][route] = res
    return out


# ----------------------------------------------------------------------------
# TABLE 3: refresh
# ----------------------------------------------------------------------------
def analyze_refresh() -> dict:
    df = pd.read_csv(REFRESH_CSV)
    df = df[df["root_label"].isin(REFRESH_MODELS)].copy()
    df = df[df["status"] == "ok"]
    df = df[df["support_definition"] == "topk:8"]
    df = df[df["object_kind"] == "support"]  # exact-support refresh; family is reported separately

    ratio_col = "refreshed_gated_mse_vs_frozen_source_gated_ratio"
    df = df.dropna(subset=[ratio_col]).copy()
    df = df[df[ratio_col] > 0]
    df["log10_ratio"] = np.log10(df[ratio_col].astype(float))

    expected_systems = sorted(df["system_key"].dropna().unique().tolist())

    out = {}
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
            out[display][f"period_{int(period)}"] = res
    return out


# ----------------------------------------------------------------------------
# TABLE 4: Dysts long horizon
# ----------------------------------------------------------------------------
def analyze_dysts() -> dict:
    primary = pd.read_csv(DYSTS_PRIMARY_CSV)
    supp = pd.read_csv(DYSTS_MLPBD_CSV)
    df = pd.concat([primary, supp], ignore_index=True, sort=False)
    df = df[df["root_label"].isin(DYSTS_DISPLAY)].copy()

    horizons = [5000, 10000, 20000, 30000]
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
    print(f"Reading routing rows from {ROUTING_CSV}")
    routing = analyze_routing()
    print(f"Reading refresh rows from {REFRESH_CSV}")
    refresh = analyze_refresh()
    print(f"Reading Dysts rows from {DYSTS_PRIMARY_CSV} + {DYSTS_MLPBD_CSV}")
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
        cell_str = ""
        if cell:
            cell_str = f" iqm={cell['iqm']:.3g} (n={cell['n']})"
        return (f"K/N = {res['K']}/{res['N']}  "
                f"sign-iqm {s['n_in_direction']}/{s['n_total']} (p={s['p_value']:.2e})"
                f"{cell_str}")

    # Print human-readable summary
    print("\n=== TABLE 2 (Routing) — cell IQM of routed/global ratio ===")
    for model, slices in routing.items():
        for slice_name, routes in slices.items():
            for route, res in routes.items():
                print(f"  {model:12s}  {slice_name:5s}  {route:25s}  {fmt(res)}")

    print("\n=== TABLE 3 (Refresh) — cell IQM of refreshed/frozen ratio ===")
    for model, periods in refresh.items():
        for period_label, res in periods.items():
            print(f"  {model:10s}  {period_label:10s}  {fmt(res)}")

    print("\n=== TABLE 4 (Dysts vs Dense MLP) — sign-iqm on per-system IQM-deltas ===")
    for model, horizons in dysts.items():
        for h_label, res in horizons.items():
            print(f"  {model:30s}  {h_label:6s}  {fmt(res)}")


if __name__ == "__main__":
    main()
