"""Build the compact paper-facing forecasting/support tables.

This builder keeps the existing Table 1 roster but replaces the stale
controlled Sparse MLP-BD packet with the repaired block-diagonal GenericKM
artifacts under transition_rich_sparse_mlp_bd_repaired_table1_20260506.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from skae.benchmarks.paper_protocol import (
    CONTROLLED_ALIGNMENT_EXCLUDED_OBSERVED_LABEL_COUNTS,
    CONTROLLED_ALIGNMENT_EXCLUDED_SYSTEM_KEYS,
    CONTROLLED_ALIGNMENT_PRIMARY_SYSTEM_KEYS,
    canonical_controlled_system_key,
)
from skae.benchmarks.paper_statistics import interquartile_mean as iqm
from tools.paper_table_rendering import write_tables


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "docs" / "figures" / "neurips_paper_2026" / "_data"
TABLE_DIR = ROOT / "docs" / "figures" / "neurips_paper_2026" / "_tables"
CONTROL_FORECAST_CSV = DATA_DIR / "controlled_forecasting_rows.csv"
CONTROL_INTERP_CSV = DATA_DIR / "controlled_support_rows.csv"
FROZEN_PROVENANCE_JSON = DATA_DIR / "main_paper_evidence_provenance.json"

REPAIRED_CONTROL_ROOT = "mlp_sparse_blockdiag_hardinit_basin_partition_control"
BASELINE_CONTROL_ROOT = "mlp_zero_sparse_hardinit_basin_partition_control"
BASELINE_DYSTS_ROOT = "dense_mlp_tanh"

CONTROL_HORIZONS = (100, 500, 1000)
DYSTS_HORIZONS = (100, 2000, 4000)
ALPHA = 0.05
ALIGNMENT_SENSITIVITY_FILENAME = "controlled_support_alignment_sensitivity.csv"

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


def verify_frozen_evidence() -> None:
    require_file(FROZEN_PROVENANCE_JSON)
    provenance = json.loads(FROZEN_PROVENANCE_JSON.read_text(encoding="utf-8"))
    output_specs = provenance.get("outputs", {})
    for path in (CONTROL_FORECAST_CSV, CONTROL_INTERP_CSV):
        require_file(path)
        spec = output_specs.get(path.name)
        if not isinstance(spec, dict) or "sha256" not in spec:
            raise ValueError(f"Missing provenance entry for {path.name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != spec["sha256"]:
            raise ValueError(
                f"Frozen evidence hash mismatch for {path.name}: "
                f"expected {spec['sha256']}, got {actual}"
            )


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


def load_control_forecasting() -> pd.DataFrame:
    require_file(CONTROL_FORECAST_CSV)
    df = pd.read_csv(CONTROL_FORECAST_CSV, low_memory=False)
    df = df[df["root_label"].isin(CONTROL_ROOTS)].copy()
    for col in ("system_name", "system_key", "train_env_name"):
        if col in df:
            df = df[~df[col].isin(EXCLUDED_SYSTEMS)].copy()
    return df


def load_control_interpretability() -> pd.DataFrame:
    require_file(CONTROL_INTERP_CSV)
    df = pd.read_csv(CONTROL_INTERP_CSV, low_memory=False)
    df = df[df["root_label"].isin(CONTROL_ROOTS)].copy()
    for col in ("system_name", "system_key", "train_env_name"):
        if col in df:
            df = df[~df[col].isin(EXCLUDED_SYSTEMS)].copy()
    return df[
        (df["support_scheme"] == "absolute:0.001")
        & (df["subset"] == "deep")
        & (pd.to_numeric(df["family_jaccard_threshold"], errors="coerce") == 0.5)
    ].copy()


def primary_alignment_rows(frame: pd.DataFrame) -> pd.DataFrame:
    canonical = frame["system_name"].map(canonical_controlled_system_key)
    observed_counts = pd.to_numeric(frame["observed_label_count"], errors="raise")
    primary = frame[observed_counts >= 2].copy()
    observed = set(primary["system_name"].map(canonical_controlled_system_key))
    if observed != set(CONTROLLED_ALIGNMENT_PRIMARY_SYSTEM_KEYS):
        raise ValueError(f"Primary alignment roster mismatch: {sorted(observed)}")
    excluded = frame[canonical.isin(CONTROLLED_ALIGNMENT_EXCLUDED_SYSTEM_KEYS)]
    excluded_counts = set(
        pd.to_numeric(excluded["observed_label_count"], errors="raise").astype(int)
    )
    if excluded_counts != {1}:
        raise ValueError(f"Unexpected excluded alignment label counts: {excluded_counts}")
    entropy = pd.to_numeric(
        excluded["family_h_basin_given_family"], errors="coerce"
    ).to_numpy(dtype=float)
    if entropy.size == 0 or not np.all(np.isfinite(entropy) & (entropy == 0.0)):
        raise ValueError("Frozen single-label alignment exclusion no longer has zero entropy")
    return primary


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
        metric_map = {"HBgivenF": "family_h_basin_given_family"}
        for key, col in metric_map.items():
            row[key] = finite_mean(sub_itp.groupby("system_name")[col].apply(iqm))
        row["FamilyUniqueCount"] = finite_mean(
            sub_itp.groupby("system_name")["family_unique_count"].mean()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_alignment_sensitivity(
    primary: pd.DataFrame, all_systems: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    excluded_system = CONTROLLED_ALIGNMENT_EXCLUDED_SYSTEM_KEYS[0]
    for root_label, meta in CONTROL_ROOTS.items():
        row: dict[str, float | int | str] = {
            "root_label": root_label,
            "model": meta.get("plain_label", meta["label"]),
            "primary_system_count": len(CONTROLLED_ALIGNMENT_PRIMARY_SYSTEM_KEYS),
            "all_system_count": len(CONTROLLED_ALIGNMENT_PRIMARY_SYSTEM_KEYS)
            + len(CONTROLLED_ALIGNMENT_EXCLUDED_SYSTEM_KEYS),
            "excluded_system": excluded_system,
            "excluded_observed_label_count": (
                CONTROLLED_ALIGNMENT_EXCLUDED_OBSERVED_LABEL_COUNTS[excluded_system]
            ),
        }
        for prefix, frame in (("primary", primary), ("all_system", all_systems)):
            subset = frame[frame["root_label"] == root_label]
            row[f"{prefix}_h_basin_given_family"] = finite_mean(
                subset.groupby("system_name")["family_h_basin_given_family"].apply(iqm)
            )
            row[f"{prefix}_mean_observed_family_count"] = finite_mean(
                subset.groupby("system_name")["family_unique_count"].mean()
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
    return sig


GENERATED_FILENAMES = (
    "table1_forecasting_multibasin_dysts.tex",
    "table2_support_alignment.tex",
    ALIGNMENT_SENSITIVITY_FILENAME,
)


def build_artifacts(output_dir: Path) -> pd.Series:
    fc = load_control_forecasting()
    all_itp = load_control_interpretability()
    primary_itp = primary_alignment_rows(all_itp)
    control = build_control_summary(fc, primary_itp)
    sig = build_control_significance(fc, primary_itp)
    sensitivity = build_alignment_sensitivity(primary_itp, all_itp)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_tables(control, sig, output_dir=output_dir, table_dir=TABLE_DIR)
    sensitivity.to_csv(
        output_dir / ALIGNMENT_SENSITIVITY_FILENAME,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )

    repaired = control[control["root"] == REPAIRED_CONTROL_ROOT].iloc[0]
    return repaired


def check_artifacts() -> None:
    with tempfile.TemporaryDirectory(prefix="skae-table1-check-") as tmp:
        generated_dir = Path(tmp)
        build_artifacts(generated_dir)
        mismatches: list[str] = []
        for name in GENERATED_FILENAMES:
            tracked = TABLE_DIR / name
            generated = generated_dir / name
            if not tracked.is_file() or tracked.read_bytes() != generated.read_bytes():
                mismatches.append(name)
        if mismatches:
            raise SystemExit(f"Headline table artifacts are stale: {', '.join(mismatches)}")
    print("Headline table artifacts are byte-identical to a clean rebuild.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Rebuild in a temporary directory and fail if tracked outputs differ.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_frozen_evidence()
    if args.check:
        check_artifacts()
        return
    repaired = build_artifacts(TABLE_DIR)
    print("Rebuilt compact forecasting and support-alignment tables.")
    print(
        "Repaired Sparse MLP-BD controlled values: "
        f"H100={repaired['H100']:.6g}, "
        f"H500={repaired['H500']:.6g}, "
        f"H1000={repaired['H1000']:.6g}, "
        f"H(B|F)={repaired['HBgivenF']:.6g}, "
        f"|F|={repaired['FamilyUniqueCount']:.6g}"
    )
    print(f"Rows in control summary: {len(CONTROL_ROOTS)}")


if __name__ == "__main__":
    main()
