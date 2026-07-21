"""Rebuild the submitted two-panel forecasting-horizon figure.

Each system first contributes one seed IQM.  The displayed point is then the
arithmetic mean across every retained system; systems are never trimmed.  The
bands reproduce the submitted figure's fixed-system, log-relative seed
bootstrap and therefore describe finite-seed uncertainty, not cross-system
heterogeneity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.neurips_2026.evidence.forecasting_horizon_rendering import (
    render_composite,
    render_panel,
)
from experiments.neurips_2026.evidence.statistics import (
    interquartile_mean,
    rowwise_interquartile_mean,
)
from experiments.neurips_2026.paths import (
    PAPER_DATA_DIR,
    PAPER_EVIDENCE_DIR,
    PAPER_TABLE_DIR,
)
from experiments.neurips_2026.protocol import (
    CONTROLLED_PAPER_PROTOCOL,
    DYSTS_PAPER_PROTOCOL,
    PAPER_MODEL_ROWS,
    canonical_controlled_system_key,
)


CONTROLLED_INPUT = PAPER_DATA_DIR / "controlled_forecasting_rows.csv"
DYSTS_INPUT = PAPER_DATA_DIR / "dysts_forecasting_rows.csv"
PROVENANCE = PAPER_DATA_DIR / "main_paper_evidence_provenance.json"
SUMMARY_PATH = PAPER_TABLE_DIR / "figure2_forecasting_horizon_summary.csv"
CONTROLLED_PANEL_PATH = PAPER_EVIDENCE_DIR / "fig_fixed17_horizon_curves.pdf"
DYSTS_PANEL_PATH = (
    PAPER_EVIDENCE_DIR
    / "fig_dysts_dt30_forecasting_performance.pdf"
)
DYSTS_PANEL_NO_LISTA_SB_PATH = (
    PAPER_EVIDENCE_DIR
    / "fig_dysts_dt30_forecasting_performance_no_lista_sb.pdf"
)
COMPOSITE_PATH = PAPER_EVIDENCE_DIR / "fig_forecasting_horizon_trends.pdf"
OUTPUT_PATHS = (
    SUMMARY_PATH,
    CONTROLLED_PANEL_PATH,
    DYSTS_PANEL_PATH,
    DYSTS_PANEL_NO_LISTA_SB_PATH,
    COMPOSITE_PATH,
)

CONTROLLED_HORIZONS = (100, 500, 1000)
DYSTS_HORIZONS = (100, 500, 1000, 1500, 2000, 3000, 4000, 5000)
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20_260_501
CSV_FLOAT_FORMAT = "%.13g"

PALETTE = {
    "LISTA": "#7B3294",
    "LISTA-BD": "#0072B2",
    "LISTA-SB": "#56B4E9",
    "Sparse MLP, BD": "#44AA99",
    "Sparse MLP": "#009E73",
    "Dense MLP": "#D55E00",
}


def _method_styles(benchmark: str) -> OrderedDict[str, tuple[str, str, str]]:
    styles: OrderedDict[str, tuple[str, str, str]] = OrderedDict()
    for row in PAPER_MODEL_ROWS:
        root_label = (
            row.controlled_variant if benchmark == "controlled" else row.dysts_variant
        )
        line_style = "-" if row.display_name.startswith("LISTA") else "--"
        styles[root_label] = (
            row.display_name,
            PALETTE[row.display_name],
            line_style,
        )
    return styles


CONTROLLED_METHODS = _method_styles("controlled")
DYSTS_METHODS = _method_styles("dysts")
DYSTS_METHODS_NO_LISTA_SB = OrderedDict(
    (root_label, style)
    for root_label, style in DYSTS_METHODS.items()
    if style[0] != "LISTA-SB"
)


def verify_frozen_inputs(
    controlled_path: Path,
    dysts_path: Path,
    provenance_path: Path,
) -> None:
    """Authenticate both compact row files against the frozen manifest."""

    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        output_specs = provenance["outputs"]
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Missing valid provenance: {provenance_path}") from error
    for path in (controlled_path, dysts_path):
        try:
            expected = output_specs[path.name]["sha256"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"No frozen provenance entry for {path.name}") from error
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Frozen evidence hash mismatch for {path.name}: "
                f"expected {expected}, got {actual}"
            )


def _load_rows(
    path: Path,
    *,
    system_column: str,
    expected_methods: tuple[str, ...],
    expected_systems: tuple[str, ...],
    expected_seeds: tuple[int, ...],
    horizons: tuple[int, ...],
    normalize_controlled_systems: bool = False,
) -> pd.DataFrame:
    rows = pd.read_csv(path, low_memory=False)
    required = {"root_label", system_column, "seed"}
    required.update(f"h{horizon}_best_periodic_mean" for horizon in horizons)
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if "status" in rows:
        rows = rows[rows["status"] == "complete"].copy()
    if normalize_controlled_systems:
        rows[system_column] = rows[system_column].map(
            canonical_controlled_system_key
        )
    rows["seed"] = pd.to_numeric(rows["seed"], errors="raise").astype(int)
    if set(rows["root_label"]) != set(expected_methods):
        raise ValueError(f"{path.name} does not contain exactly the paper methods")
    if set(rows[system_column]) != set(expected_systems):
        raise ValueError(f"{path.name} does not contain exactly the paper systems")
    if set(rows["seed"]) != set(expected_seeds):
        raise ValueError(f"{path.name} does not contain exactly the paper seeds")
    keys = ["root_label", system_column, "seed"]
    expected_rows = len(expected_methods) * len(expected_systems) * len(expected_seeds)
    if rows.duplicated(keys).any() or len(rows) != expected_rows:
        raise ValueError(f"{path.name} must have one row per method/system/seed")
    return rows


def load_controlled_rows(path: Path = CONTROLLED_INPUT) -> pd.DataFrame:
    return _load_rows(
        path,
        system_column="system_name",
        expected_methods=tuple(CONTROLLED_METHODS),
        expected_systems=CONTROLLED_PAPER_PROTOCOL.system_keys,
        expected_seeds=CONTROLLED_PAPER_PROTOCOL.seeds,
        horizons=CONTROLLED_HORIZONS,
        normalize_controlled_systems=True,
    )


def load_dysts_rows(path: Path = DYSTS_INPUT) -> pd.DataFrame:
    return _load_rows(
        path,
        system_column="system_key",
        expected_methods=tuple(DYSTS_METHODS),
        expected_systems=DYSTS_PAPER_PROTOCOL.system_keys,
        expected_seeds=DYSTS_PAPER_PROTOCOL.seeds,
        horizons=DYSTS_HORIZONS,
    )


def _log_relative_seed_interval(
    values_by_system: list[np.ndarray],
    *,
    center: float,
    rng: np.random.Generator,
    bootstrap_reps: int,
) -> tuple[float, float]:
    relative_draws = []
    for values in values_by_system:
        system_center = interquartile_mean(values)
        indices = rng.integers(
            0,
            values.size,
            size=(bootstrap_reps, values.size),
        )
        draws = rowwise_interquartile_mean(values[indices])
        draws = np.clip(draws, np.finfo(float).tiny, None)
        relative_draws.append(np.log10(draws) - math.log10(system_center))
    mean_log_relative = np.mean(np.column_stack(relative_draws), axis=1)
    low, high = np.percentile(mean_log_relative, [2.5, 97.5])
    return float(center * 10.0**low), float(center * 10.0**high)


def summarize_benchmark(
    rows: pd.DataFrame,
    *,
    benchmark: str,
    system_column: str,
    expected_systems: tuple[str, ...],
    method_styles: OrderedDict[str, tuple[str, str, str]],
    horizons: tuple[int, ...],
    bootstrap_reps: int = BOOTSTRAP_REPS,
) -> pd.DataFrame:
    """Summarize seeds within system, then average all systems without trimming."""

    if bootstrap_reps <= 0:
        raise ValueError("bootstrap_reps must be positive")
    records: list[dict[str, object]] = []
    benchmark_seed = 0 if benchmark == "controlled" else 1
    for method_index, (root_label, (display, _, _)) in enumerate(
        method_styles.items()
    ):
        method_rows = rows[rows["root_label"] == root_label]
        for horizon in horizons:
            column = f"h{horizon}_best_periodic_mean"
            values_by_system = []
            for system in expected_systems:
                values = pd.to_numeric(
                    method_rows[method_rows[system_column] == system][column],
                    errors="coerce",
                ).to_numpy(dtype=float)
                values = values[np.isfinite(values) & (values > 0.0)]
                if values.size == 0:
                    raise ValueError(
                        f"No finite positive {benchmark} values for "
                        f"{root_label}/{system}/H{horizon}"
                    )
                values_by_system.append(values)
            system_iqms = np.asarray(
                [interquartile_mean(values) for values in values_by_system]
            )
            center = float(np.mean(system_iqms))
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [BOOTSTRAP_SEED, benchmark_seed, method_index, horizon]
                )
            )
            low, high = _log_relative_seed_interval(
                values_by_system,
                center=center,
                rng=rng,
                bootstrap_reps=bootstrap_reps,
            )
            records.append(
                {
                    "benchmark": benchmark,
                    "root_label": root_label,
                    "display": display,
                    "horizon": horizon,
                    "n_systems": len(system_iqms),
                    "min_finite_seeds_per_system": min(map(len, values_by_system)),
                    "max_finite_seeds_per_system": max(map(len, values_by_system)),
                    "mean_over_system_seed_iqms": center,
                    "log_relative_seed_bootstrap_ci95_low": low,
                    "log_relative_seed_bootstrap_ci95_high": high,
                    "bootstrap_reps": bootstrap_reps,
                }
            )
    return pd.DataFrame(records)


def build_outputs(
    controlled_path: Path = CONTROLLED_INPUT,
    dysts_path: Path = DYSTS_INPUT,
    provenance_path: Path = PROVENANCE,
    *,
    bootstrap_reps: int = BOOTSTRAP_REPS,
) -> dict[Path, bytes]:
    verify_frozen_inputs(controlled_path, dysts_path, provenance_path)
    controlled = summarize_benchmark(
        load_controlled_rows(controlled_path),
        benchmark="controlled",
        system_column="system_name",
        expected_systems=CONTROLLED_PAPER_PROTOCOL.system_keys,
        method_styles=CONTROLLED_METHODS,
        horizons=CONTROLLED_HORIZONS,
        bootstrap_reps=bootstrap_reps,
    )
    dysts = summarize_benchmark(
        load_dysts_rows(dysts_path),
        benchmark="dysts_dt30",
        system_column="system_key",
        expected_systems=DYSTS_PAPER_PROTOCOL.system_keys,
        method_styles=DYSTS_METHODS,
        horizons=DYSTS_HORIZONS,
        bootstrap_reps=bootstrap_reps,
    )
    summary = pd.concat([controlled, dysts], ignore_index=True)
    contents = (
        summary.to_csv(
            index=False,
            lineterminator="\n",
            float_format=CSV_FLOAT_FORMAT,
        ).encode(),
        render_panel(
            controlled,
            CONTROLLED_METHODS,
            CONTROLLED_HORIZONS,
            title="15-system multibasin forecasting performance",
        ),
        render_panel(
            dysts,
            DYSTS_METHODS,
            DYSTS_HORIZONS,
            title=r"10-system Dysts $dt{\times}30$ forecasting performance",
        ),
        render_panel(
            dysts,
            DYSTS_METHODS_NO_LISTA_SB,
            DYSTS_HORIZONS,
            title=r"10-system Dysts $dt{\times}30$ forecasting performance",
        ),
        render_composite(
            controlled,
            dysts,
            CONTROLLED_METHODS,
            DYSTS_METHODS,
            CONTROLLED_HORIZONS,
            DYSTS_HORIZONS,
        ),
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
            raise RuntimeError(
                "Forecasting-horizon artifacts are stale: "
                + ", ".join(map(str, stale))
            )
        print(f"Verified {len(outputs)} forecasting-horizon artifacts")
        return
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlled-input", type=Path, default=CONTROLLED_INPUT)
    parser.add_argument("--dysts-input", type=Path, default=DYSTS_INPUT)
    parser.add_argument("--provenance", type=Path, default=PROVENANCE)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_outputs(
        args.controlled_input,
        args.dysts_input,
        args.provenance,
    )
    write_or_check(outputs, check=args.check)


if __name__ == "__main__":
    main()
