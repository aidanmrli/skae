"""Make the ManiSkill H125 forecasting figure from completed eval JSONs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SETTINGS = (
    "dense_tanh_sp0",
    "sparse_mlp_sp0p003",
    "lista_a0p03_sp0p003",
    "lista_a0p03_sp0p01",
)

DISPLAY_LABELS = {
    "dense_tanh_sp0": "Dense tanh",
    "dense_sp0": "Dense ReLU (legacy)",
    "sparse_mlp_sp0p003": "Sparse MLP, l1=0.003",
    "lista_a0p03_sp0p003": "LISTA, alpha=0.03, l1=0.003",
    "lista_a0p03_sp0p01": "LISTA, alpha=0.03, l1=0.01",
}

COLORS = {
    "dense_tanh_sp0": "#4C78A8",
    "dense_sp0": "#4C78A8",
    "sparse_mlp_sp0p003": "#F58518",
    "lista_a0p03_sp0p003": "#54A24B",
    "lista_a0p03_sp0p01": "#B279A2",
}

MARKERS = {
    "dense_tanh_sp0": "o",
    "dense_sp0": "o",
    "sparse_mlp_sp0p003": "s",
    "lista_a0p03_sp0p003": "^",
    "lista_a0p03_sp0p01": "D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("runs/maniskill_insertion/perturbation_e20_50k_long_eval_20260603"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/figures/neurips_paper_2026"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/maniskill_h125_forecast_20260603"),
    )
    parser.add_argument("--horizons", default="10,25,50,100,125")
    parser.add_argument("--figure-stem", default="fig_maniskill_h125_forecasting")
    parser.add_argument("--settings", default=",".join(DEFAULT_SETTINGS))
    parser.add_argument(
        "--rollout-key",
        default="rollout",
        help="Metrics key to plot, e.g. rollout or best_periodic_rollout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = [int(item) for item in args.horizons.split(",") if item.strip()]
    settings = [item.strip() for item in args.settings.split(",") if item.strip()]
    per_seed_rows = collect_rows(
        args.input_root,
        horizons,
        settings=settings,
        rollout_key=args.rollout_key,
    )
    summary_rows = summarize_rows(per_seed_rows, horizons, settings=settings)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    per_seed_csv = args.results_dir / "maniskill_h125_forecast_per_seed.csv"
    summary_csv = args.results_dir / "maniskill_h125_forecast_summary.csv"
    write_csv(per_seed_csv, per_seed_rows)
    write_csv(summary_csv, summary_rows)

    make_figure(
        summary_rows,
        horizons,
        args.output_dir / args.figure_stem,
        settings=settings,
        rollout_key=args.rollout_key,
    )

    episode_counts = [int(row["episode_count"]) for row in per_seed_rows]
    if min(episode_counts) == max(episode_counts):
        eligible_test_episodes = f"{min(episode_counts)} per seed for all plotted horizons"
    else:
        eligible_test_episodes = (
            f"{min(episode_counts)}-{max(episode_counts)} per seed across plotted horizons; "
            "see summary CSV episode_count_min/episode_count_max"
        )

    metadata = {
        "input_root": str(args.input_root),
        "output_dir": str(args.output_dir),
        "results_dir": str(args.results_dir),
        "horizons": horizons,
        "rollout_key": args.rollout_key,
        "settings": {setting: DISPLAY_LABELS.get(setting, setting) for setting in settings},
        "per_seed_csv": str(per_seed_csv),
        "summary_csv": str(summary_csv),
        "figure_pdf": str(args.output_dir / f"{args.figure_stem}.pdf"),
        "figure_png": str(args.output_dir / f"{args.figure_stem}.png"),
        "error_bar": "standard error over 3 seeds",
        "rollout_metric": "held-out state MSE",
        "eligible_test_episodes": eligible_test_episodes,
    }
    with (args.results_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps(metadata, indent=2, sort_keys=True))


def collect_rows(
    input_root: Path,
    horizons: Iterable[int],
    *,
    settings: Iterable[str],
    rollout_key: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    missing: List[str] = []
    missing_key: List[str] = []
    for setting in settings:
        for seed in (0, 1, 2):
            path = input_root / setting / f"seed{seed}" / "eval_test_long" / "metrics_summary.json"
            if not path.exists():
                missing.append(str(path))
                continue
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            rollout = payload.get(rollout_key)
            if rollout is None:
                missing_key.append(f"{path}: {rollout_key}")
                continue
            for horizon in horizons:
                key = f"h{horizon}/state_mse"
                episode_key = f"h{horizon}/episode_count"
                rows.append(
                    {
                        "setting": setting,
                        "label": DISPLAY_LABELS.get(setting, setting),
                        "seed": seed,
                        "horizon": horizon,
                        "state_mse": float(rollout[key]),
                        "episode_count": int(rollout[episode_key]),
                        "selected_mode": rollout.get(f"h{horizon}/selected_mode", ""),
                        "selected_period": rollout.get(f"h{horizon}/selected_period", ""),
                    }
                )
    if missing:
        raise FileNotFoundError("Missing eval summaries:\n" + "\n".join(missing))
    if missing_key:
        raise KeyError("Missing rollout metric keys:\n" + "\n".join(missing_key))
    return rows


def summarize_rows(
    per_seed_rows: Iterable[Mapping[str, object]],
    horizons: Iterable[int],
    *,
    settings: Iterable[str],
) -> List[Dict[str, object]]:
    by_key: Dict[tuple[str, int], List[float]] = {}
    episode_counts: Dict[tuple[str, int], List[int]] = {}
    selected_modes: Dict[tuple[str, int], Counter[str]] = {}
    for row in per_seed_rows:
        key = (str(row["setting"]), int(row["horizon"]))
        by_key.setdefault(key, []).append(float(row["state_mse"]))
        episode_counts.setdefault(key, []).append(int(row["episode_count"]))
        selected_mode = str(row.get("selected_mode", "") or "")
        if selected_mode:
            selected_modes.setdefault(key, Counter())[selected_mode] += 1

    rows: List[Dict[str, object]] = []
    for setting in settings:
        for horizon in horizons:
            key = (setting, int(horizon))
            values = np.asarray(by_key[key], dtype=np.float64)
            sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            rows.append(
                {
                    "setting": setting,
                    "label": DISPLAY_LABELS.get(setting, setting),
                    "horizon": int(horizon),
                    "seed_count": int(len(values)),
                    "episode_count_min": int(min(episode_counts[key])),
                    "episode_count_max": int(max(episode_counts[key])),
                    "mean_state_mse": float(values.mean()),
                    "sem_state_mse": sem,
                    "min_state_mse": float(values.min()),
                    "max_state_mse": float(values.max()),
                    "selected_mode_counts": ";".join(
                        f"{mode}:{count}" for mode, count in sorted(selected_modes.get(key, {}).items())
                    ),
                }
            )
    return rows


def write_csv(path: Path, rows: List[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    summary_rows: List[Mapping[str, object]],
    horizons: List[int],
    output_stem: Path,
    *,
    settings: Iterable[str],
    rollout_key: str,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 250,
            "savefig.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(5.8, 3.35), constrained_layout=True)
    for setting in settings:
        label = DISPLAY_LABELS.get(setting, setting)
        rows = [row for row in summary_rows if row["setting"] == setting]
        rows = sorted(rows, key=lambda row: int(row["horizon"]))
        x = np.asarray([int(row["horizon"]) for row in rows], dtype=np.float64)
        y = np.asarray([float(row["mean_state_mse"]) for row in rows], dtype=np.float64)
        sem = np.asarray([float(row["sem_state_mse"]) for row in rows], dtype=np.float64)
        ax.errorbar(
            x,
            y,
            yerr=sem,
            label=label,
            color=COLORS.get(setting, "#4C78A8"),
            marker=MARKERS.get(setting, "o"),
            linewidth=1.8,
            markersize=4.6,
            capsize=2.5,
            alpha=0.95,
        )

    ax.set_yscale("log")
    ax.set_xticks(horizons)
    ax.set_xlim(min(horizons) - 4, max(horizons) + 5)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Held-out state MSE")
    title = "ManiSkill perturbation forecasting"
    if rollout_key == "best_periodic_rollout":
        title += " (best periodic)"
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.35)
    ax.grid(True, which="minor", axis="y", linewidth=0.3, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False)

    for suffix in ("pdf", "png"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
