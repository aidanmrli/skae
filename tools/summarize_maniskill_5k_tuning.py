"""Summarize 5k-step ManiSkill LISTA tuning pilots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollout-key", default="best_periodic_rollout")
    parser.add_argument("--eval-dir-name", default="eval_test_periodic")
    parser.add_argument("--horizons", default="10,20,30,40,50,75,100,125")
    parser.add_argument("--primary-horizons", default="10,20,30,40,50")
    parser.add_argument("--baseline-setting", default="dense_tanh_sp0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = parse_ints(args.horizons)
    primary_horizons = parse_ints(args.primary_horizons)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(
        args.run_root,
        horizons=horizons,
        primary_horizons=primary_horizons,
        rollout_key=args.rollout_key,
        eval_dir_name=args.eval_dir_name,
    )
    if not rows:
        raise RuntimeError(f"No eval summaries found under {args.run_root}")

    rows = add_baseline_comparisons(rows, args.baseline_setting, horizons, primary_horizons)
    aggregate_rows = aggregate_rows_by_setting(rows, horizons, primary_horizons)
    telemetry = summarize_gpu_telemetry(args.run_root / "gpu_telemetry")

    per_run_csv = args.output_dir / "per_run.csv"
    aggregate_csv = args.output_dir / "aggregate.csv"
    write_csv(per_run_csv, rows)
    write_csv(aggregate_csv, aggregate_rows)

    baseline = next(
        (row for row in aggregate_rows if row["setting"] == args.baseline_setting),
        None,
    )
    lista_rows = [row for row in aggregate_rows if row["encoder_kind"] == "lista"]
    best_lista = min(
        lista_rows,
        key=lambda row: float(row["primary_mean_state_mse"]),
        default=None,
    )
    payload = {
        "run_root": str(args.run_root),
        "output_dir": str(args.output_dir),
        "rollout_key": args.rollout_key,
        "eval_dir_name": args.eval_dir_name,
        "horizons": horizons,
        "primary_horizons": primary_horizons,
        "baseline_setting": args.baseline_setting,
        "per_run_csv": str(per_run_csv),
        "aggregate_csv": str(aggregate_csv),
        "run_count": len(rows),
        "setting_count": len(aggregate_rows),
        "baseline": baseline,
        "best_lista": best_lista,
        "best_lista_beats_baseline": (
            bool(best_lista and baseline)
            and float(best_lista["primary_mean_state_mse"])
            < float(baseline["primary_mean_state_mse"])
        ),
        "gpu_telemetry": telemetry,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_ints(value: str) -> List[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_rows(
    run_root: Path,
    *,
    horizons: Iterable[int],
    primary_horizons: Iterable[int],
    rollout_key: str,
    eval_dir_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    eval_pattern = f"*/seed*/{eval_dir_name}/metrics_summary.json"
    for eval_path in sorted(run_root.glob(eval_pattern)):
        run_dir = eval_path.parent.parent
        setting = run_dir.parent.name
        seed = int(run_dir.name.removeprefix("seed"))
        payload = load_json(eval_path)
        config = load_json(run_dir / "config.json")
        model_config = dict(config.get("model_config", {}))
        loss_weights = dict(config.get("loss_weights", {}))
        rollout = payload.get(rollout_key)
        if not isinstance(rollout, Mapping):
            continue

        row: Dict[str, Any] = {
            "setting": setting,
            "seed": seed,
            "checkpoint_step": int(payload.get("checkpoint_step", -1)),
            "encoder_kind": model_config.get("encoder_kind", ""),
            "activation": model_config.get("activation", "relu"),
            "z_dim": model_config.get("z_dim", ""),
            "hidden_dim": model_config.get("hidden_dim", ""),
            "num_hidden_layers": model_config.get("num_hidden_layers", ""),
            "lista_alpha": model_config.get("lista_alpha", ""),
            "lista_loops": model_config.get("lista_loops", ""),
            "sparsity_weight": loss_weights.get("sparsity", ""),
            "prediction_weight": loss_weights.get("prediction", ""),
            "reconstruction_weight": loss_weights.get("reconstruction", ""),
            "latent_weight": loss_weights.get("latent", ""),
            "k_stability_weight": loss_weights.get("k_stability", ""),
            "num_steps": config.get("num_steps", ""),
            "batch_size": config.get("batch_size", ""),
            "sequence_length": config.get("sequence_length", ""),
            "lr": config.get("lr", ""),
            "weight_decay": config.get("weight_decay", ""),
        }
        primary_values: List[float] = []
        for horizon in horizons:
            key = f"h{horizon}/state_mse"
            value = rollout.get(key)
            row[f"h{horizon}_state_mse"] = "" if value is None else float(value)
            row[f"h{horizon}_selected_mode"] = rollout.get(f"h{horizon}/selected_mode", "")
            row[f"h{horizon}_selected_period"] = rollout.get(f"h{horizon}/selected_period", "")
            if horizon in primary_horizons and value is not None:
                primary_values.append(float(value))
        row["primary_mean_state_mse"] = mean(primary_values) if primary_values else math.nan
        rows.append(row)
    return rows


def add_baseline_comparisons(
    rows: List[Dict[str, Any]],
    baseline_setting: str,
    horizons: Iterable[int],
    primary_horizons: Iterable[int],
) -> List[Dict[str, Any]]:
    baseline_by_seed = {
        int(row["seed"]): row for row in rows if row["setting"] == baseline_setting
    }
    for row in rows:
        baseline = baseline_by_seed.get(int(row["seed"]))
        if baseline is None:
            row["primary_ratio_to_baseline"] = ""
            row["primary_beats_baseline"] = ""
            continue
        baseline_primary = float(baseline["primary_mean_state_mse"])
        primary = float(row["primary_mean_state_mse"])
        row["primary_ratio_to_baseline"] = primary / baseline_primary if baseline_primary > 0 else ""
        row["primary_beats_baseline"] = primary < baseline_primary
        for horizon in horizons:
            value = row.get(f"h{horizon}_state_mse")
            baseline_value = baseline.get(f"h{horizon}_state_mse")
            if value == "" or baseline_value == "":
                row[f"h{horizon}_ratio_to_baseline"] = ""
                continue
            baseline_float = float(baseline_value)
            row[f"h{horizon}_ratio_to_baseline"] = (
                float(value) / baseline_float if baseline_float > 0 else ""
            )
    return rows


def aggregate_rows_by_setting(
    rows: List[Mapping[str, Any]],
    horizons: Iterable[int],
    primary_horizons: Iterable[int],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["setting"])].append(row)

    aggregate: List[Dict[str, Any]] = []
    for setting, setting_rows in sorted(grouped.items()):
        first = setting_rows[0]
        out: Dict[str, Any] = {
            "setting": setting,
            "seed_count": len({int(row["seed"]) for row in setting_rows}),
            "encoder_kind": _unique_or_mixed(row["encoder_kind"] for row in setting_rows),
            "activation": _unique_or_mixed(row["activation"] for row in setting_rows),
            "z_dim": _unique_or_mixed(row["z_dim"] for row in setting_rows),
            "hidden_dim": _unique_or_mixed(row["hidden_dim"] for row in setting_rows),
            "num_hidden_layers": _unique_or_mixed(row["num_hidden_layers"] for row in setting_rows),
            "lista_alpha": _unique_or_mixed(row["lista_alpha"] for row in setting_rows),
            "lista_loops": _unique_or_mixed(row["lista_loops"] for row in setting_rows),
            "sparsity_weight": _unique_or_mixed(row["sparsity_weight"] for row in setting_rows),
            "prediction_weight": _unique_or_mixed(row["prediction_weight"] for row in setting_rows),
            "reconstruction_weight": _unique_or_mixed(row["reconstruction_weight"] for row in setting_rows),
            "latent_weight": _unique_or_mixed(row["latent_weight"] for row in setting_rows),
            "num_steps": first.get("num_steps", ""),
            "batch_size": first.get("batch_size", ""),
            "sequence_length": first.get("sequence_length", ""),
            "lr": first.get("lr", ""),
        }
        primary_values = [float(row["primary_mean_state_mse"]) for row in setting_rows]
        out["primary_mean_state_mse"] = mean(primary_values)
        out["primary_sem_state_mse"] = sem(primary_values)
        ratio_values = [
            float(row["primary_ratio_to_baseline"])
            for row in setting_rows
            if row.get("primary_ratio_to_baseline") not in ("", None)
        ]
        out["primary_mean_ratio_to_baseline"] = mean(ratio_values) if ratio_values else ""
        out["primary_beats_baseline_count"] = sum(
            1 for row in setting_rows if row.get("primary_beats_baseline") is True
        )
        for horizon in horizons:
            values = [
                float(row[f"h{horizon}_state_mse"])
                for row in setting_rows
                if row.get(f"h{horizon}_state_mse") not in ("", None)
            ]
            out[f"h{horizon}_mean_state_mse"] = mean(values) if values else ""
            out[f"h{horizon}_sem_state_mse"] = sem(values) if values else ""
            ratio_values = [
                float(row[f"h{horizon}_ratio_to_baseline"])
                for row in setting_rows
                if row.get(f"h{horizon}_ratio_to_baseline") not in ("", None)
            ]
            out[f"h{horizon}_mean_ratio_to_baseline"] = mean(ratio_values) if ratio_values else ""
        aggregate.append(out)
    aggregate.sort(key=lambda row: float(row["primary_mean_state_mse"]))
    return aggregate


def _unique_or_mixed(values: Iterable[Any]) -> Any:
    unique = sorted({str(value) for value in values})
    return unique[0] if len(unique) == 1 else "mixed:" + ",".join(unique)


def sem(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def summarize_gpu_telemetry(telemetry_dir: Path) -> Dict[str, Any]:
    if not telemetry_dir.exists():
        return {"status": "missing"}
    logs = sorted(telemetry_dir.glob("*.csv"))
    utils: List[float] = []
    memory_used: List[float] = []
    for path in logs:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                util = parse_float(row.get("gpu_utilization_percent"))
                mem = parse_float(row.get("memory_used_mib"))
                if util is not None:
                    utils.append(util)
                if mem is not None:
                    memory_used.append(mem)
    if not utils:
        return {"status": "no_samples", "log_count": len(logs)}
    utils_sorted = sorted(utils)
    return {
        "status": "ok",
        "log_count": len(logs),
        "sample_count": len(utils),
        "utilization_mean_percent": mean(utils),
        "utilization_median_percent": percentile(utils_sorted, 0.5),
        "utilization_p90_percent": percentile(utils_sorted, 0.9),
        "utilization_max_percent": max(utils),
        "active_fraction_ge_10_percent": sum(1 for value in utils if value >= 10.0) / len(utils),
        "memory_used_max_mib": max(memory_used) if memory_used else "",
    }


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def percentile(values: List[float], q: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    index = q * (len(values) - 1)
    lo = int(math.floor(index))
    hi = int(math.ceil(index))
    if lo == hi:
        return values[lo]
    weight = index - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
