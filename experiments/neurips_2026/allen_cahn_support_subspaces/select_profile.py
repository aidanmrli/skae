"""Select the smallest GPU batch satisfying the frozen telemetry contract."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from experiments.neurips_2026.allen_cahn_support_subspaces.io import (
    CARD_PATH,
    load_card,
    sha256_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source_manifest_sha256", required=True)
    parser.add_argument("--expected_card_sha256", required=True)
    parser.add_argument("--card", type=Path, default=CARD_PATH)
    return parser.parse_args()


def numeric(value: str) -> float:
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", value)
    if match is None:
        raise ValueError(f"No numeric value in telemetry cell {value!r}")
    return float(match.group(0))


def read_telemetry(path: Path) -> dict[str, float | int]:
    rows = []
    for line in path.read_text().splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != 5 or cells[2].startswith("utilization"):
            continue
        rows.append((cells[0], numeric(cells[2]), numeric(cells[3]), numeric(cells[4])))
    if not rows:
        raise RuntimeError(f"No GPU telemetry samples in {path}")
    uuids = {row[0] for row in rows}
    if len(uuids) != 1:
        raise RuntimeError(f"Telemetry contains multiple GPU UUIDs: {uuids}")
    array = np.asarray([row[1:] for row in rows], dtype=np.float64)
    active = array[:, 0] > 0
    return {
        "total_samples": int(array.shape[0]),
        "active_samples": int(active.sum()),
        "mean_active_gpu_utilization_percent": (
            float(array[active, 0].mean()) if np.any(active) else 0.0
        ),
        "mean_all_gpu_utilization_percent": float(array[:, 0].mean()),
        "peak_memory_fraction": float(np.max(array[:, 1] / array[:, 2])),
        "reported_total_memory_mib": float(np.median(array[:, 2])),
        "gpu_uuid": next(iter(uuids)),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    card, card_hash = load_card(args.card)
    if card_hash != args.expected_card_sha256:
        raise RuntimeError("Card differs from pre-profile launcher root of trust")
    contract = card["hardware_profile"]
    candidates = []
    for batch_size in contract["candidate_batch_sizes"]:
        profile_path = args.profile_dir / f"batch_{batch_size}.json"
        telemetry_path = args.profile_dir / f"batch_{batch_size}_nvidia_smi.csv"
        profile = json.loads(profile_path.read_text())
        telemetry = read_telemetry(telemetry_path)
        integrity = bool(
            profile["status"] == "completed"
            and profile["synthetic_inputs_only"]
            and not profile["outcomes_accessed"]
            and not profile["datasets_opened"]
            and int(profile["batch_size"]) == int(batch_size)
            and profile["card_sha256"] == card_hash
            and profile["source_manifest_sha256"] == args.source_manifest_sha256
            and str(profile["slurm_job_id"]) != "not_recorded"
            and str(profile["slurm_job_gpus"]) != "not_recorded"
            and int(profile["visible_cuda_device_count"])
            == int(contract["required_visible_cuda_device_count"])
            and str(contract["required_device_name_fragment"]) in str(profile["device_name"])
            and str(profile["device_uuid"]) == str(telemetry["gpu_uuid"])
            and int(profile["resident_model_count"]) == int(contract["resident_model_count"])
            and int(profile["closure_state_batch_size"])
            == int(contract["closure_state_batch_size"])
            and profile.get("historical_provenance_kernel_profiled") is True
            and int(profile.get("historical_reproduction_batch_size", -1))
            == int(card["inputs"]["ordinary_forecast_seed_rows"][
                "historical_reproduction_batch_size"
            ])
            and profile.get("historical_reproduction_horizons")
            == [int(value) for value in card["inputs"][
                "ordinary_forecast_seed_rows"
            ]["historical_evaluator_horizon_sequence"]]
            and int(profile["device_total_memory_bytes"]) > 0
            and 0.95 * float(telemetry["reported_total_memory_mib"])
            <= float(profile["device_total_memory_bytes"]) / (1024 * 1024)
            <= 1.05 * float(telemetry["reported_total_memory_mib"])
        )
        gates = {
            "integrity": integrity,
            "duration": float(profile["profile_seconds"])
            >= float(contract["minimum_profile_seconds_each"]),
            "active_samples": int(telemetry["active_samples"])
            >= int(contract["minimum_active_samples"]),
            "mean_active_utilization": float(telemetry["mean_active_gpu_utilization_percent"])
            >= float(contract["minimum_mean_active_gpu_utilization_percent"]),
            "mean_all_utilization": float(telemetry["mean_all_gpu_utilization_percent"])
            >= float(contract["minimum_mean_all_gpu_utilization_percent"]),
            "peak_memory": float(telemetry["peak_memory_fraction"])
            <= float(contract["maximum_peak_memory_fraction"]),
            "profile_peak_memory": float(profile["peak_reserved_bytes"])
            / max(1.0, float(profile["device_total_memory_bytes"]))
            <= float(contract["maximum_peak_memory_fraction"]),
        }
        candidates.append({
            "batch_size": int(batch_size),
            "passed": all(gates.values()),
            "gates": gates,
            "profile": profile,
            "telemetry": telemetry,
            "profile_filename": profile_path.name,
            "telemetry_filename": telemetry_path.name,
            "profile_sha256": sha256_path(profile_path),
            "telemetry_sha256": sha256_path(telemetry_path),
        })
    passing = [item["batch_size"] for item in candidates if item["passed"]]
    payload = {
        "schema_version": 1,
        "status": "passed" if passing else "failed",
        "selected_batch_size": min(passing) if passing else None,
        "selection_rule": "smallest passing frozen candidate",
        "synthetic_inputs_only": True,
        "outcomes_quarantined": True,
        "telemetry_interval_seconds": int(contract["telemetry_interval_seconds"]),
        "card_sha256": card_hash,
        "source_manifest_sha256": args.source_manifest_sha256,
        "candidate_batch_sizes": [int(value) for value in contract["candidate_batch_sizes"]],
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "selected": payload["selected_batch_size"]}))
    if not passing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
