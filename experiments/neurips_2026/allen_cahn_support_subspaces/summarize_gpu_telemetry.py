"""Validate one scientific shard's GPU telemetry against the frozen card."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from statistics import median

from experiments.neurips_2026.allen_cahn_support_subspaces.io import (
    CARD_PATH,
    load_card,
    sha256_path,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.evaluation_helpers import (
    verify_source_manifest,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.select_profile import read_telemetry


SOURCE_MANIFEST = Path(__file__).with_name("source_manifest.sha256")
SCIENTIFIC_TELEMETRY_CHECK_KEYS = frozenset({
    "active_samples", "mean_active_utilization", "mean_all_utilization",
    "peak_memory", "device", "visible_gpu_count", "gpu_uuid", "scope_markers",
    "samples_within_gpu_interval", "sampling_interval",
})
TELEMETRY_RECEIPT_CHECK_KEYS = frozenset({
    "status", "seed", "card", "source", "job", "gate_roster", "gates",
    "scope", "raw_path", "raw_hash", "start_path", "start_hash", "done_path",
    "done_hash", "device_metadata", "evaluator_marker_lineage",
})


def telemetry_receipt_checks(
    payload: dict,
    telemetry_dir: Path,
    *,
    card_hash: str,
    source_hash: str,
    seed: int,
    slurm_job_id: str,
    evaluator_scope: dict,
) -> dict[str, bool]:
    raw_name = str(payload.get("raw_telemetry_filename", ""))
    start_name = str(payload.get("gpu_start_marker_filename", ""))
    done_name = str(payload.get("gpu_done_marker_filename", ""))
    raw_path = telemetry_dir / raw_name
    start_path = telemetry_dir / start_name
    done_path = telemetry_dir / done_name
    gates = payload.get("checks")
    checks = {
        "status": payload.get("status") == "passed",
        "seed": int(payload.get("seed", -1)) == int(seed),
        "card": payload.get("card_sha256") == card_hash,
        "source": payload.get("source_manifest_sha256") == source_hash,
        "job": str(payload.get("slurm_job_id")) == str(slurm_job_id),
        "gate_roster": isinstance(gates, dict)
        and set(gates) == SCIENTIFIC_TELEMETRY_CHECK_KEYS,
        "gates": isinstance(gates, dict) and bool(gates)
        and all(value is True for value in gates.values()),
        "scope": payload.get("scope")
        == "evaluator-owned GPU start through final CUDA synchronization",
        "raw_path": Path(raw_name).name == raw_name and raw_path.is_file(),
        "raw_hash": raw_path.is_file()
        and sha256_path(raw_path) == payload.get("raw_telemetry_sha256"),
        "start_path": Path(start_name).name == start_name and start_path.is_file(),
        "start_hash": start_path.is_file()
        and sha256_path(start_path) == payload.get("gpu_start_marker_sha256"),
        "done_path": Path(done_name).name == done_name and done_path.is_file(),
        "done_hash": done_path.is_file()
        and sha256_path(done_path) == payload.get("gpu_done_marker_sha256"),
        "device_metadata": int(payload.get("visible_gpu_count", -1)) == 1
        and payload.get("gpu_uuid") == payload.get("telemetry", {}).get("gpu_uuid"),
        "evaluator_marker_lineage": (
            evaluator_scope.get("evaluator_owned_start_marker") == str(start_path)
            and evaluator_scope.get("evaluator_owned_done_marker") == str(done_path)
            and evaluator_scope.get("start_marker_sha256")
            == payload.get("gpu_start_marker_sha256")
            and evaluator_scope.get("done_marker_sha256")
            == payload.get("gpu_done_marker_sha256")
            and evaluator_scope.get("preload_and_serialization_excluded") is True
        ),
    }
    if set(checks) != TELEMETRY_RECEIPT_CHECK_KEYS:
        raise AssertionError("Scientific telemetry receipt roster drifted")
    return checks


def sample_unix_times(path: Path) -> list[float]:
    result = []
    for line in path.read_text().splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) == 5:
            result.append(
                datetime.strptime(cells[1], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            )
    if not result:
        raise RuntimeError("Scientific telemetry contains no timestamped samples")
    return result


def scope_timing_checks(
    start: dict,
    done: dict,
    timestamps: list[float],
    *,
    seed: int,
    maximum_median_interval: float,
) -> tuple[dict[str, bool], dict[str, float]]:
    gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
    marker_valid = bool(
        start.get("event") == "gpu_compute_start"
        and done.get("event") == "gpu_compute_done"
        and int(start.get("seed", -1)) == int(seed)
        and int(done.get("seed", -1)) == int(seed)
        and float(start.get("unix_time", 0.0)) < float(done.get("unix_time", 0.0))
    )
    checks = {
        "scope_markers": marker_valid,
        "samples_within_gpu_interval": marker_valid
        and min(timestamps) > float(start["unix_time"])
        and max(timestamps) < float(done["unix_time"]),
        "sampling_interval": bool(gaps)
        and median(gaps) <= float(maximum_median_interval),
    }
    timing = {
        "first_unix_time": min(timestamps),
        "last_unix_time": max(timestamps),
        "marker_start_unix_time": float(start["unix_time"]),
        "marker_done_unix_time": float(done["unix_time"]),
        "median_interval_seconds": median(gaps) if gaps else 0.0,
    }
    return checks, timing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--slurm_job_id", required=True)
    parser.add_argument("--device_name", required=True)
    parser.add_argument("--gpu_uuid", required=True)
    parser.add_argument("--visible_gpu_count", type=int, required=True)
    parser.add_argument("--gpu_start_file", type=Path, required=True)
    parser.add_argument("--gpu_done_file", type=Path, required=True)
    parser.add_argument("--expected_card_sha256", required=True)
    parser.add_argument("--expected_source_manifest_sha256", required=True)
    parser.add_argument("--card", type=Path, default=CARD_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    card, card_hash = load_card(args.card)
    if card_hash != args.expected_card_sha256:
        raise RuntimeError("Scientific telemetry card root mismatch")
    source_hash = verify_source_manifest(SOURCE_MANIFEST)
    if source_hash != args.expected_source_manifest_sha256:
        raise RuntimeError("Scientific telemetry source root mismatch")
    values = read_telemetry(args.telemetry)
    gates = card["scientific_hardware_gates"]
    start = json.loads(args.gpu_start_file.read_text())
    done = json.loads(args.gpu_done_file.read_text())
    timestamps = sample_unix_times(args.telemetry)
    scope_checks, sample_timing = scope_timing_checks(
        start,
        done,
        timestamps,
        seed=int(args.seed),
        maximum_median_interval=float(
            gates["maximum_median_sample_interval_seconds"]
        ),
    )
    checks = {
        "active_samples": int(values["active_samples"]) >= int(gates["minimum_active_samples"]),
        "mean_active_utilization": float(values["mean_active_gpu_utilization_percent"])
        >= float(gates["minimum_mean_active_gpu_utilization_percent"]),
        "mean_all_utilization": float(values["mean_all_gpu_utilization_percent"])
        >= float(gates["minimum_mean_all_gpu_utilization_percent"]),
        "peak_memory": float(values["peak_memory_fraction"])
        <= float(gates["maximum_peak_memory_fraction"]),
        "device": str(gates["required_device_name_fragment"]) in args.device_name,
        "visible_gpu_count": int(args.visible_gpu_count)
        == int(gates["required_visible_cuda_device_count"]),
        "gpu_uuid": str(values["gpu_uuid"]) == args.gpu_uuid,
        **scope_checks,
    }
    if set(checks) != SCIENTIFIC_TELEMETRY_CHECK_KEYS:
        raise AssertionError("Scientific telemetry check roster drifted")
    passed = all(value is True for value in checks.values())
    payload = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "seed": int(args.seed),
        "slurm_job_id": str(args.slurm_job_id),
        "device_name": args.device_name,
        "gpu_uuid": args.gpu_uuid,
        "visible_gpu_count": int(args.visible_gpu_count),
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "checks": checks,
        "telemetry": values,
        "scope": "evaluator-owned GPU start through final CUDA synchronization",
        "gpu_start_marker_filename": args.gpu_start_file.name,
        "gpu_start_marker_sha256": sha256_path(args.gpu_start_file),
        "gpu_done_marker_filename": args.gpu_done_file.name,
        "gpu_done_marker_sha256": sha256_path(args.gpu_done_file),
        "sample_timing": sample_timing,
        "raw_telemetry_filename": args.telemetry.name,
        "raw_telemetry_sha256": sha256_path(args.telemetry),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "seed": args.seed}), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
