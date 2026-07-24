"""Validate end-to-end GPU telemetry for one generated bridge dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from experiments.neurips_2026.allen_cahn_mechanistic_bridge.integrity import (
    verify_source_manifest,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.io import (
    CARD_PATH,
    load_card,
    sha256_path,
    write_json_once,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.select_profile import (
    read_telemetry,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.summarize_gpu_telemetry import (
    sample_unix_times,
)


SOURCE_MANIFEST = Path(__file__).with_name("source_manifest.sha256")
GENERATION_TELEMETRY_CHECK_KEYS = frozenset({
    "active_samples", "mean_active_utilization", "mean_all_utilization",
    "peak_memory", "device", "single_gpu_uuid", "dataset", "generation_record",
    "job_lineage", "visible_gpu_count", "gpu_uuid",
    "scope_markers", "samples_within_gpu_interval", "sampling_interval",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--generation_record", type=Path, required=True)
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
    card, card_hash = load_card(args.card)
    if card_hash != args.expected_card_sha256:
        raise RuntimeError("Bridge card differs from launcher root")
    source_hash = verify_source_manifest(SOURCE_MANIFEST)
    if source_hash != args.expected_source_manifest_sha256:
        raise RuntimeError("Bridge source manifest differs from launcher root")
    record = json.loads(args.generation_record.read_text())
    values = read_telemetry(args.raw)
    contract = card["generation_hardware"]
    start = json.loads(args.gpu_start_file.read_text())
    done = json.loads(args.gpu_done_file.read_text())
    timestamps = sample_unix_times(args.raw)
    gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
    marker_valid = bool(
        start.get("event") == "gpu_invocation_start"
        and done.get("event") == "gpu_invocation_done"
        and int(start.get("seed", -1)) == int(args.seed)
        and int(done.get("seed", -1)) == int(args.seed)
        and str(start.get("slurm_job_id")) == str(args.slurm_job_id)
        and str(done.get("slurm_job_id")) == str(args.slurm_job_id)
        and float(start.get("unix_time", 0.0)) < float(done.get("unix_time", 0.0))
    )
    checks = {
        "active_samples": int(values["active_samples"])
        >= int(contract["minimum_active_samples"]),
        "mean_active_utilization": float(values["mean_active_gpu_utilization_percent"])
        >= float(contract["minimum_mean_active_gpu_utilization_percent"]),
        "mean_all_utilization": float(values["mean_all_gpu_utilization_percent"])
        >= float(contract["minimum_mean_all_gpu_utilization_percent"]),
        "peak_memory": float(values["peak_memory_fraction"])
        <= float(contract["maximum_peak_memory_fraction"]),
        "device": str(contract["required_device_name_fragment"]) in args.device_name,
        "single_gpu_uuid": bool(str(values["gpu_uuid"]).startswith("GPU-")),
        "visible_gpu_count": int(args.visible_gpu_count)
        == int(contract["required_visible_gpu_count"]),
        "gpu_uuid": str(values["gpu_uuid"]) == args.gpu_uuid,
        "dataset": args.dataset.is_file()
        and record.get("sha256") == sha256_path(args.dataset),
        "generation_record": int(record.get("seed", -1)) == int(args.seed)
        and record.get("card_sha256") == card_hash
        and record.get("source_manifest_sha256") == source_hash,
        "job_lineage": str(record.get("slurm_job_id")) == str(args.slurm_job_id),
        "scope_markers": marker_valid,
        "samples_within_gpu_interval": marker_valid
        and min(timestamps) > float(start["unix_time"])
        and max(timestamps) < float(done["unix_time"]),
        "sampling_interval": bool(gaps)
        and median(gaps)
        <= float(contract["maximum_median_sample_interval_seconds"]),
    }
    if set(checks) != GENERATION_TELEMETRY_CHECK_KEYS:
        raise AssertionError("Generation telemetry check roster drifted")
    passed = all(value is True for value in checks.values())
    payload = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "scope": "end-to-end pinned generator invocation including save",
        "seed": int(args.seed),
        "slurm_job_id": str(args.slurm_job_id),
        "device_name": args.device_name,
        "gpu_uuid": args.gpu_uuid,
        "visible_gpu_count": int(args.visible_gpu_count),
        "checks": checks,
        "telemetry": values,
        "raw_telemetry_filename": args.raw.name,
        "raw_telemetry_sha256": sha256_path(args.raw),
        "gpu_start_marker_filename": args.gpu_start_file.name,
        "gpu_start_marker_sha256": sha256_path(args.gpu_start_file),
        "gpu_done_marker_filename": args.gpu_done_file.name,
        "gpu_done_marker_sha256": sha256_path(args.gpu_done_file),
        "sample_timing": {
            "first_unix_time": min(timestamps),
            "last_unix_time": max(timestamps),
            "marker_start_unix_time": float(start["unix_time"]),
            "marker_done_unix_time": float(done["unix_time"]),
            "median_interval_seconds": median(gaps) if gaps else 0.0,
        },
        "dataset_sha256": sha256_path(args.dataset),
        "generation_record_sha256": sha256_path(args.generation_record),
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
    }
    write_json_once(args.output, payload)
    print(json.dumps({"status": payload["status"], "checks": checks}))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
