"""Fail closed on the corrected 12-fit Dysts GPU smoke."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_out", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipts = sorted(args.base_out.glob("**/training_success.json"))
    if len(receipts) != 12:
        raise ValueError(f"expected 12 training receipts, got {len(receipts)}")
    for receipt_path in receipts:
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "training_complete":
            raise ValueError(f"invalid receipt {receipt_path}")
        metrics = json.loads((receipt_path.parent / "final_metrics.json").read_text())
        if not all(math.isfinite(float(value)) for value in metrics.values()):
            raise ValueError(f"nonfinite final metrics in {receipt_path.parent}")

    telemetry_files = sorted((args.base_out / "gpu_telemetry").glob("*.csv"))
    if len(telemetry_files) != 1:
        raise ValueError(f"expected one pack telemetry file, got {telemetry_files}")
    values = []
    with telemetry_files[0].open(newline="") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("gpu_utilization_percent", "").strip()
            if raw:
                values.append(float(raw))
    if len(values) < 3:
        raise ValueError(f"too few telemetry samples: {len(values)}")
    # The sampler writes once immediately before worker startup. Exclude that
    # declared warmup observation from the training-window utilization gate.
    measurement_values = values[1:]
    active = [value for value in measurement_values if value > 0]
    mean_all = sum(measurement_values) / len(measurement_values)
    mean_active = sum(active) / len(active) if active else 0.0
    if mean_all < 70.0 or mean_active < 80.0 or max(values, default=0.0) < 95.0:
        raise ValueError(
            f"GPU utilization gate failed: all={mean_all}, active={mean_active}, peak={max(values)}"
        )
    payload = {
        "schema_version": 1,
        "status": "passed",
        "training_receipts": len(receipts),
        "telemetry_samples": len(values),
        "measurement_samples_after_warmup": len(measurement_values),
        "mean_all_gpu_utilization_percent": mean_all,
        "mean_active_gpu_utilization_percent": mean_active,
        "peak_gpu_utilization_percent": max(values),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
