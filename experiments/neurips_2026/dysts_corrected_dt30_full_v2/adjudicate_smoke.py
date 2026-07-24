"""Fail closed on the corrected 12-fit Dysts GPU smoke."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch


def _validate_refinement_contract(config: dict, run_dir: Path) -> None:
    """Require one refinement only for an actual LISTAKM/LISTA encoder."""
    model_config = config["MODEL"]
    encoder_config = model_config["ENCODER"]
    if (
        model_config["MODEL_NAME"] == "LISTAKM"
        and encoder_config["ENCODER_TYPE"] == "lista"
        and int(encoder_config["LISTA"]["NUM_LOOPS"]) != 1
    ):
        raise ValueError(f"wrong LISTA refinement count in {run_dir}")


def _validate_final_metrics(metrics: dict, run_dir: Path) -> None:
    """Require finite scalar metrics while allowing descriptive metadata."""
    required = {
        "loss",
        "alignment_loss",
        "reconst_loss",
        "prediction_loss",
        "sparsity_loss",
        "sparsity_ratio",
    }
    missing = sorted(required - metrics.keys())
    if missing:
        raise ValueError(f"missing final metrics {missing} in {run_dir}")
    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(float(value)):
            raise ValueError(f"nonfinite final metric {key} in {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_out", type=Path, required=True)
    parser.add_argument("--expected_cache_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_cache = str(args.expected_cache_dir.resolve())
    receipts = sorted(args.base_out.glob("**/training_success.json"))
    if len(receipts) != 12:
        raise ValueError(f"expected 12 training receipts, got {len(receipts)}")
    selector_modes = set()
    for receipt_path in receipts:
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "training_complete":
            raise ValueError(f"invalid receipt {receipt_path}")
        if int(receipt.get("last_checkpoint_step", -1)) != 9_999:
            raise ValueError(f"incomplete smoke checkpoint in {receipt_path}")
        if receipt.get("checkpoint_selection_metric") != (
            "direct_strict_full_horizon_cumulative_state_summed_mse"
        ):
            raise ValueError(f"wrong selection metric in {receipt_path}")
        run_dir = receipt_path.parent
        config = json.loads((run_dir / "config.json").read_text())
        actual_cache = str(Path(config["ENV"]["DYSTS"]["CACHE_DIR"]).resolve())
        if actual_cache != expected_cache:
            raise ValueError(
                f"wrong cache root in {run_dir}: {actual_cache} != {expected_cache}"
            )
        _validate_refinement_contract(config, run_dir)
        if run_dir.parts[-5] == "dense_mlp_tanh":
            model_config = config["MODEL"]
            if (
                model_config["ENCODER"]["ACTIVATION"] != "tanh"
                or model_config["ENCODER"]["LAST_RELU"]
                or float(model_config["SPARSITY_COEFF"]) != 0.0
                or model_config["K_STRUCTURE"] != "dense"
            ):
                raise ValueError(f"dense baseline contract failed in {run_dir}")
        checkpoint = torch.load(
            run_dir / "checkpoint.pt", map_location="cpu", weights_only=False
        )
        selector_modes.add(checkpoint.get("checkpoint_selection_rollout"))
        if not math.isfinite(float(checkpoint["checkpoint_selection_score"])):
            raise ValueError(f"nonfinite direct selector in {run_dir}")
        if checkpoint["checkpoint_selection_full_horizon_finite_fraction"] != 1.0:
            raise ValueError(f"incomplete direct validation rollout in {run_dir}")
        metrics = json.loads((run_dir / "final_metrics.json").read_text())
        _validate_final_metrics(metrics, run_dir)
    if selector_modes != {"direct"}:
        raise ValueError(f"wrong checkpoint selector modes: {selector_modes}")

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
    measured = values[1:]
    active = [value for value in measured if value > 0]
    mean_all = sum(measured) / len(measured)
    mean_active = sum(active) / len(active) if active else 0.0
    peak = max(values, default=0.0)
    if mean_all < 70.0 or mean_active < 80.0 or peak < 95.0:
        raise ValueError(
            f"GPU utilization gate failed: all={mean_all}, active={mean_active}, peak={peak}"
        )
    payload = {
        "schema_version": 1,
        "status": "passed",
        "training_receipts": len(receipts),
        "expected_cache_dir": expected_cache,
        "checkpoint_selection_rollout": "direct",
        "telemetry_samples": len(values),
        "mean_all_gpu_utilization_percent": mean_all,
        "mean_active_gpu_utilization_percent": mean_active,
        "peak_gpu_utilization_percent": peak,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
