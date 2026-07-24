"""Outcome-free synthetic batch-size profile for the encoder workload."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from .features import encode_in_batches
from .io import (
    checkpoint_specs,
    load_architecture_audit,
    load_card,
    load_model,
    load_task_manifest,
    verify_source_manifest,
    write_json_once,
)


def _device_contract(card: dict[str, object]) -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The frozen profile requires exactly one visible CUDA GPU")
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    required = str(card["hardware_profile"]["required_device_name_fragment"])
    if required not in name:
        raise RuntimeError(f"Required {required!r} GPU, observed {name!r}")
    return device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-card-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-task-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    card, card_sha = load_card(expected_sha256=args.expected_card_sha256)
    source_sha = verify_source_manifest(
        card, expected_sha256=args.expected_source_manifest_sha256
    )
    task, task_sha = load_task_manifest(
        card, expected_sha256=args.expected_task_manifest_sha256
    )
    if Path(task["output_root"]) != args.output_root:
        raise RuntimeError("Profile output root differs from the frozen task")
    profile_dir = args.output_root / "profile"
    if profile_dir.exists():
        raise FileExistsError(profile_dir)
    profile_dir.mkdir(parents=True)

    device = _device_contract(card)
    specs = checkpoint_specs(card)
    audit = load_architecture_audit(card)
    models = [
        load_model(specs[(arm, seed)], card, audit, device=device)
        for arm in card["roster"]["arms"]
        for seed in card["roster"]["model_seeds"]
    ]
    generator = torch.Generator(device="cpu").manual_seed(20260721)
    observations = torch.randn(6400, 512, generator=generator).to(device)
    torch.cuda.synchronize()

    records = []
    minimum_seconds = float(card["hardware_profile"]["minimum_seconds_each"])
    for batch_size in card["hardware_profile"]["candidate_batch_sizes"]:
        torch.cuda.reset_peak_memory_stats(device)
        started_at = time.time()
        cycles = 0
        while True:
            for model in models:
                encoded = encode_in_batches(
                    model, observations, batch_size=int(batch_size)
                )
                del encoded
            cycles += 1
            torch.cuda.synchronize()
            if time.time() - started_at >= minimum_seconds:
                break
        completed_at = time.time()
        elapsed = completed_at - started_at
        encoded_states = cycles * len(models) * observations.shape[0]
        records.append(
            {
                "batch_size": int(batch_size),
                "started_at_epoch": started_at,
                "completed_at_epoch": completed_at,
                "elapsed_seconds": elapsed,
                "cycles": cycles,
                "model_calls": cycles * len(models),
                "encoded_states": int(encoded_states),
                "encoded_states_per_second": float(encoded_states / elapsed),
                "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
        )
    payload = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "status": "synthetic_outcome_free_profile_complete",
        "card_sha256": card_sha,
        "source_manifest_sha256": source_sha,
        "task_manifest_sha256": task_sha,
        "device_name": torch.cuda.get_device_name(device),
        "device_count": torch.cuda.device_count(),
        "dataset_files_opened": 0,
        "semantic_outcomes_accessed": 0,
        "model_count": len(models),
        "synthetic_shape": list(observations.shape),
        "candidates": records,
        "allocation_work_started_at_epoch": records[0]["started_at_epoch"],
        "allocation_work_completed_at_epoch": records[-1]["completed_at_epoch"],
    }
    write_json_once(profile_dir / "workload.json", payload)


if __name__ == "__main__":
    main()
