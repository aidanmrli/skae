"""Prospective field generation and field-only representation extraction."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import torch

from experiments.neurips_2026.allen_cahn_forecast_replication.core import (
    generate_all_fields,
    realized_rng_streams,
)
from experiments.neurips_2026.allen_cahn_forecast_replication.io import field_payload

from .features import encode_in_batches, observation_tensor
from .io import (
    checkpoint_specs,
    load_architecture_audit,
    load_card,
    load_field_roster,
    load_model,
    load_task_manifest,
    sha256_path,
    torch_save_once,
    verify_authenticated_v1_generator,
    verify_opened_context,
    verify_source_manifest,
    write_json_once,
)


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
        raise RuntimeError("GPU output root differs from frozen task")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    verify_opened_context(card)
    verify_authenticated_v1_generator(card)
    args.output_root.mkdir(parents=False)

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("V2 requires exactly one visible CUDA GPU")
    device = torch.device("cuda:0")
    required_name = str(card["hardware"]["required_device_name_fragment"])
    if required_name not in torch.cuda.get_device_name(device):
        raise RuntimeError("V2 did not receive the frozen A100 GPU type")
    torch.set_grad_enabled(False)
    torch.cuda.reset_peak_memory_stats(device)
    roots = {
        "card_sha256": card_sha,
        "source_manifest_sha256": source_sha,
        "task_manifest_sha256": task_sha,
    }
    job_started = time.time()

    generation_started = time.time()
    generated = generate_all_fields(card, device=device)
    generation_completed = time.time()
    if tuple(generated.shape) != (3, 256, 201, 16, 16, 2):
        raise RuntimeError("Packed generated field shape drifted")
    rng_proof = realized_rng_streams(card)
    dataset_records = []
    for dataset_index, (seed, relative_path) in enumerate(
        zip(card["prospective_datasets"]["seeds"], card["prospective_datasets"]["paths"])
    ):
        path = args.output_root / str(relative_path)
        payload = field_payload(
            generated[dataset_index], card, dataset_index=dataset_index, seed=int(seed)
        )
        torch_save_once(path, payload)
        dataset_records.append(
            {
                "dataset_index": dataset_index,
                "dataset_seed": int(seed),
                "path": str(path),
                "sha256": sha256_path(path),
                "field_shape": list(payload["fields"].shape),
                "top_level_keys": sorted(payload),
                "metadata_keys": sorted(payload["metadata"]),
            }
        )
    dataset_manifest_path = args.output_root / "dataset_manifest.json"
    dataset_manifest = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "status": "field_only_validated_and_hashed_before_encoding",
        **roots,
        "rng_stream_proof": rng_proof,
        "datasets": dataset_records,
        "semantic_target_keys_created": 0,
        "semantic_targets_derived": 0,
    }
    write_json_once(dataset_manifest_path, dataset_manifest)
    dataset_manifest_sha = sha256_path(dataset_manifest_path)

    train_fields, test_fields, reloaded_manifest = load_field_roster(
        card, expected_manifest_sha256=dataset_manifest_sha
    )
    if reloaded_manifest != dataset_manifest:
        raise RuntimeError("Reloaded prospective manifest changed")
    observations_cpu, layout = observation_tensor(
        train_fields, test_fields, card["roster"]["observation_indices"]
    )
    observations = observations_cpu.to(device)
    del generated, train_fields, test_fields, observations_cpu

    specs = checkpoint_specs(card)
    audit = load_architecture_audit(card)
    ordered_specs = [
        specs[(arm, seed)]
        for arm in card["roster"]["arms"]
        for seed in card["roster"]["model_seeds"]
    ]
    models = [load_model(spec, card, audit, device=device) for spec in ordered_specs]
    torch.cuda.synchronize()
    extraction_started = time.time()
    cycles = 0
    retained: list[torch.Tensor] = []
    minimum_seconds = float(card["hardware"]["encoding_minimum_scope_seconds"])
    batch_size = int(card["hardware"]["encoding_batch_size"])
    while True:
        current = [
            encode_in_batches(model, observations, batch_size=batch_size)
            for model in models
        ]
        torch.cuda.synchronize()
        retained = current
        cycles += 1
        if time.time() - extraction_started >= minimum_seconds:
            break
    extraction_completed = time.time()

    latents = {
        f"{spec.arm}_seed_{spec.seed}": values.detach().cpu()
        for spec, values in zip(ordered_specs, retained)
    }
    expected_latents = {
        f"{arm}_seed_{seed}"
        for arm in card["roster"]["arms"]
        for seed in card["roster"]["model_seeds"]
    }
    if set(latents) != expected_latents:
        raise RuntimeError("Latent output roster is incomplete")
    field_dir = args.output_root / "field_only"
    feature_payload = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "status": "prospective_field_only_frozen_before_semantic_access",
        **roots,
        "dataset_manifest_sha256": dataset_manifest_sha,
        "checkpoint_sha256": {
            f"{spec.arm}_seed_{spec.seed}": spec.sha256 for spec in ordered_specs
        },
        "layout": layout,
        "support_threshold": card["feature_protocol"]["support_threshold"],
        "batch_size": batch_size,
        "encoding_cycles_for_utilization": cycles,
        "semantic_target_keys_accessed": 0,
        "semantic_ties_or_labels_derived": 0,
        "latents": latents,
    }
    features_path = field_dir / "features.pt"
    torch_save_once(features_path, feature_payload)
    job_completed = time.time()
    markers = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        **roots,
        "dataset_manifest_sha256": dataset_manifest_sha,
        "features_sha256": sha256_path(features_path),
        "job_started_at_epoch": job_started,
        "generation_started_at_epoch": generation_started,
        "generation_completed_at_epoch": generation_completed,
        "extraction_started_at_epoch": extraction_started,
        "extraction_completed_at_epoch": extraction_completed,
        "job_completed_at_epoch": job_completed,
        "encoding_cycles": cycles,
        "model_count_per_cycle": len(models),
        "rows_per_model": observations.shape[0],
        "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "semantic_outcomes_accessed": 0,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_recorded"),
    }
    write_json_once(field_dir / "markers.json", markers)


if __name__ == "__main__":
    main()
