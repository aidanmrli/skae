"""Field-only GPU extraction with semantic fate labels quarantined."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from .features import encode_in_batches, observation_tensor
from .io import (
    checkpoint_specs,
    duplicate_safe_json,
    load_architecture_audit,
    load_card,
    load_field_roster,
    load_model,
    load_task_manifest,
    sha256_path,
    torch_save_once,
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
        raise RuntimeError("Extraction output root differs from frozen task")
    field_dir = args.output_root / "field_only"
    if field_dir.exists():
        raise FileExistsError(field_dir)
    field_dir.mkdir(parents=True)

    profile_path = args.output_root / "profile" / "decision.json"
    profile = duplicate_safe_json(profile_path)
    roots = {
        "card_sha256": card_sha,
        "source_manifest_sha256": source_sha,
        "task_manifest_sha256": task_sha,
    }
    if profile.get("status") != "passed" or any(
        profile.get(key) != value for key, value in roots.items()
    ):
        raise RuntimeError("Profile is not authenticated against the launch roots")
    batch_size = int(profile["selected_batch_size"])
    if batch_size not in card["hardware_profile"]["candidate_batch_sizes"]:
        raise RuntimeError("Profile selected an unfrozen batch size")

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Scientific extraction requires exactly one visible GPU")
    device = torch.device("cuda:0")
    required_name = card["scientific_hardware"]["required_device_name_fragment"]
    if required_name not in torch.cuda.get_device_name(device):
        raise RuntimeError("Scientific extraction did not receive the frozen GPU type")

    # This deserializes the training artifact but accesses only fields/split indices.
    # Semantic labels are not indexed or derived anywhere in this module.
    train_fields, test_fields, dataset_manifest = load_field_roster(card)
    observations_cpu, layout = observation_tensor(
        train_fields, test_fields, card["roster"]["observation_indices"]
    )
    observations = observations_cpu.to(device)
    specs = checkpoint_specs(card)
    audit = load_architecture_audit(card)
    ordered_specs = [
        specs[(arm, seed)]
        for arm in card["roster"]["arms"]
        for seed in card["roster"]["model_seeds"]
    ]
    models = [load_model(spec, card, audit, device=device) for spec in ordered_specs]
    torch.cuda.synchronize()

    minimum_seconds = float(card["scientific_hardware"]["minimum_gpu_scope_seconds"])
    started_at = time.time()
    cycles = 0
    retained: list[torch.Tensor] = []
    while True:
        current = [
            encode_in_batches(model, observations, batch_size=batch_size)
            for model in models
        ]
        cycles += 1
        torch.cuda.synchronize()
        retained = current
        if time.time() - started_at >= minimum_seconds:
            break
    completed_at = time.time()

    latents = {
        f"{spec.arm}_seed_{spec.seed}": values.detach().cpu()
        for spec, values in zip(ordered_specs, retained)
    }
    if set(latents) != {
        f"{arm}_seed_{seed}"
        for arm in card["roster"]["arms"]
        for seed in card["roster"]["model_seeds"]
    }:
        raise AssertionError("Latent output roster is incomplete")
    payload = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "status": "field_only_frozen_before_semantic_fate_access",
        **roots,
        "profile_decision_sha256": sha256_path(profile_path),
        "dataset_manifest_sha256": card["inputs"]["new_ic_dataset_manifest"]["sha256"],
        "dataset_manifest_status": dataset_manifest["status"],
        "checkpoint_sha256": {
            f"{spec.arm}_seed_{spec.seed}": spec.sha256 for spec in ordered_specs
        },
        "layout": layout,
        "support_threshold": card["feature_protocol"]["support_threshold"],
        "batch_size": batch_size,
        "encoding_cycles_for_utilization": cycles,
        "semantic_label_keys_accessed": 0,
        "semantic_fate_labels_derived": 0,
        "latents": latents,
    }
    features_path = field_dir / "features.pt"
    torch_save_once(features_path, payload)
    markers = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        **roots,
        "profile_decision_sha256": sha256_path(profile_path),
        "features_sha256": sha256_path(features_path),
        "started_at_epoch": started_at,
        "completed_at_epoch": completed_at,
        "duration_seconds": completed_at - started_at,
        "encoding_cycles": cycles,
        "model_count_per_cycle": len(models),
        "rows_per_model": observations.shape[0],
        "semantic_outcomes_accessed": 0,
    }
    write_json_once(field_dir / "markers.json", markers)


if __name__ == "__main__":
    main()
