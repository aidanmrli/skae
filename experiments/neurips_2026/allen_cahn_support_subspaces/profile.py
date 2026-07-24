"""Synthetic-only GPU profile of the frozen Allen--Cahn rollout kernel."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch

from experiments.neurips_2026.allen_cahn_support_subspaces.evaluation_helpers import (
    verify_source_manifest,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.io import (
    CARD_PATH,
    checkpoint_roster,
    load_card,
)


SOURCE_MANIFEST = Path(__file__).with_name("source_manifest.sha256")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--minimum_seconds", type=float, required=True)
    parser.add_argument("--card", type=Path, default=CARD_PATH)
    parser.add_argument("--expected_card_sha256", required=True)
    parser.add_argument("--expected_source_manifest_sha256", required=True)
    parser.add_argument("--gpu_selector", required=True)
    parser.add_argument("--ready_file", type=Path, required=True)
    parser.add_argument("--start_file", type=Path, required=True)
    return parser.parse_args()


@torch.no_grad()
def run_kernel(
    models: list[torch.nn.Module],
    batch_size: int,
    horizons: list[int],
    closure_batch_size: int,
    historical_batch_size: int,
) -> None:
    horizon = max(horizons)
    model = models[0]
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(20260726 + int(batch_size))
    latent_dim = int(model.cfg.z_dim)
    z0 = torch.randn(batch_size, latent_dim, generator=generator, device=device)
    cardinality = latent_dim // 2
    mask = torch.zeros(batch_size, latent_dim, device=device, dtype=z0.dtype)
    mask[:, :cardinality] = 1.0
    historical_x0 = torch.randn(
        historical_batch_size,
        int(model.observation_size),
        generator=generator,
        device=device,
    )
    for current_model in models:
        for reproduction_horizon in horizons:
            current_model.rollout_observation_discrete(
                historical_x0, horizon=int(reproduction_horizon)
            )
        states = torch.cat((z0, z0 * mask, z0 * mask), dim=0)
        for _ in range(int(horizon)):
            states = current_model.step_latent(states)
            states[2 * batch_size :] *= mask
            current_model.decode(states)
        closure_states = torch.randn(
            closure_batch_size, latent_dim, generator=generator, device=device
        )
        closure_masks = torch.zeros_like(closure_states)
        closure_masks[:, :cardinality] = 1.0
        gated = current_model.step_latent(closure_states * closure_masks)
        (gated * (1.0 - closure_masks)).square().sum()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU profiling")
    card, card_hash = load_card(args.card)
    if card_hash != args.expected_card_sha256:
        raise RuntimeError("Card differs from pre-profile launcher root of trust")
    candidates = [int(value) for value in card["hardware_profile"]["candidate_batch_sizes"]]
    if int(args.batch_size) not in candidates:
        raise ValueError("Batch size is outside the frozen candidate roster")
    if float(args.minimum_seconds) != float(card["hardware_profile"]["minimum_profile_seconds_each"]):
        raise ValueError("Profile duration differs from the frozen card")
    source_hash = verify_source_manifest(SOURCE_MANIFEST)
    if source_hash != args.expected_source_manifest_sha256:
        raise RuntimeError("Source manifest differs from pre-profile launcher root of trust")
    roster = checkpoint_roster(card)
    from experiments.neurips_2026.allen_cahn_support_subspaces.io import load_model

    models = [load_model(roster[(arm, 64)], card, "cuda")[0] for arm in ("sparse", "dense")]
    closure_batch_size = int(card["hardware_profile"]["closure_state_batch_size"])
    historical_batch_size = int(
        card["inputs"]["ordinary_forecast_seed_rows"][
            "historical_reproduction_batch_size"
        ]
    )
    reproduction_horizons = [int(value) for value in card["inputs"][
        "ordinary_forecast_seed_rows"
    ]["historical_evaluator_horizon_sequence"]]
    torch.set_float32_matmul_precision("high")
    for _ in range(2):
        run_kernel(
            models, int(args.batch_size), reproduction_horizons,
            closure_batch_size, historical_batch_size,
        )
        torch.cuda.synchronize()
    if args.ready_file.exists() or args.start_file.exists():
        raise FileExistsError("Profile telemetry handshake file already exists")
    args.ready_file.write_text("ready\n")
    deadline = time.monotonic() + 60.0
    while not args.start_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for telemetry start signal")
        time.sleep(0.05)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    iterations = 0
    while time.perf_counter() - started < float(args.minimum_seconds):
        run_kernel(
            models, int(args.batch_size), reproduction_horizons,
            closure_batch_size, historical_batch_size,
        )
        torch.cuda.synchronize()
        iterations += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    device_uuid = subprocess.run(
        ["nvidia-smi", "-i", args.gpu_selector, "--query-gpu=uuid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()[0]
    payload = {
        "schema_version": 1,
        "status": "completed",
        "synthetic_inputs_only": True,
        "outcomes_accessed": False,
        "datasets_opened": False,
        "telemetry_scope": (
            "post-warmup synchronized H80/H120/H160/H200 historical-provenance "
            "plus scientific-kernel interval"
        ),
        "historical_provenance_kernel_profiled": True,
        "historical_reproduction_batch_size": historical_batch_size,
        "historical_reproduction_horizons": reproduction_horizons,
        "batch_size": int(args.batch_size),
        "horizon": int(max(card["roster"]["horizons"])),
        "closure_state_batch_size": closure_batch_size,
        "resident_model_count": len(models),
        "profile_seconds": elapsed,
        "iterations": iterations,
        "profile_iterations_per_second": iterations / elapsed,
        "historical_trajectory_rollouts_per_iteration": (
            len(models) * historical_batch_size * len(reproduction_horizons)
        ),
        "scientific_three_mode_source_trajectories_per_iteration": (
            len(models) * int(args.batch_size)
        ),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "device_name": properties.name,
        "device_uuid": device_uuid,
        "device_total_memory_bytes": int(properties.total_memory),
        "visible_cuda_device_count": int(torch.cuda.device_count()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_recorded"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_recorded"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS", "not_recorded"),
        "nvidia_smi_gpu_selector": args.gpu_selector,
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "checkpoint_sha256": roster[("sparse", 64)].sha256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "completed", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
