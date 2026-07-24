"""Generate one untouched T40 dataset after the frozen conditional guard passes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

from experiments.neurips_2026.allen_cahn_mechanistic_bridge.conditional_guard import (
    load_and_validate,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.integrity import (
    verify_source_manifest,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.io import (
    CARD_PATH,
    load_card,
    sha256_path,
    verify_file,
    write_json_once,
)


SOURCE_MANIFEST = Path(__file__).with_name("source_manifest.sha256")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_index", type=int, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected_decision_sha256", required=True)
    parser.add_argument("--expected_card_sha256", required=True)
    parser.add_argument("--expected_source_manifest_sha256", required=True)
    parser.add_argument("--card", type=Path, default=CARD_PATH)
    return parser.parse_args()


def _load_pinned_generator(card: dict):
    source = card["inputs"]["pinned_generator_source"]
    benchmark = card["inputs"]["pinned_generator_benchmark_source"]
    verify_file(Path(source["path"]), str(source["sha256"]))
    verify_file(Path(benchmark["path"]), str(benchmark["sha256"]))
    scratch_root = Path(source["path"]).parents[2]
    if "skae" in sys.modules:
        raise RuntimeError("skae was imported before the pinned generator root was installed")
    sys.path.insert(0, str(scratch_root))
    spec = importlib.util.spec_from_file_location(
        "_pinned_allen_cahn_bridge_generator", source["path"]
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load pinned generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for vectorized data generation")
    card, card_hash = load_card(args.card)
    if card_hash != args.expected_card_sha256:
        raise RuntimeError("Bridge card differs from launcher root")
    source_hash = verify_source_manifest(SOURCE_MANIFEST)
    if source_hash != args.expected_source_manifest_sha256:
        raise RuntimeError("Bridge source manifest differs from launcher root")
    _, decision_hash, guard = load_and_validate(
        args.decision,
        expected_sha256=args.expected_decision_sha256,
        card=card,
    )
    seeds = [int(value) for value in card["new_datasets"]["seeds"]]
    if args.task_index < 0 or args.task_index >= len(seeds):
        raise ValueError("Generation task is outside frozen seed roster")
    seed = seeds[args.task_index]
    root = Path(card["new_datasets"]["output_root"])
    output = root / card["new_datasets"]["paths"][args.task_index]
    summary = Path(str(output) + ".summary.json")
    if output.exists() or summary.exists():
        raise FileExistsError(f"Refusing to overwrite frozen dataset {output}")

    module = _load_pinned_generator(card)
    frozen = card["new_datasets"]["generation"]
    protocol = module.AllenCahnRebuttalProtocol(
        grid_size=int(frozen["grid_size"]),
        diffusion=float(frozen["diffusion"]),
        rk4_dt=float(frozen["rk4_dt"]),
        substeps_per_observation=int(frozen["substeps_per_observation"]),
        trajectory_length=int(frozen["trajectory_length"]),
        label_extra_observations=int(frozen["label_extra_observations"]),
        train_trajectories=int(frozen["train_trajectories"]),
        val_trajectories=int(frozen["val_trajectories"]),
        test_trajectories=int(frozen["test_trajectories"]),
        seed=seed,
        min_regions=int(frozen["min_regions"]),
        max_regions=int(frozen["max_regions"]),
        mask_temperature=float(frozen["mask_temperature"]),
        low_frequency_cutoff=int(frozen["low_frequency_cutoff"]),
        noise_scale=float(frozen["noise_scale"]),
        require_min_area_fraction=float(frozen["require_min_area_fraction"]),
        beta=float(frozen["allen_cahn_beta"]),
        reaction_strength=float(frozen["allen_cahn_reaction_strength"]),
        center_radius=float(frozen["allen_cahn_center_radius"]),
    )
    bundle = module.generate_dataset(
        protocol,
        device=str(frozen["device"]),
        batch_size=int(frozen["batch_size"]),
        compile_step=False,
    )
    fields = bundle["fields"]
    expected_shape = (
        int(frozen["val_trajectories"]),
        int(frozen["trajectory_length"]) + 1,
        int(frozen["grid_size"]),
        int(frozen["grid_size"]),
        int(frozen["channels"]),
    )
    if tuple(fields.shape) != expected_shape or not bool(torch.isfinite(fields).all()):
        raise RuntimeError("Generated fields violate shape or finite-value contract")
    output.parent.mkdir(parents=True, exist_ok=True)
    module.save_dataset(bundle, output)
    record = {
        "schema_version": 1,
        "status": "generated_not_semantically_scored",
        "seed": seed,
        "task_index": int(args.task_index),
        "path": str(output),
        "sha256": sha256_path(output),
        "fields_shape": list(expected_shape),
        "protocol": asdict(protocol),
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "mechanism_decision_sha256": decision_hash,
        "conditional_guard": guard,
        "semantic_outcomes_printed": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_recorded"),
    }
    write_json_once(summary, record)
    print(json.dumps({"status": record["status"], "seed": seed, "sha256": record["sha256"]}))


if __name__ == "__main__":
    main()
