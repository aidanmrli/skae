"""Evaluate a refined spatial LISTA checkpoint on opened validation panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

from experiments.neurips_2026.allen_cahn_periodic_reencoding.core import (
    evaluate_model_packed,
)

# The versioned model lives in the preserved historical source tree, while the
# evaluator and validated streaming rollout kernel live in the active project.
sys.path.insert(0, "/network/scratch/l/lia/skae-rebuttal")
from skae.benchmarks.spatialized_conv_koopman_refined import (
    SpatialConvKoopman,
    SpatialConvKoopmanConfig,
)


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field", action="append", nargs=3, metavar=("SEED", "PATH", "SHA256"), required=True)
    parser.add_argument("--periods", type=int, nargs="*", default=[1, 2, 5, 10, 20, 40, 50, 100])
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=768)
    return parser


def _summarize(result: dict[str, object]) -> dict[str, object]:
    instantaneous = result["instantaneous_field_mse"]
    cumulative = result["cumulative_field_mse"]
    return {
        "horizon": int(result["horizon"]),
        "cumulative_field_mse": float(cumulative[-1]),
        "terminal_field_mse": float(instantaneous[-1]),
        "instantaneous_field_mse": instantaneous,
        "cumulative_curve": cumulative,
    }


def main() -> None:
    args = _parser().parse_args()
    checkpoint = _load(args.checkpoint)
    cfg = SpatialConvKoopmanConfig.from_mapping(checkpoint["model_config"])
    if cfg.z_dim < 4 * cfg.channels * cfg.grid_size * cfg.grid_size:
        raise RuntimeError("latent is not at least fourfold overcomplete")
    model = SpatialConvKoopman(cfg).cuda().eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset_metadata = []
    field_tensors = []
    for seed_text, path_text, expected_hash in args.field:
        path = Path(path_text)
        observed_hash = _sha256(path)
        if observed_hash != expected_hash:
            raise RuntimeError(f"field hash mismatch for {path}")
        payload = _load(path)
        fields = payload["fields"].reshape(payload["fields"].shape[0], payload["fields"].shape[1], -1)
        if fields.shape[1] < args.horizon + 1 or fields.shape[2] != model.observation_size:
            raise ValueError(f"unexpected field shape {tuple(fields.shape)}")
        dataset_metadata.append(
            {"seed": int(seed_text), "path": str(path), "sha256": observed_hash}
        )
        field_tensors.append(fields[: , : args.horizon + 1])

    packed_fields = torch.stack(field_tensors).to(device="cuda", dtype=torch.float32)
    policies_by_dataset = [dict() for _ in dataset_metadata]
    policy_specs = [("direct", None)] + [
        (f"period_{period}", period) for period in sorted(set(args.periods))
    ]
    for policy_name, period in policy_specs:
        results = evaluate_model_packed(
            model,
            packed_fields,
            horizon=args.horizon,
            period=period,
            batch_size=args.batch_size,
            max_decode_segment=100,
        )
        if len(results) != len(dataset_metadata):
            raise AssertionError("packed evaluator lost a validation panel")
        for dataset_index, result in enumerate(results):
            policies_by_dataset[dataset_index][policy_name] = _summarize(result)
    datasets = [
        metadata | {"policies": policies}
        for metadata, policies in zip(dataset_metadata, policies_by_dataset)
    ]
    output = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "model_config": cfg.to_dict(),
        "datasets": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
