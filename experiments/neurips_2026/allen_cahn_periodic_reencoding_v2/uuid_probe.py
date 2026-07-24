"""Outcome-free strict-JSON probe for the real CUDA UUID runtime type."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from experiments.neurips_2026.allen_cahn_periodic_reencoding.io import (
    load_card,
    sha256_path,
    verify_source_manifest,
    write_json_once,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--expected-card-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def strict_uuid_record(raw_uuid: object) -> dict[str, str]:
    """Require and serialize the exact runtime type that broke v1."""

    raw_type = type(raw_uuid).__name__
    value = str(raw_uuid).strip()
    if raw_type != "_CUuuid":
        raise RuntimeError(f"Expected PyTorch _CUuuid, observed {raw_type}")
    if not value.startswith("GPU-") or len(value) < 8:
        raise RuntimeError("Serialized CUDA UUID is empty or malformed")
    return {"gpu_uuid": value, "raw_uuid_type": raw_type}


def main() -> None:
    args = parse_args()
    card, card_hash = load_card(args.card, expected_sha256=args.expected_card_sha256)
    source_hash = verify_source_manifest(
        card,
        path=args.source_manifest,
        expected_sha256=args.expected_source_manifest_sha256,
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The UUID probe requires exactly one visible CUDA GPU")
    properties = torch.cuda.get_device_properties(torch.device("cuda"))
    raw_uuid = getattr(properties, "uuid", "not_recorded")
    slurm_job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    if not slurm_job_id:
        raise RuntimeError("The UUID probe must run inside its smoke SLURM job")
    payload = {
        "schema_version": 1,
        "status": "passed_real_cuda_uuid_strict_json",
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "slurm_job_id": slurm_job_id,
        "gpu_name": torch.cuda.get_device_name(),
        **strict_uuid_record(raw_uuid),
        "scientific_outcomes_accessed": False,
    }
    path = args.output_root / "lineage_uuid_probe.json"
    write_json_once(path, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "probe_sha256": sha256_path(path),
                "scientific_outcomes_accessed": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
