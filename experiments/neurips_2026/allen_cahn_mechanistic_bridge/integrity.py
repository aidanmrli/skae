"""Exact-roster source integrity checks for the dormant bridge workflow."""

from __future__ import annotations

from pathlib import Path

from experiments.neurips_2026.allen_cahn_mechanistic_bridge.io import sha256_path


REQUIRED_SOURCE_PATHS = {
    "experiments/neurips_2026/allen_cahn_mechanistic_bridge/aggregation.py",
    "experiments/neurips_2026/allen_cahn_mechanistic_bridge/conditional_guard.py",
    "experiments/neurips_2026/allen_cahn_mechanistic_bridge/extract_field_only.py",
    "experiments/neurips_2026/allen_cahn_mechanistic_bridge/families.py",
    "experiments/neurips_2026/allen_cahn_mechanistic_bridge/freeze_datasets.py",
    "experiments/neurips_2026/allen_cahn_mechanistic_bridge/freeze_field_artifacts.py",
    "experiments/neurips_2026/allen_cahn_mechanistic_bridge/generate.py",
    "experiments/neurips_2026/allen_cahn_mechanistic_bridge/generation_telemetry.py",
    "experiments/neurips_2026/allen_cahn_mechanistic_bridge/integrity.py",
    "experiments/neurips_2026/allen_cahn_mechanistic_bridge/io.py",
    "experiments/neurips_2026/allen_cahn_mechanistic_bridge/prediction_card.json",
    "experiments/neurips_2026/allen_cahn_mechanistic_bridge/probes.py",
    "experiments/neurips_2026/allen_cahn_mechanistic_bridge/reduce_label_aware.py",
    "experiments/neurips_2026/allen_cahn_mechanistic_bridge/rollouts.py",
    "experiments/neurips_2026/allen_cahn_mechanistic_bridge/statistics.py",
    "experiments/neurips_2026/allen_cahn_mechanistic_bridge/telemetry.py",
    "experiments/neurips_2026/allen_cahn_mechanistic_bridge/wrong_supports.py",
    "experiments/neurips_2026/allen_cahn_support_subspaces/evaluation_helpers.py",
    "experiments/neurips_2026/allen_cahn_support_subspaces/io.py",
    "experiments/neurips_2026/allen_cahn_support_subspaces/metrics.py",
    "experiments/neurips_2026/allen_cahn_support_subspaces/prediction_card.json",
    "experiments/neurips_2026/allen_cahn_support_subspaces/select_profile.py",
    "experiments/neurips_2026/allen_cahn_support_subspaces/summarize_gpu_telemetry.py",
    "scripts/neurips_2026/allen_cahn_mechanistic_bridge/run_dataset_manifest.sh",
    "scripts/neurips_2026/allen_cahn_mechanistic_bridge/run_extract.sh",
    "scripts/neurips_2026/allen_cahn_mechanistic_bridge/run_field_manifest.sh",
    "scripts/neurips_2026/allen_cahn_mechanistic_bridge/run_generate.sh",
    "scripts/neurips_2026/allen_cahn_mechanistic_bridge/run_reduce.sh",
    "tests/test_allen_cahn_mechanistic_bridge.py",
}


def verify_source_manifest(path: Path) -> str:
    seen: dict[str, str] = {}
    for line in path.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise RuntimeError(f"Malformed source manifest line: {line!r}")
        digest, relative = parts
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or relative in seen:
            raise RuntimeError(f"Unsafe or duplicate source path: {relative}")
        seen[relative] = digest
    if set(seen) != REQUIRED_SOURCE_PATHS:
        raise RuntimeError(
            f"Source roster mismatch: {sorted(set(seen) ^ REQUIRED_SOURCE_PATHS)}"
        )
    for relative, expected in seen.items():
        observed = sha256_path(Path(relative))
        if observed != expected:
            raise RuntimeError(f"Source hash mismatch for {relative}")
    return sha256_path(path)
