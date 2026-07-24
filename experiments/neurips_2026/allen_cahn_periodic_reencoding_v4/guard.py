"""Strict outcome-blind execution guards for periodic reencoding v4."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any

from experiments.neurips_2026.allen_cahn_periodic_reencoding.io import (
    duplicate_safe_json,
    load_card,
    sha256_path,
    verify_source_manifest,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")
_SLURM_JOB_ID = re.compile(r"[1-9][0-9]*")
_BARE_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
_SMOKE_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "card_sha256",
    "source_manifest_sha256",
    "slurm_job_id",
    "runtime_path",
    "runtime_sha256",
    "start_marker_path",
    "start_marker_sha256",
    "end_marker_path",
    "end_marker_sha256",
    "telemetry_audit_path",
    "telemetry_audit_sha256",
    "raw_telemetry_path",
    "raw_telemetry_sha256",
    "scientific_outcomes_accessed",
}
_UUID_PROBE_KEYS = {
    "schema_version",
    "status",
    "card_sha256",
    "source_manifest_sha256",
    "slurm_job_id",
    "gpu_name",
    "gpu_uuid",
    "raw_uuid_type",
    "pytorch_uuid_raw_text",
    "pytorch_uuid_canonical",
    "nvidia_smi_uuid_raw_text",
    "nvidia_smi_uuid_canonical",
    "nvidia_smi_visible_gpu_count",
    "uuid_sources_match",
    "scientific_outcomes_accessed",
}


def _digest(value: str, *, label: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _verify_file(path: Path, expected: str, *, label: str) -> None:
    _digest(expected, label=label)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is not one regular, non-symlink file")
    observed = sha256_path(path)
    if observed != expected:
        raise RuntimeError(f"{label} hash mismatch: {observed} != {expected}")


def _load_frozen_inputs(
    *,
    card_path: Path,
    expected_card_sha256: str,
    source_manifest: Path,
    expected_source_manifest_sha256: str,
) -> tuple[dict[str, Any], str, str]:
    _digest(expected_card_sha256, label="card hash")
    _digest(expected_source_manifest_sha256, label="source-manifest hash")
    card, card_hash = load_card(
        card_path, expected_sha256=expected_card_sha256
    )
    source_hash = verify_source_manifest(
        card,
        path=source_manifest,
        expected_sha256=expected_source_manifest_sha256,
    )
    return card, card_hash, source_hash


def _bound_artifact(
    payload: dict[str, Any],
    *,
    root: Path,
    path_key: str,
    hash_key: str,
    relative_path: str,
) -> None:
    expected_path = root / relative_path
    if Path(str(payload.get(path_key, ""))) != expected_path:
        raise RuntimeError(f"{path_key} escaped the frozen smoke root")
    _verify_file(
        expected_path,
        str(payload.get(hash_key, "")),
        label=hash_key,
    )


def validate_smoke_artifacts(
    *,
    card_path: Path,
    expected_card_sha256: str,
    source_manifest: Path,
    expected_source_manifest_sha256: str,
    smoke_root: Path,
    expected_smoke_receipt_sha256: str,
    expected_uuid_probe_sha256: str,
) -> None:
    """Authenticate one outcome-free smoke and its real-CUDA identity probe."""

    card, card_hash, source_hash = _load_frozen_inputs(
        card_path=card_path,
        expected_card_sha256=expected_card_sha256,
        source_manifest=source_manifest,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
    )
    if smoke_root != Path(card["outcome_free_smoke"]["output_root"]):
        raise RuntimeError("Smoke root differs from the frozen card")
    receipt_path = smoke_root / "smoke_receipt.json"
    probe_path = smoke_root / "lineage_uuid_probe.json"
    _verify_file(
        receipt_path,
        expected_smoke_receipt_sha256,
        label="smoke-receipt hash",
    )
    _verify_file(
        probe_path,
        expected_uuid_probe_sha256,
        label="UUID-probe hash",
    )
    receipt = duplicate_safe_json(receipt_path)
    if set(receipt) != _SMOKE_RECEIPT_KEYS:
        raise RuntimeError("Smoke receipt has an unexpected strict schema")
    expected_receipt = {
        "schema_version": 1,
        "status": "passed_outcome_free_gpu_smoke",
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "scientific_outcomes_accessed": False,
    }
    if any(receipt.get(key) != value for key, value in expected_receipt.items()):
        raise RuntimeError("Smoke receipt status, freeze, or outcome guard failed")
    smoke_job_id = receipt.get("slurm_job_id")
    if not isinstance(smoke_job_id, str) or _SLURM_JOB_ID.fullmatch(smoke_job_id) is None:
        raise RuntimeError("Smoke receipt has an invalid SLURM job ID")
    for path_key, hash_key, relative_path in (
        ("runtime_path", "runtime_sha256", "smoke_runtime.json"),
        ("start_marker_path", "start_marker_sha256", "markers/smoke_start.json"),
        ("end_marker_path", "end_marker_sha256", "markers/smoke_end.json"),
        (
            "telemetry_audit_path",
            "telemetry_audit_sha256",
            "smoke_telemetry_audit.json",
        ),
        ("raw_telemetry_path", "raw_telemetry_sha256", "raw_gpu_telemetry.csv"),
    ):
        _bound_artifact(
            receipt,
            root=smoke_root,
            path_key=path_key,
            hash_key=hash_key,
            relative_path=relative_path,
        )

    probe = duplicate_safe_json(probe_path)
    if set(probe) != _UUID_PROBE_KEYS:
        raise RuntimeError("UUID probe has an unexpected strict schema")
    expected_probe = {
        "schema_version": 1,
        "status": "passed_real_cuda_uuid_crosscheck_strict_json",
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "slurm_job_id": smoke_job_id,
        "raw_uuid_type": "_CUuuid",
        "nvidia_smi_visible_gpu_count": 1,
        "uuid_sources_match": True,
        "scientific_outcomes_accessed": False,
    }
    if any(probe.get(key) != value for key, value in expected_probe.items()):
        raise RuntimeError("UUID probe status, freeze, job, or outcome guard failed")
    bare = probe.get("pytorch_uuid_canonical")
    if not isinstance(bare, str) or _BARE_UUID.fullmatch(bare) is None:
        raise RuntimeError("UUID probe lacks the exact canonical PyTorch UUID")
    gpu_uuid = f"GPU-{bare}"
    if (
        probe.get("pytorch_uuid_raw_text") != bare
        or probe.get("gpu_uuid") != gpu_uuid
        or probe.get("nvidia_smi_uuid_raw_text") != gpu_uuid
        or probe.get("nvidia_smi_uuid_canonical") != gpu_uuid
        or not isinstance(probe.get("gpu_name"), str)
        or not str(probe["gpu_name"]).strip()
    ):
        raise RuntimeError("UUID probe cross-source identity fields disagree")


def validate_outcome_guard(
    *,
    card_path: Path,
    expected_card_sha256: str,
    source_manifest: Path,
    expected_source_manifest_sha256: str,
    output_root: Path,
    expected_scientific_job_id: str,
) -> str:
    """Authenticate only metric-free summary authority and return its digest."""

    card, card_hash, source_hash = _load_frozen_inputs(
        card_path=card_path,
        expected_card_sha256=expected_card_sha256,
        source_manifest=source_manifest,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
    )
    if output_root != Path(card["prospective_datasets"]["output_root"]):
        raise RuntimeError("Scientific root differs from the frozen card")
    if _SLURM_JOB_ID.fullmatch(expected_scientific_job_id) is None:
        raise ValueError("Expected scientific job ID is invalid")
    guard_path = output_root / "outcome_guard_receipt.json"
    if not guard_path.is_file() or guard_path.is_symlink():
        raise RuntimeError("Outcome guard is not one regular, non-symlink file")
    guard = duplicate_safe_json(guard_path)
    expected = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "status": "authorized_for_dependent_cpu_summary",
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "slurm_job_id": expected_scientific_job_id,
        "scientific_payload_opened": False,
        "forecast_outcomes_accessed": False,
        "both_disjoint_windows_required": True,
        "disjoint_window_ordering_verified": True,
        "evaluation_end_epoch_identity_verified": True,
    }
    if any(guard.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Outcome guard status, job, freeze, or no-access gate failed")
    if Path(str(guard.get("runtime_lineage_path", ""))) != output_root / "runtime_lineage.json":
        raise RuntimeError("Outcome guard runtime lineage escaped the scientific root")
    if Path(str(guard.get("scientific_payload_path", ""))) != output_root / "scientific_payload.json":
        raise RuntimeError("Outcome guard scientific payload escaped the scientific root")
    checks = guard.get("validity_checks")
    expected_windows = {"selection_validity", "evaluation_validity"}
    if (
        not isinstance(checks, dict)
        or set(checks) != expected_windows
        or any(
            not isinstance(row, dict)
            or not row
            or any(value is not True for value in row.values())
            for row in checks.values()
        )
    ):
        raise RuntimeError("Outcome guard lacks two passing telemetry windows")
    return sha256_path(guard_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "outcome"):
        child = subparsers.add_parser(name)
        child.add_argument("--card", type=Path, required=True)
        child.add_argument("--expected-card-sha256", required=True)
        child.add_argument("--source-manifest", type=Path, required=True)
        child.add_argument("--expected-source-manifest-sha256", required=True)
        child.add_argument("--output-root", type=Path, required=True)
        if name == "smoke":
            child.add_argument("--expected-smoke-receipt-sha256", required=True)
            child.add_argument("--expected-uuid-probe-sha256", required=True)
        else:
            child.add_argument("--expected-scientific-job-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common = {
        "card_path": args.card,
        "expected_card_sha256": args.expected_card_sha256,
        "source_manifest": args.source_manifest,
        "expected_source_manifest_sha256": args.expected_source_manifest_sha256,
    }
    if args.command == "smoke":
        validate_smoke_artifacts(
            **common,
            smoke_root=args.output_root,
            expected_smoke_receipt_sha256=args.expected_smoke_receipt_sha256,
            expected_uuid_probe_sha256=args.expected_uuid_probe_sha256,
        )
        return
    digest = validate_outcome_guard(
        **common,
        output_root=args.output_root,
        expected_scientific_job_id=args.expected_scientific_job_id,
    )
    print(digest, flush=True)


if __name__ == "__main__":
    main()
