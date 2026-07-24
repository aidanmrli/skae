"""Authenticated source contract for the negative distinct-laws V2 packet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.neurips_2026.paths import PAPER_DATA_DIR, REPO_ROOT


PACKET_ID = "global_k_distinct_laws_v2_negative"
PROTOCOL_ID = "global_k_distinct_laws_gated_local_linear_v2_new_seeds"
DEFAULT_PACKET_DIR = PAPER_DATA_DIR / PACKET_ID
SCIENTIFIC_ROOT = Path(
    "/network/scratch/l/lia/skae/global_k_distinct_laws_v2_scientific_663fd03"
)
FREEZE_ROOT = Path(
    "/network/scratch/l/lia/skae/global_k_distinct_laws_v2_20260720_freeze_663fd03"
)

SOURCE_FILES = {
    "packet": {
        "path": SCIENTIFIC_ROOT / "packet/distinct_laws_v2_packet.json",
        "sha256": "e0317c2cf02965649afa9ac627e2daf0f7f49c1448aadbee000669f0a5c7b505",
    },
    "decision": {
        "path": SCIENTIFIC_ROOT / "summary/decision.json",
        "sha256": "1200b235817c4f2f0628d64f14469e010820591e4ed0c0a8c75e2367a9c89bac",
    },
    "supplemental_audit": {
        "path": SCIENTIFIC_ROOT / "supplemental_audit/audit.json",
        "sha256": "8a8e37b10d5b9854d11732201f9cc30342b9d9949d6d30a8891a0f84c4f710ed",
    },
    "gpu_assessment": {
        "path": SCIENTIFIC_ROOT / "telemetry/scientific_gpu_assessment.json",
        "sha256": "7eabe3d7e5828b09a67968586fb7c9b3392363be3b66c0190bf93fa00c79f152",
    },
    "checkpoint_audit": {
        "path": SCIENTIFIC_ROOT / "audit/summary.json",
        "sha256": "f6fb9378d55b4e349798a644627063378911a3507d241ef3012e64a6e6c2fec4",
    },
    "card": {
        "path": REPO_ROOT
        / "experiments/neurips_2026/global_k_distinct_laws_v2_card.json",
        "sha256": "663fd03ddf9bfacabeef616f2a74f24998460d78b28413fdfeb42b012712f45b",
    },
    "task_tsv": {
        "path": FREEZE_ROOT / "tasks/full_tasks.tsv",
        "sha256": "5b1d16ec52e3cf1ea695abbfee95ccc3c5e7b7e1f8a734209791f05d940a0e83",
    },
    "task_manifest": {
        "path": FREEZE_ROOT / "tasks/full_manifest.json",
        "sha256": "0c34ab6230f0034bb93a7a7d4179d1696da6c5c96d5346c8d21685a784e8451a",
    },
    "source_lock": {
        "path": FREEZE_ROOT / "source_lock.json",
        "sha256": "a21c8e554d929dff42aeb66f91f7d5d7f1d5ac8ce7d6d3fd76668f47775779be",
    },
}

DATA_FILES = (
    "decision.json",
    "seed_rows.csv",
    "basin_rows.csv",
    "provenance.json",
)
FILES = (*DATA_FILES, "evidence_manifest.json")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"Expected a JSON object: {path}")
    return payload


def load_sources() -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for name, source in SOURCE_FILES.items():
        path = Path(source["path"])
        require(path.is_file(), f"Missing distinct-laws V2 source: {path}")
        require(
            sha256_path(path) == source["sha256"],
            f"Distinct-laws V2 source hash drifted: {name}",
        )
        if path.suffix == ".json":
            loaded[name] = load_json(path)
    return loaded


def validate_source_agreement(source: Mapping[str, Any]) -> None:
    packet = source["packet"]
    decision = source["decision"]
    audit = source["supplemental_audit"]
    gpu = source["gpu_assessment"]
    checkpoint_audit = source["checkpoint_audit"]
    card = source["card"]
    for payload in (packet, decision, gpu, checkpoint_audit, card):
        require(payload.get("protocol_id") == PROTOCOL_ID, "Protocol ID drifted")
    require(
        audit.get("protocol_id")
        == "global_k_distinct_laws_v2_supplemental_integrity_audit",
        "Supplemental protocol ID drifted",
    )
    require(
        packet["decision"]
        == {key: decision[key] for key in packet["decision"]},
        "Packet and decision values disagree",
    )
    require(
        packet["scientific_gpu_assessment"] == gpu,
        "Packet and GPU-assessment values disagree",
    )
    reproduction = audit["independent_adjudication_reproduction"]
    require(
        audit.get("status") == "passed"
        and audit.get("integrity_passed") is True
        and reproduction.get("passed") is True
        and reproduction.get("byte_equality") is True
        and reproduction.get("parsed_value_equality") is True
        and reproduction.get("original_decision_sha256")
        == SOURCE_FILES["decision"]["sha256"],
        "Independent supplemental adjudication is not exact",
    )
    require(
        decision.get("mechanism_tier") == "invalid"
        and decision.get("validity", {}).get("passed") is False
        and decision.get("sparse_gates", {}).get("joint_H_G") is False
        and decision.get("specificity", {}).get("passed") is False,
        "The frozen negative decision changed polarity",
    )
    require(
        audit.get("recommended_mechanism_text") == decision["mechanism_text"]
        and audit.get("recommended_relative_specificity_text")
        == decision["relative_specificity_text"],
        "Supplemental claim guard disagrees with the frozen decision",
    )
    require(
        checkpoint_audit.get("status") == "passed"
        and checkpoint_audit.get("passed_count") == 20
        and checkpoint_audit.get("arm_counts") == {"sparse": 10, "dense": 10},
        "Checkpoint audit roster or status drifted",
    )
    require(
        gpu.get("assessment_complete") is True
        and gpu.get("flagged_low_utilization") is False
        and gpu.get("outcomes_inspected") is False,
        "Scientific GPU assessment drifted",
    )
    require(
        card["new_seed_contract"]["scientific_seeds"] == list(range(100, 110))
        and card["benchmark"]["known_evaluation_basin_count"] == 3,
        "Scientific seed or basin roster drifted",
    )
