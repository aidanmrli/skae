"""Checks for the active Allen--Cahn support-subspace evidence packet."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.neurips_2026.evidence.allen_cahn_support_subspace_packet import (
    DEFAULT_OUTPUT_DIR,
    EXPECTED_HASHES,
    FILES,
    validate_packet,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_active_packet_is_exact_and_preserves_the_claim_boundary() -> None:
    result = validate_packet(DEFAULT_OUTPUT_DIR)
    assert result == {
        "decision": "failed",
        "seed_count": 10,
        "eligible_seed_count": 0,
        "projected_vs_dense_full_passed": True,
    }
    assert set(EXPECTED_HASHES) == set(FILES)


def test_active_manifest_wires_the_packet_and_canonical_check() -> None:
    manifest = json.loads(
        (REPO_ROOT / "docs/figures/neurips_paper_2026/manifest.json").read_text()
    )
    groups = {group["id"]: group for group in manifest["evidence_groups"]}
    group = groups["allen_cahn_support_subspaces_v4"]
    assert group["build_tool"].endswith("allen_cahn_support_subspace_packet")
    assert group["check_command_inside_allocation"] == (
        "uv run skae-paper build allen-cahn-support-subspaces --check"
    )
    assert group["manifest"].endswith("evidence_manifest.json")
