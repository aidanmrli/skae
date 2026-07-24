"""Checks for the frozen Allen--Cahn half-global/half-local negative packet."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.neurips_2026.evidence.allen_cahn_half_local_packet import (
    DEFAULT_PACKET_DIR,
    EXPECTED_HASHES,
    FILES,
    validate_packet,
)
from experiments.neurips_2026.cli import CHECKS


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_packet_is_exact_and_preserves_the_negative_scope() -> None:
    result = validate_packet(DEFAULT_PACKET_DIR)
    assert result["decision"] == "do_not_promote"
    assert result["model_seed_count"] == 1
    assert result["all_six_local_recipes_selected_zero_update"] is True
    assert result["mean_local_over_global"] > 1.0
    assert result["terminal_local_over_global"] > 1.0
    assert result["route_coverage"] < 0.90
    assert set(EXPECTED_HASHES) == set(FILES)


def test_manifest_wires_the_negative_packet_and_canonical_check() -> None:
    manifest = json.loads(
        (REPO_ROOT / "docs/figures/neurips_paper_2026/manifest.json").read_text()
    )
    groups = {group["id"]: group for group in manifest["evidence_groups"]}
    group = groups["allen_cahn_half_global_half_local_negative"]
    assert group["build_tool"].endswith("allen_cahn_half_local_packet")
    assert group["check_command_inside_allocation"] == (
        "uv run skae-paper build allen-cahn-half-local --check"
    )
    assert group["manifest"].endswith("evidence_manifest.json")
    assert (
        "experiments.neurips_2026.evidence.allen_cahn_half_local_packet",
        ("--check",),
    ) in CHECKS
