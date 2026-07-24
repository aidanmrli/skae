"""Checks for the compact authenticated negative distinct-laws V2 packet."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from experiments.neurips_2026.cli import CHECKS, COMMANDS
from experiments.neurips_2026.evidence.global_k_distinct_laws_v2_contract import (
    DEFAULT_PACKET_DIR,
    FILES,
    SOURCE_FILES,
)
from experiments.neurips_2026.evidence.global_k_distinct_laws_v2_negative import (
    EXPECTED_HASHES,
    validate_packet,
    validate_sources,
)


MODULE = "experiments.neurips_2026.evidence.global_k_distinct_laws_v2_negative"


def _csv_rows(name: str) -> list[dict[str, str]]:
    with (DEFAULT_PACKET_DIR / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_packet_is_exact_and_preserves_only_the_negative_decision() -> None:
    result = validate_packet(DEFAULT_PACKET_DIR)
    assert result == {
        "decision": "invalid_negative",
        "positive_claim_permitted": False,
        "checkpoint_count": 20,
        "basin_count": 3,
        "joint_h_g_seed_passes": 0,
    }
    assert set(EXPECTED_HASHES) == set(FILES)

    decision = json.loads((DEFAULT_PACKET_DIR / "decision.json").read_text())
    assert decision["mechanism_tier"] == "invalid"
    assert decision["relative_specificity_tier"] == (
        "dense_invalid_specificity_unresolved"
    )
    assert decision["validity"]["joint_H_G_kink_complete_seeds"] == 6
    assert decision["validity"]["joint_H_G_kink_pairs"] == 24
    assert decision["sparse_gates"]["H_nearest_rows"] == 9
    assert decision["sparse_gates"]["G_nearest_rows"] == 24
    assert decision["sparse_gates"]["joint_finite_neighborhood_rows_passing_every_gate"] == 0
    assert decision["sparse_gates"]["active_code_cloud_closure_seed_passes"] == 5
    assert decision["sparse_gates"]["active_code_cloud_rows_at_most_0.50"] == 23
    assert decision["dense_specificity"]["dense_H_global_positive_seeds"] == 9
    assert decision["dense_specificity"]["passed"] is False
    assert decision["positive_claim_permitted"] is False
    assert len(decision["rebuttal_use"]["forbidden"]) == 4


def test_seed_basin_checkpoint_and_label_firewall_evidence_is_complete() -> None:
    decision = json.loads((DEFAULT_PACKET_DIR / "decision.json").read_text())
    assert decision["benchmark"]["scientific_model_seeds"] == list(range(100, 110))
    assert decision["benchmark"]["known_evaluation_basin_count"] == 3
    assert decision["benchmark"]["label_policy"] == {
        "training_uses_basin_labels_or_count": False,
        "evaluation_only_use_of_labels_and_count": True,
    }
    assert decision["benchmark"]["family_corpus"]["corpus_seed"] == 20260726
    assert decision["benchmark"]["family_corpus"]["trajectory_split_seed"] == 20260727
    assert decision["benchmark"]["evaluation_seeds"] == {
        "calibration_seed": 20260728,
        "verification_seed": 20260729,
        "coordinate_null_seed": 20260730,
        "finite_radius_direction_seed": 20260731,
    }

    seeds = _csv_rows("seed_rows.csv")
    assert len(seeds) == 20
    assert len({row["checkpoint_sha256"] for row in seeds}) == 20
    assert all(row["checkpoint_authenticated"] == "True" for row in seeds)
    assert all(row["selector_joint_finite_count"] == "16" for row in seeds)
    assert all(row["matched_family_count"] == "3" for row in seeds)
    sparse = [row for row in seeds if row["arm"] == "sparse"]
    assert len(sparse) == 10
    assert sum(row["kink_complete_seed_pass"] == "True" for row in sparse) == 6
    assert sum(row["closure_per_seed_pass"] == "True" for row in sparse) == 8
    assert sum(row["closure_counts_toward_aggregate"] == "True" for row in sparse) == 5
    assert not any(row["joint_h_g_seed_pass"] == "True" for row in sparse)

    basins = _csv_rows("basin_rows.csv")
    assert [row["g_own_nearest"] for row in basins] == ["8", "8", "8"]
    assert [row["h_own_nearest"] for row in basins] == ["4", "3", "2"]
    assert [row["finite_every_gate"] for row in basins] == ["0", "0", "0"]


def test_packet_fails_closed_after_byte_mutation(tmp_path: Path) -> None:
    copied = tmp_path / "packet"
    shutil.copytree(DEFAULT_PACKET_DIR, copied)
    decision_path = copied / "decision.json"
    decision_path.write_bytes(decision_path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="decision.json"):
        validate_packet(copied)


def test_external_roots_regenerate_identical_bytes_when_archive_is_mounted() -> None:
    if not all(Path(item["path"]).is_file() for item in SOURCE_FILES.values()):
        pytest.skip("Authenticated raw distinct-laws V2 archive is not mounted")
    result = validate_sources(DEFAULT_PACKET_DIR)
    assert result == {"external_source_files": 9, "compact_files": 4}


def test_canonical_cli_wires_the_portable_check() -> None:
    assert COMMANDS[("build", "global-k-distinct-laws-v2-negative")] == MODULE
    assert (MODULE, ("--check",)) in CHECKS
