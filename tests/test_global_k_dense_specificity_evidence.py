"""Tests for deterministic dense-specificity paper evidence rendering."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from experiments.neurips_2026.evidence.global_k_dense_specificity import (
    check_packet,
    render_table,
    sha256_path,
)


def test_dense_specificity_table_is_deterministic_and_explicit() -> None:
    decision = {
        "sparse_system_medians": {
            "activity_leakage_true_over_null": 0.12,
            "restricted_residual_true_over_null": 0.34,
        },
        "dense_system_medians": {
            "activity_leakage_true_over_null": 0.56,
            "restricted_residual_true_over_null": 0.78,
        },
        "sparse_over_dense": {
            "activity_leakage_null_ratio": 0.21,
            "restricted_residual_null_ratio": 0.44,
        },
    }
    rendered = render_table(decision)
    assert "Sparse / null & Dense / null & Sparse / dense" in rendered
    assert "0.120 & 0.560 & 0.210" in rendered
    assert rendered.endswith("\\end{tabular}\n")


def test_dense_specificity_table_renders_invalid_missing_values() -> None:
    decision = {
        "sparse_system_medians": {
            "activity_leakage_true_over_null": 0.12,
            "restricted_residual_true_over_null": 0.34,
        },
        "dense_system_medians": {
            "activity_leakage_true_over_null": None,
            "restricted_residual_true_over_null": None,
        },
        "sparse_over_dense": {
            "activity_leakage_null_ratio": None,
            "restricted_residual_null_ratio": None,
        },
    }
    assert "0.120 & -- & --" in render_table(decision)


def _locked_bytes_exist_in_history(path: Path, expected: str) -> bool:
    commits = subprocess.run(
        ["git", "rev-list", "--all", "--", path.as_posix()],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for commit in commits:
        result = subprocess.run(
            ["git", "show", f"{commit}:{path.as_posix()}"],
            check=False,
            capture_output=True,
        )
        if result.returncode == 0 and hashlib.sha256(result.stdout).hexdigest() == expected:
            return True
    return False


def test_source_lock_authenticates_current_or_historical_local_source() -> None:
    lock_path = Path(
        "experiments/neurips_2026/global_k_dense_specificity_source_lock.json"
    )
    lock = json.loads(lock_path.read_text())
    assert lock["protocol_id"] == "global_k_dense_zero_wd_specificity_v1"
    for record in lock["sources"].values():
        path = Path(record["path"])
        assert path.is_file()
        assert (
            sha256_path(path) == record["sha256"]
            or _locked_bytes_exist_in_history(path, record["sha256"])
        )


def _write_csv(path, fields, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_compact_packet_check_does_not_require_scratch_sources(tmp_path) -> None:
    systems = [f"system_{index:02d}" for index in range(15)]
    card = {"training": {"systems": systems, "seeds": [0, 1, 2]}}
    decision = {
        "decision": "sparse_support_specific",
        "sparse_system_medians": {
            "activity_leakage_true_over_null": 0.12,
            "restricted_residual_true_over_null": 0.34,
        },
        "dense_system_medians": {
            "activity_leakage_true_over_null": 0.56,
            "restricted_residual_true_over_null": 0.78,
        },
        "sparse_over_dense": {
            "activity_leakage_null_ratio": 0.21,
            "restricted_residual_null_ratio": 0.44,
        },
    }
    card_path = tmp_path / "card.json"
    decision_path = tmp_path / "decision.json"
    run_path = tmp_path / "runs.csv"
    system_path = tmp_path / "systems.csv"
    table_path = tmp_path / "table.tex"
    card_path.write_text(json.dumps(card, sort_keys=True) + "\n")
    decision_path.write_text(json.dumps(decision, sort_keys=True) + "\n")
    run_rows = [
        {"system_key": system, "seed": seed}
        for system in systems
        for seed in (0, 1, 2)
    ]
    _write_csv(run_path, ["system_key", "seed"], run_rows)
    _write_csv(
        system_path,
        ["system_key", "run_count"],
        [{"system_key": system, "run_count": 3} for system in systems],
    )
    table_path.write_text(render_table(decision))

    normalized = {
        "run_rows": {"path": str(run_path), "sha256": sha256_path(run_path)},
        "system_rows": {"path": str(system_path), "sha256": sha256_path(system_path)},
        "decision": {"path": str(decision_path), "sha256": sha256_path(decision_path)},
        "card": {"path": str(card_path), "sha256": sha256_path(card_path)},
        "table": {"path": str(table_path), "sha256": sha256_path(table_path)},
    }
    checkpoint_rows = [
        {
            "system_key": row["system_key"],
            "seed": row["seed"],
            "dense_checkpoint": {"path": "/missing/dense", "sha256": "1" * 64},
            "sparse_checkpoint": {"path": "/missing/sparse", "sha256": "2" * 64},
            "evaluation_shard": {"path": "/missing/shard", "sha256": "3" * 64},
        }
        for row in run_rows
    ]
    provenance = {
        "card_sha256": sha256_path(card_path),
        "decision": decision["decision"],
        "source_artifacts": {
            "scratch_decision": {"path": "/missing/scratch", "sha256": "4" * 64}
        },
        "authenticated_sources": {
            "evidence_builder": {"path": "/missing/source", "sha256": "5" * 64}
        },
        "normalized_artifacts": normalized,
        "checkpoints_and_shards": checkpoint_rows,
        "complete_recipe_architecture_caveat": "This is a complete-recipe comparison.",
    }
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, sort_keys=True) + "\n")

    check_packet(provenance_path)
    table_path.write_text("tampered\n")
    with pytest.raises(RuntimeError, match="normalized_artifacts.table"):
        check_packet(provenance_path)
