from __future__ import annotations

import csv
import json

import numpy as np

from experiments.neurips_2026.allen_cahn_physics_metrics.core import METRIC_NAMES
from experiments.neurips_2026.cli import CHECKS, COMMANDS
from experiments.neurips_2026.evidence.allen_cahn_physics_metrics import (
    DEFAULT_DATA_DIR,
    DEFAULT_FIGURE_DIR,
    DEFAULT_TABLE_DIR,
    EXPECTED_RELEASE_HASHES,
    EXPECTED_SOURCE_HASHES,
    FILE_NAMES,
    validate_packet,
)
from experiments.neurips_2026.evidence.allen_cahn_physics_metrics_rendering import (
    SHORT_LABELS,
)


def _summary() -> dict[str, object]:
    return json.loads((DEFAULT_DATA_DIR / FILE_NAMES["summary"]).read_text())


def _rows(name: str) -> list[dict[str, str]]:
    with (DEFAULT_DATA_DIR / FILE_NAMES[name]).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_compact_packet_is_hash_rooted_complete_and_portable() -> None:
    result = validate_packet()
    assert result == {
        "status": "broad_secondary_concordance",
        "metric_count": 7,
        "holm_significant_count": 5,
        "paired_seed_rows": 70,
        "curve_rows": 1400,
    }
    manifest = json.loads(
        (DEFAULT_DATA_DIR / FILE_NAMES["manifest"]).read_text(encoding="utf-8")
    )
    assert manifest["source_artifact_hashes"] == EXPECTED_SOURCE_HASHES
    assert manifest["outputs"] == {
        name: digest
        for name, digest in EXPECTED_RELEASE_HASHES.items()
        if name != "manifest"
    }
    assert manifest["metric_roster"] == list(METRIC_NAMES)
    assert manifest["all_seven_metrics_rendered"] is True


def test_all_seven_h200_results_and_holm_labels_are_exact() -> None:
    summary = _summary()
    metrics = summary["metrics"]
    assert [metric["name"] for metric in metrics] == list(METRIC_NAMES)
    expected = {
        "nearest_well_pixel_disagreement": (0.0460698403829739, 9, 0.0146484375, True),
        "modal_well_accuracy": (0.2356119791666722, 5, 0.302734375, False),
        "well_area_fraction_tv_error": (0.060727123836104235, 8, 0.03515625, True),
        "interface_edge_disagreement": (0.022496643443140174, 9, 0.0146484375, True),
        "free_energy_absolute_error": (0.061973370034029696, 10, 0.0068359375, True),
        "potential_energy_absolute_error": (0.0376893685004136, 8, 0.119140625, False),
        "gradient_energy_absolute_error": (0.08581349456538312, 10, 0.0068359375, True),
    }
    for metric in metrics:
        effect, wins, holm, significant = expected[metric["name"]]
        assert np.isclose(metric["arm_mean_effect"]["value"], effect)
        assert metric["h200_cumulative_seed_wins"] == wins
        assert metric["holm_p"] == holm
        assert metric["holm_significant_0p05"] is significant
        assert metric["paired_bootstrap"]["replicates"] == 100_000
        assert metric["h200_cumulative"]["sparse"] < metric["h200_cumulative"]["dense"] or (
            metric["name"] == "modal_well_accuracy"
            and metric["h200_cumulative"]["sparse"]
            > metric["h200_cumulative"]["dense"]
        )
    assert summary["secondary_pattern"]["directionally_sparse_better_metrics"] == 7
    assert summary["secondary_pattern"]["classification"] == "broad_secondary_concordance"
    assert summary["evidence_grade"] == "outcome_aware_same_checkpoint_secondary"
    assert "does not reclassify" in summary["claim_boundary"]


def test_row_level_evidence_has_exact_seed_and_physical_time_rosters() -> None:
    seeds = _rows("seed_rows")
    curves = _rows("curve_rows")
    ties = _rows("tie_rows")
    assert len(seeds) == 7 * 10
    assert len(curves) == 7 * 200
    assert len(ties) == 200
    for name in METRIC_NAMES:
        metric_seeds = [row for row in seeds if row["metric_name"] == name]
        metric_curves = [row for row in curves if row["metric_name"] == name]
        assert [int(row["model_seed"]) for row in metric_seeds] == list(range(64, 74))
        assert [int(row["horizon_step"]) for row in metric_curves] == list(range(1, 201))
        np.testing.assert_allclose(
            [float(row["physical_time"]) for row in metric_curves],
            0.1 * np.arange(1, 201),
        )
        assert all(
            set(("dense_cumulative", "sparse_cumulative", "persistence_cumulative"))
            <= set(row)
            for row in metric_curves
        )
    assert [int(row["horizon_step"]) for row in ties] == list(range(1, 201))


def test_figure_and_table_show_every_metric_with_secondary_disclosure() -> None:
    table = (DEFAULT_TABLE_DIR / FILE_NAMES["table"]).read_text(encoding="utf-8")
    assert "Outcome-aware, same-checkpoint secondary" in table
    assert "all seven frozen metrics at the H200 cumulative endpoint" in table
    assert "Holm-adjusted across all seven metrics" in table
    assert all(table.count(SHORT_LABELS[name]) == 1 for name in METRIC_NAMES)
    assert (DEFAULT_FIGURE_DIR / FILE_NAMES["figure_pdf"]).stat().st_size > 25_000
    assert (DEFAULT_FIGURE_DIR / FILE_NAMES["figure_png"]).stat().st_size > 250_000


def test_paper_cli_exposes_builder_and_portable_checker() -> None:
    module = "experiments.neurips_2026.evidence.allen_cahn_physics_metrics"
    assert COMMANDS[("build", "allen-cahn-physics-metrics")] == module
    assert (module, ("--check",)) in CHECKS
