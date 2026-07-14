"""Contract tests for the compact headline support evidence packet."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from skae.benchmarks.controlled_alignment import (
    ENTROPY_UNITS,
    FAMILY_COUNT_SEMANTICS,
    NATIVE_LABEL_SYSTEMS,
    PROXY_LABEL_SYSTEMS,
)
from skae.benchmarks.paper_protocol import (
    CONTROLLED_ALIGNMENT_ELIGIBILITY_CRITERION,
    CONTROLLED_ALIGNMENT_OBSERVED_LABEL_COUNTS,
    CONTROLLED_ALIGNMENT_PRIMARY_SYSTEM_KEYS,
)


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "docs" / "figures" / "neurips_paper_2026"
DATA_DIR = PAPER_DIR / "_data"
SUPPORT_PATH = DATA_DIR / "controlled_support_rows.csv"
PROVENANCE_PATH = DATA_DIR / "main_paper_evidence_provenance.json"
SUPPORT_COLUMNS = (
    "root_label",
    "system_name",
    "seed",
    "support_scheme",
    "subset",
    "num_states",
    "observed_label_count",
    "family_jaccard_threshold",
    "family_h_basin_given_family",
    "family_unique_count",
)


def test_frozen_support_packet_has_only_active_alignment_fields() -> None:
    with SUPPORT_PATH.open(newline="") as handle:
        reader = csv.reader(handle)
        header = tuple(next(reader))
        row_count = sum(1 for _ in reader)
    assert header == SUPPORT_COLUMNS
    assert row_count == 1350

    provenance = json.loads(PROVENANCE_PATH.read_text())
    spec = provenance["outputs"][SUPPORT_PATH.name]
    assert provenance["schema_version"] == 4
    assert provenance["aggregation"] == {
        "within_system_seed_summary": (
            "scipy.stats.trim_mean(proportiontocut=0.25)"
        ),
        "complete_cell_retained_seed_count": 9,
        "cross_system_summary": "arithmetic_mean",
        "controlled_forecasting_system_count": 15,
        "controlled_alignment_primary_system_count": 14,
        "controlled_alignment_all_system_sensitivity_count": 15,
    }
    assert tuple(spec["columns"]) == SUPPORT_COLUMNS
    assert spec["rows"] == row_count
    assert spec["bytes"] == len(SUPPORT_PATH.read_bytes())
    assert spec["sha256"] == hashlib.sha256(SUPPORT_PATH.read_bytes()).hexdigest()
    assert "wrong-support-freeze" in provenance["schema_notes"][SUPPORT_PATH.name]
    assert provenance["filters"]["support_family_fit_population"] == (
        "all generated evaluation-trajectory states"
    )
    assert provenance["filters"]["support_score_population"] == (
        "per-observed-label center-margin >= empirical q75 subset (tie-inclusive)"
    )
    expected_counts = {
        key.replace("claude:", "claude_"): value
        for key, value in CONTROLLED_ALIGNMENT_OBSERVED_LABEL_COUNTS.items()
    }
    assert provenance["filters"]["support_alignment_observed_label_counts"] == (
        expected_counts
    )
    assert provenance["filters"]["support_alignment_primary_systems"] == [
        key.replace("claude:", "claude_")
        for key in CONTROLLED_ALIGNMENT_PRIMARY_SYSTEM_KEYS
    ]
    assert provenance["filters"]["support_alignment_excluded_systems"] == [
        "claude_duffing_triple_well"
    ]
    assert provenance["filters"]["support_alignment_eligibility_criterion"] == (
        CONTROLLED_ALIGNMENT_ELIGIBILITY_CRITERION
    )
    assert provenance["filters"]["support_alignment_raw_rows_retained"] is True
    protocol = provenance["filters"]["support_alignment_protocol"]
    assert tuple(protocol["native_label_systems"]) == NATIVE_LABEL_SYSTEMS
    assert protocol["native_label_source"] == "env.basin_label"
    assert protocol["native_center_source"] == "env.points"
    assert tuple(protocol["proxy_label_systems"]) == PROXY_LABEL_SYSTEMS
    assert protocol["proxy_basin_count_source"] == (
        "known_benchmark_count_for_evaluation_only"
    )
    assert protocol["proxy_endpoint_rollout_steps"] == 5000
    assert protocol["proxy_center_estimator"] == (
        "deterministic_farthest_first_kmeans_on_advanced_endpoints"
    )
    assert protocol["proxy_state_label_rule"] == "nearest_estimated_center"
    assert protocol["mask_visit_order"] == (
        "descending_frequency_then_ascending_packbits_bytes"
    )
    assert protocol["family_assignment_tie_break"] == "earliest_created_family"
    assert protocol["kmeans_farthest_tie_break"] == "first_endpoint_index"
    assert protocol["kmeans_assignment_tie_break"] == "first_center_index"
    assert protocol["kmeans_empty_cluster_rule"] == "retain_previous_center"
    assert protocol["center_margin_definition"] == (
        "second_nearest_center_distance_minus_nearest"
    )
    assert protocol["center_margin_quantile"] == 0.75
    assert protocol["center_margin_selection_rule"] == (
        "margin_greater_than_or_equal_to_empirical_q75_tie_inclusive"
    )
    assert "larger_than_25_percent" in protocol["center_margin_tie_semantics"]
    assert protocol["entropy_units"] == ENTROPY_UNITS
    assert protocol["family_count_semantics"] == FAMILY_COUNT_SEMANTICS


def test_frozen_num_states_exposes_tie_inflation() -> None:
    by_system = {}
    observed_label_counts = {}
    with SUPPORT_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            by_system.setdefault(row["system_name"], set()).add(int(row["num_states"]))
            observed_label_counts.setdefault(row["system_name"], set()).add(
                int(row["observed_label_count"])
            )
    assert by_system["gated_local_linear"] == {4129}
    assert by_system["claude_cal_octagon_8"] == {7167}
    assert by_system["claude_duffing_triple_well"] == {16512}
    assert all(len(counts) == 1 for counts in by_system.values())
    assert observed_label_counts == {
        key.replace("claude:", "claude_"): {value}
        for key, value in CONTROLLED_ALIGNMENT_OBSERVED_LABEL_COUNTS.items()
    }


def test_support_table_uses_clean_active_name() -> None:
    table_dir = PAPER_DIR / "_tables"
    assert (table_dir / "table2_support_alignment.tex").is_file()
    assert not (
        table_dir / "table2_support_diagnostics_per_basin_deep_no_wrong_support.tex"
    ).exists()
    manifest = json.loads((PAPER_DIR / "manifest.json").read_text())
    serialized = json.dumps(manifest)
    assert "table2_support_alignment.tex" in serialized
    assert "table2_support_diagnostics_per_basin_deep_no_wrong_support.tex" not in serialized
    table = (table_dir / "table2_support_alignment.tex").read_text()
    assert "Tie-inclusive high-margin slice, 14 systems" in table
    assert "[\\mathrm{nats}]" in table
    assert "F_{\\rm abs}^{\\rm obs}" in table

    sensitivity_path = table_dir / "controlled_support_alignment_sensitivity.csv"
    with sensitivity_path.open(newline="") as handle:
        sensitivity = list(csv.DictReader(handle))
    assert len(sensitivity) == 6
    assert {int(row["primary_system_count"]) for row in sensitivity} == {14}
    assert {int(row["all_system_count"]) for row in sensitivity} == {15}
    assert {row["excluded_system"] for row in sensitivity} == {
        "claude:duffing_triple_well"
    }
    assert {int(row["excluded_observed_label_count"]) for row in sensitivity} == {1}
    assert all(
        float(row["primary_h_basin_given_family"])
        > float(row["all_system_h_basin_given_family"])
        for row in sensitivity
    )


def test_per_system_alignment_summary_distinguishes_eligible_systems() -> None:
    table = (PAPER_DIR / "_tables" / "table_persystem_HBgivenF.tex").read_text()
    assert "Significant/eligible" in table
    assert "K/15" not in table
    assert r"K{=}14\,/\,N{=}14" in table
