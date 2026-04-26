"""Tests for the Claude catalog audit plotting utility."""

from __future__ import annotations

from pathlib import Path

from tools.plot_claude_catalog_audit import (
    build_retune_frontier,
    build_unscreened_priority_records,
    load_combined_screening_records,
    load_screening_records,
    parse_formats_arg,
    plot_catalog_audit,
)


def test_parse_formats_arg_normalizes_suffixes() -> None:
    assert parse_formats_arg("png,.svg,pdf") == ("png", "svg", "pdf")


def test_build_unscreened_priority_records_contains_expected_names() -> None:
    records = build_unscreened_priority_records()
    names = {record.name for record in records}
    assert "cal_triangle_3" in names
    assert "slow_fast_triple" in names
    assert len(records) >= 12


def test_build_retune_frontier_contains_known_screened_pass() -> None:
    frontier = build_retune_frontier(load_screening_records())
    names = {str(record["name"]) for record in frontier}
    assert "cal_triangle_3" in names


def test_load_screening_records_exposes_strict_and_relaxed_acceptance_tiers() -> None:
    records = load_screening_records()
    statuses = {str(record["status"]) for record in records}
    assert "strict_cross_pass" in statuses
    assert "accepted_relaxed_pass" in statuses
    assert "retune_frontier" in statuses


def test_load_combined_screening_records_merges_extra_packets() -> None:
    records = load_combined_screening_records(
        extra_screening_paths=(
            Path("results/claude_catalog_priority_screen_20260407/priority_screen_results.json"),
        )
    )
    names = {str(record["name"]) for record in records}
    assert "snic_multi" in names
    assert "cal_triangle_3" in names


def test_plot_catalog_audit_writes_requested_output(tmp_path: Path) -> None:
    output_paths = plot_catalog_audit(output_dir=tmp_path, formats=("png",))

    expected = tmp_path / "claude_catalog_audit_atlas.png"
    assert output_paths == [expected]
    assert expected.exists()
