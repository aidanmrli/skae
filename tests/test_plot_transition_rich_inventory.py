"""Tests for the transition-rich inventory shortlist plotting utility."""

from __future__ import annotations

from pathlib import Path

from tools.plot_transition_rich_inventory import (
    build_elite_shortlist_records,
    build_shortlist_records,
    parse_formats_arg,
    plot_elite_cards,
    plot_mechanism_atlas,
    plot_shortlist_design_map,
)


def test_build_shortlist_records_contains_expected_anchor_names():
    records = build_shortlist_records()
    names = {record.name for record in records}
    assert "Triangle central gate and sectors" in names
    assert "Rotating barrier-4" in names
    assert len(records) >= 14


def test_build_elite_shortlist_records_contains_expected_novel_names():
    records = build_elite_shortlist_records()
    names = {record.name for record in records}
    assert "Lens-warp triad" in names
    assert "Arc DAG4" in names
    assert len(records) >= 8


def test_parse_formats_arg_normalizes_suffixes():
    assert parse_formats_arg("png,.svg,pdf") == ("png", "svg", "pdf")


def test_plot_shortlist_design_map_writes_requested_output(tmp_path: Path):
    output_paths = plot_shortlist_design_map(
        output_dir=tmp_path,
        formats=("png",),
    )

    expected = tmp_path / "transition_rich_shortlist_design_map.png"
    assert output_paths == [expected]
    assert expected.exists()


def test_plot_mechanism_atlas_writes_requested_output(tmp_path: Path):
    output_paths = plot_mechanism_atlas(
        output_dir=tmp_path,
        formats=("png",),
    )

    expected = tmp_path / "transition_rich_mechanism_atlas.png"
    assert output_paths == [expected]
    assert expected.exists()


def test_plot_elite_cards_writes_requested_output(tmp_path: Path):
    output_paths = plot_elite_cards(
        output_dir=tmp_path,
        formats=("png",),
    )

    expected = tmp_path / "transition_rich_elite_cards.png"
    assert output_paths == [expected]
    assert expected.exists()
