"""Tests for the Claude catalog strict-pass gallery helper."""

from __future__ import annotations

from tools.plot_claude_catalog_strict_pass_gallery import (
    load_accepted_pass_records,
    load_strict_pass_records,
    parse_formats_arg,
)


def test_parse_formats_arg_normalizes_suffixes() -> None:
    assert parse_formats_arg("png,.svg,pdf") == ("png", "svg", "pdf")


def test_load_strict_pass_records_contains_expected_names() -> None:
    records = load_strict_pass_records()
    names = {str(record["name"]) for record in records}
    assert "cal_pentagon_5" in names
    assert "cal_triangle_3" in names
    assert "var_depth_gradient_4" in names
    assert "snic_multi" not in names
    assert len(records) >= 8


def test_load_accepted_pass_records_contains_relaxed_subset() -> None:
    records = load_accepted_pass_records()
    names = {str(record["name"]) for record in records}
    assert "snic_multi" in names
    assert "cal_square_4" in names
    assert "cal_hexagon_6" in names
    assert "var_depth_gradient_4" in names
    assert len(records) >= 12
