"""Tests for the interpretable transition-system plotting utility."""

from __future__ import annotations

from pathlib import Path

from tools.plot_interpretable_systems import (
    DEFAULT_SYSTEM_KEYS,
    parse_formats_arg,
    parse_systems_arg,
    plot_selected_systems,
    validate_layout,
)


def test_parse_systems_defaults_to_all():
    assert parse_systems_arg(None) == list(DEFAULT_SYSTEM_KEYS)
    assert parse_systems_arg("all") == list(DEFAULT_SYSTEM_KEYS)


def test_parse_systems_accepts_csv_subset():
    assert parse_systems_arg("gated_local_linear,gated_transfer_linear") == [
        "gated_local_linear",
        "gated_transfer_linear",
    ]


def test_parse_formats_arg_normalizes_suffixes():
    assert parse_formats_arg("png,.svg,pdf") == ("png", "svg", "pdf")


def test_validate_layout_accepts_catalog():
    assert validate_layout("catalog") == "catalog"


def test_plot_selected_systems_writes_only_requested_outputs(tmp_path: Path):
    output_paths = plot_selected_systems(
        systems=["gated_local_linear"],
        output_dir=tmp_path,
        grid_points=17,
        trajectory_length=20,
        start_points_per_axis=4,
        formats=("png",),
    )

    expected = tmp_path / "gated_local_linear_interpretable_overview.png"
    assert output_paths == [expected]
    assert expected.exists()
    assert not (tmp_path / "multiwell_strong_transition_interpretable_overview.png").exists()
    assert not (tmp_path / "gated_transfer_linear_interpretable_overview.png").exists()


def test_plot_selected_systems_catalog_layout_uses_system_key_filename(tmp_path: Path):
    output_paths = plot_selected_systems(
        systems=["gated_transfer_linear"],
        output_dir=tmp_path,
        grid_points=17,
        trajectory_length=20,
        start_points_per_axis=4,
        formats=("png",),
        layout="catalog",
    )

    expected = tmp_path / "gated_transfer_linear.png"
    assert output_paths == [expected]
    assert expected.exists()
    assert not (tmp_path / "gated_transfer_linear_interpretable_overview.png").exists()
