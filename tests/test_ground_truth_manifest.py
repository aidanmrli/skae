"""Checks for the compact controlled-system display manifest."""

import json

import numpy as np
import pytest

from experiments.neurips_2026.evidence.ground_truth import (
    DEFAULT_DPI,
    DEFAULT_FORMATS,
    DEFAULT_GRID_POINTS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STREAM_DENSITY,
    FieldData,
    RETAINED_15_SYSTEMS,
    _validate_active_request,
    validate_manifest,
    write_manifest,
)


def test_ground_truth_manifest_matches_the_active_pdf_inventory(tmp_path):
    spec = RETAINED_15_SYSTEMS[0]
    field = FieldData(
        spec=spec,
        dt=0.1,
        xs=np.zeros(1),
        ys=np.zeros(1),
        u=np.zeros((1, 1)),
        v=np.zeros((1, 1)),
        log_speed=np.zeros((1, 1)),
        centers=None,
        log_speed_vmin=0.0,
        log_speed_vmax=1.0,
    )

    field_pdf = tmp_path / "field.pdf"
    composite_pdf = tmp_path / "composite.pdf"
    field_pdf.write_bytes(b"field")
    composite_pdf.write_bytes(b"composite")
    write_manifest(
        tmp_path,
        [field],
        grid_points=80,
        formats=("pdf",),
        dpi=300,
        stream_density=1.05,
        individual_paths={spec.system_key: [str(field_pdf)]},
        composite_paths=[str(composite_pdf)],
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["composite"]["path"] == "composite.pdf"
    assert manifest["systems"] == [
        {
            "system_key": spec.system_key,
            "display_name": spec.title,
            "basins": spec.basin_count,
            "pdf": {
                "path": "field.pdf",
                "sha256": manifest["systems"][0]["pdf"]["sha256"],
            },
        }
    ]
    assert "--formats pdf" in manifest["reproduction_command_inside_allocation"]


def test_active_ground_truth_directory_rejects_partial_overwrites():
    with pytest.raises(ValueError, match="complete frozen roster"):
        _validate_active_request(
            DEFAULT_OUTPUT_DIR,
            [RETAINED_15_SYSTEMS[0]],
            DEFAULT_FORMATS,
            grid_points=DEFAULT_GRID_POINTS,
            stream_density=DEFAULT_STREAM_DENSITY,
            dpi=DEFAULT_DPI,
            skip_individual=False,
        )

    with pytest.raises(ValueError, match="every individual PDF"):
        _validate_active_request(
            DEFAULT_OUTPUT_DIR,
            list(RETAINED_15_SYSTEMS),
            DEFAULT_FORMATS,
            grid_points=DEFAULT_GRID_POINTS,
            stream_density=DEFAULT_STREAM_DENSITY,
            dpi=DEFAULT_DPI,
            skip_individual=True,
        )

    with pytest.raises(ValueError, match="canonical render parameters"):
        _validate_active_request(
            DEFAULT_OUTPUT_DIR,
            list(RETAINED_15_SYSTEMS),
            DEFAULT_FORMATS,
            grid_points=DEFAULT_GRID_POINTS // 2,
            stream_density=DEFAULT_STREAM_DENSITY,
            dpi=DEFAULT_DPI,
            skip_individual=False,
        )


def test_ground_truth_manifest_rejects_a_tampered_pdf(tmp_path):
    # validate_manifest requires the complete roster, so copy the active packet
    # and alter one declared artifact in the temporary directory.
    active_manifest = json.loads((DEFAULT_OUTPUT_DIR / "manifest.json").read_text())
    if active_manifest.get("schema_version") != 2:
        pytest.skip("active manifest has not yet been regenerated")
    for path in validate_manifest(DEFAULT_OUTPUT_DIR / "manifest.json"):
        (tmp_path / path.name).write_bytes(path.read_bytes())
    (tmp_path / "manifest.json").write_text(json.dumps(active_manifest) + "\n")
    first_pdf = tmp_path / active_manifest["systems"][0]["pdf"]["path"]
    first_pdf.write_bytes(first_pdf.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_manifest(tmp_path / "manifest.json")
