"""Checks for the compact controlled-system display manifest."""

import json

import numpy as np

from tools.plot_multibasin_ground_truth_vector_fields import (
    FieldData,
    RETAINED_15_SYSTEMS,
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

    write_manifest(
        tmp_path,
        [field],
        grid_points=80,
        individual_paths={spec.system_key: [str(tmp_path / "field.pdf")]},
        composite_paths=[str(tmp_path / "composite.pdf")],
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["composite"] == "composite.pdf"
    assert manifest["systems"] == [
        {
            "system_key": spec.system_key,
            "display_name": spec.title,
            "basins": spec.basin_count,
            "pdf": "field.pdf",
        }
    ]
    assert "--formats pdf" in manifest["reproduction_command_inside_allocation"]
