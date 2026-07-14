from __future__ import annotations

import json

import pytest

from tools.freeze_paper_baseline_evidence import _finite_text
from tools.summarize_paper_baseline_suite import (
    DATA_DIR,
    DEFAULT_INPUTS,
    DEFAULT_PROVENANCE,
    METADATA_NAME,
    OUTPUT_FILES,
    TABLE_DIR,
    _coverage,
    _read_csv,
    _validate_rows,
    build_outputs,
    verify_inputs,
    write_or_check,
)


def test_frozen_baseline_packet_has_exact_grid_and_metric_semantics() -> None:
    provenance = verify_inputs(DEFAULT_INPUTS, DEFAULT_PROVENANCE)
    assert provenance["source_campaigns"]["multibasin"]["file_count"] == 90
    assert provenance["source_campaigns"]["dysts"]["file_count"] == 60
    assert "/network/" not in DEFAULT_PROVENANCE.read_text(encoding="utf-8")

    multibasin = _read_csv(DEFAULT_INPUTS["multibasin"])
    dysts = _read_csv(DEFAULT_INPUTS["dysts"])
    _validate_rows(multibasin, "multibasin")
    _validate_rows(dysts, "dysts")
    assert len(multibasin) == 810
    assert len(dysts) == 540
    assert {row["metric_protocol"] for row in multibasin + dysts} == {
        "ordinary_through_h_mean_finite_starts",
        "finite_step_prefix_mean",
    }

    assert _coverage(multibasin, "multibasin")["finite_metric_rows"] == 810
    dysts_coverage = _coverage(dysts, "dysts")
    assert dysts_coverage["finite_metric_rows"] == 519
    assert dysts_coverage["finite_seed_count_distribution"] == {
        "0": 1,
        "1": 9,
        "2": 0,
        "3": 170,
    }
    assert {
        (cell["method"], cell["horizon"], cell["system"], cell["finite_seeds"])
        for cell in dysts_coverage["incomplete_cells"]
    } >= {("kmeans_hard", 4000, "dysts:LuChenCheng", 0)}


def test_frozen_rows_regenerate_all_sidecars_byte_for_byte(tmp_path) -> None:
    outputs = build_outputs(out_dir=tmp_path)
    expected_names = {*OUTPUT_FILES.values(), METADATA_NAME}
    assert {path.name for path in outputs} == expected_names
    for path, payload in outputs.items():
        assert payload == (TABLE_DIR / path.name).read_bytes()
    write_or_check(outputs, check=False)
    write_or_check(outputs, check=True)


def test_frozen_input_tampering_is_rejected(tmp_path) -> None:
    copied = {}
    for benchmark, source in DEFAULT_INPUTS.items():
        target = tmp_path / source.name
        target.write_bytes(source.read_bytes())
        copied[benchmark] = target
    copied["dysts"].write_bytes(copied["dysts"].read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Frozen baseline hash mismatch"):
        verify_inputs(copied, DEFAULT_PROVENANCE)


def test_wrong_metric_protocol_is_rejected() -> None:
    rows = _read_csv(DEFAULT_INPUTS["multibasin"])
    rows[0]["metric_protocol"] = "finite_step_prefix_mean"
    with pytest.raises(ValueError, match="Metric-protocol mismatch"):
        _validate_rows(rows, "multibasin")


def test_provenance_output_hashes_name_the_frozen_files() -> None:
    provenance = json.loads(DEFAULT_PROVENANCE.read_text(encoding="utf-8"))
    assert set(provenance["outputs"]) == {
        "paper_baseline_multibasin_rows.csv",
        "paper_baseline_dysts_rows.csv",
    }
    assert all((DATA_DIR / name).is_file() for name in provenance["outputs"])


def test_metric_sanitization_keeps_only_finite_values() -> None:
    assert _finite_text("1.25") == "1.25"
    assert _finite_text("nan") == ""
    assert _finite_text("inf") == ""
    assert _finite_text("") == ""
