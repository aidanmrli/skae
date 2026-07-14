"""Defaults for the retained Dysts dt-x30 evaluation path."""

import csv

import pytest

from skae.benchmarks.paper_protocol import DYSTS_MODEL_ROW_IDS, DYSTS_PAPER_PROTOCOL
from tools.build_dysts_long_horizon_eval_tasks import (
    DEFAULT_SEEDS,
    DYSTS_SYSTEMS,
    _read_root_specs,
)
from tools.collect_dysts_long_horizon_forecasting import (
    DEFAULT_HORIZONS as COLLECT_HORIZONS,
)
from tools.evaluate_dysts_long_horizon_run import (
    DEFAULT_HORIZONS,
    DEFAULT_PERIODIC_REENCODE_PERIODS,
)


def test_dysts_evaluation_defaults_match_the_paper_contract():
    expected_horizons = (100, 500, 1000, 1500, 2000, 3000, 4000, 5000)

    assert tuple(DYSTS_SYSTEMS) == DYSTS_PAPER_PROTOCOL.system_keys
    assert tuple(DEFAULT_SEEDS) == DYSTS_PAPER_PROTOCOL.seeds
    assert DEFAULT_HORIZONS == expected_horizons
    assert tuple(COLLECT_HORIZONS) == expected_horizons
    assert DEFAULT_PERIODIC_REENCODE_PERIODS == (10, 25, 50, 100, 150, 200)


def _write_root_specs(path, labels):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "display_name", "model_family", "root_dir"],
            delimiter="\t",
        )
        writer.writeheader()
        for label in labels:
            writer.writerow(
                {
                    "label": label,
                    "display_name": label,
                    "model_family": "global_k",
                    "root_dir": f"/tmp/{label}",
                }
            )


def test_dysts_reevaluation_accepts_only_the_six_global_k_rows(tmp_path):
    root_specs = tmp_path / "roots.tsv"
    _write_root_specs(root_specs, DYSTS_MODEL_ROW_IDS)
    assert tuple(spec.label for spec in _read_root_specs(root_specs)) == DYSTS_MODEL_ROW_IDS

    _write_root_specs(root_specs, (*DYSTS_MODEL_ROW_IDS, "staged_local_k"))
    with pytest.raises(ValueError, match="exactly the six global-K root rows"):
        _read_root_specs(root_specs)
