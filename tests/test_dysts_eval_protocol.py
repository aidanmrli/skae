"""Defaults for the retained Dysts dt-x30 evaluation path."""

import csv
import json
import math

import pytest

from experiments.neurips_2026.protocol import DYSTS_MODEL_ROW_IDS, DYSTS_PAPER_PROTOCOL
from experiments.neurips_2026.workflows.dysts_evaluation_tasks import (
    DEFAULT_SEEDS,
    DYSTS_SYSTEMS,
    _require_complete_coverage,
    _read_root_specs,
)
from experiments.neurips_2026.workflows.dysts_collection import (
    DEFAULT_HORIZONS as COLLECT_HORIZONS,
    _direct_system_effects,
    _extract_row,
    main as collection_main,
)
from experiments.neurips_2026.workflows.dysts_evaluation import (
    DEFAULT_HORIZONS,
    DEFAULT_PERIODIC_REENCODE_PERIODS,
    _has_required_horizons,
)


def test_dysts_evaluation_defaults_match_the_paper_contract():
    expected_horizons = (100, 500, 1000, 1500, 2000, 3000, 4000, 5000)

    assert tuple(DYSTS_SYSTEMS) == DYSTS_PAPER_PROTOCOL.system_keys
    assert tuple(DEFAULT_SEEDS) == DYSTS_PAPER_PROTOCOL.seeds
    assert DEFAULT_HORIZONS == expected_horizons
    assert tuple(COLLECT_HORIZONS) == expected_horizons
    assert DEFAULT_PERIODIC_REENCODE_PERIODS == (10, 25, 50, 100, 150, 200)


def test_evaluation_completeness_requires_direct_not_oracle_periodic():
    payload = {
        "modes": {
            "no_reencode": {
                "horizons": {"100": {"full_horizon_finite_fraction": 1.0}}
            }
        },
        "best_periodic": {},
    }
    assert _has_required_horizons(payload, (100,))
    assert not _has_required_horizons({"best_periodic": {"100": {"mean": 0.0}}}, (100,))


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


def test_dysts_reevaluation_allows_an_ordered_focused_replacement_row(tmp_path):
    root_specs = tmp_path / "roots.tsv"
    _write_root_specs(root_specs, ("lista_sb",))
    specs = _read_root_specs(root_specs, allow_root_subset=True)
    assert tuple(spec.label for spec in specs) == ("lista_sb",)

    _write_root_specs(root_specs, ("lista_sb", "lista"))
    with pytest.raises(ValueError, match="paper-protocol order"):
        _read_root_specs(root_specs, allow_root_subset=True)


def test_sealed_evaluation_roster_rejects_missing_receipt_backed_runs():
    _require_complete_coverage(rows=[{"task_id": 0}], missing=[], expected_rows=1)

    with pytest.raises(RuntimeError, match="found 1/2"):
        _require_complete_coverage(
            rows=[{"task_id": 0}],
            missing=[{"reason": "missing_seed_run"}],
            expected_rows=2,
        )


def test_strict_collector_writes_diagnostics_then_fails_on_pending_row(
    tmp_path, monkeypatch
):
    task_tsv = tmp_path / "tasks.tsv"
    with task_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_id",
                "root_label",
                "root_display_name",
                "model_family",
                "system_key",
                "system_slug",
                "seed",
                "run_dir",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_id": 0,
                "root_label": "lista",
                "root_display_name": "LISTA",
                "model_family": "lista",
                "system_key": "dysts:Chua",
                "system_slug": "Chua",
                "seed": 0,
                "run_dir": str(tmp_path / "missing_run"),
            }
        )
    out_dir = tmp_path / "collect"
    monkeypatch.setattr(
        "sys.argv",
        [
            "collect-dysts",
            "--task-tsv",
            str(task_tsv),
            "--out-dir",
            str(out_dir),
            "--output-tag",
            "strict_test",
            "--horizons",
            "100",
            "--require-complete",
            "--expected-task-count",
            "1",
        ],
    )

    with pytest.raises(RuntimeError, match="pending=1"):
        collection_main()

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["n_tasks"] == 1
    assert summary["n_pending"] == 1
    assert (out_dir / "pending_rows.csv").is_file()


def test_collector_does_not_promote_oracle_periodic_when_direct_is_incomplete(
    tmp_path,
):
    run_dir = tmp_path / "run"
    output = run_dir / "reeval_test"
    output.mkdir(parents=True)
    payload = {
        "dysts:Chua": {
            "modes": {
                "no_reencode": {
                    "horizons": {
                        "100": {
                            "mean": 0.01,
                            "strict_full_horizon_mean": 0.02,
                            "strict_full_horizon_num_valid": 8,
                            "full_horizon_finite_fraction": 0.5,
                            "finite_step_fraction": 0.75,
                        }
                    }
                },
                "every_step": {"horizons": {}},
            },
            "best_periodic": {
                "100": {
                    "mode": "periodic_25",
                    "mean": 1e-9,
                    "full_horizon_finite_fraction": 1.0,
                }
            },
            "best_reset": {},
            "files": {},
        }
    }
    (output / "evaluation_results_checkpoint.json").write_text(
        json.dumps(payload)
    )
    row = _extract_row(
        {
            "task_id": "0",
            "root_label": "lista",
            "root_display_name": "LISTA",
            "model_family": "lista",
            "system_key": "dysts:Chua",
            "system_slug": "dysts_Chua",
            "seed": "0",
            "run_dir": str(run_dir),
        },
        horizons=(100,),
        output_tag="test",
        checkpoint_name="checkpoint",
    )
    assert row["h100_best_periodic_mean"] == 1e-9
    assert row["status"] == "partial"


def test_system_level_effects_pair_seeds_before_aggregating_systems():
    rows = []
    for system_index in range(10):
        system = f"dysts:S{system_index}"
        for seed in range(15):
            for label, value in (("dense_mlp_tanh", 2.0), ("lista", 1.0)):
                rows.append(
                    {
                        "root_label": label,
                        "system_key": system,
                        "seed": seed,
                        "h100_no_reencode_strict_full_horizon_mean": value,
                        "h100_no_reencode_full_finite_fraction": 1.0,
                    }
                )
    record = _direct_system_effects(rows, (100,))["100"]["lista"]
    assert record["status"] == "available"
    assert record["n_systems"] == 10
    assert record["system_wins"] == 10
    assert record["mean_system_log_ratio"] == pytest.approx(math.log(0.5))
    assert record["geometric_mean_mse_ratio"] == pytest.approx(0.5)
