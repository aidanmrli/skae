"""Tests for canonical staged local-operator task derivation."""

import csv
import json

from experiments.neurips_2026.local_operators.contract import (
    ROUTE_PROTOCOL,
    TOTAL_TRAINING_STEPS,
    route_protocol_metadata,
)
from experiments.neurips_2026.local_operators.prepare_tasks import prepare_tasks


FIELDS = (
    "task_id",
    "phase",
    "model_variant",
    "system_key",
    "system_slug",
    "env_dt",
    "seed",
    "num_steps",
)


def _write_base_task(path):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "task_id": 0,
                "phase": "base",
                "model_variant": "source",
                "system_key": "gated_local_linear",
                "system_slug": "gated_local_linear",
                "env_dt": "0.05",
                "seed": 0,
                "num_steps": TOTAL_TRAINING_STEPS,
            }
        )


def test_staged_task_manifest_uses_the_shared_contract(tmp_path):
    base_tsv = tmp_path / "base.tsv"
    base_manifest = tmp_path / "base.json"
    output_tsv = tmp_path / "staged.tsv"
    output_manifest = tmp_path / "staged.json"
    _write_base_task(base_tsv)
    base_manifest.write_text('{"protocol_id": "controlled"}\n')

    rows = prepare_tasks(
        base_task_tsv=base_tsv,
        base_manifest_json=base_manifest,
        output_tsv=output_tsv,
        output_manifest_json=output_manifest,
        source_variant="source",
        target_variant="staged",
        phase_label="paper-phase",
        base_out=tmp_path / "runs",
        skip_completed=False,
    )

    manifest = json.loads(output_manifest.read_text())
    assert rows[0]["model_variant"] == "staged"
    assert rows[0]["phase"] == "paper-phase"
    assert manifest["experiment_family"] == ROUTE_PROTOCOL
    assert manifest["staged_protocol"] == route_protocol_metadata()


def test_staged_task_preparation_skips_completed_runs(tmp_path):
    base_tsv = tmp_path / "base.tsv"
    base_manifest = tmp_path / "base.json"
    _write_base_task(base_tsv)
    base_manifest.write_text("{}\n")
    completed = (
        tmp_path
        / "runs/paper-phase/staged/gated_local_linear/dt_0p05/seed_0/20260101"
    )
    completed.mkdir(parents=True)
    (completed / "evaluation_results_best.json").write_text("{}\n")

    rows = prepare_tasks(
        base_task_tsv=base_tsv,
        base_manifest_json=base_manifest,
        output_tsv=tmp_path / "staged.tsv",
        output_manifest_json=tmp_path / "staged.json",
        source_variant="source",
        target_variant="staged",
        phase_label="paper-phase",
        base_out=tmp_path / "runs",
        skip_completed=True,
    )

    assert rows == []
    assert json.loads((tmp_path / "staged.json").read_text())[
        "skipped_completed_count"
    ] == 1
