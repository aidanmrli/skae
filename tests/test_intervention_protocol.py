"""Regression tests for the frozen intervention case-study contract."""

import json
import sys

from experiments.neurips_2026.interventions import evaluate
from experiments.neurips_2026.interventions.protocol import (
    NUM_CANDIDATE_TRAJECTORIES,
    NUM_INITIAL_POINTS,
    intervention_protocol_metadata,
    validate_intervention_protocol_record,
)
from experiments.neurips_2026.paths import PAPER_DATA_DIR


def test_evaluator_defaults_match_the_frozen_case_study(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "interventions",
            "--rows_csv",
            str(tmp_path / "rows.csv"),
            "--output_dir",
            str(tmp_path / "outputs"),
        ],
    )

    args = evaluate._parse_args()

    assert args.num_initial_points == NUM_INITIAL_POINTS == 100
    assert args.num_candidate_trajectories == NUM_CANDIDATE_TRAJECTORIES == 512
    assert args.root_label == intervention_protocol_metadata()["root_label"]


def test_frozen_intervention_evidence_matches_the_contract():
    provenance = json.loads(
        (PAPER_DATA_DIR / "interventions/provenance.json").read_text()
    )
    validate_intervention_protocol_record(
        provenance["protocol"]["coordinate_dropping"]
    )
    validate_intervention_protocol_record(provenance["protocol"]["random_support"])
