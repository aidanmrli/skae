"""Integrity checks for the Allen--Cahn architecture/capacity audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "figures" / "neurips_paper_2026" / "_data"
AUDIT = DATA / "allen_cahn_global_k_forecast_optimized_architecture_audit.json"
MANIFEST = DATA / "allen_cahn_global_k_forecast_optimized_artifact_manifest.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_architecture_audit_matches_the_frozen_checkpoint_roster():
    audit = _load(AUDIT)
    manifest = _load(MANIFEST)
    expected = {
        (run["arm"], int(run["seed"])): run["checkpoint"]["sha256"]
        for run in manifest["runs"]
    }
    observed = {
        (run["arm"], int(run["seed"])): run["checkpoint_sha256"]
        for run in audit["runs"]
    }

    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == audit[
        "artifact_manifest"
    ]["sha256_at_audit"]
    assert expected == observed
    assert len(observed) == 20
    assert set(observed) == {
        (arm, seed) for arm in ("dense", "sparse") for seed in range(64, 74)
    }


def test_architecture_audit_records_exact_capacity_and_inert_lista_matrix():
    audit = _load(AUDIT)
    common = audit["common_checkpoint_structure"]
    runs = audit["runs"]

    assert common["trainable_parameter_count_from_source_and_state_elements"] == 12_698_690
    assert common["lista_s_parameter_count"] == 2_048**2
    assert common["effective_forward_parameter_count_excluding_inert_lista_s"] == (
        12_698_690 - 2_048**2
    )
    assert {run["parameter_count"] for run in runs} == {12_698_690}
    assert {run["tensor_count"] for run in runs} == {40}
    assert {run["shape_digest"] for run in runs} == {common["shape_digest"]}
    assert all(run["lista_s_nonzero_count"] == 0 for run in runs)
    assert all(run["lista_s_max_abs"] == 0.0 for run in runs)
    assert audit["forward_path_audit"]["capacity_difference"] is False
    assert audit["forward_path_audit"]["decoder_normalization_difference"] is False


def test_architecture_audit_bounds_the_sparse_treatment_claim():
    audit = _load(AUDIT)
    config = audit["configuration_audit"]
    differences = {
        item["field"]: (item["dense"], item["sparse"])
        for item in config["paired_scientific_differences"]
    }

    assert differences == {
        "model_config.encoder_kind": ("dense", "lista"),
        "model_config.lista_alpha": (0.0, 0.15),
        "loss_weights.sparsity": (0.0, 0.01),
        "training_args.model_variant": ("conv_dense", "conv_lista"),
        "training_args.lista_alpha": (0.0, 0.15),
        "training_args.sparsity_weight": (0.0, 0.01),
    }
    assert config["no_other_paired_scientific_configuration_differences"] is True
    assert config["common_model_config"]["conv_activation"] == "tanh"
    assert config["common_training_settings"]["optimizer"] == "Adam"
    assert config["common_training_settings"]["weight_decay"] == 0.0
    assert audit["forward_path_audit"]["dense_encode"] == "code"
    assert audit["forward_path_audit"]["sparse_encode"] == (
        "softshrink(code, lambda=0.15)"
    )
    assert any(
        "L1 loss alone" in statement
        for statement in audit["claim_boundary"]["does_not_warrant"]
    )
