"""Outcome-blind regression tests for the clean periodic-reencoding v2 rerun."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from experiments.neurips_2026.allen_cahn_periodic_reencoding import run as base_run
from experiments.neurips_2026.allen_cahn_periodic_reencoding.io import (
    duplicate_safe_json,
    load_card,
    sha256_path,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding_v2 import run as v2_run
from experiments.neurips_2026.allen_cahn_periodic_reencoding_v2.lineage import (
    write_runtime_lineage,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding_v2.uuid_probe import (
    strict_uuid_record,
)


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "experiments/neurips_2026/allen_cahn_periodic_reencoding"
V2 = ROOT / "experiments/neurips_2026/allen_cahn_periodic_reencoding_v2"
V1_CARD_SHA = "97716bb6dc2c0e6a4f8389d362c06ad67045ac2a85574c794f1966a388ce9e17"
V1_SOURCE_SHA = "a4729c04d1981c031d1531304668dc01b4165c1d5b1610d4d780d9730d123c6f"


def test_v1_freeze_remains_byte_exact_and_v2_science_is_unchanged() -> None:
    assert sha256_path(V1 / "prediction_card.json") == V1_CARD_SHA
    assert sha256_path(V1 / "source_manifest.sha256") == V1_SOURCE_SHA
    v1 = duplicate_safe_json(V1 / "prediction_card.json")
    v2 = duplicate_safe_json(V2 / "prediction_card.json")
    assert v2["protocol_id"] == v1["protocol_id"]
    assert v2["execution_attempt_id"] == (
        "allen_cahn_periodic_reencoding_clean_rerun_v2"
    )
    for key in (
        "question", "hypothesis", "frozen_parent", "system", "roster",
        "prior_outcome_aware_context", "scientific_distinction",
        "cadence_selection", "test_evaluation", "estimand_and_inference",
        "interpretation_branches", "visualization_plan", "hardware_plan",
    ):
        assert v2[key] == v1[key]
    for key in (
        "derivation_namespace", "derivation_role_tokens", "derivation_rule",
        "validation", "test", "rng_rule", "collision_audit", "generation_order",
        "field_only", "storage_guard", "excluded",
    ):
        assert v2["prospective_datasets"][key] == v1["prospective_datasets"][key]
    history = v2["outcome_blind_v1_operational_history"]
    assert history["scientific_choices_changed_in_v2"] is False
    assert history["scientific_outcomes_accessed"] is False


def test_v1_loader_accepts_same_protocol_clean_rerun_card() -> None:
    card_path = V2 / "prediction_card.json"
    card, digest = load_card(card_path, expected_sha256=sha256_path(card_path))
    assert digest == sha256_path(card_path)
    assert card["protocol_id"] == "allen_cahn_periodic_reencoding_confirmation_v1"
    assert card["execution_attempt_id"].endswith("clean_rerun_v2")


def test_v2_lineage_diff_is_only_docstring_and_narrow_uuid_string() -> None:
    v1 = (V1 / "lineage.py").read_text(encoding="utf-8")
    v2 = (V2 / "lineage.py").read_text(encoding="utf-8")
    normalized = v2.replace(
        '"""V2 metric-free runtime lineage with strict CUDA UUID serialization."""',
        '"""Metric-free runtime lineage helpers for the periodic packet."""',
    ).replace(
        '"gpu_uuid": str(getattr(properties, "uuid", "not_recorded")),',
        '"gpu_uuid": getattr(properties, "uuid", "not_recorded"),',
    )
    assert normalized == v1


def test_realistic_non_json_uuid_is_strictly_serialized(
    tmp_path: Path, monkeypatch,
) -> None:
    class _CUuuid:
        def __str__(self) -> str:
            return "GPU-outcome-free-probe"

    properties = SimpleNamespace(total_memory=1_000, uuid=_CUuuid())
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 100)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: properties)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "A100-test")
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 9000)
    monkeypatch.setenv("SLURM_JOB_ID", "scientific-v2-test")
    spec = SimpleNamespace(
        arm="dense", seed=64, checkpoint_step=1, path=Path("checkpoint.pt"),
        sha256="a" * 64,
    )
    path = write_runtime_lineage(
        root=tmp_path,
        card={"hardware_plan": {"maximum_peak_memory_fraction": 0.8}},
        card_hash="b" * 64,
        source_hash="c" * 64,
        smoke_receipt=Path("smoke_receipt.json"),
        smoke_hash="d" * 64,
        selection_path=Path("selection_decision.json"),
        selection_hash="e" * 64,
        validation_manifest_path=Path("validation_data_manifest.json"),
        validation_manifest_hash="f" * 64,
        test_manifest_path=Path("test_data_manifest.json"),
        test_manifest_hash="1" * 64,
        scientific_path=Path("scientific_payload.json"),
        scientific_hash="2" * 64,
        specs_and_models=[(spec, object())],
        rng_proof={"stream_count": 1536},
        precision={"autocast": False},
        device=torch.device("cuda"),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["environment"]["gpu_uuid"] == "GPU-outcome-free-probe"
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert strict_uuid_record(_CUuuid()) == {
        "gpu_uuid": "GPU-outcome-free-probe", "raw_uuid_type": "_CUuuid",
    }
    with pytest.raises(RuntimeError, match="Expected PyTorch _CUuuid"):
        strict_uuid_record("GPU-already-a-string")


def test_v2_wrapper_changes_only_lineage_binding_and_restores(monkeypatch) -> None:
    original = base_run.write_runtime_lineage
    observed = []
    monkeypatch.setattr(
        base_run, "main",
        lambda: observed.append(base_run.write_runtime_lineage is write_runtime_lineage),
    )
    v2_run.main()
    assert observed == [True]
    assert base_run.write_runtime_lineage is original


def test_v2_manifest_and_roster_are_complete() -> None:
    card = duplicate_safe_json(V2 / "prediction_card.json")
    required = set(card["source_and_outcome_guard"]["required_manifest_paths"])
    expected = {
        path.relative_to(ROOT).as_posix() for path in V2.glob("*.py")
    } | {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "scripts/neurips_2026/allen_cahn_periodic_reencoding_v2").glob("*.sh")
    } | {"tests/test_allen_cahn_periodic_v2.py"}
    assert expected <= required
    locked = {
        relative: digest
        for digest, relative in (
            line.split(maxsplit=1)
            for line in (V2 / "source_manifest.sha256").read_text().splitlines()
            if line.strip()
        )
    }
    assert expected <= set(locked)
    # The terminal V2 packet remains immutable; this regression test changed
    # later to stop treating scratch-root existence as scientific state.
    for relative, digest in locked.items():
        if relative == "tests/test_allen_cahn_periodic_v2.py":
            continue
        assert sha256_path(ROOT / relative) == digest


def test_v2_roots_are_unique_and_outcomes_remain_unopened() -> None:
    card = duplicate_safe_json(V2 / "prediction_card.json")
    science = Path(card["prospective_datasets"]["output_root"])
    smoke = Path(card["outcome_free_smoke"]["output_root"])
    assert science.name.endswith("_v2")
    assert smoke.name.endswith("_v2")
    # The smoke root is allowed to exist after the recorded operational
    # failure.  Only the absent scientific root and frozen outcome-access flag
    # are durable scientific guards; scratch existence is runtime state.
    assert not science.exists()
    assert card["source_and_outcome_guard"]["launch_state"][
        "scientific_outcomes_accessed"
    ] is False
