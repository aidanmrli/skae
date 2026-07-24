"""Outcome-blind regression tests for periodic-reencoding execution v3."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.neurips_2026.allen_cahn_periodic_reencoding import run as base_run
from experiments.neurips_2026.allen_cahn_periodic_reencoding.io import (
    duplicate_safe_json,
    load_card,
    sha256_path,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding_v3 import (
    gpu_identity,
    run as v3_run,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding_v3.lineage import (
    write_runtime_lineage,
)


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "experiments/neurips_2026/allen_cahn_periodic_reencoding"
V2 = ROOT / "experiments/neurips_2026/allen_cahn_periodic_reencoding_v2"
V3 = ROOT / "experiments/neurips_2026/allen_cahn_periodic_reencoding_v3"
V1_CARD_SHA = "97716bb6dc2c0e6a4f8389d362c06ad67045ac2a85574c794f1966a388ce9e17"
V1_SOURCE_SHA = "a4729c04d1981c031d1531304668dc01b4165c1d5b1610d4d780d9730d123c6f"
V2_CARD_SHA = "e5af3746ae9a537f4c7860221228c9f39fc92acd7ce6442b85cd48d1f35cab4f"
V2_SOURCE_SHA = "0fa8b1994c1634164224441c3baf19f48314fb9db9ae89472c2802d491d82265"
KNOWN_BYTES = bytes.fromhex("00112233445566778899aabbccddeeff")
KNOWN_BARE = "00112233-4455-6677-8899-aabbccddeeff"
KNOWN_GPU = f"GPU-{KNOWN_BARE}"


class _CUuuid:
    def __init__(self, values: object = KNOWN_BYTES, text: str = KNOWN_BARE) -> None:
        self.bytes = values
        self.text = text

    def __str__(self) -> str:
        return self.text


def _patch_nvidia(monkeypatch: pytest.MonkeyPatch, stdout: str) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
            "timeout": 15,
        }
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(gpu_identity.subprocess, "run", fake_run)
    return calls


def test_v1_v2_freezes_remain_exact_and_v3_science_is_unchanged() -> None:
    assert sha256_path(V1 / "prediction_card.json") == V1_CARD_SHA
    assert sha256_path(V1 / "source_manifest.sha256") == V1_SOURCE_SHA
    assert sha256_path(V2 / "prediction_card.json") == V2_CARD_SHA
    assert sha256_path(V2 / "source_manifest.sha256") == V2_SOURCE_SHA
    v2 = duplicate_safe_json(V2 / "prediction_card.json")
    v3 = duplicate_safe_json(V3 / "prediction_card.json")
    assert v3["protocol_id"] == v2["protocol_id"]
    assert v3["execution_attempt_id"].endswith("clean_rerun_v3")
    for key in (
        "status", "experiment_type", "question", "hypothesis",
        "scientific_distinction", "prior_outcome_aware_context", "frozen_parent",
        "roster", "system", "cadence_selection", "test_evaluation",
        "estimand_and_inference", "interpretation_branches", "visualization_plan",
        "hardware_plan",
    ):
        assert v3[key] == v2[key]
    for key, value in v2["prospective_datasets"].items():
        if key != "output_root":
            assert v3["prospective_datasets"][key] == value
    assert v3["prospective_datasets"]["output_root"].endswith("_v3")
    history = v3["outcome_blind_v2_operational_history"]
    assert history["scientific_choices_changed_in_v3"] is False
    assert history["scientific_jobs_submitted"] == 0
    assert history["scientific_outcomes_accessed"] is False


def test_v1_loader_accepts_v3_execution_card() -> None:
    card_path = V3 / "prediction_card.json"
    card, digest = load_card(card_path, expected_sha256=sha256_path(card_path))
    assert digest == sha256_path(card_path)
    assert card["protocol_id"] == "allen_cahn_periodic_reencoding_confirmation_v1"


@pytest.mark.parametrize("upper", [False, True])
def test_known_cuda_bytes_normalize_and_crosscheck_once(
    monkeypatch: pytest.MonkeyPatch, upper: bool,
) -> None:
    bare = KNOWN_BARE.upper() if upper else KNOWN_BARE
    nvidia = f"GPU-{bare}\n"
    calls = _patch_nvidia(monkeypatch, nvidia)
    record = gpu_identity.verified_cuda_uuid_record(_CUuuid(text=bare))
    assert record["gpu_uuid"] == KNOWN_GPU
    assert record["pytorch_uuid_canonical"] == KNOWN_BARE
    assert record["nvidia_smi_uuid_canonical"] == KNOWN_GPU
    assert record["raw_uuid_type"] == "_CUuuid"
    assert record["uuid_sources_match"] is True
    assert calls == [[
        "nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader",
    ]]
    assert json.loads(json.dumps(record, allow_nan=False)) == record


def test_cuda_identity_rejects_wrong_type_or_malformed_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_nvidia(monkeypatch, f"{KNOWN_GPU}\n")
    with pytest.raises(RuntimeError, match="Expected PyTorch _CUuuid"):
        gpu_identity.verified_cuda_uuid_record(KNOWN_BARE)
    with pytest.raises(RuntimeError, match="must contain 16 bytes"):
        gpu_identity.verified_cuda_uuid_record(_CUuuid(values=KNOWN_BYTES[:-1]))
    with pytest.raises(RuntimeError, match="not a byte sequence"):
        gpu_identity.verified_cuda_uuid_record(_CUuuid(values=[256] * 16))


def test_cuda_identity_rejects_string_or_cross_source_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_nvidia(monkeypatch, f"{KNOWN_GPU}\n")
    other = "10112233-4455-6677-8899-aabbccddeeff"
    with pytest.raises(RuntimeError, match="bytes and string disagree"):
        gpu_identity.verified_cuda_uuid_record(_CUuuid(text=other))
    with pytest.raises(RuntimeError, match="not canonical hyphenated"):
        gpu_identity.verified_cuda_uuid_record(_CUuuid(text="{" + KNOWN_BARE + "}"))
    _patch_nvidia(monkeypatch, f"GPU-{other}\n")
    with pytest.raises(RuntimeError, match="UUIDs disagree"):
        gpu_identity.verified_cuda_uuid_record(_CUuuid())
    _patch_nvidia(monkeypatch, f"{KNOWN_GPU}\n{KNOWN_GPU}\n")
    with pytest.raises(RuntimeError, match="exactly one"):
        gpu_identity.verified_cuda_uuid_record(_CUuuid())


def test_lineage_uses_same_verified_identity_and_strict_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_nvidia(monkeypatch, f"{KNOWN_GPU}\n")
    properties = SimpleNamespace(total_memory=1_000, uuid=_CUuuid())
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 100)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: properties)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "A100-test")
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 9000)
    monkeypatch.setenv("SLURM_JOB_ID", "scientific-v3-test")
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
    assert payload["environment"]["gpu_uuid"] == KNOWN_GPU
    assert payload["environment"]["uuid_sources_match"] is True
    assert len(calls) == 1
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_v3_wrapper_changes_only_lineage_binding_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = base_run.write_runtime_lineage
    observed: list[bool] = []
    monkeypatch.setattr(
        base_run, "main",
        lambda: observed.append(base_run.write_runtime_lineage is write_runtime_lineage),
    )
    v3_run.main()
    assert observed == [True]
    assert base_run.write_runtime_lineage is original


def test_v3_manifest_roster_and_unique_roots() -> None:
    card = duplicate_safe_json(V3 / "prediction_card.json")
    required = set(card["source_and_outcome_guard"]["required_manifest_paths"])
    expected = {
        path.relative_to(ROOT).as_posix() for path in V3.glob("*.py")
    } | {
        path.relative_to(ROOT).as_posix()
        for path in (
            ROOT / "scripts/neurips_2026/allen_cahn_periodic_reencoding_v3"
        ).glob("*.sh")
    } | {
        "experiments/neurips_2026/allen_cahn_periodic_reencoding_v3/prediction_card.json",
        "experiments/neurips_2026/allen_cahn_periodic_reencoding_v2/source_manifest.sha256",
        "tests/test_allen_cahn_periodic_v3.py",
        "docs/archive/allen_cahn_periodic_reencoding_v2_smoke_uuid_failure_20260721.md",
    }
    assert expected <= required
    locked = {
        relative: digest
        for digest, relative in (
            line.split(maxsplit=1)
            for line in (V3 / "source_manifest.sha256").read_text().splitlines()
            if line.strip()
        )
    }
    assert expected <= set(locked)
    for relative, digest in locked.items():
        if relative.startswith("tests/"):
            continue
        assert sha256_path(ROOT / relative) == digest
    science = Path(card["prospective_datasets"]["output_root"])
    smoke = Path(card["outcome_free_smoke"]["output_root"])
    assert science.name.endswith("_v3") and smoke.name.endswith("_v3")
    v2 = duplicate_safe_json(V2 / "prediction_card.json")
    historical_roots = {
        Path(v2["prospective_datasets"]["output_root"]),
        Path(v2["outcome_free_smoke"]["output_root"]),
    }
    assert science != smoke
    assert science not in historical_roots and smoke not in historical_roots
    launch = card["source_and_outcome_guard"]["launch_state"]
    assert launch == {
        "scientific_jobs_submitted": 0,
        "datasets_generated": 0,
        "checkpoints_evaluated": 0,
        "scientific_outcomes_accessed": False,
    }
