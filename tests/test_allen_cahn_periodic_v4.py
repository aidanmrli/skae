"""Outcome-blind regression tests for periodic-reencoding execution v4."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.neurips_2026.allen_cahn_periodic_reencoding.io import (
    duplicate_safe_json,
    parse_source_manifest,
    sha256_path,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding_v4 import guard


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "experiments/neurips_2026/allen_cahn_periodic_reencoding"
V2 = ROOT / "experiments/neurips_2026/allen_cahn_periodic_reencoding_v2"
V3 = ROOT / "experiments/neurips_2026/allen_cahn_periodic_reencoding_v3"
V4 = ROOT / "experiments/neurips_2026/allen_cahn_periodic_reencoding_v4"
SCRIPTS = ROOT / "scripts/neurips_2026/allen_cahn_periodic_reencoding_v4"
CARD_HASH = "a" * 64
SOURCE_HASH = "b" * 64
BARE_UUID = "49925033-cf14-40cd-4a4f-12fad21d1629"
GPU_UUID = f"GPU-{BARE_UUID}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_smoke_fixture(root: Path) -> tuple[str, str]:
    artifacts = {
        "runtime_path": ("runtime_sha256", "smoke_runtime.json"),
        "start_marker_path": ("start_marker_sha256", "markers/smoke_start.json"),
        "end_marker_path": ("end_marker_sha256", "markers/smoke_end.json"),
        "telemetry_audit_path": (
            "telemetry_audit_sha256", "smoke_telemetry_audit.json"
        ),
        "raw_telemetry_path": ("raw_telemetry_sha256", "raw_gpu_telemetry.csv"),
    }
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed_outcome_free_gpu_smoke",
        "card_sha256": CARD_HASH,
        "source_manifest_sha256": SOURCE_HASH,
        "slurm_job_id": "10170001",
        "scientific_outcomes_accessed": False,
    }
    for index, (path_key, (hash_key, relative)) in enumerate(artifacts.items()):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"metric-free-{index}\n", encoding="utf-8")
        receipt[path_key] = str(path)
        receipt[hash_key] = sha256_path(path)
    receipt_path = root / "smoke_receipt.json"
    _write_json(receipt_path, receipt)
    probe_path = root / "lineage_uuid_probe.json"
    _write_json(
        probe_path,
        {
            "schema_version": 1,
            "status": "passed_real_cuda_uuid_crosscheck_strict_json",
            "card_sha256": CARD_HASH,
            "source_manifest_sha256": SOURCE_HASH,
            "slurm_job_id": "10170001",
            "gpu_name": "NVIDIA A100-SXM4-80GB",
            "gpu_uuid": GPU_UUID,
            "raw_uuid_type": "_CUuuid",
            "pytorch_uuid_raw_text": BARE_UUID,
            "pytorch_uuid_canonical": BARE_UUID,
            "nvidia_smi_uuid_raw_text": GPU_UUID,
            "nvidia_smi_uuid_canonical": GPU_UUID,
            "nvidia_smi_visible_gpu_count": 1,
            "uuid_sources_match": True,
            "scientific_outcomes_accessed": False,
        },
    )
    return sha256_path(receipt_path), sha256_path(probe_path)


def _patch_smoke_freeze(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(
        guard,
        "_load_frozen_inputs",
        lambda **_kwargs: (
            {"outcome_free_smoke": {"output_root": str(root)}},
            CARD_HASH,
            SOURCE_HASH,
        ),
    )


def _validate_smoke(root: Path, receipt_hash: str, probe_hash: str) -> None:
    guard.validate_smoke_artifacts(
        card_path=Path("card.json"),
        expected_card_sha256=CARD_HASH,
        source_manifest=Path("source.sha256"),
        expected_source_manifest_sha256=SOURCE_HASH,
        smoke_root=root,
        expected_smoke_receipt_sha256=receipt_hash,
        expected_uuid_probe_sha256=probe_hash,
    )


def _write_outcome_guard(root: Path, **updates: Any) -> Path:
    payload = {
        "schema_version": 1,
        "protocol_id": "allen_cahn_periodic_reencoding_confirmation_v1",
        "status": "authorized_for_dependent_cpu_summary",
        "card_sha256": CARD_HASH,
        "source_manifest_sha256": SOURCE_HASH,
        "slurm_job_id": "10170002",
        "scientific_payload_opened": False,
        "forecast_outcomes_accessed": False,
        "both_disjoint_windows_required": True,
        "disjoint_window_ordering_verified": True,
        "evaluation_end_epoch_identity_verified": True,
        "runtime_lineage_path": str(root / "runtime_lineage.json"),
        "scientific_payload_path": str(root / "scientific_payload.json"),
        "validity_checks": {
            "selection_validity": {"mean": True, "p10": True},
            "evaluation_validity": {"mean": True, "p10": True},
        },
    }
    payload.update(updates)
    path = root / "outcome_guard_receipt.json"
    _write_json(path, payload)
    return path


def _patch_outcome_freeze(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(
        guard,
        "_load_frozen_inputs",
        lambda **_kwargs: (
            {
                "protocol_id": "allen_cahn_periodic_reencoding_confirmation_v1",
                "prospective_datasets": {"output_root": str(root)},
            },
            CARD_HASH,
            SOURCE_HASH,
        ),
    )


def _validate_outcome(root: Path, job_id: str = "10170002") -> str:
    return guard.validate_outcome_guard(
        card_path=Path("card.json"),
        expected_card_sha256=CARD_HASH,
        source_manifest=Path("source.sha256"),
        expected_source_manifest_sha256=SOURCE_HASH,
        output_root=root,
        expected_scientific_job_id=job_id,
    )


def test_v1_through_v3_remain_frozen_and_v4_science_is_exact() -> None:
    expected = {
        V1: (
            "97716bb6dc2c0e6a4f8389d362c06ad67045ac2a85574c794f1966a388ce9e17",
            "a4729c04d1981c031d1531304668dc01b4165c1d5b1610d4d780d9730d123c6f",
        ),
        V2: (
            "e5af3746ae9a537f4c7860221228c9f39fc92acd7ce6442b85cd48d1f35cab4f",
            "0fa8b1994c1634164224441c3baf19f48314fb9db9ae89472c2802d491d82265",
        ),
        V3: (
            "a8636606f3248135759efe18f82b6dd62c95ad0d731f979611bf2093bdee5f48",
            "6e738758bcb1fc6d3041e0cd46696588a668085df53e722e7a5203af54ad8a68",
        ),
    }
    for package, (card_hash, source_hash) in expected.items():
        assert sha256_path(package / "prediction_card.json") == card_hash
        assert sha256_path(package / "source_manifest.sha256") == source_hash
    v3 = duplicate_safe_json(V3 / "prediction_card.json")
    v4 = duplicate_safe_json(V4 / "prediction_card.json")
    scientific_keys = (
        "status", "experiment_type", "question", "hypothesis",
        "scientific_distinction", "prior_outcome_aware_context", "frozen_parent",
        "roster", "system", "cadence_selection", "test_evaluation",
        "estimand_and_inference", "interpretation_branches", "visualization_plan",
        "hardware_plan",
    )
    assert all(v4[key] == v3[key] for key in scientific_keys)
    assert all(
        key == "output_root" or v4["prospective_datasets"][key] == value
        for key, value in v3["prospective_datasets"].items()
    )
    assert v4["outcome_free_smoke"]["workload"] == v3["outcome_free_smoke"]["workload"]
    assert v4["outcome_free_smoke"]["forbidden"] == v3["outcome_free_smoke"]["forbidden"]
    assert v4["outcome_blind_v3_operational_history"]["scientific_choices_changed_in_v4"] is False


def test_v4_manifest_is_exact_v3_transitive_extension() -> None:
    v3_entries = set(parse_source_manifest(V3 / "source_manifest.sha256"))
    additions = {
        "experiments/neurips_2026/allen_cahn_periodic_reencoding_v3/source_manifest.sha256",
        "experiments/neurips_2026/allen_cahn_periodic_reencoding_v4/__init__.py",
        "experiments/neurips_2026/allen_cahn_periodic_reencoding_v4/guard.py",
        "experiments/neurips_2026/allen_cahn_periodic_reencoding_v4/prediction_card.json",
        "scripts/neurips_2026/allen_cahn_periodic_reencoding_v4/queue.sh",
        "scripts/neurips_2026/allen_cahn_periodic_reencoding_v4/run_generate_select_evaluate.sh",
        "scripts/neurips_2026/allen_cahn_periodic_reencoding_v4/run_smoke.sh",
        "scripts/neurips_2026/allen_cahn_periodic_reencoding_v4/run_summary.sh",
        "tests/test_allen_cahn_periodic_v4.py",
        "docs/archive/allen_cahn_periodic_reencoding_v3_jq_startup_failure_20260721.md",
    }
    card = duplicate_safe_json(V4 / "prediction_card.json")
    assert set(card["source_and_outcome_guard"]["required_manifest_paths"]) == v3_entries | additions
    assert set(parse_source_manifest(V4 / "source_manifest.sha256")) == v3_entries | additions
    locked = parse_source_manifest(V4 / "source_manifest.sha256")
    for relative, digest in locked.items():
        if relative.startswith("tests/"):
            continue
        assert sha256_path(ROOT / relative) == digest


def test_scripts_are_portable_and_reuse_v3_science() -> None:
    sources = {path.name: path.read_text(encoding="utf-8") for path in SCRIPTS.glob("*.sh")}
    assert set(sources) == {
        "queue.sh", "run_generate_select_evaluate.sh", "run_smoke.sh", "run_summary.sh"
    }
    assert all("jq" not in source for source in sources.values())
    assert all("#SBATCH --partition=long" in source for source in sources.values())
    assert "#SBATCH --gres=gpu:a100l:1" in sources["run_generate_select_evaluate.sh"]
    assert "#SBATCH --gres=gpu:a100l:1" in sources["run_smoke.sh"]
    smoke_guard = "allen_cahn_periodic_reencoding_v4.guard smoke"
    assert smoke_guard in sources["queue.sh"]
    assert smoke_guard in sources["run_generate_select_evaluate.sh"]
    assert smoke_guard in sources["run_smoke.sh"]
    assert "allen_cahn_periodic_reencoding_v3.run" in sources["run_generate_select_evaluate.sh"]
    assert "allen_cahn_periodic_reencoding_v3.uuid_probe" in sources["run_smoke.sh"]
    assert "allen_cahn_periodic_reencoding_v4.guard outcome" in sources["run_summary.sh"]


def test_unique_roots_and_fail_closed_lifecycle() -> None:
    cards = [duplicate_safe_json(package / "prediction_card.json") for package in (V1, V2, V3, V4)]
    science = [Path(card["prospective_datasets"]["output_root"]) for card in cards]
    smoke = [Path(card["outcome_free_smoke"]["output_root"]) for card in cards]
    assert len(set(science + smoke)) == 8
    assert science[-1].name.endswith("_v4") and smoke[-1].name.endswith("_v4")
    queue = (SCRIPTS / "queue.sh").read_text(encoding="utf-8")
    gpu = (SCRIPTS / "run_generate_select_evaluate.sh").read_text(encoding="utf-8")
    smoke_script = (SCRIPTS / "run_smoke.sh").read_text(encoding="utf-8")
    assert queue.index('if [[ -e "${OUTPUT_ROOT}" ]]') < queue.index("guard smoke")
    assert gpu.index('if [[ -e "${OUTPUT_ROOT}" ]]') < gpu.index("guard smoke")
    assert 'if [[ -e "${OUTPUT_ROOT}" ]]' in smoke_script


def test_smoke_guard_accepts_exact_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_smoke_freeze(monkeypatch, tmp_path)
    receipt_hash, probe_hash = _write_smoke_fixture(tmp_path)
    _validate_smoke(tmp_path, receipt_hash, probe_hash)


def test_smoke_guard_rejects_duplicate_and_malformed_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_smoke_freeze(monkeypatch, tmp_path)
    receipt_hash, _ = _write_smoke_fixture(tmp_path)
    probe = tmp_path / "lineage_uuid_probe.json"
    probe.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        _validate_smoke(tmp_path, receipt_hash, sha256_path(probe))
    probe.write_text('{"schema_version":', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        _validate_smoke(tmp_path, receipt_hash, sha256_path(probe))


@pytest.mark.parametrize(
    ("artifact", "field", "value", "message"),
    [
        ("smoke_receipt.json", "card_sha256", "c" * 64, "freeze"),
        ("lineage_uuid_probe.json", "slurm_job_id", "10170009", "job"),
        ("lineage_uuid_probe.json", "scientific_outcomes_accessed", True, "outcome"),
        ("lineage_uuid_probe.json", "pytorch_uuid_raw_text", BARE_UUID.upper(), "identity"),
    ],
)
def test_smoke_guard_rejects_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    field: str,
    value: Any,
    message: str,
) -> None:
    _patch_smoke_freeze(monkeypatch, tmp_path)
    _write_smoke_fixture(tmp_path)
    path = tmp_path / artifact
    payload = duplicate_safe_json(path)
    payload[field] = value
    _write_json(path, payload)
    with pytest.raises(RuntimeError, match=message):
        _validate_smoke(
            tmp_path,
            sha256_path(tmp_path / "smoke_receipt.json"),
            sha256_path(tmp_path / "lineage_uuid_probe.json"),
        )


def test_smoke_guard_rejects_hash_and_root_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_smoke_freeze(monkeypatch, tmp_path)
    receipt_hash, probe_hash = _write_smoke_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        _validate_smoke(tmp_path, "0" * 64, probe_hash)
    with pytest.raises(RuntimeError, match="Smoke root differs"):
        _validate_smoke(tmp_path / "wrong", receipt_hash, probe_hash)


def test_outcome_guard_returns_only_authenticated_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_outcome_freeze(monkeypatch, tmp_path)
    path = _write_outcome_guard(tmp_path)
    assert _validate_outcome(tmp_path) == sha256_path(path)


@pytest.mark.parametrize(
    ("updates", "job_id"),
    [
        ({"status": "failed"}, "10170002"),
        ({"slurm_job_id": "10170003"}, "10170002"),
        ({"scientific_payload_opened": True}, "10170002"),
        ({"forecast_outcomes_accessed": True}, "10170002"),
    ],
)
def test_outcome_guard_rejects_status_job_or_access_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, Any],
    job_id: str,
) -> None:
    _patch_outcome_freeze(monkeypatch, tmp_path)
    _write_outcome_guard(tmp_path, **updates)
    with pytest.raises(RuntimeError, match="status, job, freeze, or no-access"):
        _validate_outcome(tmp_path, job_id)


def test_outcome_guard_rejects_duplicate_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_outcome_freeze(monkeypatch, tmp_path)
    path = tmp_path / "outcome_guard_receipt.json"
    path.write_text('{"status":"a","status":"b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        _validate_outcome(tmp_path)


def test_outcome_cli_prints_exactly_one_digest(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    args = SimpleNamespace(
        command="outcome", card=Path("card"), expected_card_sha256=CARD_HASH,
        source_manifest=Path("source"), expected_source_manifest_sha256=SOURCE_HASH,
        output_root=Path("root"), expected_scientific_job_id="10170002",
    )
    monkeypatch.setattr(guard, "parse_args", lambda: args)
    monkeypatch.setattr(guard, "validate_outcome_guard", lambda **_kwargs: "d" * 64)
    guard.main()
    captured = capsys.readouterr()
    assert captured.out == "d" * 64 + "\n"
    assert captured.err == ""
