from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.neurips_2026.global_k_residual_forecast import (
    evaluate,
    preflight,
    protocol,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments" / "neurips_2026" / "global_k_residual_forecast"
SCRIPTS = ROOT / "scripts" / "neurips_2026" / "global_k_residual_forecast"


def _card() -> dict:
    return json.loads((PACKAGE / "prediction_card.json").read_text())


def _tasks() -> dict:
    return json.loads((PACKAGE / "task_manifest.json").read_text())


@pytest.mark.parametrize("race_role", ["card", "task", "source"])
def test_residual_roots_are_decoded_from_the_exact_authenticated_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, race_role: str,
) -> None:
    task_path = tmp_path / "tasks.json"
    source_path = tmp_path / "sources.sha256"
    task_bytes = (PACKAGE / "task_manifest.json").read_bytes()
    stable_source = ROOT / "pyproject.toml"
    source_bytes = (
        f"{protocol.sha256_path(stable_source)}  pyproject.toml\n".encode()
    )
    task_path.write_bytes(task_bytes)
    source_path.write_bytes(source_bytes)
    card = copy.deepcopy(_card())
    card["freeze"]["launch_authorized"] = True
    card["freeze"]["task_manifest_sha256"] = protocol.sha256_bytes(task_bytes)
    card["freeze"]["source_manifest_sha256"] = protocol.sha256_bytes(source_bytes)
    card_path = tmp_path / "card.json"
    card_bytes = json.dumps(card, sort_keys=True).encode()
    card_path.write_bytes(card_bytes)
    expected = {
        card_path: protocol.sha256_bytes(card_bytes),
        task_path: protocol.sha256_bytes(task_bytes),
        source_path: protocol.sha256_bytes(source_bytes),
    }
    target = {"card": card_path, "task": task_path, "source": source_path}[
        race_role
    ]
    original = protocol.read_verified_bytes

    def mutate_after_read(path: Path, digest: str, role: str) -> bytes:
        value = original(path, digest, role)
        if path == target:
            path.write_bytes(b"mutated after authenticated read")
        return value

    monkeypatch.setattr(protocol, "read_verified_bytes", mutate_after_read)
    loaded, tasks, _ = protocol.load_frozen_protocol(
        card_path=card_path,
        task_path=task_path,
        source_manifest_path=source_path,
        expected_card_sha256=expected[card_path],
        expected_task_sha256=expected[task_path],
        expected_source_manifest_sha256=expected[source_path],
    )
    assert loaded["freeze"]["launch_authorized"] is True
    assert len(tasks["tasks"]) == 10


def test_checkpoint_loader_deserializes_only_authenticated_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "seed_100" / "checkpoint.pt"
    path.parent.mkdir()
    path.write_bytes(b"authenticated checkpoint")
    events = []

    def read_once(_path: Path, _digest: str, _role: str) -> bytes:
        value = _path.read_bytes()
        _path.write_bytes(b"replacement after read")
        events.append("authenticated")
        return value

    def load_bytes(value: bytes, map_location: str) -> dict:
        assert value == b"authenticated checkpoint"
        assert map_location == "cpu"
        events.append("deserialized_authenticated_bytes")
        return {"config": {}, "model_state_dict": {}}

    class DummyConfig:
        @staticmethod
        def from_dict(_value: dict) -> object:
            return object()

    class DummyModel:
        def to(self, _device: str):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(evaluate, "read_verified_bytes", read_once)
    monkeypatch.setattr(evaluate, "load_torch_payload", load_bytes)
    monkeypatch.setattr(evaluate, "Config", DummyConfig)
    monkeypatch.setattr(
        evaluate, "make_env", lambda _cfg: SimpleNamespace(observation_size=2)
    )
    monkeypatch.setattr(evaluate, "make_model", lambda _cfg, _size: DummyModel())
    monkeypatch.setattr(evaluate, "load_model_state_dict_compat", lambda *_: None)

    def audit(_cfg, _model, _checkpoint, _card, _spec, representative):
        assert representative == {"already": "decoded"}
        events.append("audited_in_memory")
        return {"checkpoint_step": 10}

    monkeypatch.setattr(evaluate, "audit_authenticated_checkpoint", audit)
    monkeypatch.setattr(
        evaluate,
        "trainable_parameter_counts",
        lambda _model: {"total": 1},
    )
    row = {
        "task_id": 0,
        "model_seed": 100,
        "sparse_checkpoint": {"path": str(path), "sha256": "a" * 64},
    }
    evaluate._load_model(
        row, "sparse", {"training_arms": {}}, "cpu", {"already": "decoded"}
    )
    assert path.read_bytes() == b"replacement after read"
    assert events == [
        "authenticated", "deserialized_authenticated_bytes", "audited_in_memory"
    ]


def test_residual_in_memory_audit_rejects_schema_drift_after_terminal_packet() -> None:
    bundle = protocol.authenticate_v2_inputs(_card())
    task = _tasks()["tasks"][0]
    expected = _tasks()["provenance_contract"]
    with pytest.raises(AssertionError, match="checkpoint audit failed"):
        evaluate._load_model(
            task,
            "sparse",
            bundle["card"],
            "cpu",
            bundle["representative_config"],
        )


def _portable_script_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    repo = tmp_path / "repo"
    package = repo / "experiments" / "neurips_2026" / "global_k_residual_forecast"
    common = repo / "scripts" / "common"
    package.mkdir(parents=True)
    common.mkdir(parents=True)
    event_log = tmp_path / "events.log"
    cluster = common / "cluster_env.sh"
    cluster.write_text('printf "cluster\\n" >> "${EVENT_LOG}"\n')
    card = package / "prediction_card.json"
    tasks = package / "task_manifest.json"
    card.write_text("{}\n")
    tasks.write_text("{}\n")
    manifest = package / "source_manifest.sha256"
    manifest.write_text(
        f"{hashlib.sha256(cluster.read_bytes()).hexdigest()}  "
        "scripts/common/cluster_env.sh\n"
    )
    expected = {
        "card": hashlib.sha256(card.read_bytes()).hexdigest(),
        "task": hashlib.sha256(tasks.read_bytes()).hexdigest(),
        "source": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, event, status in (
        ("uv", "preflight", 42),
        ("mkdir", "mkdir", 0),
        ("nvidia-smi", "gpu", 0),
        ("sbatch", "sbatch", 0),
        ("git", "git", 0),
    ):
        stub = bindir / name
        stub.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '{event}\\n' >> \"${{EVENT_LOG}}\"\n"
            f"exit {status}\n"
        )
        stub.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "EVENT_LOG": str(event_log),
        "EXPECTED_CARD_SHA256": expected["card"],
        "EXPECTED_TASK_SHA256": expected["task"],
        "EXPECTED_SOURCE_MANIFEST_SHA256": expected["source"],
        "AUTHORIZE_GLOBAL_K_RESIDUAL_FORECAST": (
            f"root-redteam-approved:{expected['card']}:{expected['source']}:"
            f"{expected['task']}"
        ),
        "OUTPUT_ROOT": str(tmp_path / "output"),
        "MODE": "smoke",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    return repo, event_log, environment


@pytest.mark.parametrize(
    "name",
    ["queue.sh", "run_prepare.sh", "run_forecast.sh", "run_telemetry.sh", "run_summary.sh"],
)
def test_failed_cpu_preflight_precedes_every_mutation_or_gpu_event(
    tmp_path: Path, name: str,
) -> None:
    repo, event_log, environment = _portable_script_fixture(tmp_path)
    source = (SCRIPTS / name).read_text().replace(
        '/home/mila/l/lia/skae', str(repo)
    )
    if name == "queue.sh":
        source = source.replace(
            '/network/scratch/l/lia/skae/global_k_residual_forecast_v3_20260721',
            str(tmp_path / "output"),
        )
    script = tmp_path / name
    script.write_text(source)
    result = subprocess.run(
        ["bash", str(script)], cwd=repo, env=environment,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 42
    assert event_log.read_text().splitlines() == ["cluster", "preflight"]


@pytest.mark.parametrize("mutation", ["card", "listed_source"])
def test_shell_root_or_manifest_mutation_fails_before_cluster_or_preflight(
    tmp_path: Path, mutation: str,
) -> None:
    repo, event_log, environment = _portable_script_fixture(tmp_path)
    package = repo / "experiments" / "neurips_2026" / "global_k_residual_forecast"
    if mutation == "card":
        (package / "prediction_card.json").write_text("mutated\n")
    else:
        (repo / "scripts" / "common" / "cluster_env.sh").write_text("mutated\n")
    script = tmp_path / "run_forecast.sh"
    script.write_text(
        (SCRIPTS / "run_forecast.sh").read_text().replace(
            '/home/mila/l/lia/skae', str(repo)
        )
    )
    result = subprocess.run(
        ["bash", str(script)], cwd=repo, env=environment,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert not event_log.exists()


def test_static_shell_and_internal_reauthentication_order() -> None:
    for name in (
        "queue.sh", "run_prepare.sh", "run_forecast.sh", "run_telemetry.sh",
        "run_summary.sh",
    ):
        text = (SCRIPTS / name).read_text()
        roots = text.index("printf '%s  %s\\n'")
        manifest = text.index('sha256sum --check --strict --status "${SOURCE_PATH}"')
        cluster = text.index("source scripts/common/cluster_env.sh")
        cpu = text.index("global_k_residual_forecast.preflight")
        assert roots < manifest < cluster < cpu
        if name == "queue.sh":
            assert cpu < text.index('mkdir --mode=700 "${OUTPUT_ROOT}"')
            assert cpu < text.index("sbatch --parsable")
        if name == "run_forecast.sh":
            assert cpu < text.index('mkdir -p "${MODE_ROOT}')
            assert cpu < text.index("nvidia-smi --query-gpu")
            assert cpu < text.index("global_k_residual_forecast.evaluate")

    evaluate_source = (PACKAGE / "evaluate.py").read_text()
    prepare_source = (PACKAGE / "prepare_data.py").read_text()
    telemetry_source = (PACKAGE / "telemetry.py").read_text()
    summary_source = (PACKAGE / "summarize.py").read_text()
    assert evaluate_source.index("authenticate_checkpoint_roster(tasks)") < (
        evaluate_source.index("torch.cuda.is_available()")
    )
    assert prepare_source.index("authenticate_checkpoint_roster(tasks)") < (
        prepare_source.index("data_dir.mkdir")
    )
    assert telemetry_source.index("authenticate_checkpoint_roster(tasks)") < (
        telemetry_source.index("_assess_one(", telemetry_source.index("def main"))
    )
    assert summary_source.index("authenticate_checkpoint_roster(tasks)") < (
        summary_source.index("validate_gate(", summary_source.index("def main"))
    )


def _temporary_checkpoint_tasks(tmp_path: Path) -> dict:
    tasks = copy.deepcopy(_tasks())
    for row in tasks["tasks"]:
        for arm in ("sparse", "dense"):
            path = (
                tmp_path / arm / f"seed_{row['model_seed']}" / "checkpoint.pt"
            )
            value = f"{arm}-{row['model_seed']}".encode()
            path.parent.mkdir(parents=True)
            path.write_bytes(value)
            row[f"{arm}_checkpoint"] = {
                "path": str(path),
                "sha256": hashlib.sha256(value).hexdigest(),
            }
    return tasks


def test_complete_checkpoint_roster_rejects_every_mutation_and_swap(
    tmp_path: Path,
) -> None:
    tasks = _temporary_checkpoint_tasks(tmp_path)
    assert protocol.authenticate_checkpoint_roster(tasks) == 20
    for row in tasks["tasks"]:
        for arm in ("sparse", "dense"):
            path = Path(row[f"{arm}_checkpoint"]["path"])
            original = path.read_bytes()
            path.write_bytes(original + b"tampered")
            with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
                protocol.authenticate_checkpoint_roster(tasks)
            path.write_bytes(original)
    swapped = copy.deepcopy(tasks)
    swapped["tasks"][0]["sparse_checkpoint"], swapped["tasks"][1][
        "sparse_checkpoint"
    ] = (
        swapped["tasks"][1]["sparse_checkpoint"],
        swapped["tasks"][0]["sparse_checkpoint"],
    )
    with pytest.raises(RuntimeError, match="provenance"):
        protocol.authenticate_checkpoint_roster(swapped)


def test_preflight_rejects_stage_or_output_before_bundle_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    card = _card()
    card["freeze"]["launch_authorized"] = True
    tasks = _tasks()
    monkeypatch.setattr(
        preflight, "load_frozen_protocol", lambda **_: (card, tasks, {})
    )
    events = []
    monkeypatch.setattr(
        preflight, "authenticate_v2_inputs", lambda _card: events.append("v2")
    )
    monkeypatch.setattr(
        preflight, "authenticate_checkpoint_roster", lambda _tasks: 20
    )
    common = {
        "card_path": tmp_path / "card",
        "task_path": tmp_path / "tasks",
        "source_manifest_path": tmp_path / "sources",
        "expected_card_sha256": "a" * 64,
        "expected_task_sha256": "b" * 64,
        "expected_source_manifest_sha256": "c" * 64,
    }
    with pytest.raises(RuntimeError, match="Output root"):
        preflight.authenticate_complete_bundle(
            **common, stage="queue", output_root=tmp_path / "wrong"
        )
    assert events == []


def test_preflight_attestation_is_explicitly_outcome_blind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    card = _card()
    card["freeze"]["launch_authorized"] = True
    card["freeze"]["output_root"] = str(tmp_path)
    tasks = _tasks()
    monkeypatch.setattr(
        preflight, "load_frozen_protocol", lambda **_: (card, tasks, {"root": "fixed"})
    )
    monkeypatch.setattr(
        preflight,
        "authenticate_v2_inputs",
        lambda _card: {"card": {"protocol_id": "authenticated-v2"}},
    )
    monkeypatch.setattr(
        preflight, "authenticate_checkpoint_roster", lambda _tasks: 20
    )
    payload = preflight.authenticate_complete_bundle(
        card_path=tmp_path / "card",
        task_path=tmp_path / "tasks",
        source_manifest_path=tmp_path / "sources",
        expected_card_sha256="a" * 64,
        expected_task_sha256="b" * 64,
        expected_source_manifest_sha256="c" * 64,
        stage="queue",
        output_root=tmp_path,
    )
    assert payload["checkpoint_count"] == 20
    assert payload["outcomes_inspected"] is False
    assert "forecast_outcomes_read" not in payload


def test_cpu_preflight_rejects_visible_cuda(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight", "--stage", "queue", "--output-root", str(tmp_path),
            "--expected-card-sha256", "a" * 64,
            "--expected-task-sha256", "b" * 64,
            "--expected-source-manifest-sha256", "c" * 64,
        ],
    )
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        preflight.main()
