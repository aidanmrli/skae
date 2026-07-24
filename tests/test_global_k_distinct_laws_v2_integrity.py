import json
from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.neurips_2026.global_k_distinct_laws_v2_math import (
    decoder_linearity_diagnostics,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_preflight import (
    preflight_mixed_pack,
    preflight_scientific_queue,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_source_lock import (
    EXTERNAL_INPUT_KEYS,
    SOURCE_PATHS,
    build_lock,
    verify_source_lock,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_tasks import (
    build_rows,
    load_card,
    sha256_path,
    write_tsv,
)


ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = (
    ROOT / "experiments/neurips_2026/global_k_distinct_laws_v2_card.json"
)


class OddNonlinearDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def decode(self, latent):
        return latent[..., :2] ** 3 + 0.0 * self.anchor


def _fresh_lock(tmp_path):
    inputs = []
    for name in (
        "smoke_tasks.tsv", "smoke_manifest.json",
        "full_tasks.tsv", "full_manifest.json",
    ):
        path = tmp_path / name
        path.write_text(name + "\n")
        inputs.append(path)
    return build_lock(CARD_PATH, *inputs)


def _valid_protocol_files(tmp_path):
    card, card_sha = load_card(CARD_PATH)
    paths = {}
    for mode in ("smoke", "full"):
        task_path = tmp_path / f"{mode}_tasks.tsv"
        manifest_path = tmp_path / f"{mode}_manifest.json"
        rows = build_rows(card, mode)
        write_tsv(task_path, rows)
        manifest = {
            "schema_version": 1, "protocol_id": card["protocol_id"],
            "mode": mode, "card_sha256": card_sha,
            "task_tsv_sha256": sha256_path(task_path), "task_count": len(rows),
            "arms": [row["arm"] for row in rows],
            "seeds": [row["seed"] for row in rows],
            "outcomes_quarantined": mode == "smoke",
        }
        manifest_path.write_text(json.dumps(manifest))
        paths[mode] = (task_path, manifest_path)
    lock = build_lock(
        CARD_PATH, paths["smoke"][0], paths["smoke"][1],
        paths["full"][0], paths["full"][1],
    )
    lock_path = tmp_path / "source_lock.json"
    lock_path.write_text(json.dumps(lock))
    return card, lock, lock_path, paths


def test_decoder_probe_rejects_an_odd_nonlinearity():
    diagnostics = decoder_linearity_diagnostics(OddNonlinearDecoder(), 256)
    assert diagnostics["zero_preserving"] is True
    assert diagnostics["additive"] is False
    assert diagnostics["homogeneous"] is False
    assert diagnostics["linear"] is False


@pytest.mark.parametrize(
    ("section", "key", "message"),
    [
        ("sources", SOURCE_PATHS[0], "source roster"),
        ("external_inputs", sorted(EXTERNAL_INPUT_KEYS)[0], "external roster"),
    ],
)
def test_source_lock_rejects_a_deleted_required_record(
    tmp_path, section, key, message,
):
    payload = _fresh_lock(tmp_path)
    del payload[section][key]
    path = tmp_path / f"bad_{section}.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match=message):
        verify_source_lock(path)


def test_shell_boundaries_have_no_jq_dependency_and_initialize_before_preflight():
    card = json.loads(CARD_PATH.read_text())
    keys = card["gpu_utilization_and_schedule"]["smoke"][
        "required_decision_check_keys"
    ]
    assert len(keys) == len(set(keys)) == 8
    assert set(keys) == {
        "all_processes_exit_zero", "minimum_active_samples",
        "telemetry_covers_complete_active_window",
        "mean_active_gpu_utilization", "p10_active_gpu_utilization",
        "peak_memory", "projected_full_pack_wall_time", "one_a100_80gb",
    }
    script_dir = ROOT / "scripts/neurips_2026/global_k_distinct_laws_v2"
    mixed = (script_dir / "run_mixed_pack.sh").read_text()
    queue = (script_dir / "queue_scientific_chain.sh").read_text()
    for source in (mixed, queue):
        assert "jq " not in source
        assert source.index("source scripts/common/cluster_env.sh") < source.index(
            "global_k_distinct_laws_v2_preflight"
        )
    assert mixed.index("global_k_distinct_laws_v2_preflight") < mixed.index(
        'STATUS_DIR="${PACK_ROOT}/status"'
    )
    assert queue.index("global_k_distinct_laws_v2_preflight") < queue.index(
        "TRAIN_JOB_ID=$("
    )


def test_python_preflights_authenticate_exact_inputs_without_jq(tmp_path):
    card, lock, lock_path, paths = _valid_protocol_files(tmp_path)
    mixed = preflight_mixed_pack(
        mode="smoke", card_path=CARD_PATH, source_lock_path=lock_path,
        expected_source_lock_sha=sha256_path(lock_path),
        task_tsv=paths["smoke"][0], task_manifest=paths["smoke"][1],
    )
    assert mixed["status"] == "passed"
    checks = {
        key: True for key in card["gpu_utilization_and_schedule"]["smoke"][
            "required_decision_check_keys"
        ]
    }
    decision = {
        "protocol_id": card["protocol_id"], "passed": True,
        "outcomes_inspected": False, "checks": checks,
        "provenance": {
            "card_sha256": lock["card_sha256"],
            "source_lock_sha256": sha256_path(lock_path),
            "task_tsv_sha256": lock["external_inputs"]["smoke_task_tsv"]["sha256"],
            "assessor_sha256": lock["sources"][
                "experiments/neurips_2026/assess_global_k_distinct_laws_v2_smoke.py"
            ]["sha256"],
        },
    }
    decision_path = tmp_path / "smoke_decision.json"
    decision_path.write_text(json.dumps(decision))
    queued = preflight_scientific_queue(
        card_path=CARD_PATH, source_lock_path=lock_path,
        smoke_decision=decision_path, task_tsv=paths["full"][0],
        task_manifest=paths["full"][1],
    )
    assert queued["status"] == "passed"
    del decision["checks"][next(iter(decision["checks"]))]
    decision_path.write_text(json.dumps(decision))
    with pytest.raises(RuntimeError, match="missing, extra"):
        preflight_scientific_queue(
            card_path=CARD_PATH, source_lock_path=lock_path,
            smoke_decision=decision_path, task_tsv=paths["full"][0],
            task_manifest=paths["full"][1],
        )
