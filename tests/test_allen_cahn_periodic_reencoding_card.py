from __future__ import annotations

import hashlib
import json

import torch

from experiments.neurips_2026.allen_cahn_periodic_reencoding.generator import (
    realized_rng_streams,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.io import (
    CARD_PATH,
    REPO_ROOT,
    _exact_owned_storage,
    duplicate_safe_json,
    sha256_path,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.run import (
    _cadence_grid,
)


def _card() -> dict:
    return duplicate_safe_json(CARD_PATH)


def test_card_freezes_exact_overcomplete_dense_sparse_roster() -> None:
    card = _card()
    assert card["status"] == "frozen_before_generation_or_evaluation"
    assert card["roster"] == {
        "arms": ["dense", "sparse"],
        "model_seeds": list(range(64, 74)),
        "checkpoint_count": 20,
        "pairing": "same model seed across arms",
    }
    assert card["system"]["state_dim"] == 512
    assert card["system"]["latent_dim"] == 2048
    assert card["system"]["overcomplete_ratio"] == 4
    assert card["frozen_parent"]["dense_control"].startswith(
        "The parent loader asserts encoder_kind=dense"
    )
    parent = REPO_ROOT / card["frozen_parent"]["checkpoint_card"]
    assert sha256_path(parent) == card["frozen_parent"]["checkpoint_card_sha256"]


def test_hash_derived_dataset_seeds_and_streams_are_exact_and_disjoint() -> None:
    card = _card()
    namespace = card["prospective_datasets"]["derivation_namespace"]
    observed = []
    for role, token in (("validation", "cadence_validation"), ("test", "sealed_test")):
        for record in card["prospective_datasets"][role]:
            label = namespace.replace("ROLE", token).replace("INDEX", str(record["index"]))
            digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
            value = int(digest[:8], 16) & 0x7FFFFFFF
            assert digest == record["sha256_derivation"]
            assert value == record["seed"]
            observed.append(value)
    assert len(observed) == len(set(observed)) == 6
    proof = realized_rng_streams(card)
    assert proof["stream_count"] == 6 * 256
    assert proof["excluded_intersection_empty"] is True
    assert proof["maximum"] < 2**31


def test_cadence_grid_and_horizon_policy_are_frozen() -> None:
    card = _card()
    assert _cadence_grid(card) == ["direct", 1, 2, 5, 10, 20, 25, 50, 100]
    assert card["system"]["validation_horizon_steps"] == 200
    assert card["system"]["test_horizon_steps"] == 400
    assert "per-seed" in card["cadence_selection"]["forbidden"]
    assert "No finite-prefix" in card["test_evaluation"]["strict_finiteness"]


def test_exact_owned_storage_rejects_views_with_packed_backing_storage() -> None:
    packed = torch.zeros(3, 4, 5, dtype=torch.float32)
    view = packed[0]
    assert not _exact_owned_storage(view)
    owned = view.clone().contiguous()
    assert _exact_owned_storage(owned)
    assert owned.untyped_storage().nbytes() == owned.numel() * owned.element_size()


def test_card_json_has_no_duplicate_keys_and_declares_zero_launch_state() -> None:
    card = _card()
    parsed = json.loads(CARD_PATH.read_text(encoding="utf-8"))
    assert parsed == card
    state = card["source_and_outcome_guard"]["launch_state"]
    assert state == {
        "scientific_jobs_submitted": 0,
        "datasets_generated": 0,
        "checkpoints_evaluated": 0,
        "scientific_outcomes_accessed": False,
    }
