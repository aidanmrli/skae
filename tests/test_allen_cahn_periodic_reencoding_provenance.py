from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.neurips_2026.allen_cahn_forecast_replication import io as parent_io
from experiments.neurips_2026.allen_cahn_periodic_reencoding import field_integrity
from experiments.neurips_2026.allen_cahn_periodic_reencoding.io import (
    ARCHITECTURE_AUDIT_RELATIVE,
    CARD_PATH,
    PACKAGE_DIR,
    REPO_ROOT,
    duplicate_safe_json,
    load_parent_card,
    sha256_path,
    validate_architecture_audit,
    verify_source_manifest,
)


def _card() -> dict:
    return duplicate_safe_json(CARD_PATH)


def _audit_inputs() -> tuple[dict, dict, dict]:
    card = _card()
    parent, _ = parent_io.load_card(parent_io.CARD_PATH)
    audit = duplicate_safe_json(REPO_ROOT / ARCHITECTURE_AUDIT_RELATIVE)
    artifact = duplicate_safe_json(REPO_ROOT / audit["artifact_manifest"]["path"])
    return audit, artifact, parent


def test_parent_binding_requires_exact_source_records_and_passed_audit() -> None:
    card = _card()
    parent = load_parent_card(card)
    assert card["frozen_parent"]["pinned_model_source"] == parent_io.pinned_source(
        parent, "checkpoint_model"
    )
    assert card["frozen_parent"]["pinned_generator_source"] == parent_io.pinned_source(
        parent, "physics_and_initial_conditions"
    )

    drifted = deepcopy(card)
    drifted["frozen_parent"]["pinned_model_source"].pop("historical_commit")
    with pytest.raises(RuntimeError, match="not the exact parent source record"):
        load_parent_card(drifted)

    audit_drifted = deepcopy(card)
    audit_drifted["frozen_parent"]["architecture_audit_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load_parent_card(audit_drifted)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda audit, _artifact: audit["configuration_audit"][
                "common_model_config"
            ].__setitem__("dense_activation", "relu"),
            "dense-control configuration audit failed",
        ),
        (
            lambda audit, _artifact: audit["configuration_audit"][
                "common_training_settings"
            ].__setitem__("weight_decay", 1e-4),
            "dense-control configuration audit failed",
        ),
        (
            lambda audit, _artifact: audit["runs"][0].__setitem__(
                "checkpoint_sha256", "0" * 64
            ),
            "run roster drifted",
        ),
        (
            lambda _audit, artifact: artifact["runs"][0]["checkpoint"].__setitem__(
                "sha256", "0" * 64
            ),
            "checkpoint roster drifted",
        ),
    ],
)
def test_architecture_audit_fails_closed_on_scientific_drift(
    mutation, message: str
) -> None:
    audit, artifact, parent = _audit_inputs()
    mutation(audit, artifact)
    with pytest.raises(RuntimeError, match=message):
        validate_architecture_audit(audit, artifact, parent)


def test_card_manifest_roster_covers_every_periodic_source_script_and_test() -> None:
    required = set(_card()["source_and_outcome_guard"]["required_manifest_paths"])
    package_sources = {
        path.relative_to(REPO_ROOT).as_posix() for path in PACKAGE_DIR.glob("*.py")
    }
    script_sources = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (
            REPO_ROOT / "scripts/neurips_2026/allen_cahn_periodic_reencoding"
        ).glob("*.sh")
    }
    test_sources = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "tests").glob(
            "test_allen_cahn_periodic_reencoding_*.py"
        )
    }
    assert package_sources | script_sources | test_sources <= required
    assert ARCHITECTURE_AUDIT_RELATIVE.as_posix() in required
    assert (
        "docs/figures/neurips_paper_2026/_data/"
        "allen_cahn_global_k_forecast_optimized_artifact_manifest.json"
    ) in required


def test_source_roster_rejects_duplicate_required_paths(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("frozen\n", encoding="utf-8")
    manifest = tmp_path / "source_manifest.sha256"
    manifest.write_text(f"{sha256_path(source)}  {source}\n", encoding="utf-8")
    card = {
        "source_and_outcome_guard": {
            "required_manifest_paths": [str(source), str(source)]
        }
    }
    with pytest.raises(RuntimeError, match="malformed or duplicated"):
        verify_source_manifest(card, path=manifest)


def _write_field_manifest(root: Path, card: dict) -> tuple[Path, list[Path]]:
    role = "validation"
    shape = [2, 3, 1, 1, 1]
    paths: list[Path] = []
    rows = []
    for index, record in enumerate(card["prospective_datasets"][role]):
        seed = int(record["seed"])
        path = root / "data" / f"{role}_seed{seed}_fields.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"opaque-field-{index}".encode())
        paths.append(path)
        rows.append(
            {
                "role": role,
                "dataset_index": index,
                "dataset_seed": seed,
                "path": str(path),
                "sha256": sha256_path(path),
                "shape": shape,
                "storage_bytes": 4 * 2 * 3,
            }
        )
    manifest = root / "validation_data_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": card["protocol_id"],
                "role": role,
                "datasets": rows,
            }
        ),
        encoding="utf-8",
    )
    return manifest, paths


def test_field_manifest_binds_hashes_and_invokes_payload_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = deepcopy(_card())
    card["protocol_id"] = "field-integrity-test"
    card["system"].update(
        trajectories_per_dataset=2,
        validation_horizon_steps=2,
        grid_size=1,
        channels=1,
    )
    card["prospective_datasets"]["validation"] = [
        {"index": index, "seed": seed} for index, seed in enumerate((101, 102, 103))
    ]
    root = (tmp_path / "packet").resolve()
    root.mkdir()
    manifest, paths = _write_field_manifest(root, card)
    calls: list[tuple[str, int, int]] = []
    monkeypatch.setattr(field_integrity, "torch_load", lambda path: {"loaded": str(path)})
    monkeypatch.setattr(
        field_integrity,
        "validate_field_payload",
        lambda _payload, _card, *, role, dataset_index, seed: calls.append(
            (role, dataset_index, seed)
        ),
    )
    result = field_integrity.verify_field_artifact_manifest(
        manifest,
        card,
        expected_sha256=sha256_path(manifest),
        role="validation",
        root=root,
    )
    assert calls == [("validation", 0, 101), ("validation", 1, 102), ("validation", 2, 103)]
    assert all(row["payload_validated"] for row in result["field_files"])

    paths[0].write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        field_integrity.verify_field_artifact_manifest(
            manifest,
            card,
            expected_sha256=sha256_path(manifest),
            role="validation",
            root=root,
        )
