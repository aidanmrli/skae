"""Hash locks and provenance checks for the April local-law audit."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


CENTERED_DIR = Path(
    "/network/scratch/l/lia/skae/results/"
    "transition_rich_centered_chart_mechanism_20260420"
)
SELF_ROUTED_DIR = Path(
    "/network/scratch/l/lia/skae/results/"
    "transition_rich_self_routed_forecasting_20260420"
)
GEOMETRY_DIR = Path(
    "/network/scratch/l/lia/skae/results/"
    "true_jacobian_geometry_fixed17_seed0_20260424_reaudit"
)
RUNTIME_COMMIT = "207e6a5c74d4ef8ba06b97a3a76205da46a09f7b"
ARCHIVE_COMMIT = "7e93a7239991a17eb24d5968cc09f012a88a4435"
CLEANUP_COMMIT = "6a05022ec75eb52ba374e09626d967ee1ab4a15c"

FILE_LOCKS = {
    CENTERED_DIR / "centered_chart_mechanism_rows.csv":
        "3bceca30f383a2aee68296b47bef30ee901dd614a8aea527d44c3a07947d9ebe",
    CENTERED_DIR / "centered_chart_mechanism_summary.md":
        "fde117f74c9873339805bf3bb7ea0d57f935b9e95723969991837dbd6c5e058d",
    CENTERED_DIR / "manifest.json":
        "b25e4121516f05a7543375de279613f5f07623d8b80eda5376c2d6d6e9659202",
    CENTERED_DIR / "failures.json":
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    CENTERED_DIR / "automation/centered_chart_mechanism_queue.json":
        "6b75a2d91094c006a0d1fcb51c679cddb8d5c3a9e902f6d6f962584543db1645",
    SELF_ROUTED_DIR / "self_routed_forecasting_rows.csv":
        "e96ac4417e9a3d3aad068907ddb156feffa32507eace5f76c25559c5e8b2cb7f",
    SELF_ROUTED_DIR / "self_routed_forecasting_summary.md":
        "446be82bc6efc2b1d346be9735ce5e923847f6cf42b9d009d1625f959e2299a7",
    SELF_ROUTED_DIR / "manifest.json":
        "03215759100774a18929c495e9b40be5cc545a164f65afa5d486bca3e540e2f6",
    SELF_ROUTED_DIR / "failures.json":
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    SELF_ROUTED_DIR / "automation/self_routed_forecasting_queue.json":
        "fa724ed563946ad1b71378f40a173629d1db20e3a3d4332fa1510511431cc509",
    GEOMETRY_DIR / "true_jacobian_geometry_rows.csv":
        "8907582b5ea48e269fc08277c2bd8f9acd2d4f19cef8ff293b0b02305b420f08",
    GEOMETRY_DIR / "manifest.json":
        "882789bcc6d5ad39f2c0bc7aef1deaa336ca51d46e317fc9753087d152dcb55b",
    GEOMETRY_DIR / "failures.json":
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
}

ARCHIVE_FILE_LOCKS = {
    "tools/evaluate_transition_rich_centered_chart_mechanism.py":
        "3ed8dada6924115f269bc5ce659c3297dd05fd3193f0b9a7b0190858e0a50886",
    "tools/merge_transition_rich_centered_chart_mechanism_shards.py":
        "cb20c94d42ba2d306e1155c94432aa1876d56fe23f965f7c28304d9454b1bad3",
    "tools/evaluate_transition_rich_self_routed_forecasting.py":
        "fd10d29410d6cf84e4dfc44526efea0ea73abda4e38df6d905908b9d7942ef66",
    "tools/merge_transition_rich_self_routed_forecasting_shards.py":
        "e2b0a64716c74f015751dbd1927ed630728ac0fdbe20d6e8849792829e574613",
    "scripts/run_transition_rich_centered_chart_mechanism.sh":
        "5306e0157c15b01454162fe70b288d7994583c6c630052cec5878077f5442c33",
    "scripts/run_transition_rich_self_routed_forecasting.sh":
        "77b2b9cc17e2d6f8990a907fa07f24b5416739d67dda3ae6b43f009c96727721",
    "tools/evaluate_transition_rich_true_jacobian_geometry.py":
        "eea8575b4c4afb957138d3288914e99c16cf3f0edb84f935d6a663437a7fa91e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_bytes(path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{ARCHIVE_COMMIT}:{path}"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def verify_sources() -> dict[str, Any]:
    """Verify locked artifacts and disclose the dirty-runtime source gap."""
    for path, expected in FILE_LOCKS.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")
    archive_hashes = {}
    for path, expected in ARCHIVE_FILE_LOCKS.items():
        actual = hashlib.sha256(_archive_bytes(path)).hexdigest()
        if actual != expected:
            raise ValueError(f"Archived source mismatch for {path}: {actual}")
        archive_hashes[path] = actual

    centered_manifest = json.loads((CENTERED_DIR / "manifest.json").read_text())
    self_manifest = json.loads((SELF_ROUTED_DIR / "manifest.json").read_text())
    geometry_manifest = json.loads((GEOMETRY_DIR / "manifest.json").read_text())
    assert centered_manifest["num_rows"] == 74369
    assert centered_manifest["num_failures"] == 0
    assert centered_manifest["shard_count"] == 3
    assert self_manifest["num_rows"] == 24600
    assert self_manifest["completed_runs"] == self_manifest["num_runs"] == 510
    assert self_manifest["num_failures"] == 0
    assert geometry_manifest["num_rows"] == 198302
    assert geometry_manifest["completed_runs"] == geometry_manifest["num_runs"] == 49
    assert geometry_manifest["num_failures"] == 0
    for directory in (CENTERED_DIR, SELF_ROUTED_DIR, GEOMETRY_DIR):
        assert json.loads((directory / "failures.json").read_text()) == []

    log_commits: dict[str, list[str]] = {}
    for name, directory in (
        ("centered", CENTERED_DIR),
        ("self_routed", SELF_ROUTED_DIR),
        ("geometry", GEOMETRY_DIR),
    ):
        commits = []
        for path in sorted((directory / "logs").glob("*.out")):
            match = re.search(r"^Git commit: ([0-9a-f]{40})$", path.read_text(), re.M)
            if match:
                commits.append(match.group(1))
        log_commits[name] = sorted(set(commits))
        assert log_commits[name] == [RUNTIME_COMMIT]

    missing_at_runtime = []
    for path in ARCHIVE_FILE_LOCKS:
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{RUNTIME_COMMIT}:{path}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode:
            missing_at_runtime.append(path)
    return {
        "file_sha256": {str(path): digest for path, digest in FILE_LOCKS.items()},
        "worker_log_commits": log_commits,
        "logged_runtime_commit": RUNTIME_COMMIT,
        "archived_implementation_commit": ARCHIVE_COMMIT,
        "cleanup_commit": CLEANUP_COMMIT,
        "archive_file_sha256": archive_hashes,
        "archive_files_missing_at_logged_runtime_commit": missing_at_runtime,
        "attestation_level": "data_exact_source_compatible_not_runtime_exact",
        "attestation_limit": (
            "Rows and manifests are hash-locked, and compatible evaluators are preserved "
            "at the parent of the cleanup commit. Worker logs name a commit that lacks "
            "those evaluator paths, so the exact dirty runtime source is not recoverable."
        ),
    }
