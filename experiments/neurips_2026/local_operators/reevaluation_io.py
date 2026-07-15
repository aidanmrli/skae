"""Completed-run discovery and cache identity for local/global reevaluation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


RunKey = Tuple[str, int]
CACHE_SCHEMA_VERSION = 2
COMPLETION_MARKER = "evaluation_results_best.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_from_path(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("seed_") and part[5:].isdigit():
            return int(part[5:])
    return None


def _system_from_run(root: Path, run_dir: Path) -> str | None:
    try:
        relative = run_dir.relative_to(root).parts
    except ValueError:
        relative = run_dir.parts
    seed_index = next(
        (index for index, part in enumerate(relative) if part.startswith("seed_")),
        None,
    )
    if seed_index is None:
        return None
    system_index = seed_index - 1
    if system_index >= 0 and relative[system_index].startswith("dt_"):
        system_index -= 1
    if system_index < 0:
        return None
    slug = relative[system_index]
    return "claude:" + slug[7:] if slug.startswith("claude_") else slug


def _config_identity(run_dir: Path) -> RunKey:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Completed run lacks config.json: {run_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        return str(config["ENV"]["ENV_NAME"]), int(config["SEED"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Malformed run identity in {config_path}") from error


def _discover_runs(root: Path) -> Dict[RunKey, Path]:
    """Discover only completed runs and fail on ambiguous system/seed keys."""
    candidates: Dict[RunKey, List[Path]] = defaultdict(list)
    for checkpoint in root.rglob("checkpoint.pt"):
        run_dir = checkpoint.parent
        if not (run_dir / COMPLETION_MARKER).is_file():
            continue
        key = (_system_from_run(root, run_dir), _seed_from_path(run_dir))
        if key[0] is None or key[1] is None:
            continue
        config_key = _config_identity(run_dir)
        if config_key != key:
            raise ValueError(
                "Run path/config identity mismatch: "
                f"path={key!r}, config={config_key!r}, run={run_dir}"
            )
        candidates[(str(key[0]), int(key[1]))].append(run_dir)
    ambiguous = {key: paths for key, paths in candidates.items() if len(paths) != 1}
    if ambiguous:
        details = "; ".join(
            f"{key}: {[str(path) for path in sorted(paths)]}"
            for key, paths in sorted(ambiguous.items())
        )
        raise RuntimeError(f"Ambiguous completed runs; resolve explicitly: {details}")
    return {key: paths[0] for key, paths in candidates.items()}


def _cache_fingerprint(
    *,
    kind: str,
    system: str,
    seed: int,
    run_dir: Path,
    horizons: Sequence[int],
    periods: Sequence[int],
    batch_size: int,
    support_definition: str,
    family_jaccard_threshold: float,
) -> Dict[str, object]:
    artifact_path = run_dir / "stage2_artifacts.pt"
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": kind,
        "system": system,
        "seed": int(seed),
        "run_dir": str(run_dir.resolve()),
        "checkpoint_sha256": _sha256(run_dir / "checkpoint.pt"),
        "config_sha256": _sha256(run_dir / "config.json"),
        "stage2_artifacts_sha256": (
            _sha256(artifact_path) if artifact_path.is_file() else None
        ),
        "horizons": [int(value) for value in horizons],
        "periods": [int(value) for value in periods],
        "batch_size": int(batch_size),
        "evaluation_seed_offset": 12_345,
        "support_definition": support_definition,
        "family_jaccard_threshold": float(family_jaccard_threshold),
    }


def _run_record(kind: str, key: RunKey, run_dir: Path) -> Dict[str, object]:
    artifact_path = run_dir / "stage2_artifacts.pt"
    return {
        "kind": kind,
        "system_key": key[0],
        "seed": key[1],
        "run_dir": str(run_dir.resolve()),
        "checkpoint_sha256": _sha256(run_dir / "checkpoint.pt"),
        "config_sha256": _sha256(run_dir / "config.json"),
        "stage2_artifacts_sha256": (
            _sha256(artifact_path) if artifact_path.is_file() else None
        ),
        "completion_marker": COMPLETION_MARKER,
    }
