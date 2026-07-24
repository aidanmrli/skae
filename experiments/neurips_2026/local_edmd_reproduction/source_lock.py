"""Fail-closed verification for reproduction sources and historical inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.neurips_2026.local_edmd_reproduction.contract import LOCK_PATH


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_tree_digest(root: Path, pattern: str) -> tuple[str, int]:
    """Hash sorted ``sha256sum``-style relative-path records."""

    paths = sorted(path for path in root.glob(pattern) if path.is_file())
    records = "".join(
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
        for path in paths
    ).encode()
    return hashlib.sha256(records).hexdigest(), len(paths)


def load_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid reproduction source lock: {path}") from error
    if payload.get("protocol_id") != "local_edmd_poly_historical_reproduction_v1":
        raise ValueError("Unexpected reproduction source-lock protocol")
    return payload


def verify_source_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    """Authenticate every mutable source and external input before use."""

    lock = load_lock(path)
    failures: list[str] = []
    for name, item in lock.get("sources", {}).items():
        source = REPOSITORY_ROOT / item["path"]
        actual = sha256_file(source) if source.is_file() else "missing"
        if actual != item["sha256"]:
            failures.append(
                f"source {name}: expected {item['sha256']}, got {actual}"
            )
    for name, item in lock.get("external_files", {}).items():
        source = Path(item["path"])
        actual = sha256_file(source) if source.is_file() else "missing"
        if actual != item["sha256"]:
            failures.append(
                f"external file {name}: expected {item['sha256']}, got {actual}"
            )
    for name, item in lock.get("external_trees", {}).items():
        root = Path(item["root"])
        actual, count = portable_tree_digest(root, item["pattern"])
        if actual != item["sha256"] or count != int(item["file_count"]):
            failures.append(
                f"external tree {name}: expected {item['sha256']}/{item['file_count']}, "
                f"got {actual}/{count}"
            )
    if failures:
        raise RuntimeError("Source-lock verification failed:\n" + "\n".join(failures))
    return lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock = verify_source_lock(args.lock)
    print(
        f"Verified reproduction source lock with {len(lock['sources'])} sources, "
        f"{len(lock['external_files'])} files, and "
        f"{len(lock['external_trees'])} trees"
    )


if __name__ == "__main__":
    main()
