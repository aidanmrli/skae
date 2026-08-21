"""Small launch, identity, and GPU-discovery helpers for pilot orchestration."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .pilot import atomic_write_json


def git_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def gpu_identity() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 6:
            continue
        rows.append(
            dict(
                zip(
                    ("index", "uuid", "name", "memory_total_mib", "driver", "compute_cap"),
                    values,
                )
            )
        )
    if len(rows) != 1:
        raise RuntimeError(f"pilot requires exactly one visible GPU, found {len(rows)}")
    if rows[0]["uuid"] == "" or not rows[0]["uuid"].startswith("GPU-"):
        raise RuntimeError("nvidia-smi did not report a GPU UUID")
    if "RTX 8000" not in rows[0]["name"] and "RTX8000" not in rows[0]["name"]:
        raise RuntimeError(f"pilot requires one RTX 8000, found {rows[0]['name']}")
    return rows


def identity_hash(identity: dict[str, Any]) -> str:
    return str(identity["task_identity_sha256"])
