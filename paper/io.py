"""Small file helpers shared by the reproduction code."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Yield a temporary path next to ``path``; it replaces ``path`` when the block succeeds.

    Readers never see a partly written file, and an interrupted write leaves the old file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_csv(path: Path) -> List[Dict[str, str]]:
    """Rows of a CSV file (``.csv`` or gzip-compressed ``.csv.gz``) as dictionaries of strings."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write dictionaries as CSV, atomically and with ``\\n`` line ends; columns in first-seen key order."""
    fields: List[str] = []
    for row in rows:
        fields += [key for key in row if key not in fields]
    with atomic_path(path) as temporary, open(temporary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any, indent: int = 2, **options) -> None:
    """Write JSON atomically, with a trailing newline; ``options`` go to :func:`json.dumps`."""
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, indent=indent, **options) + "\n", encoding="utf-8")


def write_gzip(path: Path, text: str) -> None:
    """Write ``text`` gzip-compressed and atomically, with no file name or timestamp in the header."""
    with atomic_path(path) as temporary, open(temporary, "wb") as raw, \
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
        handle.write(text.encode())


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
