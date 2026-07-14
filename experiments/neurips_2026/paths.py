"""Repository and cluster-path resolution for the versioned paper workflow."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
PAPER_SOURCE = DOCS_DIR / "neurips_sparse_koopman_multibasin.tex"
PAPER_EVIDENCE_DIR = DOCS_DIR / "figures" / "neurips_paper_2026"
PAPER_DATA_DIR = PAPER_EVIDENCE_DIR / "_data"
PAPER_TABLE_DIR = PAPER_EVIDENCE_DIR / "_tables"


def results_root() -> Path:
    """Return the explicit or repository-local collected-results root."""

    return Path(os.environ.get("SKAE_RESULTS_ROOT", REPO_ROOT / "results")).expanduser()


def scratch_root() -> Path:
    """Return a portable scratch root without embedding a contributor name."""

    configured = os.environ.get("SKAE_SCRATCH_ROOT")
    if configured:
        return Path(configured).expanduser()
    user = os.environ.get("USER", "user")
    mila_root = Path("/network/scratch") / user[:1] / user / "skae"
    if mila_root.parent.exists():
        return mila_root
    return REPO_ROOT / "runs"


__all__ = [
    "REPO_ROOT",
    "DOCS_DIR",
    "PAPER_SOURCE",
    "PAPER_EVIDENCE_DIR",
    "PAPER_DATA_DIR",
    "PAPER_TABLE_DIR",
    "results_root",
    "scratch_root",
]
