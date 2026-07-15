"""Repository and cluster-path resolution for the versioned paper workflow."""

from __future__ import annotations

import os
from pathlib import Path

from skae.runtime_paths import resolve_scratch_root


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

    return resolve_scratch_root(fallback=REPO_ROOT / "runs")


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
