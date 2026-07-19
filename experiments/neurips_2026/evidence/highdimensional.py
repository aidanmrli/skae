"""Build and verify the paper's high-dimensional confirmation displays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from experiments.neurips_2026.evidence.allen_cahn_contingency import (
    render_contingency,
)
from experiments.neurips_2026.evidence.highdimensional_rendering import (
    render_highdimensional,
)
from experiments.neurips_2026.paths import PAPER_DATA_DIR, PAPER_EVIDENCE_DIR


LORENZ_SUMMARY = PAPER_DATA_DIR / "lorenz96_highdim_confirmation_summary.json"
LORENZ_ROWS = PAPER_DATA_DIR / "lorenz96_highdim_confirmation_seed_rows.csv"
ALLEN_FORECAST = (
    PAPER_DATA_DIR / "allen_cahn_temporal_group_holdout_forecast_seed_rows.csv"
)
ALLEN_SUPPORT = (
    PAPER_DATA_DIR / "allen_cahn_temporal_group_holdout_support_seed_rows.csv"
)
ALLEN_FIXED_RECORDS = (
    PAPER_DATA_DIR / "allen_cahn_temporal_group_holdout_fixed_seed_records.csv"
)
TEMPORAL_PROVENANCE = (
    PAPER_DATA_DIR / "allen_cahn_temporal_group_holdout_provenance.json"
)
COMBINED_PROVENANCE = PAPER_DATA_DIR / "highdimensional_confirmation_provenance.json"
MAIN_PDF = "fig_highdimensional_confirmation.pdf"
MAIN_PNG = "fig_highdimensional_confirmation.png"
CONTINGENCY_PDF = "fig_allen_cahn_support_contingency.pdf"
CONTINGENCY_PNG = "fig_allen_cahn_support_contingency.png"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Hash mismatch for {path}: expected {expected}, got {actual}")


def verify_compact_inputs() -> None:
    provenance = json.loads(TEMPORAL_PROVENANCE.read_text())
    for record in provenance["evidence"].values():
        relative = str(record["path"])
        if relative.startswith("../"):
            continue
        require_hash(PAPER_DATA_DIR / relative, str(record["sha256"]))


def verify_declared_provenance() -> None:
    combined = json.loads(COMBINED_PROVENANCE.read_text())
    for component in combined["components"]:
        require_hash(
            PAPER_DATA_DIR / str(component["provenance"]),
            str(component["provenance_sha256"]),
        )
    display = combined["display"]
    require_hash(
        PAPER_DATA_DIR / str(display["pdf"]), str(display["pdf_sha256"])
    )
    require_hash(
        PAPER_DATA_DIR / str(display["png"]), str(display["png_sha256"])
    )
    supplementary = combined["supplementary_displays"]
    for prefix in ("support_contingency", "time20_snapshots"):
        require_hash(
            PAPER_DATA_DIR / str(supplementary[f"{prefix}_pdf"]),
            str(supplementary[f"{prefix}_pdf_sha256"]),
        )


def build(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    render_highdimensional(
        lorenz_summary=json.loads(LORENZ_SUMMARY.read_text()),
        lorenz_rows=pd.read_csv(LORENZ_ROWS),
        allen_forecast=pd.read_csv(ALLEN_FORECAST),
        allen_support=pd.read_csv(ALLEN_SUPPORT),
        output_pdf=output_dir / MAIN_PDF,
        output_png=output_dir / MAIN_PNG,
    )
    render_contingency(
        pd.read_csv(ALLEN_FIXED_RECORDS),
        output_pdf=output_dir / CONTINGENCY_PDF,
        output_png=output_dir / CONTINGENCY_PNG,
    )


def check() -> None:
    verify_compact_inputs()
    verify_declared_provenance()
    with TemporaryDirectory() as temporary:
        rebuilt = Path(temporary)
        build(rebuilt)
        for name in (MAIN_PDF, MAIN_PNG, CONTINGENCY_PDF, CONTINGENCY_PNG):
            active = PAPER_EVIDENCE_DIR / name
            candidate = rebuilt / name
            if active.read_bytes() != candidate.read_bytes():
                raise ValueError(f"High-dimensional artifact is not reproducible: {name}")
    print("High-dimensional displays are byte-identical to a clean rebuild.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Rebuild in a temporary directory and compare with tracked outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        check()
        return
    verify_compact_inputs()
    build(PAPER_EVIDENCE_DIR)
    print("Built high-dimensional confirmation displays.")


if __name__ == "__main__":
    main()
