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
LORENZ_PROVENANCE = (
    PAPER_DATA_DIR / "lorenz96_highdim_confirmation_provenance.json"
)
ALLEN_FORECAST = (
    PAPER_DATA_DIR / "allen_cahn_temporal_group_holdout_forecast_seed_rows.csv"
)
ALLEN_SUPPORT = (
    PAPER_DATA_DIR / "allen_cahn_temporal_group_holdout_support_seed_rows.csv"
)
ALLEN_FIXED_RECORDS = (
    PAPER_DATA_DIR / "allen_cahn_temporal_group_holdout_fixed_seed_records.csv"
)
ALLEN_TIME20_PROVENANCE = (
    PAPER_DATA_DIR / "allen_cahn_time20_confirmation_provenance.json"
)
TEMPORAL_PROVENANCE = (
    PAPER_DATA_DIR / "allen_cahn_temporal_group_holdout_provenance.json"
)
ALLEN_FORECAST_PROVENANCE = (
    PAPER_DATA_DIR / "allen_cahn_global_k_forecast_optimized_provenance.json"
)
COMBINED_PROVENANCE = PAPER_DATA_DIR / "highdimensional_confirmation_provenance.json"
MAIN_PDF = "fig_highdimensional_confirmation.pdf"
MAIN_PNG = "fig_highdimensional_confirmation.png"
CONTINGENCY_PDF = "fig_allen_cahn_support_contingency.pdf"
CONTINGENCY_PNG = "fig_allen_cahn_support_contingency.png"
TIME20_PDF = "fig_allen_cahn_time20_snapshots.pdf"

COMPONENTS = (
    ("lorenz96_D128", LORENZ_PROVENANCE),
    ("allen_cahn_D512_time20", ALLEN_TIME20_PROVENANCE),
    (
        "allen_cahn_D512_temporal_group_independent_holdout",
        TEMPORAL_PROVENANCE,
    ),
    (
        "allen_cahn_D512_global_k_forecast_optimized",
        ALLEN_FORECAST_PROVENANCE,
    ),
)

DESCRIPTION = (
    "Component-level provenance for fixed-data Lorenz-96, "
    "representation-optimized Allen-Cahn, and the separate "
    "forecast-optimized Allen-Cahn packet."
)
CLAIM_SPLIT = (
    "Lorenz-96 supplies a complete-recipe direct-rollout sparse forecasting "
    "advantage. The representation-optimized Allen-Cahn packet supplies strong "
    "alignment between T=20 final-state support families and T=20 modal-well "
    "fate labels across all trajectories, and exactly one transferred family per "
    "fate on a single-well-dominated final-field slice prospectively frozen for "
    "the new holdout after earlier mixed-domain fragmentation. Validation fate "
    "metrics selected T=20 scoring, Jaccard 0.40, the temporal weight, and whole-"
    "packet advancement; conditional on that label-assisted recipe, training, "
    "representative fitting, and independent-holdout assignment remain label-"
    "free. The packet retains the 0.009 primary uniqueness-gate miss and negative "
    "long-horizon sparse forecasting. With matched trainable tensor shapes and "
    "effective forward-path parameter count, the separate forecast-optimized, no-reencoding "
    "Allen-Cahn packet supplies secondary through-horizon mean-error improvements "
    "for the joint soft-thresholding-plus-L1 treatment at physical times 16 and "
    "20, while preserving its failed terminal gate and unopened holdout."
)
CONTINGENCY_DISPLAY_CONTRACT = {
    "display_model_seed": 21,
    "family_identity": "raw transferred training-codebook index",
    "family_order": "ascending nonnegative raw codebook index, then unknown (-1)",
    "ordering_uses_evaluation_fate": False,
    "sparse_all_and_slice_share_complete_axis": True,
    "panel_specific_fate_row_counts_annotated": True,
    "slice": (
        "final modal-well occupancy >= 0.90, prospectively frozen for the new "
        "holdout after prior mixed-domain fragmentation"
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Hash mismatch for {path}: expected {expected}, got {actual}")


def resolve_child_evidence_path(provenance_path: Path, relative: str) -> Path:
    """Resolve a child record without allowing it to escape the paper packet."""

    candidate = (provenance_path.parent / relative).resolve()
    evidence_root = PAPER_EVIDENCE_DIR.resolve()
    if not candidate.is_relative_to(evidence_root):
        raise ValueError(
            f"Evidence path escapes the paper packet: {provenance_path}: {relative}"
        )
    return candidate


def verify_compact_inputs() -> None:
    lorenz = json.loads(LORENZ_PROVENANCE.read_text())
    protocol = lorenz["protocol"]
    if protocol.get("rollout_mode") != "no_reencode_repeated_K":
        raise ValueError("Lorenz--96 rollout mode is not fail-closed as repeated K")
    if protocol.get("rollout_implementation") != (
        "skae/model.py::GenericKoopmanModel.rollout_observation_discrete"
    ):
        raise ValueError("Lorenz--96 rollout implementation drifted")
    for field in (
        "model_source_sha256_at_training_commit",
        "evaluator_source_sha256_at_training_commit",
    ):
        digest = str(protocol.get(field, ""))
        if len(digest) != 64:
            raise ValueError(f"Lorenz--96 provenance lacks a full {field}")
        int(digest, 16)
    provenance = json.loads(TEMPORAL_PROVENANCE.read_text())
    for record in provenance["evidence"].values():
        relative = str(record["path"])
        require_hash(
            resolve_child_evidence_path(TEMPORAL_PROVENANCE, relative),
            str(record["sha256"]),
        )


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
    require_hash(
        PAPER_DATA_DIR / str(supplementary["support_contingency_png"]),
        str(supplementary["support_contingency_png_sha256"]),
    )


def combined_provenance(output_dir: Path) -> dict[str, object]:
    """Derive the parent packet's provenance from its current child artifacts."""

    time20_path = output_dir / TIME20_PDF
    if not time20_path.is_file():
        # The time-20 snapshot has its own builder and is only referenced here.
        # During a temporary parent rebuild, authenticate the active child output.
        time20_path = PAPER_EVIDENCE_DIR / TIME20_PDF
    return {
        "schema_version": 1,
        "status": "complete",
        "description": DESCRIPTION,
        "components": [
            {
                "id": component_id,
                "provenance": provenance_path.name,
                "provenance_sha256": sha256(provenance_path),
            }
            for component_id, provenance_path in COMPONENTS
        ],
        "display": {
            "pdf": f"../{MAIN_PDF}",
            "pdf_sha256": sha256(output_dir / MAIN_PDF),
            "png": f"../{MAIN_PNG}",
            "png_sha256": sha256(output_dir / MAIN_PNG),
        },
        "supplementary_displays": {
            "support_contingency_pdf": f"../{CONTINGENCY_PDF}",
            "support_contingency_pdf_sha256": sha256(
                output_dir / CONTINGENCY_PDF
            ),
            "support_contingency_png": f"../{CONTINGENCY_PNG}",
            "support_contingency_png_sha256": sha256(
                output_dir / CONTINGENCY_PNG
            ),
            "time20_snapshots_pdf": f"../{TIME20_PDF}",
            "time20_snapshots_pdf_sha256": sha256(time20_path),
        },
        "support_contingency_display_contract": CONTINGENCY_DISPLAY_CONTRACT,
        "claim_split": CLAIM_SPLIT,
    }


def build(output_dir: Path) -> tuple[Path, ...]:
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
    provenance_path = output_dir / "_data" / COMBINED_PROVENANCE.name
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(combined_provenance(output_dir), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return (
        output_dir / MAIN_PDF,
        output_dir / MAIN_PNG,
        output_dir / CONTINGENCY_PDF,
        output_dir / CONTINGENCY_PNG,
        provenance_path,
    )


def check() -> None:
    verify_compact_inputs()
    verify_declared_provenance()
    with TemporaryDirectory() as temporary:
        rebuilt = Path(temporary)
        build(rebuilt)
        for relative in (
            Path(MAIN_PDF),
            Path(MAIN_PNG),
            Path(CONTINGENCY_PDF),
            Path(CONTINGENCY_PNG),
            Path("_data") / COMBINED_PROVENANCE.name,
        ):
            active = PAPER_EVIDENCE_DIR / relative
            candidate = rebuilt / relative
            if active.read_bytes() != candidate.read_bytes():
                raise ValueError(
                    f"High-dimensional artifact is not reproducible: {relative}"
                )
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
