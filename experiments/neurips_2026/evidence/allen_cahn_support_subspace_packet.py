"""Freeze and verify the released Allen--Cahn support-subspace audit packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.neurips_2026.paths import PAPER_DATA_DIR


PACKET_ID = "allen_cahn_support_subspaces_v4"
DEFAULT_OUTPUT_DIR = PAPER_DATA_DIR / PACKET_ID
FILES = ("decision.json", "seed_rows.csv", "provenance.json", "evidence_manifest.json")
EXPECTED_HASHES = {
    "decision.json": "fe64ea1eaca12bfb4c20a583b3da94c8f9f2a3bc735c3f4e2dca3ab319c87b1a",
    "seed_rows.csv": "12dc1860d4f1e74b7fc9847a0a086525f8a6a8fc95e0d705a2fa1090e1d7ef20",
    "provenance.json": "fbc4df60c0f2840a6abe5793d27a625ae2f6aea72c97d3e882f060a8d2979ec8",
    "evidence_manifest.json": "b7287a588b571670b645fc81edbfc5ef3531cf5b73270ed7e16aee88d14384ea",
}
EXPECTED_ROOTS = {
    "card_sha256": "fafa3b1a0e8f63095c3926171673fa62f2baec6e2af36a954cbca83d35f35743",
    "source_manifest_sha256": "e4219ecb3b2e25d08f9f1e5afc51a16f84d94409baf62280651cc101fc3f7024",
    "profile_decision_sha256": "043ee246bdfcc8d4ef50431d234274404bfd2438114c8755d513a62f5a04b993",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_packet(directory: Path) -> dict[str, Any]:
    for name in FILES:
        path = directory / name
        _require(path.is_file(), f"Missing support-subspace evidence: {path}")
        _require(
            sha256_path(path) == EXPECTED_HASHES[name],
            f"Released support-subspace evidence hash drifted: {name}",
        )

    manifest = json.loads((directory / "evidence_manifest.json").read_text())
    _require(manifest.get("schema_version") == 1, "Unexpected evidence schema")
    expected_members = {
        name: EXPECTED_HASHES[name]
        for name in ("decision.json", "seed_rows.csv", "provenance.json")
    }
    _require(manifest.get("files") == expected_members, "Evidence member roster drifted")

    decision = json.loads((directory / "decision.json").read_text())
    _require(decision.get("decision") == "failed", "Released decision must remain failed")
    _require(
        all(decision.get(key) == value for key, value in EXPECTED_ROOTS.items()),
        "Released decision root of trust drifted",
    )
    validity = decision["validity"]
    _require(
        validity["passed"] is True
        and validity["provenance_and_firewall"] is True
        and validity["outcome_blind_canary_release"] is True,
        "Released validity/firewall checks no longer pass",
    )
    _require(
        decision["exact_fixed_P0_closure"]["passed"] is False
        and decision["decoded_forecast"]["passed"] is False,
        "Negative mechanism decision boundary drifted",
    )
    _require(
        decision["decoded_forecast"]["projected_vs_dense_full_passed"] is True,
        "Projected sparse-versus-dense forecast result drifted",
    )
    family = decision["family"]
    _require(
        family["family_checks"]["eligible_seeds"] is False
        and family["qualification"]["eligible_seed_count"] == 0
        and family["qualification"]["eligible_seeds"] == [],
        "Conditional bridge eligibility drifted",
    )

    with (directory / "seed_rows.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require([int(row["seed"]) for row in rows] == list(range(64, 74)), "Seed roster drifted")
    _require(
        all(
            row["family_eligible"] == "False"
            and int(row["family_train_fit_count"]) == 1
            and int(row["family_qualified_count"]) == 1
            for row in rows
        ),
        "Released one-family result drifted",
    )
    _require(
        all(0.0 < float(row["sparse_active_density"]) < 1.0 for row in rows),
        "Sparse active-density evidence is malformed",
    )

    provenance = json.loads((directory / "provenance.json").read_text())
    _require(
        all(provenance.get(key) == value for key, value in EXPECTED_ROOTS.items()),
        "Released provenance root of trust drifted",
    )
    _require(
        len(provenance.get("shards", {})) == 10
        and len(provenance.get("telemetry", {})) == 10
        and len(provenance.get("raw_telemetry", {})) == 10,
        "Released provenance roster is incomplete",
    )
    return {
        "decision": decision["decision"],
        "seed_count": len(rows),
        "eligible_seed_count": family["qualification"]["eligible_seed_count"],
        "projected_vs_dense_full_passed": decision["decoded_forecast"]
        ["projected_vs_dense_full_passed"],
    }


def build(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"Refusing to overwrite evidence: {output_dir}")
    result = validate_packet(source_dir)
    output_dir.mkdir(parents=True)
    for name in FILES:
        (output_dir / name).write_bytes((source_dir / name).read_bytes())
    return validate_packet(output_dir) | result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-evidence-dir", type=Path)
    parser.add_argument("--output-data-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        if args.source_evidence_dir is not None:
            raise ValueError("--source-evidence-dir is not used with --check")
        result = validate_packet(args.output_data_dir)
    else:
        if args.source_evidence_dir is None:
            raise ValueError("--source-evidence-dir is required when building")
        result = build(args.source_evidence_dir, args.output_data_dir)
    print(json.dumps({"status": "passed", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
