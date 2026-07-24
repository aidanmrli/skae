"""Write or fail-closed check the compact negative distinct-laws V2 packet."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from experiments.neurips_2026.evidence.global_k_distinct_laws_v2_contract import (
    DATA_FILES,
    DEFAULT_PACKET_DIR,
    FILES,
    PACKET_ID,
    SOURCE_FILES,
    json_bytes,
    load_json,
    require,
    sha256_bytes,
    sha256_path,
)
from experiments.neurips_2026.evidence.global_k_distinct_laws_v2_extract import (
    build_payloads,
)


# Reviewed roots of the deterministic compact extraction. These make portable
# checks fail closed even when the external scratch archive is unavailable.
EXPECTED_HASHES = {
    "decision.json": "3b963c802f9bb38c856d451d4541e3ceac4660c54e6eea0d97fa58b8b7a3adc7",
    "seed_rows.csv": "52bbb46a1fc14accd6e58ee858f2478da83eb36fa048a55f7d1bd01b593ad261",
    "basin_rows.csv": "1a0df8314079eab8dce7eac5190c72ddb618ff730359a5bca5a5d6538e490878",
    "provenance.json": "1d16b3822fd85944115885dcdd1df6b96c4641837a01e4a6c7fdbc95ddb20162",
    "evidence_manifest.json": "08c55c2d299b31f8ba0e9d239104ad0a61f13812d7cd81158b3cb088a8419359",
}


def write_packet(directory: Path = DEFAULT_PACKET_DIR) -> dict[str, str]:
    payloads = build_payloads()
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (directory / name).write_bytes(payload)
    hashes = {name: sha256_bytes(payload) for name, payload in payloads.items()}
    manifest = {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "status": "invalid_negative",
        "positive_claim_permitted": False,
        "files": hashes,
        "external_source_roots": {
            name: item["sha256"] for name, item in SOURCE_FILES.items()
        },
        "check_command_inside_allocation": (
            "uv run skae-paper build global-k-distinct-laws-v2-negative --check"
        ),
        "source_check_command_inside_allocation": (
            "uv run skae-paper build global-k-distinct-laws-v2-negative "
            "--check-sources"
        ),
    }
    manifest_bytes = json_bytes(manifest)
    (directory / "evidence_manifest.json").write_bytes(manifest_bytes)
    return {**hashes, "evidence_manifest.json": sha256_bytes(manifest_bytes)}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_packet(directory: Path = DEFAULT_PACKET_DIR) -> dict[str, Any]:
    require(set(EXPECTED_HASHES) == set(FILES), "Expected packet roots are incomplete")
    for name in FILES:
        path = directory / name
        require(path.is_file(), f"Missing distinct-laws V2 evidence: {path}")
        require(
            sha256_path(path) == EXPECTED_HASHES[name],
            f"Released distinct-laws V2 evidence hash drifted: {name}",
        )
    manifest = load_json(directory / "evidence_manifest.json")
    require(
        manifest.get("packet_id") == PACKET_ID
        and manifest.get("status") == "invalid_negative"
        and manifest.get("positive_claim_permitted") is False
        and manifest.get("files")
        == {name: EXPECTED_HASHES[name] for name in DATA_FILES},
        "Distinct-laws V2 evidence manifest drifted",
    )
    decision = load_json(directory / "decision.json")
    require(
        decision.get("status") == "invalid_negative"
        and decision.get("positive_claim_permitted") is False
        and decision.get("mechanism_tier") == "invalid"
        and decision.get("validity", {}).get("passed") is False
        and decision.get("sparse_gates", {}).get("joint_H_G") is False
        and decision.get("dense_specificity", {}).get("passed") is False,
        "Distinct-laws V2 negative claim boundary drifted",
    )
    require(
        decision["benchmark"]["scientific_model_seeds"] == list(range(100, 110))
        and decision["benchmark"]["known_evaluation_basin_count"] == 3
        and decision["validity"]["joint_H_G_kink_pairs"] == 24
        and decision["sparse_gates"]["H_seed_passes"] == 0
        and decision["sparse_gates"]["G_seed_passes"] == 0
        and decision["sparse_gates"]["joint_seed_passes"] == 0,
        "Distinct-laws V2 core counts drifted",
    )
    seed_rows = _read_csv(directory / "seed_rows.csv")
    require(
        len(seed_rows) == 20
        and [(row["arm"], int(row["seed"])) for row in seed_rows]
        == [("sparse", seed) for seed in range(100, 110)]
        + [("dense", seed) for seed in range(100, 110)]
        and len({row["checkpoint_sha256"] for row in seed_rows}) == 20
        and all(row["checkpoint_validation_passed"] == "True" for row in seed_rows)
        and all(row["selector_joint_finite_count"] == "16" for row in seed_rows),
        "Distinct-laws V2 seed/checkpoint roster drifted",
    )
    basin_rows = _read_csv(directory / "basin_rows.csv")
    require(
        len(basin_rows) == 3
        and [int(row["basin_index"]) for row in basin_rows] == [0, 1, 2]
        and [int(row["g_own_nearest"]) for row in basin_rows] == [8, 8, 8]
        and [int(row["h_own_nearest"]) for row in basin_rows] == [4, 3, 2]
        and all(int(row["finite_every_gate"]) == 0 for row in basin_rows),
        "Distinct-laws V2 per-basin negative counts drifted",
    )
    provenance = load_json(directory / "provenance.json")
    require(
        provenance.get("external_source_roots")
        == {
            name: {"path": str(item["path"]), "sha256": item["sha256"]}
            for name, item in SOURCE_FILES.items()
        }
        and provenance.get("authentication", {}).get(
            "raw_external_artifacts_modified"
        )
        is False,
        "Distinct-laws V2 provenance drifted",
    )
    return {
        "decision": "invalid_negative",
        "positive_claim_permitted": False,
        "checkpoint_count": len(seed_rows),
        "basin_count": len(basin_rows),
        "joint_h_g_seed_passes": decision["sparse_gates"]["joint_seed_passes"],
    }


def validate_sources(directory: Path = DEFAULT_PACKET_DIR) -> dict[str, int]:
    validate_packet(directory)
    rebuilt = build_payloads()
    for name, payload in rebuilt.items():
        require(
            (directory / name).read_bytes() == payload,
            f"Compact distinct-laws V2 extraction drifted from raw source: {name}",
        )
    return {"external_source_files": len(SOURCE_FILES), "compact_files": len(rebuilt)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET_DIR)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-sources", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(
        sum((args.write, args.check, args.check_sources)) == 1,
        "Choose exactly one of --write, --check, or --check-sources",
    )
    if args.write:
        result: dict[str, Any] = {"written_hashes": write_packet(args.packet_dir)}
    elif args.check_sources:
        result = {**validate_packet(args.packet_dir), **validate_sources(args.packet_dir)}
    else:
        result = validate_packet(args.packet_dir)
    print(json.dumps({"status": "passed", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
