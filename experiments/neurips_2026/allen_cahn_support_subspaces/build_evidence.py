"""Build or clean-clone-check compact support-subspace evidence artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.neurips_2026.allen_cahn_support_subspaces.io import sha256_path


FILES = ("decision.json", "seed_rows.csv", "provenance.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "check"))
    parser.add_argument("--result_dir", type=Path)
    parser.add_argument("--evidence_dir", type=Path, required=True)
    return parser.parse_args()


def validate(directory: Path) -> dict[str, object]:
    manifest_path = directory / "evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = set(FILES)
    if set(manifest["files"]) != expected:
        raise AssertionError("Compact evidence roster mismatch")
    for name, digest in manifest["files"].items():
        if sha256_path(directory / name) != digest:
            raise AssertionError(f"Compact evidence hash mismatch: {name}")
    decision = json.loads((directory / "decision.json").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    with (directory / "seed_rows.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 10 or sorted(int(row["seed"]) for row in rows) != list(range(64, 74)):
        raise AssertionError("Compact seed roster mismatch")
    if decision["card_sha256"] != provenance["card_sha256"]:
        raise AssertionError("Decision/provenance card hash mismatch")
    for path in directory.iterdir():
        if path.is_file() and "/network/scratch/" in path.read_text(errors="ignore"):
            raise AssertionError(f"Absolute scratch dependency leaked into {path.name}")
    return {"decision": decision["decision"], "seed_count": len(rows)}


def main() -> None:
    args = parse_args()
    if args.mode == "build":
        if args.result_dir is None:
            raise ValueError("--result_dir is required in build mode")
        if args.evidence_dir.exists():
            raise FileExistsError(args.evidence_dir)
        args.evidence_dir.mkdir(parents=True)
        for name in FILES:
            source = args.result_dir / name
            target = args.evidence_dir / name
            if name.endswith(".json"):
                target.write_text(json.dumps(
                    json.loads(source.read_text()), indent=2, sort_keys=True, allow_nan=False
                ) + "\n")
            else:
                target.write_text(source.read_text())
        manifest = {
            "schema_version": 1,
            "files": {name: sha256_path(args.evidence_dir / name) for name in FILES},
        }
        (args.evidence_dir / "evidence_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    result = validate(args.evidence_dir)
    print(json.dumps({"status": "passed", **result}), flush=True)


if __name__ == "__main__":
    main()
