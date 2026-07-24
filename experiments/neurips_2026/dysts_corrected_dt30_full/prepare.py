"""Validate corrected Dysts task tables and write evaluation root specs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.neurips_2026.protocol import DYSTS_MODEL_ROW_IDS


DISPLAY = {
    "lista": ("LISTA", "lista"),
    "lista_bd": ("LISTA-BD", "lista"),
    "lista_sb": ("LISTA-SB", "lista"),
    "sparse_mlp": ("Sparse MLP", "mlp"),
    "sparse_mlp_bd": ("Sparse MLP-BD", "mlp"),
    "dense_mlp_tanh": ("Dense MLP tanh", "mlp"),
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _validate(rows: list[dict[str, str]], *, expected: int) -> None:
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, got {len(rows)}")
    variants = tuple(dict.fromkeys(row["model_variant"] for row in rows))
    if variants != tuple(DYSTS_MODEL_ROW_IDS):
        raise ValueError(f"variant order mismatch: {variants}")
    if any(float(row["dt_multiplier"]) != 30.0 for row in rows):
        raise ValueError("all rows must use dt multiplier 30")
    for row in rows:
        if row["model_variant"] in {"lista", "lista_bd", "lista_sb"}:
            if int(row["lista_num_loops"]) != 1:
                raise ValueError("every LISTA row must use one refinement")
        if row["model_variant"] == "dense_mlp_tanh":
            checks = {
                "config_name": "generic_no_shrink",
                "sparsity_coeff": "0.0",
                "k_structure": "dense",
            }
            for key, expected_value in checks.items():
                if row[key] != expected_value:
                    raise ValueError(f"dense baseline {key}={row[key]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_tsv", type=Path, required=True)
    parser.add_argument("--full_tsv", type=Path, required=True)
    parser.add_argument("--base_out", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--root_specs", type=Path, required=True)
    parser.add_argument("--systems_file", type=Path, required=True)
    parser.add_argument("--smoke_systems_file", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    smoke = _read(args.smoke_tsv)
    full = _read(args.full_tsv)
    _validate(smoke, expected=12)
    _validate(full, expected=900)

    systems = tuple(dict.fromkeys(row["system_key"] for row in full))
    if len(systems) != 10:
        raise ValueError(f"expected ten systems, got {systems}")
    args.systems_file.parent.mkdir(parents=True, exist_ok=True)
    args.systems_file.write_text(
        "".join(system.split(":", 1)[1] + "\n" for system in systems)
    )
    args.smoke_systems_file.write_text("Chua\n")
    with args.root_specs.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "display_name", "model_family", "root_dir"],
            delimiter="\t",
        )
        writer.writeheader()
        for variant in DYSTS_MODEL_ROW_IDS:
            display, family = DISPLAY[variant]
            writer.writerow({
                "label": variant,
                "display_name": display,
                "model_family": family,
                "root_dir": str(args.base_out / args.phase / variant),
            })
    payload = {
        "schema_version": 1,
        "status": "prepared",
        "smoke_rows": len(smoke),
        "full_rows": len(full),
        "systems": list(systems),
        "variants": list(DYSTS_MODEL_ROW_IDS),
    }
    args.receipt.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
