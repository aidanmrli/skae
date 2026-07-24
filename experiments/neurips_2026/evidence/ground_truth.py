"""Build and validate the canonical controlled-system vector-field packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

from experiments.neurips_2026.protocol import CONTROLLED_PAPER_PROTOCOL
from experiments.neurips_2026.evidence.ground_truth_rendering import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DPI,
    DEFAULT_FORMATS,
    DEFAULT_GRID_POINTS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STREAM_DENSITY,
    DIRECTION_DECIMALS,
    FIELD_DECIMALS,
    GENERATOR_ID,
    LOG_SPEED_DECIMALS,
    FieldData,
    RETAINED_15_SYSTEMS,
    SystemSpec,
    compute_field,
    parse_formats,
    parse_systems,
    render_composite,
    render_individual,
)


def write_manifest(
    output_dir: Path,
    fields: list[FieldData],
    *,
    grid_points: int,
    formats: tuple[str, ...],
    dpi: int,
    stream_density: float,
    individual_paths: dict[str, list[str]],
    composite_paths: list[str],
) -> None:
    def pdf_record(paths: Iterable[str]) -> dict[str, str] | None:
        pdf = next((Path(path) for path in paths if Path(path).suffix == ".pdf"), None)
        if pdf is None:
            return None
        return {
            "path": pdf.name,
            "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        }

    payload = {
        "schema_version": 3,
        "description": (
            f"PDF vector-field displays for {len(fields)} controlled multibasin "
            "systems."
        ),
        "generated_by": GENERATOR_ID,
        "compute_allocation": "salloc --mem=8G -c 4 --partition=long",
        "reproduction_command_inside_allocation": (
            "uv run skae-paper build ground-truth --formats pdf"
        ),
        "render_parameters": {
            "grid_points": int(grid_points),
            "formats": list(formats),
            "dpi": int(dpi),
            "stream_density": float(stream_density),
            "field_decimals": FIELD_DECIMALS,
            "log_speed_decimals": LOG_SPEED_DECIMALS,
            "direction_decimals": DIRECTION_DECIMALS,
        },
        "composite": pdf_record(composite_paths),
        "systems": [
            {
                "system_key": field.spec.system_key,
                "display_name": field.spec.title,
                "basins": field.spec.basin_count,
                "pdf": pdf_record(individual_paths.get(field.spec.system_key, [])),
            }
            for field in fields
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_manifest(manifest_path: Path) -> tuple[Path, ...]:
    """Validate the active roster and hashes, returning every declared PDF."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 3 or payload.get("generated_by") != GENERATOR_ID:
        raise ValueError("Ground-truth manifest is not the canonical schema")
    expected_render_parameters = {
        "grid_points": DEFAULT_GRID_POINTS,
        "formats": list(DEFAULT_FORMATS),
        "dpi": DEFAULT_DPI,
        "stream_density": DEFAULT_STREAM_DENSITY,
        "field_decimals": FIELD_DECIMALS,
        "log_speed_decimals": LOG_SPEED_DECIMALS,
        "direction_decimals": DIRECTION_DECIMALS,
    }
    if payload.get("render_parameters") != expected_render_parameters:
        raise ValueError("Ground-truth render parameters drifted from the paper contract")
    systems = payload.get("systems")
    if not isinstance(systems, list):
        raise ValueError("Ground-truth manifest has no system inventory")
    expected = [
        (spec.system_key, spec.title, spec.basin_count) for spec in RETAINED_15_SYSTEMS
    ]
    observed = [
        (row.get("system_key"), row.get("display_name"), row.get("basins"))
        for row in systems
        if isinstance(row, dict)
    ]
    if observed != expected:
        raise ValueError("Ground-truth manifest roster drifted from the paper protocol")

    records = [payload.get("composite"), *(row.get("pdf") for row in systems)]
    paths: list[Path] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Ground-truth manifest is missing a PDF record")
        name = record.get("path")
        expected_hash = record.get("sha256")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"Unsafe ground-truth artifact name: {name!r}")
        path = manifest_path.parent / name
        if not path.is_file():
            raise ValueError(f"Missing ground-truth artifact: {path}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Ground-truth artifact hash mismatch: {path}")
        paths.append(path)
    return tuple(paths)


def _validate_active_request(
    output_dir: Path,
    specs: list[SystemSpec],
    formats: tuple[str, ...],
    *,
    grid_points: int,
    stream_density: float,
    dpi: int,
    skip_individual: bool,
) -> None:
    if output_dir.resolve() != DEFAULT_OUTPUT_DIR.resolve():
        return
    if tuple(spec.system_key for spec in specs) != CONTROLLED_PAPER_PROTOCOL.system_keys:
        raise ValueError("The active paper directory requires the complete frozen roster")
    if formats != DEFAULT_FORMATS:
        raise ValueError("The active paper directory contains only canonical PDF outputs")
    if (
        grid_points != DEFAULT_GRID_POINTS
        or stream_density != DEFAULT_STREAM_DENSITY
        or dpi != DEFAULT_DPI
    ):
        raise ValueError("The active paper directory requires canonical render parameters")
    if skip_individual:
        raise ValueError("The active paper directory requires every individual PDF")


def build_outputs(
    output_dir: Path,
    specs: list[SystemSpec],
    *,
    grid_points: int,
    chunk_size: int,
    stream_density: float,
    formats: tuple[str, ...],
    dpi: int,
    skip_individual: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = []
    for spec in specs:
        print(f"Computing vector field: {spec.system_key}", flush=True)
        fields.append(compute_field(spec, grid_points=grid_points, chunk_size=chunk_size))

    individual_paths: dict[str, list[str]] = {}
    if not skip_individual:
        for field in fields:
            print(f"Rendering individual plot: {field.spec.system_key}", flush=True)
            individual_paths[field.spec.system_key] = render_individual(
                field,
                output_dir,
                formats=formats,
                dpi=dpi,
                stream_density=stream_density,
            )

    print("Rendering composite", flush=True)
    composite_paths = render_composite(
        fields,
        output_dir,
        formats=formats,
        dpi=dpi,
        stream_density=max(0.75, stream_density * 0.82),
    )
    write_manifest(
        output_dir,
        fields,
        grid_points=grid_points,
        formats=formats,
        dpi=dpi,
        stream_density=stream_density,
        individual_paths=individual_paths,
        composite_paths=composite_paths,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        default="retained15",
        help="retained15/all or comma-separated retained-15 system keys",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid_points", type=int, default=DEFAULT_GRID_POINTS)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--stream_density",
        type=float,
        default=DEFAULT_STREAM_DENSITY,
    )
    parser.add_argument("--formats", default=",".join(DEFAULT_FORMATS))
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--skip_individual", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = parse_systems(args.systems)
    formats = parse_formats(args.formats)
    _validate_active_request(
        args.output_dir,
        specs,
        formats,
        grid_points=args.grid_points,
        stream_density=args.stream_density,
        dpi=args.dpi,
        skip_individual=args.skip_individual,
    )
    if args.check:
        if args.output_dir.resolve() != DEFAULT_OUTPUT_DIR.resolve():
            raise ValueError("--check compares only the active paper output directory")
        with TemporaryDirectory(
            prefix="skae-ground-truth-",
            dir=os.environ.get("SLURM_TMPDIR") or None,
        ) as temp_dir:
            generated_dir = Path(temp_dir)
            build_outputs(
                generated_dir,
                specs,
                grid_points=args.grid_points,
                chunk_size=args.chunk_size,
                stream_density=args.stream_density,
                formats=formats,
                dpi=args.dpi,
                skip_individual=args.skip_individual,
            )
            generated_paths = validate_manifest(generated_dir / "manifest.json")
            stale = [
                path.name
                for path in generated_paths
                if not (args.output_dir / path.name).is_file()
                or (args.output_dir / path.name).read_bytes() != path.read_bytes()
            ]
            if (args.output_dir / "manifest.json").read_bytes() != (
                generated_dir / "manifest.json"
            ).read_bytes():
                stale.append("manifest.json")
            if stale:
                raise RuntimeError(f"Ground-truth artifacts are stale: {stale}")
        print(f"Verified {len(RETAINED_15_SYSTEMS) + 1} ground-truth PDFs")
        return

    build_outputs(
        args.output_dir,
        specs,
        grid_points=args.grid_points,
        chunk_size=args.chunk_size,
        stream_density=args.stream_density,
        formats=formats,
        dpi=args.dpi,
        skip_individual=args.skip_individual,
    )
    print(f"Wrote outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
