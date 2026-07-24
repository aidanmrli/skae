"""Run unchanged v1 scientific computation with repaired v3 lineage writing."""

from __future__ import annotations

from experiments.neurips_2026.allen_cahn_periodic_reencoding import run as base_run
from experiments.neurips_2026.allen_cahn_periodic_reencoding_v3.lineage import (
    write_runtime_lineage,
)


def main() -> None:
    """Replace only post-compute lineage writing for the duration of v1 main."""

    original = base_run.write_runtime_lineage
    base_run.write_runtime_lineage = write_runtime_lineage
    try:
        base_run.main()
    finally:
        base_run.write_runtime_lineage = original


if __name__ == "__main__":
    main()
