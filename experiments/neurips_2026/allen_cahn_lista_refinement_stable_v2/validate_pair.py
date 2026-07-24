"""Validate one seed's paired Q=2/Q=3 branches before external evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.neurips_2026.allen_cahn_lista_refinement_stable.validate_smoke import (
    validate_run,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run_dir) != 2:
        raise ValueError("exactly two paired depth runs are required")
    rows = [validate_run(path) for path in args.run_dir]
    if sorted(int(row["refinements"]) for row in rows) != [2, 3]:
        raise ValueError("production pair must contain Q=2 and Q=3")
    if any(int(row["max_forecast_step"]) < 3499 for row in rows):
        raise ValueError("production pair did not complete 3500 forecast updates")
    for key in (
        "source_checkpoint_sha256",
        "source_model_state_sha256",
        "source_training_generator_sha256",
        "first_forecast_batch_sha256",
        "final_training_generator_sha256",
    ):
        if len({row[key] for row in rows}) != 1:
            raise ValueError(f"paired branches differ in {key}")
    payload = {"schema_version": 1, "status": "passed", "runs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
