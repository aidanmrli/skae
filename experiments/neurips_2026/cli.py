"""Discoverable command surface for the complete NeurIPS paper workflow."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict
from typing import Sequence

from experiments.neurips_2026.controlled import (
    get_transition_rich_basin_partition_model,
    transition_rich_basin_partition_manifest_jsonable,
)
from experiments.neurips_2026.protocol import (
    CONTROLLED_MODEL_ROW_IDS,
    CONTROLLED_PAPER_PROTOCOL,
    DYSTS_MODEL_ROW_IDS,
    DYSTS_PAPER_PROTOCOL,
    PAPER_MODEL_ROWS,
    PAPER_SEEDS,
)
from experiments.neurips_2026.paths import PAPER_EVIDENCE_DIR, REPO_ROOT
from skae.config import get_config


COMMANDS = {
    ("tasks", "controlled"): "experiments.neurips_2026.workflows.controlled_tasks",
    ("tasks", "dysts"): "experiments.neurips_2026.workflows.dysts_tasks",
    (
        "tasks",
        "dysts-evaluation",
    ): "experiments.neurips_2026.workflows.dysts_evaluation_tasks",
    ("tasks", "baselines"): "experiments.neurips_2026.baselines.tasks",
    ("train", "local-operators"): "experiments.neurips_2026.local_operators.train",
    ("evaluate", "checkpoints"): "skae.cli.evaluate",
    ("evaluate", "dysts"): "experiments.neurips_2026.workflows.dysts_evaluation",
    ("evaluate", "classical"): "experiments.neurips_2026.baselines.classical",
    ("evaluate", "local-linear"): "experiments.neurips_2026.baselines.local_linear",
    ("evaluate", "interventions"): "experiments.neurips_2026.interventions.evaluate",
    (
        "evaluate",
        "local-operators",
    ): "experiments.neurips_2026.local_operators.reevaluate",
    ("collect", "controlled"): "experiments.neurips_2026.workflows.controlled_collection",
    ("collect", "dysts"): "experiments.neurips_2026.workflows.dysts_collection",
    ("collect", "compare"): "experiments.neurips_2026.workflows.comparison",
    ("alignment", "reduce"): "experiments.neurips_2026.workflows.alignment_reduction",
    ("alignment", "merge"): "experiments.neurips_2026.workflows.alignment_merge",
    ("freeze", "headline"): "experiments.neurips_2026.evidence.freeze",
    ("freeze", "baselines"): "experiments.neurips_2026.baselines.freeze",
    ("freeze", "interventions"): "experiments.neurips_2026.interventions.freeze",
    ("build", "headline"): "experiments.neurips_2026.evidence.headline_tables",
    ("build", "controlled"): "experiments.neurips_2026.evidence.controlled_tables",
    ("build", "dysts"): "experiments.neurips_2026.evidence.dysts",
    ("build", "baselines"): "experiments.neurips_2026.baselines.summarize",
    (
        "build",
        "local-operators",
    ): "experiments.neurips_2026.evidence.local_operator_tables",
    ("build", "interventions"): "experiments.neurips_2026.interventions.artifacts",
    ("build", "ground-truth"): "experiments.neurips_2026.evidence.ground_truth",
    ("cache", "dysts"): "experiments.neurips_2026.workflows.dysts_cache",
}


CHECKS = (
    (
        "experiments.neurips_2026.evidence.freeze",
        ("--compact-existing-support", "--check"),
    ),
    ("experiments.neurips_2026.evidence.headline_tables", ("--check",)),
    ("experiments.neurips_2026.evidence.controlled_tables", ("--check",)),
    ("experiments.neurips_2026.evidence.dysts", ("--check",)),
    ("experiments.neurips_2026.baselines.summarize", ("--check",)),
    ("experiments.neurips_2026.evidence.local_operator_tables", ("--check",)),
    ("experiments.neurips_2026.interventions.artifacts", ("--check",)),
)


def _invoke(module_name: str, args: Sequence[str]) -> None:
    """Run a legacy-style ``main()`` with an isolated argument vector."""

    module = importlib.import_module(module_name)
    previous_argv = sys.argv
    sys.argv = [module_name, *args]
    try:
        try:
            module.main()
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
    finally:
        sys.argv = previous_argv


def protocol_summary() -> dict[str, object]:
    """Return the human- and machine-readable frozen experiment roster."""

    return {
        "controlled": transition_rich_basin_partition_manifest_jsonable(),
        "dysts": asdict(DYSTS_PAPER_PROTOCOL),
        "model_rows": [asdict(row) for row in PAPER_MODEL_ROWS],
    }


def validate_protocol() -> None:
    """Fail fast when a paper invariant or evidence reference has drifted."""

    if len(CONTROLLED_PAPER_PROTOCOL.system_keys) != 15 or len(
        set(CONTROLLED_PAPER_PROTOCOL.system_keys)
    ) != 15:
        raise RuntimeError("Controlled paper roster must contain 15 unique systems")
    if len(DYSTS_PAPER_PROTOCOL.system_keys) != 10 or len(
        set(DYSTS_PAPER_PROTOCOL.system_keys)
    ) != 10:
        raise RuntimeError("Dysts paper roster must contain 10 unique systems")
    if PAPER_SEEDS != tuple(range(15)):
        raise RuntimeError("The paper seed roster must be exactly seeds 0 through 14")
    if len(PAPER_MODEL_ROWS) != 6:
        raise RuntimeError("The paper must map exactly six neural model rows")
    if tuple(row.controlled_variant for row in PAPER_MODEL_ROWS) != CONTROLLED_MODEL_ROW_IDS:
        raise RuntimeError("Controlled model order has drifted from PAPER_MODEL_ROWS")
    if tuple(row.dysts_variant for row in PAPER_MODEL_ROWS) != DYSTS_MODEL_ROW_IDS:
        raise RuntimeError("Dysts model order has drifted from PAPER_MODEL_ROWS")
    if CONTROLLED_PAPER_PROTOCOL.seeds != DYSTS_PAPER_PROTOCOL.seeds:
        raise RuntimeError("The two main benchmarks no longer share the frozen seed roster")
    if (
        CONTROLLED_PAPER_PROTOCOL.num_steps != 200_000
        or CONTROLLED_PAPER_PROTOCOL.sequence_length != 8
        or DYSTS_PAPER_PROTOCOL.num_steps != 100_000
        or DYSTS_PAPER_PROTOCOL.sequence_length != 10
        or DYSTS_PAPER_PROTOCOL.dt_multiplier != 30.0
    ):
        raise RuntimeError("A frozen paper training budget has drifted")

    dense_recipe = get_transition_rich_basin_partition_model(
        "mlp_zero_sparse_hardinit_basin_partition_control"
    )
    dense_cfg = get_config(dense_recipe.config_name)
    if dense_cfg.MODEL.ENCODER.ACTIVATION.lower() != "tanh":
        raise RuntimeError("Dense no-sparsity baseline must use tanh activation")
    if dense_cfg.MODEL.SPARSITY_COEFF != 0.0:
        raise RuntimeError("Dense no-sparsity baseline has a nonzero sparsity coefficient")

    manifest_path = PAPER_EVIDENCE_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 3:
        raise RuntimeError("Unsupported active-evidence manifest schema")
    paper_source = REPO_ROOT / str(manifest["paper_source"])
    if not paper_source.is_file():
        raise RuntimeError(f"Missing paper source declared by manifest: {paper_source}")
    workflow_guide = REPO_ROOT / str(manifest["workflow_guide"])
    if not workflow_guide.is_file():
        raise RuntimeError(
            f"Missing reproduction guide declared by manifest: {workflow_guide}"
        )
    if manifest.get("complete_check_inside_allocation") != "uv run skae-paper check":
        raise RuntimeError("The active-evidence manifest lacks the canonical full check")
    for group in manifest["evidence_groups"]:
        build_tool = group.get("build_tool", "")
        if not build_tool.startswith("experiments.neurips_2026."):
            raise RuntimeError(
                f"Evidence group {group['id']!r} has a noncanonical builder"
            )
        for field in ("inputs",):
            for relative_path in group.get(field, []):
                declared_path = manifest_path.parent / relative_path
                if not declared_path.is_file():
                    raise RuntimeError(
                        f"Evidence group {group['id']!r} is missing {field[:-1]} "
                        f"{relative_path!r}"
                    )
        for field in ("provenance", "manifest"):
            relative_path = group.get(field)
            if relative_path and not (manifest_path.parent / relative_path).is_file():
                raise RuntimeError(
                    f"Evidence group {group['id']!r} is missing {field} "
                    f"{relative_path!r}"
                )
        for output in group.get("outputs", []):
            if output.startswith("15 individual PDFs"):
                continue
            output_path = manifest_path.parent / output
            if not output_path.exists():
                raise RuntimeError(
                    f"Evidence group {group['id']!r} is missing output {output!r}"
                )


def _parser() -> argparse.ArgumentParser:
    targets = sorted(f"{command} {target}" for command, target in COMMANDS)
    parser = argparse.ArgumentParser(
        prog="skae-paper",
        description=(
            "Run the frozen sparse-Koopman NeurIPS workflow. Remaining arguments "
            "are forwarded to the selected operation.\n\nOperations:\n  "
            + "\n  ".join(targets)
            + "\n  protocol show\n  protocol validate\n  check"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?")
    parser.add_argument("target", nargs="?")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch a paper workflow without exposing implementation filenames."""

    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    known, forwarded = parser.parse_known_args(raw_args)
    if known.command is None:
        parser.print_help()
        return
    if known.command == "protocol":
        if known.target == "show":
            print(json.dumps(protocol_summary(), indent=2, sort_keys=True))
            return
        if known.target == "validate":
            validate_protocol()
            print("Paper protocol and declared evidence outputs are consistent.")
            return
        parser.error("protocol requires target 'show' or 'validate'")
    if known.command == "check":
        if known.target is not None or forwarded:
            parser.error("check accepts no additional arguments")
        validate_protocol()
        for module_name, check_args in CHECKS:
            _invoke(module_name, check_args)
        print("All frozen paper evidence checks passed.")
        return

    module_name = COMMANDS.get((known.command, known.target))
    if module_name is None:
        parser.error(f"unknown operation: {known.command} {known.target or ''}".rstrip())
    _invoke(module_name, forwarded)


if __name__ == "__main__":
    main()
