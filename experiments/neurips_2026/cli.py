"""Discoverable command surface for the complete NeurIPS paper workflow."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from dataclasses import asdict
from typing import Sequence

from experiments.neurips_2026.controlled import (
    controlled_manifest_jsonable,
    get_controlled_model,
)
from experiments.neurips_2026.interventions.protocol import (
    intervention_protocol_metadata,
)
from experiments.neurips_2026.local_operators.contract import (
    route_protocol_metadata,
)
from experiments.neurips_2026.alignment import alignment_protocol_metadata
from experiments.neurips_2026.protocol import (
    CLASSICAL_BASELINE_METHOD_IDS,
    CONTROLLED_MODEL_ROW_IDS,
    CONTROLLED_PAPER_PROTOCOL,
    DYSTS_MODEL_ROW_IDS,
    DYSTS_PAPER_PROTOCOL,
    LOCAL_LINEAR_BASELINE_METHOD_IDS,
    PAPER_CONTROLLED_SYSTEMS,
    PAPER_MODEL_ROWS,
    PAPER_SEEDS,
    STANDALONE_BASELINE_SEEDS,
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
    (
        "tasks",
        "local-operators",
    ): "experiments.neurips_2026.local_operators.prepare_tasks",
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
    (
        "build",
        "high-dimensional",
    ): "experiments.neurips_2026.evidence.highdimensional",
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
    ("experiments.neurips_2026.evidence.highdimensional", ("--check",)),
    ("experiments.neurips_2026.evidence.controlled_tables", ("--check",)),
    ("experiments.neurips_2026.evidence.dysts", ("--check",)),
    ("experiments.neurips_2026.baselines.summarize", ("--check",)),
    ("experiments.neurips_2026.evidence.local_operator_tables", ("--check",)),
    ("experiments.neurips_2026.interventions.artifacts", ("--check",)),
    ("experiments.neurips_2026.evidence.ground_truth", ("--check",)),
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


def _invoke_isolated(module_name: str, args: Sequence[str]) -> None:
    """Run one evidence builder without leaking process-global plot state."""

    subprocess.run(
        [sys.executable, "-m", module_name, *args],
        cwd=REPO_ROOT,
        check=True,
    )


def protocol_summary() -> dict[str, object]:
    """Return the human- and machine-readable frozen experiment roster."""

    return {
        "controlled": controlled_manifest_jsonable(),
        "dysts": asdict(DYSTS_PAPER_PROTOCOL),
        "model_rows": [asdict(row) for row in PAPER_MODEL_ROWS],
        "alignment": alignment_protocol_metadata(),
        "intervention_case_study": intervention_protocol_metadata(),
        "local_operator": route_protocol_metadata(),
        "standalone_baselines": {
            "systems": list(CONTROLLED_PAPER_PROTOCOL.system_keys),
            "seeds": list(STANDALONE_BASELINE_SEEDS),
            "classical_methods": list(CLASSICAL_BASELINE_METHOD_IDS),
            "local_linear_methods": list(LOCAL_LINEAR_BASELINE_METHOD_IDS),
        },
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
    if tuple(system.system_key for system in PAPER_CONTROLLED_SYSTEMS) != (
        CONTROLLED_PAPER_PROTOCOL.system_keys
    ):
        raise RuntimeError("Controlled system metadata drifted from the frozen roster")
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

    dense_recipe = get_controlled_model(
        "mlp_zero_sparse_hardinit_basin_partition_control"
    )
    dense_cfg = get_config(dense_recipe.config_name)
    if dense_cfg.MODEL.ENCODER.ACTIVATION.lower() != "tanh":
        raise RuntimeError("Dense no-sparsity baseline must use tanh activation")
    if dense_cfg.MODEL.SPARSITY_COEFF != 0.0:
        raise RuntimeError("Dense no-sparsity baseline has a nonzero sparsity coefficient")

    dysts_tasks = importlib.import_module(
        "experiments.neurips_2026.workflows.dysts_tasks"
    )
    if tuple(dysts_tasks.DYSTS_SYSTEM_SPECS) != DYSTS_PAPER_PROTOCOL.system_keys:
        raise RuntimeError("Dysts task metadata drifted from the frozen paper roster")

    baseline_modules = (
        importlib.import_module("experiments.neurips_2026.baselines.tasks"),
        importlib.import_module("experiments.neurips_2026.baselines.classical"),
        importlib.import_module("experiments.neurips_2026.baselines.local_linear"),
    )
    if any(
        tuple(module.DEFAULT_SYSTEMS) != CONTROLLED_PAPER_PROTOCOL.system_keys
        for module in baseline_modules
    ):
        raise RuntimeError("A standalone baseline roster drifted from the protocol")

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
    groups = manifest.get("evidence_groups")
    if not isinstance(groups, list) or not all(
        isinstance(group, dict) and isinstance(group.get("id"), str)
        for group in groups
    ):
        raise RuntimeError("Active evidence groups must have string IDs")
    if len({group["id"] for group in groups}) != len(groups):
        raise RuntimeError("Active evidence groups must have unique IDs")
    canonical_commands = set(COMMANDS.values())
    for group in groups:
        build_tool = group.get("build_tool", "")
        if build_tool not in canonical_commands:
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
            if not isinstance(output, str):
                raise RuntimeError(f"Evidence group {group['id']!r} has an invalid output")
            output_path = manifest_path.parent / output
            if not output_path.is_file():
                raise RuntimeError(
                    f"Evidence group {group['id']!r} is missing output {output!r}"
                )
        if group.get("id") == "controlled_ground_truth_vector_fields":
            ground_truth = importlib.import_module(
                "experiments.neurips_2026.evidence.ground_truth"
            )
            ground_truth.validate_manifest(
                manifest_path.parent / str(group["manifest"])
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
    if len(raw_args) >= 2:
        module_name = COMMANDS.get((raw_args[0], raw_args[1]))
        if module_name is not None:
            _invoke(module_name, raw_args[2:])
            return
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
            _invoke_isolated(module_name, check_args)
        print("All frozen paper evidence checks passed.")
        return

    parser.error(f"unknown operation: {known.command} {known.target or ''}".rstrip())


if __name__ == "__main__":
    main()
