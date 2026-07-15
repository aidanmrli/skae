"""Regression tests for the repository's maintained ownership boundaries."""

import ast
import importlib
import json

from pathlib import Path

from experiments.neurips_2026 import cli
from experiments.neurips_2026.baselines import classical, local_linear, tasks
from experiments.neurips_2026 import controlled
from experiments.neurips_2026.local_operators import contract as local_operator_contract
from experiments.neurips_2026.protocol import (
    CONTROLLED_PAPER_PROTOCOL,
    DYSTS_PAPER_PROTOCOL,
    PAPER_MODEL_ROWS,
)
from skae.benchmarks import paper_protocol as legacy_protocol
from skae.benchmarks import (
    transition_rich_basin_partition_manifest as legacy_controlled,
)
from skae.support import routing as reusable_support_routing


ROOT = Path(__file__).resolve().parents[1]


def test_paper_cli_dispatches_to_the_canonical_module(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        cli,
        "_invoke",
        lambda module_name, args: calls.append((module_name, tuple(args))),
    )

    cli.main(["build", "controlled", "--check"])

    assert calls == [
        ("experiments.neurips_2026.evidence.controlled_tables", ("--check",))
    ]


def test_paper_cli_forwards_subcommand_help(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        cli,
        "_invoke",
        lambda module_name, args: calls.append((module_name, tuple(args))),
    )

    cli.main(["build", "controlled", "--help"])

    assert calls == [
        ("experiments.neurips_2026.evidence.controlled_tables", ("--help",))
    ]


def test_full_paper_check_isolates_each_builder(monkeypatch, capsys) -> None:
    calls = []
    monkeypatch.setattr(cli, "validate_protocol", lambda: None)
    monkeypatch.setattr(
        cli,
        "_invoke_isolated",
        lambda module_name, args: calls.append((module_name, tuple(args))),
    )

    cli.main(["check"])

    assert calls == [
        (module_name, tuple(args)) for module_name, args in cli.CHECKS
    ]
    assert "All frozen paper evidence checks passed." in capsys.readouterr().out


def test_every_paper_command_resolves_to_a_main_function() -> None:
    for module_name in set(cli.COMMANDS.values()):
        assert callable(importlib.import_module(module_name).main)


def test_frozen_protocol_is_unique_and_legacy_import_is_identical() -> None:
    assert len(CONTROLLED_PAPER_PROTOCOL.system_keys) == 15
    assert len(set(CONTROLLED_PAPER_PROTOCOL.system_keys)) == 15
    assert len(DYSTS_PAPER_PROTOCOL.system_keys) == 10
    assert len(set(DYSTS_PAPER_PROTOCOL.system_keys)) == 10
    assert len(PAPER_MODEL_ROWS) == 6
    assert legacy_protocol.CONTROLLED_PAPER_PROTOCOL is CONTROLLED_PAPER_PROTOCOL
    assert tasks.DEFAULT_SYSTEMS is CONTROLLED_PAPER_PROTOCOL.system_keys
    assert classical.DEFAULT_SYSTEMS is CONTROLLED_PAPER_PROTOCOL.system_keys
    assert local_linear.DEFAULT_SYSTEMS is CONTROLLED_PAPER_PROTOCOL.system_keys


def test_controlled_api_is_clean_and_legacy_names_are_shim_only() -> None:
    assert not any("transition_rich" in name.lower() for name in controlled.__all__)
    assert legacy_controlled.transition_rich_basin_partition_systems is (
        controlled.controlled_systems
    )
    assert legacy_controlled.get_transition_rich_basin_partition_model is (
        controlled.get_controlled_model
    )


def test_protocol_validation_covers_declared_paper_outputs() -> None:
    cli.validate_protocol()


def test_tools_are_compatibility_shims_only() -> None:
    tool_modules = sorted((ROOT / "tools").glob("*.py"))
    assert tool_modules
    assert all(len(path.read_text().splitlines()) <= 50 for path in tool_modules)
    for path in tool_modules:
        importlib.import_module(f"tools.{path.stem}")


def test_reusable_library_does_not_depend_on_the_paper_package() -> None:
    compatibility_shims = {
        ROOT / "skae/benchmarks/controlled_alignment.py",
        ROOT / "skae/benchmarks/paper_protocol.py",
        ROOT / "skae/benchmarks/paper_statistics.py",
        ROOT / "skae/benchmarks/transition_rich_basin_partition_manifest.py",
    }
    for path in (ROOT / "skae").rglob("*.py"):
        if path in compatibility_shims:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
        assert not any(
            name.startswith("experiments.neurips_2026")
            for name in imported_modules
        ), path


def test_reusable_support_routing_owns_no_paper_protocol_constants() -> None:
    frozen_names = {
        "SUPPORT_DEFINITION",
        "SUPPORT_THRESHOLD",
        "FAMILY_JACCARD_THRESHOLD",
        "FIT_CONFIGURED_ROWS",
        "FIT_SEED_OFFSET",
    }
    assert frozen_names.isdisjoint(vars(reusable_support_routing))
    assert local_operator_contract.SUPPORT_DEFINITION == "absolute:0.001"
    assert local_operator_contract.FAMILY_JACCARD_THRESHOLD == 0.40
    assert local_operator_contract.FIT_CONFIGURED_ROWS == 512


def test_maintained_scripts_have_no_contributor_specific_scratch_path() -> None:
    scripts = sorted((ROOT / "scripts").rglob("*.sh"))
    assert scripts
    assert not list((ROOT / "scripts").glob("*.sh"))
    combined = "\n".join(path.read_text() for path in scripts)
    assert "/network/scratch/l/lia" not in combined
    assert "scripts/common/cluster_env.sh" in combined
    assert '${SLURM_SUBMIT_DIR:-$PWD}' in combined
    assert 'git -C "$(dirname "${BASH_SOURCE[0]}")"' not in combined
    assert '"alignment_protocol": {' not in combined
    assert 'SYSTEMS_CSV="${SYSTEMS_CSV:-gated_local_linear' not in combined
    assert 'SEEDS_CSV="${SEEDS_CSV:-0,1,2}"' not in combined
    assert "support=absolute:0.001" not in combined
    assert '"staged_protocol": {' not in combined
    assert (
        'FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD:-0.4}"'
        not in combined
    )
    intervention = (
        ROOT / "scripts/neurips_2026/interventions/run.sh"
    ).read_text()
    assert 'NUM_INITIAL_POINTS="${NUM_INITIAL_POINTS:-100}"' not in intervention
    assert (
        'SUPPORT_DEFINITION="${SUPPORT_DEFINITION:-absolute:0.001}"'
        not in intervention
    )


def test_active_manifest_has_only_machine_resolvable_outputs() -> None:
    manifest_path = ROOT / "docs/figures/neurips_paper_2026/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for group in manifest["evidence_groups"]:
        assert all((manifest_path.parent / output).is_file() for output in group["outputs"])
        assert not any("individual PDFs" in output for output in group["outputs"])


def test_active_provenance_names_canonical_generators() -> None:
    provenance_path = (
        ROOT
        / "docs/figures/neurips_paper_2026/_data/local_map_forecasting_provenance.json"
    )
    provenance = json.loads(provenance_path.read_text())
    assert provenance["generated_by"] == (
        "experiments.neurips_2026.evidence.local_operator_tables"
    )
