"""Regression tests for the repository's maintained ownership boundaries."""

import ast
import importlib

from pathlib import Path

from experiments.neurips_2026 import cli
from experiments.neurips_2026.protocol import (
    CONTROLLED_PAPER_PROTOCOL,
    DYSTS_PAPER_PROTOCOL,
    PAPER_MODEL_ROWS,
)
from skae.benchmarks import paper_protocol as legacy_protocol


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


def test_maintained_scripts_have_no_contributor_specific_scratch_path() -> None:
    scripts = sorted((ROOT / "scripts").rglob("*.sh"))
    assert scripts
    assert not list((ROOT / "scripts").glob("*.sh"))
    combined = "\n".join(path.read_text() for path in scripts)
    assert "/network/scratch/l/lia" not in combined
    assert "scripts/common/cluster_env.sh" in combined
    assert '${SLURM_SUBMIT_DIR:-$PWD}' in combined
    assert 'git -C "$(dirname "${BASH_SOURCE[0]}")"' not in combined
