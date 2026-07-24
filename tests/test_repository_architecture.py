"""Regression tests for the repository's maintained ownership boundaries."""

import ast
import hashlib
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

RESIDUAL_V1_HISTORICAL_ROOTS = {
    "card": "d60d833d84961da0c5931e6ee6cf3dbf763c12ccf3607764c5700f7cba2808dc",
    "source": "03e19d4efa6e98cc1b429403223de6159a8a787841e3218bed690af27cc750b8",
    "task": "86a3dce2ce8fd6ca569aebcccb6812ac6c3ee206ec21ba8e2ccf2642305fb024",
    "queue": "c84c7af99e44d6087a91d397b3c4c61ff44cf93f7f0d2bcc8d67878c2fe9b252",
}
PROVENANCE_LOCKED_SCRIPT_GROUPS = (
    {
        "ownership": "invalid_historical_provenance",
        "lock": (
            "experiments/neurips_2026/allen_cahn_direct_baseline/"
            "source_manifest.sha256"
        ),
        "lock_sha256": (
            "0cdd665641374e5b38ee711636beb877480f6877fb06f51c67ed0e40eae2106f"
        ),
        "card_sha": "b621250ac6f24ab35db2668a5aa07126a723b1088e0e873c13e585f4688329c7",
        "card_in_lock": False,
        "format": "sha256_manifest",
        "entry_count": 18,
        "additional_roots": {
            (
                "experiments/neurips_2026/allen_cahn_direct_baseline/"
                "task_lock.json"
            ): (
                "269081d3a4b53d665e7c3aa449605d2c456795d5712299eb82ec0aab14088201"
            ),
            "docs/archive/allen_cahn_direct_baseline_process_correction_20260721.md": (
                "adc57f7f544a97eeb09270082847e89fbbf399f7fc6ed85cca017b2b0404991c"
            ),
        },
        "script_dir": "scripts/neurips_2026/allen_cahn_direct_baseline/",
        "scripts": {
            "queue.sh": (
                "50f7b5a3d7310dad26e6d24b078032e22eff4d39540b1815a19614e83f5bf12f"
            ),
            "run_smoke.sh": (
                "88020fbfa68a70d333910b5f1a8a3db6f9d1b7e319c50a4b2f96e36721a723a6"
            ),
            "run_train_array.sh": (
                "cd6bba7cea938613d761f9b1445e9d5e1a40c3f4d6bb7d5df783227d045b3194"
            ),
            "run_evaluate_array.sh": (
                "21e430a57129ed31be2bb0fb6fb1d6fb94ca93d6ae8e0bbc5c4476c2d738b51f"
            ),
            "run_summary.sh": (
                "acda5758ba6df63dc3cfee82a4c66f6bcab14b28e9126429ce0dc0aa4db8a04b"
            ),
        },
    },
    {
        "ownership": "invalid_historical_provenance",
        "lock": (
            "experiments/neurips_2026/allen_cahn_early_fate_probe/"
            "source_manifest.sha256"
        ),
        "lock_sha256": (
            "28dd4626283e2629f391d9618e48b672ff266ecfd82925f68680712a37e2c496"
        ),
        "card_sha": "17987b43e40acbbc0c59d7a9d12ae2f1efa1343ff8904bbd9f58037a9be43bb2",
        "format": "sha256_manifest",
        "entry_count": 19,
        "additional_roots": {
            "docs/archive/allen_cahn_early_fate_probe_v1_invalid_20260721.md": (
                "778c6b47c9fdbc38aac23477cd915b8b62bc44903deb0b3aaacc56c6bc0ebf71"
            ),
        },
        "script_dir": "scripts/neurips_2026/allen_cahn_early_fate_probe/",
        "scripts": {
            "run_profile.sh": (
                "3b37ce6242ac3a2cb1cd8e23d5c45c9b6227d83ee0b4e6f5f527eef4b7db2cb0"
            ),
            "run_extract.sh": (
                "2d339ab5b5fd6768e979a1446b7f4a34ce0b024f4c1261e1f09d1c16b596704a"
            ),
            "run_summary.sh": (
                "f85a943d891af58e697dd5c470cca73f4e8d907719af25eebfe2ed95bea07d60"
            ),
            "queue.sh": (
                "6e1f61046e4454a064237b166783ffeb577a58db9e0f00c8dce4f317830da0ac"
            ),
        },
    },
    {
        "ownership": "invalid_historical_provenance",
        "lock": (
            "experiments/neurips_2026/allen_cahn_early_fate_probe_v2/"
            "source_manifest.sha256"
        ),
        "lock_sha256": (
            "dfe1c66166488befe119757b0de9e3bcf7e4be564fb81dc74038aa1ceef9618d"
        ),
        "card_sha": "db6f57e45568284e8903520df0dc2838ac2410d8adf8839d21ce366dc446a20c",
        "format": "sha256_manifest",
        "entry_count": 26,
        "additional_roots": {
            (
                "docs/archive/"
                "allen_cahn_early_fate_probe_v2_telemetry_failure_20260721.md"
            ): (
                "3817a77d0b89dadb910bf942b5739a441ad71174f06d96413b474f5d5537283f"
            ),
        },
        "script_dir": "scripts/neurips_2026/allen_cahn_early_fate_probe_v2/",
        "scripts": {
            "run_generate_extract.sh": (
                "b6f3e5165c8db346e19c30249b1b7e90dc022b6a947ff4fa9a75fc4de684ef30"
            ),
            "run_telemetry.sh": (
                "ac9b1a0bcce7cfc83392391ad50f4ca9121b95bfb81991f1f6885b69566da62a"
            ),
            "run_summary.sh": (
                "f254f1b01d30cd203b18cfc62852489481a04f43406f5a26958fc7458ad40824"
            ),
            "queue.sh": (
                "6a8736a1104a06efe335e6ac12f5ff0a24d413dc82ea3ac23c16d8beb28bb8ce"
            ),
        },
    },
    {
        "ownership": "completed_packet_provenance",
        "lock": (
            "experiments/neurips_2026/allen_cahn_support_subspaces/"
            "source_manifest.sha256"
        ),
        "lock_sha256": (
            "e4219ecb3b2e25d08f9f1e5afc51a16f84d94409baf62280651cc101fc3f7024"
        ),
        "card_sha": "fafa3b1a0e8f63095c3926171673fa62f2baec6e2af36a954cbca83d35f35743",
        "format": "sha256_manifest",
        "entry_count": 21,
        "script_dir": "scripts/neurips_2026/allen_cahn_support_subspaces/",
        "scripts": {
            "run_array.sh": (
                "969d678b2be5b4c9845c08e5d58870757342b8ce9877893c20fd0a67486c3ea0"
            ),
            "run_canary_validation.sh": (
                "46e09c6a38675b71988370315cc4e3897154aec141788a9181a549e0482574b2"
            ),
            "run_profile.sh": (
                "964f42ee5ba8ba2c8b65b2d4205d6c7e2813de7841b41851f8fba1e7f7c02fc2"
            ),
            "run_summary.sh": (
                "e576387574ed1064f7c22e559a079c0723cece535c10c3a732677436e5540149"
            ),
            "run_v4_validation.sh": (
                "009e94e3198c2c4803d7d4c422c3b1792f99bda912c116e66ca21a515e9baeb5"
            ),
        },
    },
    {
        "ownership": "completed_packet_provenance",
        "lock": (
            "experiments/neurips_2026/allen_cahn_forecast_replication/"
            "source_manifest.sha256"
        ),
        "lock_sha256": (
            "8add4eb16eea0f1e4b6d1483bf96149e092549f20977d93cb94b566502587595"
        ),
        "card_sha": "5519644cbbc8992a356045e68ff496818dceed500300432fd985febf80a555de",
        "format": "sha256_manifest",
        "entry_count": 21,
        "script_dir": "scripts/neurips_2026/allen_cahn_forecast_replication/",
        "scripts": {
            "run_generate_evaluate.sh": (
                "11f2b2ad43d5fbb12a0c72c212adcf0180443f8de44f47798ea74c67f4b392ed"
            ),
            "run_summary.sh": (
                "b2c4ac457a55a2b8d5c9e21da8ba2569e413ff59c54dd66d64932ac10884783e"
            ),
            "queue.sh": (
                "8dd179183a2f94c315c16a3d6de500c0c8f5821733b6f10a0105599c62ee2253"
            ),
        },
    },
    {
        "ownership": "completed_packet_provenance",
        "lock": (
            "experiments/neurips_2026/allen_cahn_physics_metrics/"
            "source_manifest.sha256"
        ),
        "lock_sha256": (
            "2c0af1fac15f182b3dd6538d909b77b68e123566a46f488a93b89068c31c3221"
        ),
        "card_sha": "d4748ec37aaf6d10de0c02eb988c5278840eacbbcf649007e1675a0c788bcb88",
        "format": "sha256_manifest",
        "entry_count": 23,
        "script_dir": "scripts/neurips_2026/allen_cahn_physics_metrics/",
        "scripts": {
            "run_evaluate.sh": (
                "13568b5ca663f8a5115f518f81d8f2a2f13f9c3b288a7230ee3537621d279566"
            ),
            "run_summary.sh": (
                "8d9882f8b238a320e1c87b5d57658db36677b49cc91342982dc7cd5d7aff7c9d"
            ),
        },
    },
    {
        "ownership": "invalid_historical_provenance",
        "lock": "experiments/neurips_2026/global_k_residual_forecast/source_manifest.sha256",
        "lock_sha256": "2c7439ca57c61e74c9f05b1dbb4d9f9c19c0e32efe60587063e27ae4ab8bd8e8",
        "card_sha": "fdb48269a6a0f7f964fcbf27271f54a67f195f6ef46d2e5c83ebcf67046629ca",
        "card_in_lock": False,
        "format": "sha256_manifest",
        "entry_count": 43,
        "historical_roots": RESIDUAL_V1_HISTORICAL_ROOTS,
        "additional_roots": {
            "experiments/neurips_2026/global_k_residual_forecast/task_manifest.json": "86a3dce2ce8fd6ca569aebcccb6812ac6c3ee206ec21ba8e2ccf2642305fb024",
            "docs/archive/global_k_residual_forecast_v1_smoke_gate_failure_20260721.md": "349cf839a59c09fc9efd4c14daf090dc131d0e8db340be052ebece140cfe5bd9",
            "docs/archive/global_k_residual_forecast_v2_smoke_gate_failure_20260721.md": "bb63a5d03076513449c9309a32152784499d24feacbe4c20404d6941b68e923b",
            "docs/archive/global_k_residual_forecast_v3_finiteness_failure_20260721.md": "f905c7511b1c9bac0ad2a3cabaa9afa2955090783bafcf4a6b1ee4cd7a85cfec",
        },
        "script_dir": "scripts/neurips_2026/global_k_residual_forecast/",
        "scripts": {
            "queue.sh": "db0222b88401214a34010e67ef0fdbf07d5d36d3ba9bc763249451a42afff8d4",
        },
    },
    {
        "ownership": "historical_reproduction_provenance",
        "lock": "experiments/neurips_2026/local_edmd_reproduction/source_lock.json",
        "lock_sha256": (
            "16e229e85c72536b1fea6824f12ceddec4d90c9ffece21641c2ae2cf2b887f03"
        ),
        "card_sha": "5a103beb9821405c4a5b762a781421a572b85a2108efbc4f78c010f3b57db3ed",
        "format": "json_source_lock",
        "entry_count": 27,
        "script_dir": "scripts/neurips_2026/local_edmd_reproduction/",
        "scripts": {
            "queue_full_chain.sh": (
                "fe9ca5c89845c19769ddc38fcfe482a56a804a97d55c531a1f323e30397e3ba4"
            ),
            "run_array.sh": (
                "1b1f51dc75a64f385fadb8533854cabbdaae71016a7a97ed4c36b2af254517d3"
            ),
            "run_check.sh": (
                "bb0931ef5dd265cd7036867ea6b144b5c01ab18e88addfe57b4489a1b0c2c5e4"
            ),
            "run_collect.sh": (
                "1260a1c07dd5f76c5eb29f8f0fe21c07d175f64f0cec5898eb1f329bed570483"
            ),
        },
    },
) + tuple(json.loads((ROOT / "tests/periodic_provenance_group.json").read_text()))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _locked_script_paths() -> set[Path]:
    valid_ownership = {
        "completed_packet_provenance",
        "historical_reproduction_provenance",
        "invalid_historical_provenance",
        "authorized_prospective_packet_provenance",
    }
    locked_paths: set[Path] = set()
    for group in PROVENANCE_LOCKED_SCRIPT_GROUPS:
        assert group["ownership"] in valid_ownership
        assert all(len(digest) == 64 for digest in group.get("historical_roots", {}).values())
        lock_path = ROOT / group["lock"]
        assert _sha256(lock_path) == group["lock_sha256"]
        card_path = lock_path.parent / "prediction_card.json"
        assert _sha256(card_path) == group["card_sha"]
        if group["format"] == "sha256_manifest":
            entries = {
                relative_path: digest
                for digest, relative_path in (
                    line.split(maxsplit=1)
                    for line in lock_path.read_text().splitlines()
                )
            }
            assert len(entries) == group["entry_count"]
        else:
            payload = json.loads(lock_path.read_text())
            entries = {
                source["path"]: source["sha256"]
                for source in payload["sources"].values()
            }
            assert len(entries) == group["entry_count"]
        if group.get("card_in_lock", True):
            assert entries[card_path.relative_to(ROOT).as_posix()] == group["card_sha"]
        for relative_path, expected_hash in group.get("additional_roots", {}).items():
            assert _sha256(ROOT / relative_path) == expected_hash
        for script_name, expected_hash in group["scripts"].items():
            relative_path = f"{group['script_dir']}{script_name}"
            script_path = ROOT / relative_path
            assert entries.get(relative_path) == expected_hash
            assert _sha256(script_path) == expected_hash
            assert script_path not in locked_paths
            locked_paths.add(script_path)
    return locked_paths


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
    provenance_locked_scripts = _locked_script_paths()
    maintained_scripts = [
        path for path in scripts if path not in provenance_locked_scripts
    ]
    for path in maintained_scripts:
        assert "/network/scratch/l/lia" not in path.read_text(), path
    contributor_specific_scripts = {
        path
        for path in scripts
        if "/network/scratch/l/lia" in path.read_text()
    }
    assert contributor_specific_scripts == provenance_locked_scripts
    combined = "\n".join(path.read_text() for path in scripts)
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
