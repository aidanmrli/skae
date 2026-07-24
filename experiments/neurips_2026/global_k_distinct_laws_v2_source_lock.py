#!/usr/bin/env python3
"""Freeze and verify mutable sources for the approved distinct-law V2 protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.neurips_2026.global_k_distinct_laws_v2_tasks import (
    load_card,
    sha256_path,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ID = "global_k_distinct_laws_gated_local_linear_v2_new_seeds"
SOURCE_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "experiments/neurips_2026/global_k_distinct_laws_v2_card.json",
    "experiments/neurips_2026/global_k_distinct_laws_v2_tasks.py",
    "experiments/neurips_2026/global_k_distinct_laws_v2_preflight.py",
    "experiments/neurips_2026/global_k_distinct_laws_v2_checkpoint_audit.py",
    "experiments/neurips_2026/global_k_distinct_laws_v2.py",
    "experiments/neurips_2026/global_k_distinct_laws_v2_math.py",
    "experiments/neurips_2026/global_k_distinct_laws_v2_routing.py",
    "experiments/neurips_2026/summarize_global_k_distinct_laws_v2.py",
    "experiments/neurips_2026/assess_global_k_distinct_laws_v2_smoke.py",
    "experiments/neurips_2026/assess_global_k_distinct_laws_v2_scientific_gpu.py",
    "experiments/neurips_2026/build_global_k_distinct_laws_v2_packet.py",
    "experiments/neurips_2026/global_k_distinct_laws_v2_source_lock.py",
    "experiments/neurips_2026/global_k_support_invariance.py",
    "experiments/neurips_2026/global_k_support_invariance_card.json",
    "experiments/neurips_2026/global_k_dense_zero_wd_card.json",
    "skae/checkpoint_compat.py",
    "skae/config.py",
    "skae/model.py",
    "skae/data.py",
    "skae/evaluation.py",
    "skae/training/runner.py",
    "scripts/common/cluster_env.sh",
    "scripts/common/run_benchmark_task.sh",
    "scripts/common/gpu_guard.sh",
    "scripts/neurips_2026/global_k_distinct_laws_v2/build_tasks.sh",
    "scripts/neurips_2026/global_k_distinct_laws_v2/run_mixed_pack.sh",
    "scripts/neurips_2026/global_k_distinct_laws_v2/assess_smoke.sh",
    "scripts/neurips_2026/global_k_distinct_laws_v2/assess_scientific_gpu.sh",
    "scripts/neurips_2026/global_k_distinct_laws_v2/run_checkpoint_audit.sh",
    "scripts/neurips_2026/global_k_distinct_laws_v2/run_evaluation.sh",
    "scripts/neurips_2026/global_k_distinct_laws_v2/run_summary.sh",
    "scripts/neurips_2026/global_k_distinct_laws_v2/run_packet.sh",
    "scripts/neurips_2026/global_k_distinct_laws_v2/queue_scientific_chain.sh",
)
EXTERNAL_INPUT_KEYS = frozenset({
    "smoke_task_tsv", "smoke_manifest", "full_task_tsv", "full_manifest",
})


def build_lock(
    card_path: Path, smoke_task_tsv: Path, smoke_manifest: Path,
    full_task_tsv: Path, full_manifest: Path,
) -> dict[str, Any]:
    card, card_hash = load_card(card_path)
    sources: dict[str, dict[str, str]] = {}
    for relative in SOURCE_PATHS:
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        sources[relative] = {"path": relative, "sha256": sha256_path(path)}
    external = {}
    for name, path in (
        ("smoke_task_tsv", smoke_task_tsv),
        ("smoke_manifest", smoke_manifest),
        ("full_task_tsv", full_task_tsv),
        ("full_manifest", full_manifest),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        external[name] = {"path": str(path), "sha256": sha256_path(path)}
    return {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "card_sha256": card_hash,
        "sources": sources,
        "external_inputs": external,
    }


def load_lock(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Unexpected distinct-law V2 source-lock protocol")
    return payload


def verify_source_lock(path: Path) -> dict[str, Any]:
    lock = load_lock(path)
    failures: list[str] = []
    sources = lock.get("sources", {})
    external = lock.get("external_inputs", {})
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_PATHS):
        observed = set(sources) if isinstance(sources, dict) else set()
        failures.append(
            f"source roster: missing={sorted(set(SOURCE_PATHS) - observed)}, "
            f"extra={sorted(observed - set(SOURCE_PATHS))}"
        )
    if not isinstance(external, dict) or set(external) != EXTERNAL_INPUT_KEYS:
        observed = set(external) if isinstance(external, dict) else set()
        failures.append(
            f"external roster: missing={sorted(EXTERNAL_INPUT_KEYS - observed)}, "
            f"extra={sorted(observed - EXTERNAL_INPUT_KEYS)}"
        )
    for name, item in sources.items() if isinstance(sources, dict) else ():
        if item.get("path") != name:
            failures.append(f"source {name}: path field is {item.get('path')!r}")
            continue
        source = REPOSITORY_ROOT / item["path"]
        actual = sha256_path(source) if source.is_file() else "missing"
        if actual != item["sha256"]:
            failures.append(f"source {name}: {actual} != {item['sha256']}")
    for name, item in external.items() if isinstance(external, dict) else ():
        source = Path(item["path"])
        actual = sha256_path(source) if source.is_file() else "missing"
        if actual != item["sha256"]:
            failures.append(f"external {name}: {actual} != {item['sha256']}")
    if failures:
        raise RuntimeError("Distinct-law V2 source-lock failure:\n" + "\n".join(failures))
    card_source = lock["sources"][
        "experiments/neurips_2026/global_k_distinct_laws_v2_card.json"
    ]
    if card_source["sha256"] != lock["card_sha256"]:
        raise RuntimeError("Source lock contains inconsistent card hashes")
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--card", type=Path)
    parser.add_argument("--smoke_task_tsv", type=Path)
    parser.add_argument("--smoke_manifest", type=Path)
    parser.add_argument("--full_task_tsv", type=Path)
    parser.add_argument("--full_manifest", type=Path)
    args = parser.parse_args()
    if args.freeze:
        required = (
            args.card, args.smoke_task_tsv, args.smoke_manifest,
            args.full_task_tsv, args.full_manifest,
        )
        if any(value is None for value in required):
            raise ValueError("Freeze mode requires card and both task tables/manifests")
        if args.lock.exists():
            raise FileExistsError(f"Refusing to overwrite {args.lock}")
        payload = build_lock(*required)  # type: ignore[arg-type]
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        args.lock.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        status = "frozen"
    else:
        payload = verify_source_lock(args.lock)
        status = "verified"
    print(
        json.dumps(
            {
                "status": status,
                "lock": str(args.lock),
                "lock_sha256": sha256_path(args.lock),
                "source_count": len(payload["sources"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
