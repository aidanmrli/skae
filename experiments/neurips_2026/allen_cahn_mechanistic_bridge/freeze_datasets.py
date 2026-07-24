"""Hash generated datasets without opening their semantic contents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.neurips_2026.allen_cahn_mechanistic_bridge.conditional_guard import (
    load_and_validate,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.integrity import (
    verify_source_manifest,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.generation_telemetry import (
    GENERATION_TELEMETRY_CHECK_KEYS,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.io import (
    CARD_PATH,
    load_card,
    sha256_path,
    write_json_once,
)


SOURCE_MANIFEST = Path(__file__).with_name("source_manifest.sha256")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected_decision_sha256", required=True)
    parser.add_argument("--expected_card_sha256", required=True)
    parser.add_argument("--expected_source_manifest_sha256", required=True)
    parser.add_argument("--card", type=Path, default=CARD_PATH)
    return parser.parse_args()


def build_manifest(
    card: dict,
    *,
    card_hash: str,
    source_hash: str,
    decision_hash: str,
) -> dict:
    root = Path(card["new_datasets"]["output_root"])
    records = []
    for seed, relative in zip(
        card["new_datasets"]["seeds"], card["new_datasets"]["paths"]
    ):
        path = root / relative
        summary_path = Path(str(path) + ".summary.json")
        telemetry_path = root / "telemetry" / f"generate_seed_{seed}.json"
        if not path.is_file() or not summary_path.is_file() or not telemetry_path.is_file():
            raise FileNotFoundError(path)
        summary = json.loads(summary_path.read_text())
        telemetry = json.loads(telemetry_path.read_text())
        observed = sha256_path(path)
        if (
            int(summary.get("seed", -1)) != int(seed)
            or Path(summary.get("path", "")) != path
            or summary.get("sha256") != observed
            or summary.get("card_sha256") != card_hash
            or summary.get("source_manifest_sha256") != source_hash
            or summary.get("mechanism_decision_sha256") != decision_hash
            or summary.get("semantic_outcomes_printed") is not False
        ):
            raise RuntimeError(f"Generation record failed integrity checks: {summary_path}")
        checks = telemetry.get("checks")
        if (
            telemetry.get("status") != "passed"
            or int(telemetry.get("seed", -1)) != int(seed)
            or telemetry.get("card_sha256") != card_hash
            or telemetry.get("source_manifest_sha256") != source_hash
            or telemetry.get("dataset_sha256") != observed
            or telemetry.get("generation_record_sha256") != sha256_path(summary_path)
            or str(telemetry.get("slurm_job_id")) != str(summary.get("slurm_job_id"))
            or not isinstance(checks, dict)
            or set(checks) != GENERATION_TELEMETRY_CHECK_KEYS
            or not checks
            or not all(value is True for value in checks.values())
        ):
            raise RuntimeError(f"Generation telemetry failed: {telemetry_path}")
        raw_name = str(telemetry.get("raw_telemetry_filename", ""))
        raw_path = root / "telemetry" / raw_name
        if Path(raw_name).name != raw_name or not raw_path.is_file():
            raise RuntimeError(f"Unsafe generation telemetry path: {raw_name}")
        if sha256_path(raw_path) != telemetry.get("raw_telemetry_sha256"):
            raise RuntimeError(f"Generation raw telemetry hash failed: {raw_path}")
        start_path = root / "telemetry" / f"generate_seed_{seed}.start"
        done_path = root / "telemetry" / f"generate_seed_{seed}.done"
        if (
            telemetry.get("gpu_start_marker_filename") != start_path.name
            or telemetry.get("gpu_done_marker_filename") != done_path.name
            or not start_path.is_file()
            or not done_path.is_file()
            or telemetry.get("gpu_start_marker_sha256") != sha256_path(start_path)
            or telemetry.get("gpu_done_marker_sha256") != sha256_path(done_path)
        ):
            raise RuntimeError(f"Generation scope markers failed: {telemetry_path}")
        records.append(
            {
                "seed": int(seed),
                "path": str(path),
                "sha256": observed,
                "summary_path": str(summary_path),
                "summary_sha256": sha256_path(summary_path),
                "telemetry_path": str(telemetry_path),
                "telemetry_sha256": sha256_path(telemetry_path),
                "raw_telemetry_path": str(raw_path),
                "raw_telemetry_sha256": telemetry["raw_telemetry_sha256"],
                "gpu_start_marker_path": str(start_path),
                "gpu_start_marker_sha256": telemetry["gpu_start_marker_sha256"],
                "gpu_done_marker_path": str(done_path),
                "gpu_done_marker_sha256": telemetry["gpu_done_marker_sha256"],
                "slurm_job_id": summary["slurm_job_id"],
            }
        )
    return {
        "schema_version": 1,
        "status": "frozen_before_semantic_access",
        "semantic_dataset_access": False,
        "generation_gpu_telemetry_passed": True,
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "mechanism_decision_sha256": decision_hash,
        "datasets": records,
    }


def main() -> None:
    args = parse_args()
    card, card_hash = load_card(args.card)
    if card_hash != args.expected_card_sha256:
        raise RuntimeError("Bridge card differs from launcher root")
    source_hash = verify_source_manifest(SOURCE_MANIFEST)
    if source_hash != args.expected_source_manifest_sha256:
        raise RuntimeError("Bridge source manifest differs from launcher root")
    _, decision_hash, _ = load_and_validate(
        args.decision,
        expected_sha256=args.expected_decision_sha256,
        card=card,
    )
    payload = build_manifest(
        card,
        card_hash=card_hash,
        source_hash=source_hash,
        decision_hash=decision_hash,
    )
    write_json_once(args.output, payload)
    print(json.dumps({"status": payload["status"], "sha256": sha256_path(args.output)}))


if __name__ == "__main__":
    main()
