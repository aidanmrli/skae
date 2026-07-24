"""Outcome-independent launch guard for the Allen--Cahn bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.neurips_2026.allen_cahn_mechanistic_bridge.io import (
    CARD_PATH,
    load_card,
    sha256_path,
)


def validate_mechanism_decision(
    decision: dict[str, Any], card: dict[str, Any]
) -> dict[str, Any]:
    contract = card["conditional_launch"]
    checks = {
        "mechanism_card": decision.get("card_sha256")
        == contract["required_card_sha256"],
        "mechanism_source": decision.get("source_manifest_sha256")
        == contract["required_source_manifest_sha256"],
        "validity": decision.get("validity", {}).get("passed") is True,
        "provenance_and_firewall": decision.get("validity", {}).get(
            "provenance_and_firewall"
        )
        is True,
        "train_fit_family_qualification": decision.get("family", {})
        .get("family_checks", {})
        .get("eligible_seeds")
        is True,
        "minimum_eligible_model_seeds": int(
            decision.get("family", {})
            .get("qualification", {})
            .get("eligible_seed_count", -1)
        )
        >= 8,
    }
    return {"passed": all(checks.values()), "checks": checks}


def load_and_validate(
    path: Path,
    *,
    expected_sha256: str,
    card: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if path != Path(card["conditional_launch"]["decision_path"]):
        raise RuntimeError("Mechanism decision path differs from frozen card")
    observed = sha256_path(path)
    if observed != expected_sha256:
        raise RuntimeError("Mechanism decision hash differs from launcher root")
    decision = json.loads(path.read_text())
    result = validate_mechanism_decision(decision, card)
    if not result["passed"]:
        raise RuntimeError(f"Conditional bridge launch failed: {result['checks']}")
    return decision, observed, result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected_decision_sha256", required=True)
    parser.add_argument("--card", type=Path, default=CARD_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    card, _ = load_card(args.card)
    _, observed, result = load_and_validate(
        args.decision, expected_sha256=args.expected_decision_sha256, card=card
    )
    print(json.dumps({"status": "passed", "decision_sha256": observed, **result}))


if __name__ == "__main__":
    main()
