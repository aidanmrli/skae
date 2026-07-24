"""Validate the frozen periodic-reencoding card and transitive source manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.neurips_2026.allen_cahn_periodic_reencoding.io import (
    CARD_PATH,
    MANIFEST_PATH,
    checkpoint_specs,
    load_card,
    load_parent_card,
    verify_source_manifest,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.generator import (
    realized_rng_streams,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, default=CARD_PATH)
    parser.add_argument("--expected-card-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    card, card_hash = load_card(args.card, expected_sha256=args.expected_card_sha256)
    source_hash = verify_source_manifest(
        card,
        path=args.source_manifest,
        expected_sha256=args.expected_source_manifest_sha256,
    )
    parent = load_parent_card(card)
    specs = checkpoint_specs(card)
    if len(specs) != 20 or len({(spec.arm, spec.seed) for spec in specs}) != 20:
        raise RuntimeError("Checkpoint roster is incomplete")
    rng = realized_rng_streams(card)
    print(
        json.dumps(
            {
                "status": "passed",
                "card_sha256": card_hash,
                "source_manifest_sha256": source_hash,
                "parent_protocol_id": parent["protocol_id"],
                "checkpoint_count": len(specs),
                "rng_stream_count": rng["stream_count"],
                "scientific_outcomes_accessed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
