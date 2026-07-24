"""Validate the exact card and executable/transitive source root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.neurips_2026.allen_cahn_forecast_replication.io import (
    CARD_PATH,
    MANIFEST_PATH,
    assert_runtime_values_safe,
    load_card,
    verify_source_manifest,
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
    assert_runtime_values_safe(
        [
            args.card,
            args.source_manifest,
            args.expected_card_sha256,
            args.expected_source_manifest_sha256,
        ]
    )
    card, card_hash = load_card(args.card, expected_sha256=args.expected_card_sha256)
    manifest_hash = verify_source_manifest(
        card,
        path=args.source_manifest,
        expected_sha256=args.expected_source_manifest_sha256,
    )
    print(
        json.dumps(
            {
                "status": "source_lock_passed",
                "card_sha256": card_hash,
                "source_manifest_sha256": manifest_hash,
                "scientific_outcomes_accessed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
