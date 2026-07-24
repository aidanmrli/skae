"""Tests for the portable dense-specificity assertion contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.neurips_2026.evidence.global_k_dense_specificity_portable import (
    ASSERTION_CONTRACT,
    validate_assertion_contract,
)


def test_dense_specificity_assertion_contract_accepts_negative_facts() -> None:
    validate_assertion_contract(dict(ASSERTION_CONTRACT), Path("task_000.json"))


@pytest.mark.parametrize("key", sorted(ASSERTION_CONTRACT))
def test_dense_specificity_assertion_contract_rejects_inverted_polarity(key: str) -> None:
    assertions = dict(ASSERTION_CONTRACT)
    assertions[key] = not assertions[key]
    with pytest.raises(RuntimeError, match="polarity mismatch"):
        validate_assertion_contract(assertions, Path("shard.json"))


def test_dense_specificity_assertion_contract_rejects_missing_or_extra_keys() -> None:
    missing = dict(ASSERTION_CONTRACT)
    missing.pop("basin_labels_or_counts_used")
    with pytest.raises(RuntimeError, match="schema mismatch"):
        validate_assertion_contract(missing, Path("missing.json"))

    extra = {**ASSERTION_CONTRACT, "unknown_assertion": True}
    with pytest.raises(RuntimeError, match="schema mismatch"):
        validate_assertion_contract(extra, Path("extra.json"))


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_dense_specificity_assertion_contract_rejects_non_boolean_values(
    value: object,
) -> None:
    assertions = dict(ASSERTION_CONTRACT)
    assertions["global_k_unmodified"] = value
    with pytest.raises(RuntimeError, match="not boolean"):
        validate_assertion_contract(assertions, Path("typed.json"))


def test_dense_specificity_frozen_source_lock_digest_is_unchanged() -> None:
    source_lock = Path(
        "experiments/neurips_2026/global_k_dense_specificity_source_lock.json"
    )
    from experiments.neurips_2026.evidence.global_k_dense_specificity import (
        sha256_path,
    )

    assert sha256_path(source_lock) == (
        "cc2c37f431ee0a577265b5c36698e249d4b4f30354c1722ebe71511e7d05dd9f"
    )
