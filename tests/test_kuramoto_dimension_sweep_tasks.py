"""Tests for Kuramoto dimension-sweep task generation."""

from __future__ import annotations

from argparse import Namespace
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.build_kuramoto_dimension_sweep_tasks import _build_rows


def test_kuramoto_dimension_sweep_builds_expected_grid():
    args = Namespace(
        phase_label="kuramoto_dimension_sweep_dt00625_200k",
        dimensions_csv="8,16",
        model_variants_csv="generic_sparse,lista_dense_promoted,lista_blockdiag",
        seeds_csv="0,1",
        num_steps=200000,
        env_dt=0.00625,
        eval_profile="full",
    )

    rows = _build_rows(args)

    assert len(rows) == 2 * 3 * 2
    assert {row["kuramoto_num_oscillators"] for row in rows} == {8, 16}
    assert {row["model_variant"] for row in rows} == {
        "generic_sparse",
        "lista_dense_promoted",
        "lista_blockdiag",
    }
    dense_rows = [row for row in rows if row["model_variant"] == "lista_dense_promoted"]
    assert {row["sparsity_coeff"] for row in dense_rows} == {0.003}
    assert {row["lr"] for row in dense_rows} == {5e-5}
    assert {row["k_matrix_lr"] for row in dense_rows} == {5e-6}
    assert {row["weight_decay"] for row in dense_rows} == {1e-4}


def test_kuramoto_dimension_sweep_keeps_benchmark_defaults_for_generic_and_blockdiag():
    args = Namespace(
        phase_label="kuramoto_dimension_sweep_dt00625_200k",
        dimensions_csv="32",
        model_variants_csv="generic_sparse,lista_blockdiag",
        seeds_csv="0",
        num_steps=200000,
        env_dt=0.00625,
        eval_profile="full",
    )

    rows = _build_rows(args)

    generic = next(row for row in rows if row["model_variant"] == "generic_sparse")
    blockdiag = next(row for row in rows if row["model_variant"] == "lista_blockdiag")

    assert generic["lr"] == ""
    assert generic["sparsity_coeff"] == 0.0025
    assert blockdiag["k_structure"] == "block_diagonal"
    assert blockdiag["k_block_size"] == 16
    assert blockdiag["sparsity_coeff"] == 0.006
