"""Tests for the Dysts dt-x30 basin-block task builder."""

from __future__ import annotations

from argparse import Namespace

from experiments.neurips_2026.workflows.dysts_tasks import (
    DYSTS_SYSTEM_SPECS,
    _build_parser,
    _build_rows,
)
from experiments.neurips_2026.workflows.dysts_cache import _apply_dysts_dt_multiplier
from skae.config import get_config


def _args() -> Namespace:
    return Namespace(
        phase_label="dysts_dt30_clean",
        output_tsv="unused.tsv",
        output_manifest_json=None,
        systems_csv="dysts:Chua",
        model_variants_csv="lista_bd,lista_sb",
        seeds_csv="0",
        num_steps=100000,
        batch_size=256,
        target_size=256,
        sequence_length=10,
        dt_multiplier=30.0,
        sparsity_coeff=0.006,
        dense_sparsity_coeff=0.0,
        generic_lr=1e-4,
        generic_k_matrix_lr=1e-5,
        lista_alpha=0.15,
        lista_sb_num_loops=2,
        lista_lr=5e-5,
        lista_k_matrix_lr=5e-6,
        weight_decay=1e-4,
        soft_block_weight=1e-4,
        soft_block_norm="l1",
        eval_profile="full",
        dysts_cache_profile="full",
        dysts_ic_noise_scale=0.2,
    )


def test_lista_sb_encodes_the_reported_row_specific_ablation():
    rows = _build_rows(_args())

    assert len(rows) == 2
    by_variant = {row["model_variant"]: row for row in rows}

    bd = by_variant["lista_bd"]
    sb = by_variant["lista_sb"]

    for field in [
        "config_name",
        "target_size",
        "sequence_length",
        "sparsity_coeff",
        "lista_alpha",
        "lr",
        "k_matrix_lr",
        "weight_decay",
    ]:
        assert sb[field] == bd[field]

    assert bd["lista_num_loops"] == 1
    assert bd["lista_final_op"] == "relu"
    assert bd["k_structure"] == "block_diagonal"
    assert bd["k_num_blocks"] == 2
    assert bd["soft_block"] == 0

    assert sb["k_structure"] == "dense"
    assert sb["k_num_blocks"] == ""
    assert sb["soft_block"] == 1
    assert sb["soft_block_num_blocks"] == 2
    assert sb["soft_block_weight"] == 1e-4
    assert sb["soft_block_norm"] == "l1"
    assert sb["lista_num_loops"] == 2
    assert sb["lista_final_op"] == "sign_split"
    assert bd["diagnostic_structure_count"] == 2
    assert "lobe" in bd["structure_count_note"]
    assert "basin_count" not in bd
    assert "block_count_note" not in bd


def test_default_arguments_are_the_exact_dysts_paper_protocol():
    args = _build_parser().parse_args(["--output_tsv", "unused.tsv"])
    rows = _build_rows(args)

    assert len(rows) == 10 * 6 * 15
    assert list(dict.fromkeys(row["system_key"] for row in rows)) == [
        "dysts:Chua",
        "dysts:Dadras",
        "dysts:DequanLi",
        "dysts:Hadley",
        "dysts:LuChenCheng",
        "dysts:QiChen",
        "dysts:Sakarya",
        "dysts:SanUmSrisuchinwong",
        "dysts:ShimizuMorioka",
        "dysts:WangSun",
    ]
    assert tuple(DYSTS_SYSTEM_SPECS) == tuple(
        dict.fromkeys(row["system_key"] for row in rows)
    )
    assert list(dict.fromkeys(row["model_variant"] for row in rows)) == [
        "lista",
        "lista_bd",
        "lista_sb",
        "sparse_mlp_bd",
        "sparse_mlp",
        "dense_mlp_tanh",
    ]
    assert {row["seed"] for row in rows} == set(range(15))
    assert {row["num_steps"] for row in rows} == {100_000}
    assert {row["sequence_length"] for row in rows} == {10}
    assert {row["dt_multiplier"] for row in rows} == {"30"}
    by_variant = {row["model_variant"]: row for row in rows}
    assert by_variant["lista"]["lista_num_loops"] == 1
    assert by_variant["lista"]["lista_final_op"] == "relu"
    assert by_variant["lista_bd"]["lista_num_loops"] == 1
    assert by_variant["lista_bd"]["lista_final_op"] == "relu"
    assert by_variant["lista_sb"]["lista_num_loops"] == 2
    assert by_variant["lista_sb"]["lista_final_op"] == "sign_split"


def test_lista_sb_can_match_the_one_refinement_lista_rows():
    args = _args()
    args.lista_sb_num_loops = 1
    rows = _build_rows(args)
    by_variant = {row["model_variant"]: row for row in rows}

    assert by_variant["lista_bd"]["lista_num_loops"] == 1
    assert by_variant["lista_sb"]["lista_num_loops"] == 1
    assert by_variant["lista_sb"]["lista_final_op"] == "sign_split"
    assert by_variant["lista_sb"]["soft_block"] == 1


def test_cache_prebuild_uses_the_exact_frozen_task_timestep():
    cfg = get_config("lista_nonlinear")
    cfg.ENV.ENV_NAME = "dysts:Chua"
    dt = _apply_dysts_dt_multiplier(cfg, 30.0)
    assert dt == float(DYSTS_SYSTEM_SPECS["dysts:Chua"]["base_dt"]) * 30.0
