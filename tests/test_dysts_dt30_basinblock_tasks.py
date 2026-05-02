"""Tests for the Dysts dt-x30 basin-block task builder."""

from __future__ import annotations

from argparse import Namespace

from tools.build_dysts_dt30_basinblock_tasks import _build_rows


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
        lista_lr=5e-5,
        lista_k_matrix_lr=5e-6,
        weight_decay=1e-4,
        soft_block_weight=1e-4,
        soft_block_norm="l1",
        eval_profile="full",
        dysts_cache_profile="full",
        dysts_ic_noise_scale=0.2,
    )


def test_lista_sb_matches_lista_bd_encoder_and_only_softens_k_structure():
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
        "lista_num_loops",
        "lista_final_op",
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
