"""Frozen protocol constants for the experiments reported in the paper.

This module deliberately contains only the small, final paper-facing contract.
Exploratory benchmark manifests remain separate so their defaults cannot drift
the system rosters, row identities, or training budgets reported in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class PaperBenchmarkProtocol:
    """Immutable training contract for one paper benchmark."""

    protocol_id: str
    system_keys: Tuple[str, ...]
    seeds: Tuple[int, ...]
    num_steps: int
    batch_size: int
    target_size: int
    sequence_length: int
    dt_multiplier: Optional[float] = None


@dataclass(frozen=True)
class PaperModelRow:
    """Map one displayed KAE row to each benchmark's task-table ID."""

    display_name: str
    controlled_variant: str
    dysts_variant: str


@dataclass(frozen=True)
class DystsPaperRowOverride:
    """A reviewed Dysts row exception outside the shared base recipe."""

    variant: str
    lista_num_loops: int
    lista_final_op: str
    source_campaign_system_count: int
    retained_paper_system_count: int


PAPER_MODEL_ROWS: Tuple[PaperModelRow, ...] = (
    PaperModelRow(
        "LISTA",
        "lista_dense_signsplit_p256_hardinit_basin_partition",
        "lista",
    ),
    PaperModelRow(
        "LISTA-BD",
        "lista_blockdiag_signsplit_hardinit_basin_partition",
        "lista_bd",
    ),
    PaperModelRow(
        "LISTA-SB",
        "lista_dense_softblock_signsplit_p256_hardinit_basin_partition",
        "lista_sb",
    ),
    PaperModelRow(
        "Sparse MLP, BD",
        "mlp_sparse_blockdiag_hardinit_basin_partition_control",
        "sparse_mlp_bd",
    ),
    PaperModelRow(
        "Sparse MLP",
        "mlp_sparse_hardinit_basin_partition_control",
        "sparse_mlp",
    ),
    PaperModelRow(
        "Dense MLP",
        "mlp_zero_sparse_hardinit_basin_partition_control",
        "dense_mlp_tanh",
    ),
)

CONTROLLED_MODEL_ROW_IDS: Tuple[str, ...] = tuple(
    row.controlled_variant for row in PAPER_MODEL_ROWS
)
DYSTS_MODEL_ROW_IDS: Tuple[str, ...] = tuple(
    row.dysts_variant for row in PAPER_MODEL_ROWS
)
PAPER_SEEDS: Tuple[int, ...] = tuple(range(15))

CONTROLLED_SYSTEM_DISPLAY_NAMES = {
    "gated_local_linear": "Local-linear gates",
    "gated_transfer_linear": "Transfer-gated local-linear",
    "claude:arrested_spiral": "Arrested spiral",
    "claude:cal_asymmetric_3": "Asymmetric three-well",
    "claude:cal_high_cross_3": "High-cross three-well",
    "claude:cal_hexagon_6": "Hexagonal six-well",
    "claude:cal_octagon_8": "Octagonal eight-well",
    "claude:cal_pentagon_5": "Pentagonal five-well",
    "claude:cal_square_4": "Square four-well",
    "claude:duffing_triple_well": "Triple-well Duffing",
    "claude:snic_multi": "SNIC multi-attractor",
    "claude:transition_routes_4": "Transition-routes four-well",
    "claude:var_depth_gradient_4": "Depth-gradient four-well",
    "claude:var_diamond_4": "Diamond four-well",
    "claude:var_l_shape_5": "L-shaped five-well",
}


def canonical_controlled_system_key(system_key: str) -> str:
    """Normalize collector keys to the canonical paper-protocol spelling."""

    if system_key.startswith("claude_"):
        return "claude:" + system_key.removeprefix("claude_")
    return system_key


def controlled_system_display_name(system_key: str) -> str:
    """Return the paper-facing name for one retained controlled system."""

    return CONTROLLED_SYSTEM_DISPLAY_NAMES[canonical_controlled_system_key(system_key)]

# LISTA-SB is a retained historical ablation: unlike LISTA and LISTA-BD, it
# uses two LISTA loops and a sign-split output. Independent models were run for
# 12 systems in its source campaign; the paper reports the retained 10 systems.
DYSTS_PAPER_ROW_OVERRIDES: Tuple[DystsPaperRowOverride, ...] = (
    DystsPaperRowOverride(
        variant="lista_sb",
        lista_num_loops=2,
        lista_final_op="sign_split",
        source_campaign_system_count=12,
        retained_paper_system_count=10,
    ),
)

CONTROLLED_PAPER_PROTOCOL = PaperBenchmarkProtocol(
    protocol_id="neurips_2026_controlled_multibasin_v1",
    system_keys=(
        "gated_local_linear",
        "gated_transfer_linear",
        "claude:arrested_spiral",
        "claude:cal_asymmetric_3",
        "claude:cal_high_cross_3",
        "claude:cal_hexagon_6",
        "claude:cal_octagon_8",
        "claude:cal_pentagon_5",
        "claude:cal_square_4",
        "claude:duffing_triple_well",
        "claude:snic_multi",
        "claude:transition_routes_4",
        "claude:var_depth_gradient_4",
        "claude:var_diamond_4",
        "claude:var_l_shape_5",
    ),
    seeds=PAPER_SEEDS,
    num_steps=200_000,
    batch_size=256,
    target_size=256,
    sequence_length=8,
)

# Recomputed from the frozen seed-42, 128 x 129-state label assignment.  Label
# construction is model-independent.  Triple-well Duffing's proxy collapses
# to one center, making conditional entropy identically zero for every model.
CONTROLLED_ALIGNMENT_OBSERVED_LABEL_COUNTS = {
    "gated_local_linear": 3,
    "gated_transfer_linear": 3,
    "claude:arrested_spiral": 5,
    "claude:cal_asymmetric_3": 3,
    "claude:cal_high_cross_3": 3,
    "claude:cal_hexagon_6": 6,
    "claude:cal_octagon_8": 8,
    "claude:cal_pentagon_5": 5,
    "claude:cal_square_4": 4,
    "claude:duffing_triple_well": 1,
    "claude:snic_multi": 3,
    "claude:transition_routes_4": 4,
    "claude:var_depth_gradient_4": 4,
    "claude:var_diamond_4": 4,
    "claude:var_l_shape_5": 5,
}
CONTROLLED_ALIGNMENT_ELIGIBILITY_CRITERION = (
    "at_least_two_observed_evaluation_labels_in_the_frozen_proxy_assignment"
)
CONTROLLED_ALIGNMENT_EXCLUDED_SYSTEM_KEYS: Tuple[str, ...] = tuple(
    key
    for key in CONTROLLED_PAPER_PROTOCOL.system_keys
    if CONTROLLED_ALIGNMENT_OBSERVED_LABEL_COUNTS[key] < 2
)
CONTROLLED_ALIGNMENT_PRIMARY_SYSTEM_KEYS: Tuple[str, ...] = tuple(
    key
    for key in CONTROLLED_PAPER_PROTOCOL.system_keys
    if key not in CONTROLLED_ALIGNMENT_EXCLUDED_SYSTEM_KEYS
)
CONTROLLED_ALIGNMENT_EXCLUDED_OBSERVED_LABEL_COUNTS = {
    key: CONTROLLED_ALIGNMENT_OBSERVED_LABEL_COUNTS[key]
    for key in CONTROLLED_ALIGNMENT_EXCLUDED_SYSTEM_KEYS
}

DYSTS_PAPER_PROTOCOL = PaperBenchmarkProtocol(
    protocol_id="neurips_2026_dysts_dt30_v1",
    system_keys=(
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
    ),
    seeds=PAPER_SEEDS,
    num_steps=100_000,
    batch_size=256,
    target_size=256,
    sequence_length=10,
    dt_multiplier=30.0,
)
