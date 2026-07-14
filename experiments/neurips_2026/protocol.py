"""Frozen protocol constants for the experiments reported in the NeurIPS paper.

This module deliberately contains only the small, final paper-facing contract.
Exploratory benchmark manifests remain separate so their defaults cannot drift
the system rosters, row identities, or training budgets reported in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


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


@dataclass(frozen=True)
class ControlledSystemContract:
    """Paper-facing metadata for one retained controlled benchmark system."""

    system_key: str
    display_name: str
    basin_count: int
    paper_role: str
    alignment_label_source: Literal["native", "proxy"]


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
CONTROLLED_MODEL_DISPLAY_NAMES = {
    row.controlled_variant: row.display_name for row in PAPER_MODEL_ROWS
}
DYSTS_MODEL_DISPLAY_NAMES = {
    row.dysts_variant: row.display_name for row in PAPER_MODEL_ROWS
}
# The retained Dysts appendix historically uses a hyphenated compact label.
DYSTS_MODEL_DISPLAY_NAMES["sparse_mlp_bd"] = "Sparse MLP-BD"
PAPER_SEEDS: Tuple[int, ...] = tuple(range(15))

PAPER_CONTROLLED_SYSTEMS: Tuple[ControlledSystemContract, ...] = (
    ControlledSystemContract(
        "gated_local_linear",
        "Local-linear gates",
        3,
        "clean mechanistic chart-switch positive",
        "native",
    ),
    ControlledSystemContract(
        "gated_transfer_linear",
        "Transfer-gated local-linear",
        3,
        "explicit-transfer native stress test",
        "native",
    ),
    ControlledSystemContract(
        "claude:arrested_spiral",
        "Arrested spiral",
        5,
        "spiral-to-capture control",
        "proxy",
    ),
    ControlledSystemContract(
        "claude:cal_asymmetric_3",
        "Asymmetric three-well",
        3,
        "asymmetric three-basin control",
        "proxy",
    ),
    ControlledSystemContract(
        "claude:cal_high_cross_3",
        "High-cross three-well",
        3,
        "high-crossing three-basin control",
        "proxy",
    ),
    ControlledSystemContract(
        "claude:cal_hexagon_6",
        "Hexagonal six-well",
        6,
        "mid-high basin polygon control",
        "proxy",
    ),
    ControlledSystemContract(
        "claude:cal_octagon_8",
        "Octagonal eight-well",
        8,
        "high-basin polygon control",
        "proxy",
    ),
    ControlledSystemContract(
        "claude:cal_pentagon_5",
        "Pentagonal five-well",
        5,
        "mid-count polygon control",
        "proxy",
    ),
    ControlledSystemContract(
        "claude:cal_square_4",
        "Square four-well",
        4,
        "clean four-basin baseline",
        "proxy",
    ),
    ControlledSystemContract(
        "claude:duffing_triple_well",
        "Triple-well Duffing",
        3,
        "triple-well Duffing control",
        "proxy",
    ),
    ControlledSystemContract(
        "claude:snic_multi",
        "SNIC multi-attractor",
        3,
        "non-multiwell mechanistic outlier",
        "proxy",
    ),
    ControlledSystemContract(
        "claude:transition_routes_4",
        "Transition-routes four-well",
        4,
        "explicit route-choice benchmark",
        "proxy",
    ),
    ControlledSystemContract(
        "claude:var_depth_gradient_4",
        "Depth-gradient four-well",
        4,
        "occupancy-skew stress test",
        "proxy",
    ),
    ControlledSystemContract(
        "claude:var_diamond_4",
        "Diamond four-well",
        4,
        "rotated-separatrix geometry mismatch",
        "proxy",
    ),
    ControlledSystemContract(
        "claude:var_l_shape_5",
        "L-shaped five-well",
        5,
        "non-convex geometry case",
        "proxy",
    ),
)
CONTROLLED_SYSTEM_DISPLAY_NAMES = {
    system.system_key: system.display_name for system in PAPER_CONTROLLED_SYSTEMS
}
CONTROLLED_BENCHMARK_BASIN_COUNTS = {
    system.system_key: system.basin_count for system in PAPER_CONTROLLED_SYSTEMS
}
CONTROLLED_NATIVE_LABEL_SYSTEM_KEYS: Tuple[str, ...] = tuple(
    system.system_key
    for system in PAPER_CONTROLLED_SYSTEMS
    if system.alignment_label_source == "native"
)
CONTROLLED_PROXY_LABEL_SYSTEM_KEYS: Tuple[str, ...] = tuple(
    system.system_key
    for system in PAPER_CONTROLLED_SYSTEMS
    if system.alignment_label_source == "proxy"
)

CLASSICAL_BASELINE_METHOD_IDS: Tuple[str, ...] = (
    "dmd",
    "edmd_poly",
    "rbf_dictionary_edmd",
)
LOCAL_LINEAR_BASELINE_METHOD_IDS: Tuple[str, ...] = (
    "kmeans_hard",
    "gmm_hard",
    "gmm_soft",
)
STANDALONE_BASELINE_METHOD_IDS: Tuple[str, ...] = (
    *CLASSICAL_BASELINE_METHOD_IDS,
    *LOCAL_LINEAR_BASELINE_METHOD_IDS,
)
STANDALONE_BASELINE_SEEDS: Tuple[int, ...] = (0, 1, 2)


def canonical_controlled_system_key(system_key: str) -> str:
    """Normalize collector keys to the canonical paper-protocol spelling."""

    if system_key.startswith("claude_"):
        return "claude:" + system_key.removeprefix("claude_")
    return system_key


def controlled_system_display_name(system_key: str) -> str:
    """Return the paper-facing name for one retained controlled system."""

    return CONTROLLED_SYSTEM_DISPLAY_NAMES[canonical_controlled_system_key(system_key)]


def model_display_name(variant: str) -> str:
    """Return the paper label for either benchmark's artifact-level row ID."""

    if variant in CONTROLLED_MODEL_DISPLAY_NAMES:
        return CONTROLLED_MODEL_DISPLAY_NAMES[variant]
    if variant in DYSTS_MODEL_DISPLAY_NAMES:
        return DYSTS_MODEL_DISPLAY_NAMES[variant]
    raise KeyError(f"Unknown paper model variant {variant!r}")

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
    system_keys=tuple(system.system_key for system in PAPER_CONTROLLED_SYSTEMS),
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


__all__ = [
    "PaperBenchmarkProtocol",
    "PaperModelRow",
    "DystsPaperRowOverride",
    "ControlledSystemContract",
    "PAPER_MODEL_ROWS",
    "CONTROLLED_MODEL_ROW_IDS",
    "DYSTS_MODEL_ROW_IDS",
    "CONTROLLED_MODEL_DISPLAY_NAMES",
    "DYSTS_MODEL_DISPLAY_NAMES",
    "PAPER_SEEDS",
    "PAPER_CONTROLLED_SYSTEMS",
    "CONTROLLED_SYSTEM_DISPLAY_NAMES",
    "CONTROLLED_BENCHMARK_BASIN_COUNTS",
    "CONTROLLED_NATIVE_LABEL_SYSTEM_KEYS",
    "CONTROLLED_PROXY_LABEL_SYSTEM_KEYS",
    "CLASSICAL_BASELINE_METHOD_IDS",
    "LOCAL_LINEAR_BASELINE_METHOD_IDS",
    "STANDALONE_BASELINE_METHOD_IDS",
    "STANDALONE_BASELINE_SEEDS",
    "canonical_controlled_system_key",
    "controlled_system_display_name",
    "model_display_name",
    "DYSTS_PAPER_ROW_OVERRIDES",
    "CONTROLLED_PAPER_PROTOCOL",
    "CONTROLLED_ALIGNMENT_OBSERVED_LABEL_COUNTS",
    "CONTROLLED_ALIGNMENT_ELIGIBILITY_CRITERION",
    "CONTROLLED_ALIGNMENT_EXCLUDED_SYSTEM_KEYS",
    "CONTROLLED_ALIGNMENT_PRIMARY_SYSTEM_KEYS",
    "CONTROLLED_ALIGNMENT_EXCLUDED_OBSERVED_LABEL_COUNTS",
    "DYSTS_PAPER_PROTOCOL",
]
