"""Manifest for the fixed transition-rich basin-partition LISTA sweep."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

from skae.config import Config, get_env_dt


TRANSITION_RICH_BASIN_PARTITION_NUM_STEPS = 200_000
TRANSITION_RICH_BASIN_PARTITION_BATCH_SIZE = 256
TRANSITION_RICH_BASIN_PARTITION_TARGET_SIZE = 256
TRANSITION_RICH_BASIN_PARTITION_SEQUENCE_LENGTH = 8
TRANSITION_RICH_BASIN_PARTITION_SEEDS: Sequence[int] = (0, 1, 2)


@dataclass(frozen=True)
class TransitionRichBasinPartitionSystem:
    """Definition of one system in the fixed transition-rich shortlist."""

    system_key: str
    env_name: str
    system_group: str
    basin_count: int
    paper_role: str

    @property
    def system_slug(self) -> str:
        return self.system_key.replace(":", "_")


@dataclass(frozen=True)
class TransitionRichBasinPartitionModel:
    """Definition of one LISTA model variant in the transition-rich sweep."""

    variant: str
    config_name: str
    num_steps: int
    lr: Optional[float]
    k_matrix_lr: Optional[float]
    weight_decay: Optional[float]
    res_coeff: float
    reconst_coeff: float
    pred_coeff: float
    sparsity_coeff: float
    lista_alpha: float
    lista_num_loops: int
    lista_final_op: str
    k_structure: str
    use_basin_count_for_blocks: bool = False


TRANSITION_RICH_BASIN_PARTITION_SYSTEMS: Sequence[TransitionRichBasinPartitionSystem] = (
    TransitionRichBasinPartitionSystem(
        system_key="multiwell_strong_transition",
        env_name="multiwell_strong_transition",
        system_group="native",
        basin_count=5,
        paper_role="shared-corridor native toy",
    ),
    TransitionRichBasinPartitionSystem(
        system_key="gated_local_linear",
        env_name="gated_local_linear",
        system_group="native",
        basin_count=3,
        paper_role="clean mechanistic chart-switch positive",
    ),
    TransitionRichBasinPartitionSystem(
        system_key="gated_transfer_linear",
        env_name="gated_transfer_linear",
        system_group="native",
        basin_count=3,
        paper_role="explicit-transfer native stress test",
    ),
    TransitionRichBasinPartitionSystem(
        system_key="claude:arrested_spiral",
        env_name="claude:arrested_spiral",
        system_group="claude_catalog",
        basin_count=5,
        paper_role="spiral-to-capture Claude control",
    ),
    TransitionRichBasinPartitionSystem(
        system_key="claude:cal_asymmetric_3",
        env_name="claude:cal_asymmetric_3",
        system_group="claude_catalog",
        basin_count=3,
        paper_role="asymmetric 3-basin Claude control",
    ),
    TransitionRichBasinPartitionSystem(
        system_key="claude:cal_high_cross_3",
        env_name="claude:cal_high_cross_3",
        system_group="claude_catalog",
        basin_count=3,
        paper_role="high-crossing 3-basin Claude control",
    ),
    TransitionRichBasinPartitionSystem(
        system_key="claude:cal_hexagon_6",
        env_name="claude:cal_hexagon_6",
        system_group="claude_catalog",
        basin_count=6,
        paper_role="mid-high basin polygon control",
    ),
    TransitionRichBasinPartitionSystem(
        system_key="claude:cal_octagon_8",
        env_name="claude:cal_octagon_8",
        system_group="claude_catalog",
        basin_count=8,
        paper_role="high-basin polygon control",
    ),
    TransitionRichBasinPartitionSystem(
        system_key="claude:cal_pentagon_5",
        env_name="claude:cal_pentagon_5",
        system_group="claude_catalog",
        basin_count=5,
        paper_role="mid-count polygon control",
    ),
    TransitionRichBasinPartitionSystem(
        system_key="claude:cal_square_4",
        env_name="claude:cal_square_4",
        system_group="claude_catalog",
        basin_count=4,
        paper_role="clean four-basin baseline",
    ),
    TransitionRichBasinPartitionSystem(
        system_key="claude:checkerboard_potential",
        env_name="claude:checkerboard_potential",
        system_group="claude_catalog",
        basin_count=4,
        paper_role="alternating checkerboard routing control",
    ),
    TransitionRichBasinPartitionSystem(
        system_key="claude:duffing_triple_well",
        env_name="claude:duffing_triple_well",
        system_group="claude_catalog",
        basin_count=3,
        paper_role="triple-well Duffing control",
    ),
    TransitionRichBasinPartitionSystem(
        system_key="claude:snic_multi",
        env_name="claude:snic_multi",
        system_group="claude_catalog",
        basin_count=3,
        paper_role="non-multiwell mechanistic outlier",
    ),
    TransitionRichBasinPartitionSystem(
        system_key="claude:transition_routes_4",
        env_name="claude:transition_routes_4",
        system_group="claude_catalog",
        basin_count=4,
        paper_role="explicit route-choice benchmark",
    ),
    TransitionRichBasinPartitionSystem(
        system_key="claude:var_depth_gradient_4",
        env_name="claude:var_depth_gradient_4",
        system_group="claude_catalog",
        basin_count=4,
        paper_role="occupancy-skew stress test",
    ),
    TransitionRichBasinPartitionSystem(
        system_key="claude:var_diamond_4",
        env_name="claude:var_diamond_4",
        system_group="claude_catalog",
        basin_count=4,
        paper_role="rotated-separatrix geometry mismatch",
    ),
    TransitionRichBasinPartitionSystem(
        system_key="claude:var_l_shape_5",
        env_name="claude:var_l_shape_5",
        system_group="claude_catalog",
        basin_count=5,
        paper_role="non-convex geometry case",
    ),
)


TRANSITION_RICH_BASIN_PARTITION_MODELS: Sequence[TransitionRichBasinPartitionModel] = (
    TransitionRichBasinPartitionModel(
        variant="lista_dense_basin_partition",
        config_name="lista_parity_generic_sparse",
        num_steps=TRANSITION_RICH_BASIN_PARTITION_NUM_STEPS,
        lr=5e-5,
        k_matrix_lr=5e-6,
        weight_decay=1e-4,
        res_coeff=1.0,
        reconst_coeff=0.03,
        pred_coeff=1.0,
        sparsity_coeff=0.003,
        lista_alpha=0.15,
        lista_num_loops=1,
        lista_final_op="relu",
        k_structure="dense",
    ),
    TransitionRichBasinPartitionModel(
        variant="lista_blockdiag_basin_partition",
        config_name="lista_parity_generic_sparse",
        num_steps=TRANSITION_RICH_BASIN_PARTITION_NUM_STEPS,
        lr=5e-5,
        k_matrix_lr=5e-6,
        weight_decay=1e-4,
        res_coeff=1.0,
        reconst_coeff=0.03,
        pred_coeff=1.0,
        sparsity_coeff=0.003,
        lista_alpha=0.15,
        lista_num_loops=1,
        lista_final_op="relu",
        k_structure="block_diagonal",
        use_basin_count_for_blocks=True,
    ),
)


def transition_rich_basin_partition_systems() -> List[TransitionRichBasinPartitionSystem]:
    """Return the ordered fixed shortlist."""
    return list(TRANSITION_RICH_BASIN_PARTITION_SYSTEMS)


def transition_rich_basin_partition_models() -> List[TransitionRichBasinPartitionModel]:
    """Return the ordered LISTA variants for the shortlist."""
    return list(TRANSITION_RICH_BASIN_PARTITION_MODELS)


def get_transition_rich_basin_partition_system(system_key: str) -> TransitionRichBasinPartitionSystem:
    """Lookup a fixed-shortlist system by key."""
    for spec in TRANSITION_RICH_BASIN_PARTITION_SYSTEMS:
        if spec.system_key == system_key:
            return spec
    raise KeyError(f"Unknown transition-rich basin-partition system '{system_key}'")


def get_transition_rich_basin_partition_model(
    variant: str,
) -> TransitionRichBasinPartitionModel:
    """Lookup a LISTA sweep model variant by name."""
    for spec in TRANSITION_RICH_BASIN_PARTITION_MODELS:
        if spec.variant == variant:
            return spec
    raise KeyError(f"Unknown transition-rich basin-partition model '{variant}'")


def resolve_transition_rich_default_dt(system_key: str) -> float:
    """Resolve the configured default dt for a fixed-shortlist system."""
    spec = get_transition_rich_basin_partition_system(system_key)
    cfg = Config()
    cfg.ENV.ENV_NAME = spec.env_name
    return float(get_env_dt(cfg))


def get_transition_rich_basin_count(system_key: str) -> int:
    """Return the benchmark basin count for a fixed-shortlist system."""
    return int(get_transition_rich_basin_partition_system(system_key).basin_count)


def transition_rich_basin_partition_manifest_jsonable() -> Dict[str, object]:
    """Return a JSON-serializable snapshot of the fixed shortlist and recipe."""
    return {
        "num_steps": TRANSITION_RICH_BASIN_PARTITION_NUM_STEPS,
        "batch_size": TRANSITION_RICH_BASIN_PARTITION_BATCH_SIZE,
        "target_size": TRANSITION_RICH_BASIN_PARTITION_TARGET_SIZE,
        "sequence_length": TRANSITION_RICH_BASIN_PARTITION_SEQUENCE_LENGTH,
        "seeds": list(TRANSITION_RICH_BASIN_PARTITION_SEEDS),
        "systems": [
            {
                **asdict(spec),
                "system_slug": spec.system_slug,
                "resolved_default_dt": resolve_transition_rich_default_dt(spec.system_key),
            }
            for spec in TRANSITION_RICH_BASIN_PARTITION_SYSTEMS
        ],
        "models": [asdict(spec) for spec in TRANSITION_RICH_BASIN_PARTITION_MODELS],
    }
