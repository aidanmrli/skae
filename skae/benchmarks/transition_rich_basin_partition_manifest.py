"""Paper-facing controlled multibasin systems and model recipes.

The exploratory transition-rich sweep has been retired.  This module now
contains only the 15 systems and six KAE rows reported in the paper.  Basin
counts are benchmark metadata used to size the two structured-transition
diagnostics and to score post-hoc alignment; they are never training labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

from skae.benchmarks.paper_protocol import (
    CONTROLLED_MODEL_ROW_IDS,
    CONTROLLED_PAPER_PROTOCOL,
)
from skae.config import Config, get_env_dt


TRANSITION_RICH_BASIN_PARTITION_NUM_STEPS = CONTROLLED_PAPER_PROTOCOL.num_steps
TRANSITION_RICH_BASIN_PARTITION_BATCH_SIZE = CONTROLLED_PAPER_PROTOCOL.batch_size
TRANSITION_RICH_BASIN_PARTITION_TARGET_SIZE = CONTROLLED_PAPER_PROTOCOL.target_size
TRANSITION_RICH_BASIN_PARTITION_SEQUENCE_LENGTH = (
    CONTROLLED_PAPER_PROTOCOL.sequence_length
)
TRANSITION_RICH_BASIN_PARTITION_SEEDS: Sequence[int] = (
    CONTROLLED_PAPER_PROTOCOL.seeds
)


@dataclass(frozen=True)
class TransitionRichBasinPartitionSystem:
    """One retained controlled benchmark system."""

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
    """One retained controlled KAE recipe."""

    variant: str
    config_name: str
    num_steps: int = CONTROLLED_PAPER_PROTOCOL.num_steps
    target_size: int = CONTROLLED_PAPER_PROTOCOL.target_size
    lr: float = 5e-5
    k_matrix_lr: float = 5e-6
    weight_decay: float = 1e-4
    res_coeff: float = 1.0
    reconst_coeff: float = 0.03
    pred_coeff: float = 1.0
    sparsity_coeff: float = 0.003
    k_structure: str = "dense"
    hard_init_oversample: bool = True
    hard_init_fraction: float = 0.5
    hard_init_pool_size: int = 1024
    hard_init_num_candidates: int = 4096
    hard_init_probe_steps: int = 32
    hard_init_num_perturbations: int = 4
    hard_init_perturb_scale: float = 0.04
    hard_init_transient_window: int = 8
    hard_init_transient_weight: float = 0.5
    hard_init_jitter_scale: float = 0.25
    lista_alpha: Optional[float] = None
    lista_num_loops: Optional[int] = None
    lista_final_op: Optional[str] = None
    use_basin_count_for_blocks: bool = False
    soft_block: bool = False
    use_basin_count_for_soft_block_num_blocks: bool = False
    soft_block_weight: Optional[float] = None
    soft_block_norm: Optional[str] = None


def _system(
    system_key: str,
    basin_count: int,
    paper_role: str,
) -> TransitionRichBasinPartitionSystem:
    group = "native" if not system_key.startswith("claude:") else "claude_catalog"
    return TransitionRichBasinPartitionSystem(
        system_key=system_key,
        env_name=system_key,
        system_group=group,
        basin_count=basin_count,
        paper_role=paper_role,
    )


TRANSITION_RICH_BASIN_PARTITION_SYSTEMS: Sequence[
    TransitionRichBasinPartitionSystem
] = (
    _system("gated_local_linear", 3, "clean mechanistic chart-switch positive"),
    _system("gated_transfer_linear", 3, "explicit-transfer native stress test"),
    _system("claude:arrested_spiral", 5, "spiral-to-capture control"),
    _system("claude:cal_asymmetric_3", 3, "asymmetric three-basin control"),
    _system("claude:cal_high_cross_3", 3, "high-crossing three-basin control"),
    _system("claude:cal_hexagon_6", 6, "mid-high basin polygon control"),
    _system("claude:cal_octagon_8", 8, "high-basin polygon control"),
    _system("claude:cal_pentagon_5", 5, "mid-count polygon control"),
    _system("claude:cal_square_4", 4, "clean four-basin baseline"),
    _system("claude:duffing_triple_well", 3, "triple-well Duffing control"),
    _system("claude:snic_multi", 3, "non-multiwell mechanistic outlier"),
    _system("claude:transition_routes_4", 4, "explicit route-choice benchmark"),
    _system("claude:var_depth_gradient_4", 4, "occupancy-skew stress test"),
    _system("claude:var_diamond_4", 4, "rotated-separatrix geometry mismatch"),
    _system("claude:var_l_shape_5", 5, "non-convex geometry case"),
)


_LISTA_COMMON = {
    "config_name": "lista_parity_generic_sparse",
    "lista_alpha": 0.15,
    "lista_num_loops": 2,
    "lista_final_op": "sign_split",
}

TRANSITION_RICH_BASIN_PARTITION_MODELS: Sequence[
    TransitionRichBasinPartitionModel
] = (
    TransitionRichBasinPartitionModel(
        variant="lista_dense_signsplit_p256_hardinit_basin_partition",
        **_LISTA_COMMON,
    ),
    TransitionRichBasinPartitionModel(
        variant="lista_blockdiag_signsplit_hardinit_basin_partition",
        k_structure="block_diagonal",
        use_basin_count_for_blocks=True,
        **_LISTA_COMMON,
    ),
    TransitionRichBasinPartitionModel(
        variant="lista_dense_softblock_signsplit_p256_hardinit_basin_partition",
        soft_block=True,
        use_basin_count_for_soft_block_num_blocks=True,
        soft_block_weight=1e-4,
        soft_block_norm="l1",
        **_LISTA_COMMON,
    ),
    TransitionRichBasinPartitionModel(
        variant="mlp_sparse_blockdiag_hardinit_basin_partition_control",
        config_name="generic_sparse",
        k_structure="block_diagonal",
        use_basin_count_for_blocks=True,
    ),
    TransitionRichBasinPartitionModel(
        variant="mlp_sparse_hardinit_basin_partition_control",
        config_name="generic_sparse",
    ),
    TransitionRichBasinPartitionModel(
        variant="mlp_zero_sparse_hardinit_basin_partition_control",
        config_name="generic_no_shrink",
        sparsity_coeff=0.0,
    ),
)

if tuple(model.variant for model in TRANSITION_RICH_BASIN_PARTITION_MODELS) != (
    CONTROLLED_MODEL_ROW_IDS
):
    raise RuntimeError("Controlled model recipes do not match PAPER_MODEL_ROWS")
if tuple(system.system_key for system in TRANSITION_RICH_BASIN_PARTITION_SYSTEMS) != (
    CONTROLLED_PAPER_PROTOCOL.system_keys
):
    raise RuntimeError("Controlled system metadata does not match the paper protocol")


def transition_rich_basin_partition_systems() -> List[
    TransitionRichBasinPartitionSystem
]:
    """Return the ordered 15-system paper roster."""

    return list(TRANSITION_RICH_BASIN_PARTITION_SYSTEMS)


def transition_rich_basin_partition_models() -> List[
    TransitionRichBasinPartitionModel
]:
    """Return the ordered six-row paper roster."""

    return list(TRANSITION_RICH_BASIN_PARTITION_MODELS)


def get_transition_rich_basin_partition_system(
    system_key: str,
) -> TransitionRichBasinPartitionSystem:
    """Look up one retained controlled system."""

    for spec in TRANSITION_RICH_BASIN_PARTITION_SYSTEMS:
        if spec.system_key == system_key:
            return spec
    raise KeyError(f"Unknown controlled paper system '{system_key}'")


def get_transition_rich_basin_partition_model(
    variant: str,
) -> TransitionRichBasinPartitionModel:
    """Look up one retained controlled model row."""

    for spec in TRANSITION_RICH_BASIN_PARTITION_MODELS:
        if spec.variant == variant:
            return spec
    raise KeyError(f"Unknown controlled paper model '{variant}'")


def resolve_transition_rich_default_dt(system_key: str) -> float:
    """Resolve the configured observation timestep for a retained system."""

    spec = get_transition_rich_basin_partition_system(system_key)
    cfg = Config()
    cfg.ENV.ENV_NAME = spec.env_name
    return float(get_env_dt(cfg))


def get_transition_rich_basin_count(system_key: str) -> int:
    """Return the known basin count used by evaluation and block diagnostics."""

    return int(get_transition_rich_basin_partition_system(system_key).basin_count)


def transition_rich_basin_partition_manifest_jsonable() -> Dict[str, object]:
    """Return a JSON-serializable snapshot of the frozen paper protocol."""

    return {
        "protocol_id": CONTROLLED_PAPER_PROTOCOL.protocol_id,
        "num_steps": TRANSITION_RICH_BASIN_PARTITION_NUM_STEPS,
        "batch_size": TRANSITION_RICH_BASIN_PARTITION_BATCH_SIZE,
        "target_size": TRANSITION_RICH_BASIN_PARTITION_TARGET_SIZE,
        "sequence_length": TRANSITION_RICH_BASIN_PARTITION_SEQUENCE_LENGTH,
        "seeds": list(TRANSITION_RICH_BASIN_PARTITION_SEEDS),
        "systems": [
            {
                **asdict(spec),
                "system_slug": spec.system_slug,
                "resolved_default_dt": resolve_transition_rich_default_dt(
                    spec.system_key
                ),
            }
            for spec in TRANSITION_RICH_BASIN_PARTITION_SYSTEMS
        ],
        "models": [asdict(spec) for spec in TRANSITION_RICH_BASIN_PARTITION_MODELS],
    }
