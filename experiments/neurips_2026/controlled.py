"""Frozen controlled multibasin systems and model recipes for the paper.

Earlier exploratory sweeps have been retired. This module contains only the
15 systems and six KAE rows reported in the paper. Basin
counts are benchmark metadata used to size the two structured-transition
diagnostics and to score post-hoc alignment; they are never training labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

from experiments.neurips_2026.protocol import (
    CONTROLLED_MODEL_ROW_IDS,
    CONTROLLED_PAPER_PROTOCOL,
    PAPER_CONTROLLED_SYSTEMS,
)
from skae.config import Config, get_env_dt


CONTROLLED_NUM_STEPS = CONTROLLED_PAPER_PROTOCOL.num_steps
CONTROLLED_BATCH_SIZE = CONTROLLED_PAPER_PROTOCOL.batch_size
CONTROLLED_TARGET_SIZE = CONTROLLED_PAPER_PROTOCOL.target_size
CONTROLLED_SEQUENCE_LENGTH = CONTROLLED_PAPER_PROTOCOL.sequence_length
CONTROLLED_SEEDS: Sequence[int] = CONTROLLED_PAPER_PROTOCOL.seeds


@dataclass(frozen=True)
class ControlledSystem:
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
class ControlledModel:
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


CONTROLLED_SYSTEMS: Sequence[ControlledSystem] = tuple(
    ControlledSystem(
        system_key=system.system_key,
        env_name=system.system_key,
        system_group=(
            "native" if system.alignment_label_source == "native" else "analytic"
        ),
        basin_count=system.basin_count,
        paper_role=system.paper_role,
    )
    for system in PAPER_CONTROLLED_SYSTEMS
)


_LISTA_COMMON = {
    "config_name": "lista_parity_generic_sparse",
    "lista_alpha": 0.15,
    "lista_num_loops": 2,
    "lista_final_op": "sign_split",
}

CONTROLLED_MODELS: Sequence[ControlledModel] = (
    ControlledModel(
        variant="lista_dense_signsplit_p256_hardinit_basin_partition",
        **_LISTA_COMMON,
    ),
    ControlledModel(
        variant="lista_blockdiag_signsplit_hardinit_basin_partition",
        k_structure="block_diagonal",
        use_basin_count_for_blocks=True,
        **_LISTA_COMMON,
    ),
    ControlledModel(
        variant="lista_dense_softblock_signsplit_p256_hardinit_basin_partition",
        soft_block=True,
        use_basin_count_for_soft_block_num_blocks=True,
        soft_block_weight=1e-4,
        soft_block_norm="l1",
        **_LISTA_COMMON,
    ),
    ControlledModel(
        variant="mlp_sparse_blockdiag_hardinit_basin_partition_control",
        config_name="generic_sparse",
        k_structure="block_diagonal",
        use_basin_count_for_blocks=True,
    ),
    ControlledModel(
        variant="mlp_sparse_hardinit_basin_partition_control",
        config_name="generic_sparse",
    ),
    ControlledModel(
        variant="mlp_zero_sparse_hardinit_basin_partition_control",
        config_name="generic_no_shrink",
        sparsity_coeff=0.0,
    ),
)

if tuple(model.variant for model in CONTROLLED_MODELS) != (
    CONTROLLED_MODEL_ROW_IDS
):
    raise RuntimeError("Controlled model recipes do not match PAPER_MODEL_ROWS")
if tuple(system.system_key for system in CONTROLLED_SYSTEMS) != (
    CONTROLLED_PAPER_PROTOCOL.system_keys
):
    raise RuntimeError("Controlled system metadata does not match the paper protocol")


def controlled_systems() -> List[ControlledSystem]:
    """Return the ordered 15-system paper roster."""

    return list(CONTROLLED_SYSTEMS)


def controlled_models() -> List[ControlledModel]:
    """Return the ordered six-row paper roster."""

    return list(CONTROLLED_MODELS)


def get_controlled_system(system_key: str) -> ControlledSystem:
    """Look up one retained controlled system."""

    for spec in CONTROLLED_SYSTEMS:
        if spec.system_key == system_key:
            return spec
    raise KeyError(f"Unknown controlled paper system '{system_key}'")


def get_controlled_model(variant: str) -> ControlledModel:
    """Look up one retained controlled model row."""

    for spec in CONTROLLED_MODELS:
        if spec.variant == variant:
            return spec
    raise KeyError(f"Unknown controlled paper model '{variant}'")


def resolve_controlled_default_dt(system_key: str) -> float:
    """Resolve the configured observation timestep for a retained system."""

    spec = get_controlled_system(system_key)
    cfg = Config()
    cfg.ENV.ENV_NAME = spec.env_name
    return float(get_env_dt(cfg))


def get_controlled_basin_count(system_key: str) -> int:
    """Return the known basin count used by evaluation and block diagnostics."""

    return int(get_controlled_system(system_key).basin_count)


def controlled_manifest_jsonable() -> Dict[str, object]:
    """Return a JSON-serializable snapshot of the frozen paper protocol."""

    return {
        "protocol_id": CONTROLLED_PAPER_PROTOCOL.protocol_id,
        "num_steps": CONTROLLED_NUM_STEPS,
        "batch_size": CONTROLLED_BATCH_SIZE,
        "target_size": CONTROLLED_TARGET_SIZE,
        "sequence_length": CONTROLLED_SEQUENCE_LENGTH,
        "seeds": list(CONTROLLED_SEEDS),
        "systems": [
            {
                **asdict(spec),
                "system_slug": spec.system_slug,
                "resolved_default_dt": resolve_controlled_default_dt(spec.system_key),
            }
            for spec in CONTROLLED_SYSTEMS
        ],
        "models": [asdict(spec) for spec in CONTROLLED_MODELS],
    }


__all__ = [
    "CONTROLLED_NUM_STEPS",
    "CONTROLLED_BATCH_SIZE",
    "CONTROLLED_TARGET_SIZE",
    "CONTROLLED_SEQUENCE_LENGTH",
    "CONTROLLED_SEEDS",
    "ControlledSystem",
    "ControlledModel",
    "CONTROLLED_SYSTEMS",
    "CONTROLLED_MODELS",
    "controlled_systems",
    "controlled_models",
    "get_controlled_system",
    "get_controlled_model",
    "resolve_controlled_default_dt",
    "get_controlled_basin_count",
    "controlled_manifest_jsonable",
]
