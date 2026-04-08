"""Manifest for the first Claude catalog training packet.

This packet is intentionally small. It operationalizes the paper-facing
recommendation from the Claude catalog handoff:

- start with the strict six-system core
- use the same three model families already used for the paper branch
- keep the `200k`, `target_size=256`, `sequence_length=8`, `seed in {0,1,2}`
  recipe fixed
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

from skae.config import Config, get_env_dt


CLAUDE_CATALOG_PACKET_NUM_STEPS = 200_000
CLAUDE_CATALOG_PACKET_BATCH_SIZE = 256
CLAUDE_CATALOG_PACKET_TARGET_SIZE = 256
CLAUDE_CATALOG_PACKET_SEQUENCE_LENGTH = 8
CLAUDE_CATALOG_PACKET_SEEDS: Sequence[int] = (0, 1, 2)


@dataclass(frozen=True)
class ClaudeCatalogPacketSystem:
    """Definition of a single Claude-catalog packet environment."""

    system_key: str
    env_name: str
    system_group: str
    paper_role: str

    @property
    def system_slug(self) -> str:
        return self.system_key.replace(":", "_")


@dataclass(frozen=True)
class ClaudeCatalogPacketModel:
    """Definition of a single model variant for the Claude packet."""

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
    lista_alpha: Optional[float] = None
    lista_num_loops: Optional[int] = None
    lista_final_op: Optional[str] = None
    k_structure: Optional[str] = None
    k_block_size: Optional[int] = None


CLAUDE_CATALOG_PACKET_STRICT_SYSTEMS: Sequence[ClaudeCatalogPacketSystem] = (
    ClaudeCatalogPacketSystem(
        system_key="claude:cal_triangle_3",
        env_name="claude:cal_triangle_3",
        system_group="strict_core",
        paper_role="minimal symmetric control",
    ),
    ClaudeCatalogPacketSystem(
        system_key="claude:cal_pentagon_5",
        env_name="claude:cal_pentagon_5",
        system_group="strict_core",
        paper_role="mid-count polygon control",
    ),
    ClaudeCatalogPacketSystem(
        system_key="claude:var_depth_gradient_4",
        env_name="claude:var_depth_gradient_4",
        system_group="strict_core",
        paper_role="asymmetry and occupancy-skew stress test",
    ),
    ClaudeCatalogPacketSystem(
        system_key="claude:var_diamond_4",
        env_name="claude:var_diamond_4",
        system_group="strict_core",
        paper_role="rotated-separatrix geometry mismatch",
    ),
    ClaudeCatalogPacketSystem(
        system_key="claude:var_l_shape_5",
        env_name="claude:var_l_shape_5",
        system_group="strict_core",
        paper_role="non-convex geometry case",
    ),
    ClaudeCatalogPacketSystem(
        system_key="claude:transition_routes_4",
        env_name="claude:transition_routes_4",
        system_group="strict_core",
        paper_role="explicit route-choice / shared-corridor case",
    ),
)


CLAUDE_CATALOG_PACKET_SECOND_WAVE_SYSTEMS: Sequence[ClaudeCatalogPacketSystem] = (
    ClaudeCatalogPacketSystem(
        system_key="claude:hybrid_state_dep_rot_5",
        env_name="claude:hybrid_state_dep_rot_5",
        system_group="second_wave",
        paper_role="strict-pass hybrid mechanism",
    ),
    ClaudeCatalogPacketSystem(
        system_key="claude:cal_hexagon_6",
        env_name="claude:cal_hexagon_6",
        system_group="second_wave",
        paper_role="first grounded higher-basin control",
    ),
    ClaudeCatalogPacketSystem(
        system_key="claude:snic_multi",
        env_name="claude:snic_multi",
        system_group="second_wave",
        paper_role="non-well mechanistic outlier",
    ),
)


CLAUDE_CATALOG_PACKET_MODELS: Sequence[ClaudeCatalogPacketModel] = (
    ClaudeCatalogPacketModel(
        variant="generic_sparse_ns200k_best",
        config_name="generic_sparse",
        num_steps=CLAUDE_CATALOG_PACKET_NUM_STEPS,
        lr=1e-4,
        k_matrix_lr=1e-5,
        weight_decay=1e-4,
        res_coeff=1.0,
        reconst_coeff=0.03,
        pred_coeff=1.0,
        sparsity_coeff=0.0025,
    ),
    ClaudeCatalogPacketModel(
        variant="generic_sparse_sc0_ns200k_best",
        config_name="generic_sparse",
        num_steps=CLAUDE_CATALOG_PACKET_NUM_STEPS,
        lr=1e-4,
        k_matrix_lr=1e-5,
        weight_decay=1e-4,
        res_coeff=1.0,
        reconst_coeff=0.03,
        pred_coeff=1.0,
        sparsity_coeff=0.0,
    ),
    ClaudeCatalogPacketModel(
        variant="lista_dense_promoted_stage4",
        config_name="lista_parity_generic_sparse",
        num_steps=CLAUDE_CATALOG_PACKET_NUM_STEPS,
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
)


def claude_catalog_packet_systems(include_second_wave: bool = False) -> List[ClaudeCatalogPacketSystem]:
    """Return the ordered list of Claude-catalog packet systems."""
    systems = list(CLAUDE_CATALOG_PACKET_STRICT_SYSTEMS)
    if include_second_wave:
        systems.extend(CLAUDE_CATALOG_PACKET_SECOND_WAVE_SYSTEMS)
    return systems


def claude_catalog_packet_models() -> List[ClaudeCatalogPacketModel]:
    """Return the ordered list of model variants for the Claude packet."""
    return list(CLAUDE_CATALOG_PACKET_MODELS)


def get_claude_catalog_packet_system(system_key: str) -> ClaudeCatalogPacketSystem:
    """Lookup a Claude packet system by system key."""
    for spec in claude_catalog_packet_systems(include_second_wave=True):
        if spec.system_key == system_key:
            return spec
    raise KeyError(f"Unknown Claude catalog packet system '{system_key}'")


def get_claude_catalog_packet_model(variant: str) -> ClaudeCatalogPacketModel:
    """Lookup a Claude packet model by variant name."""
    for spec in CLAUDE_CATALOG_PACKET_MODELS:
        if spec.variant == variant:
            return spec
    raise KeyError(f"Unknown Claude catalog packet model '{variant}'")


def resolve_claude_catalog_packet_dt(system_key: str) -> float:
    """Resolve the numeric default timestep for a Claude packet system."""
    spec = get_claude_catalog_packet_system(system_key)
    cfg = Config()
    cfg.ENV.ENV_NAME = spec.env_name
    return float(get_env_dt(cfg))


def claude_catalog_packet_manifest_jsonable(include_second_wave: bool = False) -> Dict[str, object]:
    """Return a JSON-serializable snapshot of the Claude packet manifest."""
    systems = claude_catalog_packet_systems(include_second_wave=include_second_wave)
    return {
        "num_steps": CLAUDE_CATALOG_PACKET_NUM_STEPS,
        "batch_size": CLAUDE_CATALOG_PACKET_BATCH_SIZE,
        "target_size": CLAUDE_CATALOG_PACKET_TARGET_SIZE,
        "sequence_length": CLAUDE_CATALOG_PACKET_SEQUENCE_LENGTH,
        "seeds": list(CLAUDE_CATALOG_PACKET_SEEDS),
        "include_second_wave": include_second_wave,
        "systems": [
            {
                **asdict(spec),
                "system_slug": spec.system_slug,
                "resolved_default_dt": resolve_claude_catalog_packet_dt(spec.system_key),
            }
            for spec in systems
        ],
        "models": [asdict(spec) for spec in CLAUDE_CATALOG_PACKET_MODELS],
    }
