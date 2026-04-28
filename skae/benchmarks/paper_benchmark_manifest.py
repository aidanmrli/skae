"""Canonical manifest for the paper benchmark experiments.

This module defines the benchmark systems and model variants that should be
used for the final research-paper experiments. The intent is to keep one
source of truth for the paper benchmark rather than scattering lists across
scripts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence


PAPER_BENCHMARK_NUM_STEPS = 50_000
PAPER_BENCHMARK_BATCH_SIZE = 256
PAPER_BENCHMARK_TARGET_SIZE = 256
PAPER_BENCHMARK_SEQUENCE_LENGTH = 10
PAPER_BENCHMARK_SEEDS: Sequence[int] = (0, 1, 2)
PAPER_BENCHMARK_DT_HALVING_FACTORS: Sequence[float] = (1.0, 0.5, 0.25)


@dataclass(frozen=True)
class PaperBenchmarkSystem:
    """Definition of a single paper-benchmark environment."""

    system_key: str
    env_name: str
    system_group: str
    default_dt: Optional[float]
    is_dysts: bool

    @property
    def system_slug(self) -> str:
        return self.system_key.replace(":", "_")

    @property
    def dt_halving_schedule(self) -> List[Optional[float]]:
        if self.default_dt is None:
            return [None]
        return [self.default_dt * factor for factor in PAPER_BENCHMARK_DT_HALVING_FACTORS]


@dataclass(frozen=True)
class PaperBenchmarkModel:
    """Definition of a single paper-benchmark model variant."""

    variant: str
    config_name: str
    res_coeff: float
    reconst_coeff: float
    pred_coeff: float
    sparsity_coeff: float
    lista_alpha: Optional[float] = None
    lista_num_loops: Optional[int] = None
    lista_final_op: Optional[str] = None
    k_structure: Optional[str] = None
    k_block_size: Optional[int] = None


PAPER_BENCHMARK_SYSTEMS: Sequence[PaperBenchmarkSystem] = (
    PaperBenchmarkSystem("duffing", "duffing", "builtin_low_dim", 0.01, False),
    PaperBenchmarkSystem("lotka_volterra", "lotka_volterra", "builtin_low_dim", 0.01, False),
    PaperBenchmarkSystem("blended", "blended", "builtin_low_dim", 0.05, False),
    PaperBenchmarkSystem("multiwell_gradient", "multiwell_gradient", "builtin_low_dim", 0.02, False),
    PaperBenchmarkSystem("multiwell_rotational", "multiwell_rotational", "builtin_low_dim", 0.02, False),
    PaperBenchmarkSystem("multiwell_energy", "multiwell_energy", "builtin_low_dim", 0.02, False),
    PaperBenchmarkSystem("multiwell_strong_transition", "multiwell_strong_transition", "builtin_low_dim", 0.02, False),
    PaperBenchmarkSystem("multiwell_gradient_hd", "multiwell_gradient_hd", "builtin_high_dim", 0.02, False),
    PaperBenchmarkSystem("multiwell_rotational_hd", "multiwell_rotational_hd", "builtin_high_dim", 0.02, False),
    PaperBenchmarkSystem("multiwell_energy_hd", "multiwell_energy_hd", "builtin_high_dim", 0.02, False),
    PaperBenchmarkSystem("multiwell_strong_transition_hd", "multiwell_strong_transition_hd", "builtin_high_dim", 0.02, False),
    PaperBenchmarkSystem("kuramoto", "kuramoto", "builtin_high_dim", 0.05, False),
    PaperBenchmarkSystem("hopfield", "hopfield", "builtin_high_dim", 0.05, False),
    PaperBenchmarkSystem("competitive_lv", "competitive_lv", "builtin_high_dim", 0.01, False),
    PaperBenchmarkSystem("dysts:Dadras", "dysts:Dadras", "dysts_multi_basin", None, True),
    PaperBenchmarkSystem("dysts:Duffing", "dysts:Duffing", "dysts_multi_basin", None, True),
    PaperBenchmarkSystem("dysts:QiChen", "dysts:QiChen", "dysts_multi_basin", None, True),
    PaperBenchmarkSystem("dysts:Sakarya", "dysts:Sakarya", "dysts_multi_basin", None, True),
    PaperBenchmarkSystem("dysts:SprottTorus", "dysts:SprottTorus", "dysts_multi_basin", None, True),
    PaperBenchmarkSystem("dysts:Chua", "dysts:Chua", "dysts_multi_basin", None, True),
    PaperBenchmarkSystem("dysts:MultiChua", "dysts:MultiChua", "dysts_multi_basin", None, True),
    PaperBenchmarkSystem("dysts:DequanLi", "dysts:DequanLi", "dysts_multi_basin", None, True),
    PaperBenchmarkSystem("dysts:LuChenCheng", "dysts:LuChenCheng", "dysts_multi_basin", None, True),
    PaperBenchmarkSystem("dysts:SanUmSrisuchinwong", "dysts:SanUmSrisuchinwong", "dysts_multi_basin", None, True),
    PaperBenchmarkSystem("dysts:WangSun", "dysts:WangSun", "dysts_multi_basin", None, True),
    PaperBenchmarkSystem("dysts:ShimizuMorioka", "dysts:ShimizuMorioka", "dysts_multi_basin", None, True),
    PaperBenchmarkSystem("dysts:LorenzCoupled", "dysts:LorenzCoupled", "dysts_multi_basin", None, True),
    PaperBenchmarkSystem("dysts:RikitakeDynamo", "dysts:RikitakeDynamo", "dysts_multi_basin", None, True),
    PaperBenchmarkSystem("dysts:Hadley", "dysts:Hadley", "dysts_multi_basin", None, True),
)


PAPER_BENCHMARK_MODELS: Sequence[PaperBenchmarkModel] = (
    PaperBenchmarkModel(
        variant="generic_sparse",
        config_name="generic_sparse",
        res_coeff=1.0,
        reconst_coeff=0.03,
        pred_coeff=1.0,
        sparsity_coeff=0.0025,
    ),
    PaperBenchmarkModel(
        variant="generic_sparse_blockdiag",
        config_name="generic_sparse",
        res_coeff=1.0,
        reconst_coeff=0.03,
        pred_coeff=1.0,
        sparsity_coeff=0.0025,
        k_structure="block_diagonal",
        k_block_size=16,
    ),
    PaperBenchmarkModel(
        variant="lista_dense",
        config_name="lista_parity_generic_sparse",
        res_coeff=1.0,
        reconst_coeff=0.03,
        pred_coeff=1.0,
        sparsity_coeff=0.0060,
        lista_alpha=0.15,
        lista_num_loops=1,
        lista_final_op="relu",
        k_structure="dense",
    ),
    PaperBenchmarkModel(
        variant="lista_diagonal",
        config_name="lista_parity_generic_sparse",
        res_coeff=1.0,
        reconst_coeff=0.03,
        pred_coeff=1.0,
        sparsity_coeff=0.0060,
        lista_alpha=0.15,
        lista_num_loops=1,
        lista_final_op="relu",
        k_structure="diagonal",
    ),
    PaperBenchmarkModel(
        variant="lista_blockdiag",
        config_name="lista_parity_generic_sparse",
        res_coeff=1.0,
        reconst_coeff=0.03,
        pred_coeff=1.0,
        sparsity_coeff=0.0060,
        lista_alpha=0.15,
        lista_num_loops=1,
        lista_final_op="relu",
        k_structure="block_diagonal",
        k_block_size=16,
    ),
)


def paper_benchmark_systems() -> List[PaperBenchmarkSystem]:
    """Return the ordered list of paper-benchmark systems."""
    return list(PAPER_BENCHMARK_SYSTEMS)


def paper_benchmark_models() -> List[PaperBenchmarkModel]:
    """Return the ordered list of paper-benchmark model variants."""
    return list(PAPER_BENCHMARK_MODELS)


def get_paper_benchmark_system(system_key: str) -> PaperBenchmarkSystem:
    """Lookup a paper-benchmark system by system key."""
    for spec in PAPER_BENCHMARK_SYSTEMS:
        if spec.system_key == system_key:
            return spec
    raise KeyError(f"Unknown paper benchmark system '{system_key}'")


def get_paper_benchmark_model(variant: str) -> PaperBenchmarkModel:
    """Lookup a paper-benchmark model by variant name."""
    for spec in PAPER_BENCHMARK_MODELS:
        if spec.variant == variant:
            return spec
    raise KeyError(f"Unknown paper benchmark model '{variant}'")


def resolve_system_default_dt(system_key: str) -> float:
    """Resolve the numeric default timestep for a benchmark system."""
    spec = get_paper_benchmark_system(system_key)
    if spec.default_dt is not None:
        return float(spec.default_dt)

    from skae.benchmarks.dysts_adapter import get_dysts_system_metadata

    dysts_name = spec.env_name.split(":", 1)[1]
    metadata = get_dysts_system_metadata(dysts_name)
    dt = metadata.get("dt")
    if dt is None:
        raise ValueError(f"Dysts metadata for '{dysts_name}' does not define dt")
    return float(dt)


def paper_benchmark_manifest_jsonable() -> Dict[str, object]:
    """Return a JSON-serializable snapshot of the paper benchmark manifest."""
    return {
        "num_steps": PAPER_BENCHMARK_NUM_STEPS,
        "batch_size": PAPER_BENCHMARK_BATCH_SIZE,
        "target_size": PAPER_BENCHMARK_TARGET_SIZE,
        "sequence_length": PAPER_BENCHMARK_SEQUENCE_LENGTH,
        "seeds": list(PAPER_BENCHMARK_SEEDS),
        "systems": [
            {
                **asdict(spec),
                "system_slug": spec.system_slug,
                "resolved_default_dt": resolve_system_default_dt(spec.system_key),
            }
            for spec in PAPER_BENCHMARK_SYSTEMS
        ],
        "models": [asdict(spec) for spec in PAPER_BENCHMARK_MODELS],
    }
