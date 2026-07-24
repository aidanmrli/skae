"""Scientific contract for the historical local-polynomial-EDMD reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple

from experiments.neurips_2026.protocol import (
    CONTROLLED_PAPER_PROTOCOL,
    DYSTS_PAPER_PROTOCOL,
    STANDALONE_BASELINE_SEEDS,
)


PROTOCOL_ID = "local_edmd_poly_historical_reproduction_v1"
METHOD_ID = "local_edmd_poly_kmeans"
FEATURE_METHOD = "edmd_poly"
ROUTE_SPACE = "state"
NUM_COMPONENTS_GRID: Tuple[int, ...] = (1, 2, 4, 8, 16)
VALIDATION_FRACTION = 0.25
RIDGE_LAMBDA = 1e-6
EDMD_DEGREE = 3
KERNEL_CENTERS = 128
KERNEL_GAMMA = 0.0
MAX_TRAIN_PAIRS = 0
MIN_COMPONENT_TRANSITIONS = 64
MAX_ABS_STATE_FOR_FIT = 1e6
ENV_DT = 0.0
CONFIG_NAME = "default"
TORCH_THREADS = 1
RESULT_ROOT = Path(
    "/network/scratch/l/lia/skae/local_edmd_poly_reproduction_20260720"
)
CARD_PATH = Path(__file__).with_name("prediction_card.json")
LOCK_PATH = Path(__file__).with_name("source_lock.json")


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark: str
    protocol_id: str
    systems: Tuple[str, ...]
    horizons: Tuple[int, ...]
    num_trajectories: int
    trajectory_length: int
    train_fraction: float
    dysts_dt_multiplier: float
    dysts_standardize: int


BENCHMARKS: Mapping[str, BenchmarkSpec] = {
    "controlled": BenchmarkSpec(
        benchmark="controlled",
        protocol_id=CONTROLLED_PAPER_PROTOCOL.protocol_id,
        systems=CONTROLLED_PAPER_PROTOCOL.system_keys,
        horizons=(100, 500, 1000),
        num_trajectories=256,
        trajectory_length=1000,
        train_fraction=0.6,
        dysts_dt_multiplier=0.0,
        dysts_standardize=0,
    ),
    "dysts": BenchmarkSpec(
        benchmark="dysts",
        protocol_id=DYSTS_PAPER_PROTOCOL.protocol_id,
        systems=DYSTS_PAPER_PROTOCOL.system_keys,
        horizons=(100, 2000, 4000),
        num_trajectories=200,
        trajectory_length=4000,
        train_fraction=0.5,
        dysts_dt_multiplier=30.0,
        dysts_standardize=1,
    ),
}
SEEDS: Tuple[int, ...] = STANDALONE_BASELINE_SEEDS

# Known June outcomes are comparison targets, not prospective predictions.
EXPECTED_AGGREGATES = {
    "controlled": {
        100: 0.14953663670168432,
        500: 0.25242123852375753,
        1000: 0.2747299334457285,
    },
    "dysts": {
        100: 0.0005005032778234641,
        2000: 2.1705718905105225,
        4000: 2.966046746017425,
    },
}


def expected_keys() -> set[tuple[str, str, int, int]]:
    """Return the frozen benchmark/system/seed/horizon grid."""

    return {
        (benchmark, system, seed, horizon)
        for benchmark, spec in BENCHMARKS.items()
        for system in spec.systems
        for seed in SEEDS
        for horizon in spec.horizons
    }


def expected_task_count() -> int:
    """Return the number of independent system/seed jobs."""

    return sum(len(spec.systems) * len(SEEDS) for spec in BENCHMARKS.values())
