"""Benchmarking module for SKAE with dysts integration.

This module provides:
- DystsEnv: Adapter wrapping dysts systems to SKAE Env interface
- System catalog: Categorization of dysts systems
- Benchmark runner: Large-scale benchmarking infrastructure
"""

from skae.benchmarks.dysts_adapter import DystsEnv, get_dysts_systems, is_dysts_available
from skae.benchmarks.system_catalog import (
    QUICK_TEST,
    STANDARD_BENCHMARK,
    MULTI_ATTRACTOR_SYSTEMS,
    MULTI_BASIN_SYSTEMS,
    MULTI_SCROLL_SYSTEMS,
    WELL_STUDIED_CHAOTIC,
    HIGH_DIMENSIONAL,
    get_all_systems,
    get_multi_attractor_systems,
    get_multi_basin_systems,
    get_multiscroll_systems,
    get_system_info,
)
from skae.benchmarks.paper_benchmark_manifest import (
    PAPER_BENCHMARK_BATCH_SIZE,
    PAPER_BENCHMARK_NUM_STEPS,
    PAPER_BENCHMARK_SEQUENCE_LENGTH,
    PAPER_BENCHMARK_TARGET_SIZE,
    PAPER_BENCHMARK_SEEDS,
    paper_benchmark_manifest_jsonable,
    paper_benchmark_models,
    paper_benchmark_systems,
)

__all__ = [
    "DystsEnv",
    "get_dysts_systems",
    "is_dysts_available",
    "QUICK_TEST",
    "STANDARD_BENCHMARK", 
    "MULTI_ATTRACTOR_SYSTEMS",
    "MULTI_BASIN_SYSTEMS",
    "MULTI_SCROLL_SYSTEMS",
    "WELL_STUDIED_CHAOTIC",
    "HIGH_DIMENSIONAL",
    "get_all_systems",
    "get_multi_attractor_systems",
    "get_multi_basin_systems",
    "get_multiscroll_systems",
    "get_system_info",
    "PAPER_BENCHMARK_BATCH_SIZE",
    "PAPER_BENCHMARK_NUM_STEPS",
    "PAPER_BENCHMARK_SEQUENCE_LENGTH",
    "PAPER_BENCHMARK_TARGET_SIZE",
    "PAPER_BENCHMARK_SEEDS",
    "paper_benchmark_manifest_jsonable",
    "paper_benchmark_models",
    "paper_benchmark_systems",
]
