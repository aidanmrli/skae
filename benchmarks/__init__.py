"""Benchmarking module for SKAE with dysts integration.

This module provides:
- DystsEnv: Adapter wrapping dysts systems to SKAE Env interface
- System catalog: Categorization of dysts systems
- Benchmark runner: Large-scale benchmarking infrastructure
"""

from benchmarks.dysts_adapter import DystsEnv, get_dysts_systems, is_dysts_available
from benchmarks.system_catalog import (
    QUICK_TEST,
    STANDARD_BENCHMARK,
    MULTI_BASIN_SYSTEMS,
    WELL_STUDIED_CHAOTIC,
    HIGH_DIMENSIONAL,
    get_all_systems,
    get_system_info,
)

__all__ = [
    "DystsEnv",
    "get_dysts_systems",
    "is_dysts_available",
    "QUICK_TEST",
    "STANDARD_BENCHMARK", 
    "MULTI_BASIN_SYSTEMS",
    "WELL_STUDIED_CHAOTIC",
    "HIGH_DIMENSIONAL",
    "get_all_systems",
    "get_system_info",
]
