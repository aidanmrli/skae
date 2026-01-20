"""Categorization and metadata for dysts systems.

This module organizes the 135+ dysts systems into meaningful categories
for structured experimentation and benchmarking.

Categories:
- MULTI_ATTRACTOR_SYSTEMS: Multiple attractors/basins (multistability)
- MULTI_SCROLL_SYSTEMS: Multi-scroll/double-wing attractors (lobe switching)
- MULTI_BASIN_SYSTEMS: Combined multi-basin candidates (manual baseline)
- WELL_STUDIED_CHAOTIC: Canonical chaotic systems for benchmarking
- HIGH_DIMENSIONAL: Systems with dimension > 3
- QUASI_PERIODIC: Systems with quasi-periodic behavior
- HAMILTONIAN_LIKE: Conservative or nearly-conservative systems

Curated test sets:
- QUICK_TEST: 4 systems for rapid validation
- STANDARD_BENCHMARK: 10-15 systems for paper benchmarks
"""

from typing import List, Dict, Any


def get_all_systems(include_delay: bool = False) -> List[str]:
    """Get list of all continuous dysts systems.
    
    Args:
        include_delay: Whether to include delay differential equation systems.
        
    Returns:
        Sorted list of system names.
    """
    try:
        from benchmarks.dysts_adapter import get_dysts_systems
        return get_dysts_systems(include_delay=include_delay)
    except ImportError:
        return []


def get_system_info(system_name: str) -> Dict[str, Any]:
    """Get metadata for a specific system.
    
    Args:
        system_name: Name of the dysts system.
        
    Returns:
        Dictionary with system metadata.
    """
    try:
        from benchmarks.dysts_adapter import get_dysts_system_metadata
        return get_dysts_system_metadata(system_name)
    except ImportError:
        return {}


def get_all_system_metadata(include_delay: bool = False) -> Dict[str, Any]:
    """Get metadata for all available systems.
    
    Args:
        include_delay: Whether to include delay differential equation systems.
        
    Returns:
        Dictionary mapping system name to metadata.
    """
    try:
        from benchmarks.dysts_adapter import get_dysts_system_data
        return get_dysts_system_data(include_delay=include_delay)
    except ImportError:
        return {}


# =============================================================================
# System Categorization
# =============================================================================

# Systems with multiple basins/attractors (multistability)
# These are key for testing block sparsity hypothesis
MULTI_ATTRACTOR_SYSTEMS = [
    "Dadras",            # Multiple attractors under bifurcations
    "Duffing",           # Double-well forcing yields distinct basins
    "QiChen",            # Double-wing from bistable attractors
    "Sakarya",           # Merging of two disjoint bistable attractors
    "SprottTorus",       # Multiattractor (torus vs complex attractor)
]

# Systems with multiple scrolls/lobes (switching within a single attractor)
MULTI_SCROLL_SYSTEMS = [
    "Chua",              # Double scroll attractor with multiple lobes
    "MultiChua",         # Multiple scroll attractors
    "DequanLi",          # Three-scroll unified attractor
    "LuChenCheng",       # Four-scroll attractor
    "SanUmSrisuchinwong",  # Two-scroll attractor
    "WangSun",           # Four-scroll attractor
    "ShimizuMorioka",    # Two-lobe structure similar to Lorenz
]

# Coupled or switching systems with multiple equilibria
MULTI_BASIN_MANUAL = [
    "LorenzCoupled",     # Two coupled Lorenz systems
    "RikitakeDynamo",    # Coupled disk dynamos with reversals
    "Hadley",            # Multiple equilibria
]

# Combined list of multi-basin candidates (manual baseline)
MULTI_BASIN_SYSTEMS = (
    MULTI_ATTRACTOR_SYSTEMS
    + MULTI_SCROLL_SYSTEMS
    + MULTI_BASIN_MANUAL
)

# Description keywords for automatic multi-basin discovery
MULTI_ATTRACTOR_KEYWORDS = [
    "bistable",
    "bistability",
    "multiattractor",
    "multiple attractor",
    "multiple attractors",
    "coexisting",
    "disjoint",
    "multistable",
]
MULTI_SCROLL_KEYWORDS = [
    "scroll",
    "double-wing",
    "double wing",
    "two-scroll",
    "two scroll",
    "three scroll",
    "four scroll",
    "multiscroll",
    "multi-scroll",
]

# Canonical chaotic systems well-studied in literature
WELL_STUDIED_CHAOTIC = [
    "Lorenz",            # The classic butterfly attractor
    "Rossler",           # Simple 3D chaotic flow
    "Chen",              # Chen system (Lorenz family)
    "Lu",                # Lu system (Lorenz family)
    "Thomas",            # Thomas' cyclically symmetric attractor
    "Halvorsen",         # Halvorsen attractor
    "Sprott",            # Sprott systems (simplest chaotic flows)
    "SprottA",
    "SprottB",
    "SprottC",
    "Burke",             # Burke-Shaw attractor
    "Aizawa",            # Aizawa attractor
    "NoseHoover",        # Nose-Hoover oscillator
]

# Systems with quasi-periodic or periodic behavior
QUASI_PERIODIC = [
    "Torus",             # Flow on torus
    "VanDerPol",         # Van der Pol oscillator (limit cycle)
    "ForcedVanDerPol",   # Forced Van der Pol
]

# High-dimensional systems (d > 3)
HIGH_DIMENSIONAL = [
    "Lorenz96",          # Lorenz 96 model (variable dimension, default 5)
    "MacArthur",         # MacArthur competition model (4D)
    "Laser",             # Laser dynamics (4D)
    "HenonHeiles",       # Henon-Heiles Hamiltonian (4D)
    "DoublePendulum",    # Double pendulum (4D)
    "SwingingAtwood",    # Swinging Atwood machine (4D)
    "SprottLinz",        # Sprott-Linz system
]

# Hamiltonian or nearly-conservative systems
HAMILTONIAN_LIKE = [
    "HenonHeiles",       # Conservative Hamiltonian chaos
    "DoublePendulum",    # Conservative mechanical system
    "SwingingAtwood",    # Conservative mechanical system
    "ThreeBodyCircular", # Restricted 3-body problem
    "NoseHoover",        # Thermostat dynamics
]

# Biological/ecological systems
BIOLOGICAL = [
    "LotkaVolterra",     # Predator-prey (matches SKAE's lotka_volterra)
    "Oregonator",        # Oregonator chemical oscillator
    "BelousovZhabotinsky", # BZ reaction
    "GlycolyticOscillation", # Glycolysis oscillations
    "HodgkinHuxley",     # Neuron dynamics
    "HindmarshRose",     # Neuron model
    "FitzHughNagumo",    # Simplified neuron model
    "MacArthur",         # Competition model
]

# Circuit/electrical systems
CIRCUIT_SYSTEMS = [
    "Chua",
    "MultiChua",
    "Colpitts",          # Colpitts oscillator
    "Laser",
    "RabinovichFabrikant",
]


# =============================================================================
# Curated Test Sets
# =============================================================================

# Quick test: 4 diverse systems for rapid validation
QUICK_TEST = [
    "Lorenz",     # 3D chaotic, single attractor
    "Rossler",    # 3D chaotic, simpler dynamics
    "Chua",       # 3D, multi-basin (double scroll)
    "Chen",       # 3D chaotic, Lorenz family
]

# Standard benchmark: ~12 systems for paper benchmarks
STANDARD_BENCHMARK = [
    # Well-studied chaotic
    "Lorenz",
    "Rossler", 
    "Chen",
    "Lu",
    "Thomas",
    "Halvorsen",
    # Multi-basin (key for block sparsity)
    "Chua",
    "ShimizuMorioka",
    # Different dynamics
    "NoseHoover",
    "Aizawa",
    "SprottB",
    "Burke",
]

# Extended benchmark for comprehensive evaluation
EXTENDED_BENCHMARK = STANDARD_BENCHMARK + [
    # Additional chaotic
    "SprottA",
    "SprottC",
    "Dadras",
    "Finance",
    # Biological
    "LotkaVolterra",
    "Oregonator",
    # Higher dimensional
    "HenonHeiles",
    "DoublePendulum",
]

# Block sparsity experiment: multi-basin vs single-basin comparison
BLOCK_SPARSITY_EXPERIMENT = {
    "multi_basin": [
        "Chua",              # Multiple scroll regions
        "ShimizuMorioka",    # Two-lobe structure
        "LorenzCoupled",     # Coupled attractors (if available)
        "Hadley",            # Multiple equilibria
    ],
    "single_basin_control": [
        "Lorenz",            # Single strange attractor
        "Rossler",           # Single band attractor
        "Thomas",            # Single attractor
        "Chen",              # Single attractor
    ],
}

# Dimension scaling experiment
DIMENSION_SCALING_EXPERIMENT = {
    "2D": ["VanDerPol", "FitzHughNagumo"],  # Note: these might not be in dysts
    "3D": ["Lorenz", "Rossler", "Chen", "Thomas"],
    "4D": ["HenonHeiles", "DoublePendulum", "SwingingAtwood"],
    "5D+": ["Lorenz96"],  # Lorenz96 default is 5D
}


# =============================================================================
# Utility Functions
# =============================================================================

def filter_available_systems(system_list: List[str], include_delay: bool = False) -> List[str]:
    """Filter a list to only include systems that are actually available.
    
    Args:
        system_list: List of desired system names.
        include_delay: Whether to include delay differential equation systems.
        
    Returns:
        Filtered list containing only available systems.
    """
    available = set(get_all_systems(include_delay=include_delay))
    return [s for s in system_list if s in available]


def get_systems_by_description_keywords(
    keywords: List[str],
    include_delay: bool = False,
) -> List[str]:
    """Find systems whose descriptions match any keyword.
    
    Args:
        keywords: Case-insensitive keyword list for matching descriptions.
        include_delay: Whether to include delay differential equation systems.
        
    Returns:
        Sorted list of matching system names.
    """
    metadata = get_all_system_metadata(include_delay=include_delay)
    if not metadata:
        return []
    
    matches = []
    for name, meta in metadata.items():
        desc = (meta.get("description") or "").lower()
        if any(keyword in desc for keyword in keywords):
            matches.append(name)
    
    return sorted(matches)


def get_multi_attractor_systems(
    include_auto: bool = True,
    include_manual: bool = True,
    include_delay: bool = False,
) -> List[str]:
    """Get multistable/multi-attractor systems.
    
    Args:
        include_auto: Include metadata keyword matches.
        include_manual: Include manual curation list.
        include_delay: Whether to include delay differential equation systems.
        
    Returns:
        Filtered list of system names.
    """
    systems = set()
    if include_manual:
        systems.update(MULTI_ATTRACTOR_SYSTEMS)
    if include_auto:
        systems.update(
            get_systems_by_description_keywords(
                MULTI_ATTRACTOR_KEYWORDS,
                include_delay=include_delay,
            )
        )
    return filter_available_systems(sorted(systems), include_delay=include_delay)


def get_multiscroll_systems(
    include_auto: bool = True,
    include_manual: bool = True,
    include_delay: bool = False,
) -> List[str]:
    """Get multi-scroll/double-wing systems with lobe switching.
    
    Args:
        include_auto: Include metadata keyword matches.
        include_manual: Include manual curation list.
        include_delay: Whether to include delay differential equation systems.
        
    Returns:
        Filtered list of system names.
    """
    systems = set()
    if include_manual:
        systems.update(MULTI_SCROLL_SYSTEMS)
    if include_auto:
        systems.update(
            get_systems_by_description_keywords(
                MULTI_SCROLL_KEYWORDS,
                include_delay=include_delay,
            )
        )
    return filter_available_systems(sorted(systems), include_delay=include_delay)


def get_multi_basin_systems(
    include_auto: bool = True,
    include_manual: bool = True,
    include_delay: bool = False,
) -> List[str]:
    """Get combined multi-basin candidates (multistable + multiscroll).
    
    Args:
        include_auto: Include metadata keyword matches.
        include_manual: Include manual curation lists.
        include_delay: Whether to include delay differential equation systems.
        
    Returns:
        Filtered list of system names.
    """
    systems = set()
    systems.update(
        get_multi_attractor_systems(
            include_auto=include_auto,
            include_manual=include_manual,
            include_delay=include_delay,
        )
    )
    systems.update(
        get_multiscroll_systems(
            include_auto=include_auto,
            include_manual=include_manual,
            include_delay=include_delay,
        )
    )
    if include_manual:
        systems.update(MULTI_BASIN_MANUAL)
    return filter_available_systems(sorted(systems), include_delay=include_delay)


def get_systems_by_dimension(min_dim: int = 1, max_dim: int = 10) -> List[str]:
    """Get systems within a dimension range.
    
    Args:
        min_dim: Minimum dimension (inclusive).
        max_dim: Maximum dimension (inclusive).
        
    Returns:
        List of system names with dimension in [min_dim, max_dim].
    """
    result = []
    for system_name in get_all_systems():
        try:
            info = get_system_info(system_name)
            dim = info.get("dimension", 3)
            if min_dim <= dim <= max_dim:
                result.append(system_name)
        except Exception:
            continue
    return result


def print_catalog_summary():
    """Print a summary of the system catalog."""
    all_systems = get_all_systems()
    multi_attractor = get_multi_attractor_systems()
    multi_scroll = get_multiscroll_systems()
    multi_basin = get_multi_basin_systems()
    
    print("=" * 60)
    print("DYSTS SYSTEM CATALOG SUMMARY")
    print("=" * 60)
    print(f"\nTotal available systems: {len(all_systems)}")
    
    print(f"\nMulti-basin candidates: {len(multi_basin)}")
    print(f"  Multi-attractor: {len(multi_attractor)}")
    print(f"  Multi-scroll: {len(multi_scroll)}")
    for s in multi_basin:
        print(f"  - {s}")
    
    print(f"\nWell-studied chaotic: {len(filter_available_systems(WELL_STUDIED_CHAOTIC))}")
    for s in filter_available_systems(WELL_STUDIED_CHAOTIC)[:5]:
        print(f"  - {s}")
    print("  ...")
    
    print(f"\nQuick test set: {filter_available_systems(QUICK_TEST)}")
    print(f"Standard benchmark: {len(filter_available_systems(STANDARD_BENCHMARK))} systems")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_catalog_summary()
