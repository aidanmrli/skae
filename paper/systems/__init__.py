"""Benchmark systems of the paper: the name tables, and ``make_system``.

* ``MULTIBASIN_SYSTEMS``: 15 two-dimensional multibasin systems, two gated
  local-linear systems (``gated.py``) and thirteen analytic vector fields
  (``analytic.py``), simulated by ``multibasin.MultibasinSystem``.
* ``DYSTS_SYSTEMS``: 10 three-dimensional chaotic flows (dysts class names), with
  display names in ``DYSTS_NAMES``; ``paper.systems.dysts`` builds and loads
  their trajectory data.

``make_system`` raises ``ValueError`` for a name outside these tables.

This module imports nothing heavy, so command-line help that lists systems is fast.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemSpec:
    """A multibasin system: its key, its name in the paper's tables, its stored step and its basin count."""

    name: str          # key in configs, run directories and evidence files
    display_name: str  # name in the paper's tables
    dt: float          # stored step: one RK4 step of this size per observation
    num_basins: int    # designed attractors; also the K blocks of the -BD and -SB models


MULTIBASIN_SYSTEMS = {
    spec.name: spec
    for spec in (
        SystemSpec("gated_local_linear", "Local-linear gates", 0.04, 3),
        SystemSpec("gated_transfer_linear", "Transfer-gated local-linear", 0.04, 3),
        SystemSpec("arrested_spiral", "Arrested spiral", 0.02, 5),
        SystemSpec("cal_asymmetric_3", "Asymmetric three-well", 0.03, 3),
        SystemSpec("cal_high_cross_3", "High-cross three-well", 0.03, 3),
        SystemSpec("cal_hexagon_6", "Hexagonal six-well", 0.03, 6),
        SystemSpec("cal_octagon_8", "Octagonal eight-well", 0.03, 8),
        SystemSpec("cal_pentagon_5", "Pentagonal five-well", 0.03, 5),
        SystemSpec("cal_square_4", "Square four-well", 0.03, 4),
        SystemSpec("duffing_triple_well_no_rotation", "Triple-well Duffing", 0.02, 3),
        SystemSpec("snic_multi", "SNIC multi-attractor", 0.02, 3),
        SystemSpec("transition_routes_4", "Transition-routes four-well", 0.02, 4),
        SystemSpec("var_depth_gradient_4", "Depth-gradient four-well", 0.03, 4),
        SystemSpec("var_diamond_4", "Diamond four-well", 0.03, 4),
        SystemSpec("var_l_shape_5", "L-shaped five-well", 0.03, 5),
    )
}

# dysts class name -> display name (the LaTeX tables write each en dash as --).
DYSTS_NAMES = {
    "Chua": "Chua",
    "Dadras": "Dadras",
    "DequanLi": "Dequan Li",
    "Hadley": "Hadley",
    "LuChenCheng": "Lu–Chen–Cheng",
    "QiChen": "Qi–Chen",
    "Sakarya": "Sakarya",
    "SanUmSrisuchinwong": "San-Um–Srisuchinwong",
    "ShimizuMorioka": "Shimizu–Morioka",
    "WangSun": "Wang–Sun",
}
DYSTS_SYSTEMS = tuple(DYSTS_NAMES)


def make_system(benchmark: str, name: str):
    """A ``MultibasinSystem`` or, for ``benchmark="dysts"``, a ``DystsSystem`` (runs read its ``dim``).

    Raises ``ValueError`` for an unknown benchmark or system.
    """
    if benchmark == "multibasin":
        from paper.systems.multibasin import MultibasinSystem

        return MultibasinSystem(name)
    if benchmark == "dysts":
        from paper.systems.dysts import DystsSystem

        return DystsSystem(name)
    raise ValueError(f"unknown benchmark {benchmark!r}")
