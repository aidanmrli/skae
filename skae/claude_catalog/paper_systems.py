"""Analytic two-dimensional systems retained by the controlled paper roster.

The historical catalog contained many screening candidates.  These are the 13
catalog systems that survived into ``CONTROLLED_PAPER_PROTOCOL``.  Equations,
default timesteps, initial-condition boxes, and metadata match the original
implementations.
"""

from __future__ import annotations

import math

import torch

from skae.claude_catalog.base import CatalogSystem
from skae.claude_catalog.registry import register


class GaussianWellIndepRotation(CatalogSystem):
    """Gaussian multi-well potential with independent rotation."""

    def __init__(
        self,
        wells,
        omega=1.0,
        conf=0.03,
        dt=0.03,
        ic_box=None,
        **kwargs,
    ):
        self._wells = wells
        self.omega = omega
        self.conf = conf
        if ic_box is None:
            xs = [well[0] for well in wells]
            ys = [well[1] for well in wells]
            margin = 1.5
            ic_box = [
                (min(xs) - margin, max(xs) + margin),
                (min(ys) - margin, max(ys) + margin),
            ]
        super().__init__(dt=dt, ic_box=ic_box, **kwargs)

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        d_v_dx = torch.zeros_like(x)
        d_v_dy = torch.zeros_like(y)
        for center_x, center_y, amplitude, sigma in self._wells:
            delta_x = x - center_x
            delta_y = y - center_y
            distance_squared = delta_x**2 + delta_y**2
            gaussian = amplitude * torch.exp(
                -distance_squared / (2 * sigma**2)
            )
            d_v_dx = d_v_dx + gaussian * delta_x / sigma**2
            d_v_dy = d_v_dy + gaussian * delta_y / sigma**2
        d_v_dx = d_v_dx + 4 * self.conf * x**3
        d_v_dy = d_v_dy + 4 * self.conf * y**3
        return torch.stack(
            [-d_v_dx + self.omega * y, -d_v_dy - self.omega * x]
        )


def _polygon_wells(count, radius, amplitude, phase=0.0):
    return [
        (
            radius * math.cos(2 * math.pi * index / count + phase),
            radius * math.sin(2 * math.pi * index / count + phase),
            amplitude,
            0.5,
        )
        for index in range(count)
    ]


@register
class CalibratedSquare4(GaussianWellIndepRotation):
    name = "cal_square_4"
    category = "B"
    description = "4 balanced wells on square with independent rotation"

    def __init__(self, **kwargs):
        super().__init__(
            wells=_polygon_wells(4, 1.8, 3.0, math.pi / 4),
            omega=1.0,
            conf=0.03,
            ic_box=[(-3.3, 3.3), (-3.3, 3.3)],
            **kwargs,
        )


@register
class CalibratedPentagon5(GaussianWellIndepRotation):
    name = "cal_pentagon_5"
    category = "B"
    description = "5 balanced wells on pentagon with independent rotation"

    def __init__(self, **kwargs):
        super().__init__(
            wells=_polygon_wells(5, 1.8, 2.0, math.pi / 2),
            omega=1.1,
            conf=0.03,
            ic_box=[(-3.3, 3.3), (-3.3, 3.3)],
            **kwargs,
        )


@register
class CalibratedHexagon6(GaussianWellIndepRotation):
    name = "cal_hexagon_6"
    category = "B"
    description = "6 balanced wells on hexagon with independent rotation"

    def __init__(self, **kwargs):
        super().__init__(
            wells=_polygon_wells(6, 1.7, 2.0),
            omega=1.0,
            conf=0.03,
            ic_box=[(-3.3, 3.3), (-3.3, 3.3)],
            **kwargs,
        )


@register
class CalibratedOctagon8(GaussianWellIndepRotation):
    name = "cal_octagon_8"
    category = "B"
    description = "8 balanced wells on octagon with independent rotation"

    def __init__(self, **kwargs):
        super().__init__(
            wells=_polygon_wells(8, 2.2, 3.0),
            omega=0.90,
            conf=0.02,
            ic_box=[(-4.0, 4.0), (-4.0, 4.0)],
            **kwargs,
        )


@register
class AsymmetricTriangle3(GaussianWellIndepRotation):
    name = "cal_asymmetric_3"
    category = "B"
    description = "3 asymmetric wells (different depths) with rotation"

    def __init__(self, **kwargs):
        super().__init__(
            wells=[
                (0.0, 1.8, 2.5, 0.55),
                (-1.5, -0.9, 1.5, 0.4),
                (1.5, -0.9, 2.0, 0.5),
            ],
            omega=1.0,
            conf=0.03,
            ic_box=[(-3.0, 3.0), (-2.5, 3.0)],
            **kwargs,
        )


@register
class HighCrossingTriangle3(GaussianWellIndepRotation):
    name = "cal_high_cross_3"
    category = "B"
    description = "3 wells with high rotation for 60% crossing fraction"

    def __init__(self, **kwargs):
        super().__init__(
            wells=_polygon_wells(3, 1.8, 3.0, math.pi / 2),
            omega=2.0,
            conf=0.03,
            ic_box=[(-3.3, 3.3), (-3.3, 3.3)],
            **kwargs,
        )


@register
class DepthVariation4(GaussianWellIndepRotation):
    name = "var_depth_gradient_4"
    category = "B"
    description = "4 wells with depth gradient testing occupancy balance"

    def __init__(self, **kwargs):
        super().__init__(
            wells=[
                (-1.3, 1.3, 2.2, 0.55),
                (1.3, 1.3, 2.5, 0.5),
                (1.3, -1.3, 3.0, 0.5),
                (-1.3, -1.3, 3.5, 0.5),
            ],
            omega=1.3,
            conf=0.03,
            ic_box=[(-3.0, 3.0), (-3.0, 3.0)],
            **kwargs,
        )


@register
class DiamondWells4(GaussianWellIndepRotation):
    name = "var_diamond_4"
    category = "B"
    description = "4 wells in diamond pattern with rotation"

    def __init__(self, **kwargs):
        super().__init__(
            wells=[
                (0.0, 2.2, 2.5, 0.5),
                (2.2, 0.0, 2.5, 0.5),
                (0.0, -2.2, 2.5, 0.5),
                (-2.2, 0.0, 2.5, 0.5),
            ],
            omega=1.0,
            conf=0.02,
            ic_box=[(-3.5, 3.5), (-3.5, 3.5)],
            **kwargs,
        )


@register
class LShapeWells5(GaussianWellIndepRotation):
    name = "var_l_shape_5"
    category = "B"
    description = "5 wells in L-shape (asymmetric topology)"

    def __init__(self, **kwargs):
        super().__init__(
            wells=[
                (-1.5, 1.5, 2.5, 0.5),
                (-1.5, 0.0, 2.5, 0.5),
                (-1.5, -1.5, 2.5, 0.5),
                (0.0, -1.5, 2.5, 0.5),
                (1.5, -1.5, 2.5, 0.5),
            ],
            omega=1.0,
            conf=0.03,
            ic_box=[(-3.5, 3.5), (-3.5, 3.5)],
            **kwargs,
        )


@register
class TransitionRoutes4(CatalogSystem):
    name = "transition_routes_4"
    category = "H"
    description = "4 basins with specific fast/slow transition corridors"

    def __init__(self, dt=0.02, **kwargs):
        super().__init__(dt=dt, ic_box=[(-3.5, 3.5), (-3.5, 3.5)], **kwargs)
        self.wells = [
            (-1.8, 1.8, 3.0, 0.6),
            (1.8, 1.8, 3.0, 0.6),
            (-1.8, -1.8, 3.0, 0.6),
            (1.8, -1.8, 3.0, 0.6),
        ]

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        d_v_dx = torch.zeros_like(x)
        d_v_dy = torch.zeros_like(y)
        for center_x, center_y, amplitude, sigma in self.wells:
            delta_x = x - center_x
            delta_y = y - center_y
            gaussian = amplitude * torch.exp(
                -(delta_x**2 + delta_y**2) / (2 * sigma**2)
            )
            d_v_dx += gaussian * delta_x / sigma**2
            d_v_dy += gaussian * delta_y / sigma**2
        d_v_dx += 0.12 * x**3
        d_v_dy += 0.12 * y**3
        dx_dt = -d_v_dx + y
        dy_dt = -d_v_dy - x
        corridor = torch.exp(-(y - 1.8) ** 2 / 0.3) + torch.exp(
            -(y + 1.8) ** 2 / 0.3
        )
        dx_dt += 0.3 * corridor * y
        dy_dt -= 0.3 * corridor * x
        return torch.stack([dx_dt, dy_dt])


def _soft_confine(value, scale=10.0, strength=0.01):
    return -strength * value**3 / scale**3


@register
class DuffingTripleWell(CatalogSystem):
    name = "duffing_triple_well"
    category = "D"
    description = "Triple-well Duffing oscillator with 3 potential minima"

    def __init__(self, dt=0.02, **kwargs):
        super().__init__(dt=dt, ic_box=[(-2.0, 2.0), (-2.0, 2.0)], **kwargs)
        self.a = 0.3
        self.delta = 0.5
        self.omega = 1.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        d_v_dx = x**5 - 2.0 * x**3 + 2.0 * self.a * x
        dx_dt = y + _soft_confine(x, scale=4.0, strength=0.003)
        dy_dt = -d_v_dx - self.delta * y
        dy_dt = dy_dt + _soft_confine(y, scale=4.0, strength=0.003)
        dx_dt = dx_dt + self.omega * y
        dy_dt = dy_dt - self.omega * x
        return torch.stack([dx_dt, dy_dt])


_EPS = 1e-8


@register
class SNICMulti(CatalogSystem):
    name = "snic_multi"
    category = "E"
    description = (
        "Multiple saddle-node on invariant circle bifurcation with 3 fixed points"
    )

    def __init__(self, dt=0.02, **kwargs):
        super().__init__(dt=dt, ic_box=[(-2.0, 2.0), (-2.0, 2.0)], **kwargs)
        self.a = 1.2
        self.eps = 0.3
        self.omega = 0.5

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        radius_squared = x**2 + y**2 + _EPS
        radius = torch.sqrt(radius_squared)
        theta = torch.atan2(y, x)
        radial_rate = radius * (1.0 - radius_squared) - self.eps * torch.cos(
            3.0 * theta
        )
        angular_rate = 1.0 - self.a * torch.cos(3.0 * theta)
        cosine = x / (radius + _EPS)
        sine = y / (radius + _EPS)
        dx_dt = radial_rate * cosine - radius * angular_rate * sine
        dy_dt = radial_rate * sine + radius * angular_rate * cosine
        dx_dt = dx_dt - 0.01 * x * (x**2 + y**2)
        dy_dt = dy_dt - 0.01 * y * (x**2 + y**2)
        dx_dt = dx_dt + self.omega * y
        dy_dt = dy_dt - self.omega * x
        return torch.stack([dx_dt, dy_dt])


@register
class ArrestedSpiral(CatalogSystem):
    name = "arrested_spiral"
    category = "H"
    description = (
        "Global spiral inflow with local Gaussian wells that capture trajectories "
        "at different radii"
    )

    def __init__(self, dt=0.02, **kwargs):
        super().__init__(dt=dt, ic_box=[(-3.0, 3.0), (-3.0, 3.0)], **kwargs)
        self.alpha_damp = 0.3
        self.omega = 2.0
        self.well_centers = torch.tensor(
            [[1.5, 0.0], [0.0, 1.8], [-1.0, -1.0], [0.8, -1.5]],
            dtype=torch.float64,
        )
        self.well_depth = 4.0
        self.well_sigma2 = 0.25

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        dx_dt = -self.alpha_damp * x + self.omega * y
        dy_dt = -self.alpha_damp * y - self.omega * x
        for center in self.well_centers:
            delta_x = x - center[0]
            delta_y = y - center[1]
            distance_squared = delta_x**2 + delta_y**2
            gaussian = torch.exp(-distance_squared / (2.0 * self.well_sigma2))
            dx_dt = dx_dt - self.well_depth * gaussian * delta_x / self.well_sigma2
            dy_dt = dy_dt - self.well_depth * gaussian * delta_y / self.well_sigma2
        radius_squared = x**2 + y**2
        origin_gaussian = torch.exp(-radius_squared / 0.3)
        dx_dt = dx_dt - 1.5 * origin_gaussian * x / 0.15
        dy_dt = dy_dt - 1.5 * origin_gaussian * y / 0.15
        dx_dt = dx_dt - 0.02 * x**3
        dy_dt = dy_dt - 0.02 * y**3
        return torch.stack([dx_dt, dy_dt])
