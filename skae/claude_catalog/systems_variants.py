"""Variant systems: diverse well configurations all tuned to pass acceptance gates.

All use independent rotation: dx/dt = -dV/dx + omega*y, dy/dt = -dV/dy - omega*x
Calibrated with omega/amp ratios from parameter sweep.
"""

import torch
import math
from skae.claude_catalog.base import CatalogSystem, rk4_step
from skae.claude_catalog.registry import register
from skae.claude_catalog.systems_tuned import GaussianWellIndepRotation


# =========================================================================
# Random well placements (non-symmetric)
# =========================================================================

@register
class RandomWells3v1(GaussianWellIndepRotation):
    """3 non-symmetric wells at hand-picked positions."""
    name = "var_random_3a"
    category = "B"
    description = "3 non-symmetric wells (random placement A)"

    def __init__(self, **kw):
        wells = [
            (-1.0, 2.0, 2.0, 0.5),
            (1.5, -0.5, 2.0, 0.5),
            (-0.5, -1.8, 2.0, 0.5),
        ]
        super().__init__(wells=wells, omega=1.0, conf=0.03, dt=0.03,
                         ic_box=[(-3.0, 3.5), (-3.5, 3.5)], **kw)


@register
class RandomWells4v1(GaussianWellIndepRotation):
    """4 non-symmetric wells."""
    name = "var_random_4a"
    category = "B"
    description = "4 non-symmetric wells (random placement A)"

    def __init__(self, **kw):
        wells = [
            (-2.0, 1.0, 2.5, 0.5),
            (0.5, 2.0, 2.5, 0.5),
            (2.0, -0.5, 2.5, 0.5),
            (-0.5, -2.0, 2.5, 0.5),
        ]
        super().__init__(wells=wells, omega=1.0, conf=0.03, dt=0.03,
                         ic_box=[(-3.5, 3.5), (-3.5, 3.5)], **kw)


@register
class RandomWells5v1(GaussianWellIndepRotation):
    """5 non-symmetric wells."""
    name = "var_random_5a"
    category = "B"
    description = "5 non-symmetric wells (random placement A)"

    def __init__(self, **kw):
        wells = [
            (-1.8, 1.5, 2.0, 0.5),
            (0.0, 2.2, 2.0, 0.5),
            (2.0, 0.5, 2.0, 0.5),
            (1.0, -1.8, 2.0, 0.5),
            (-1.5, -1.0, 2.0, 0.5),
        ]
        super().__init__(wells=wells, omega=1.0, conf=0.03, dt=0.03,
                         ic_box=[(-3.5, 3.5), (-3.5, 3.5)], **kw)


# =========================================================================
# Depth variations (some deep, some shallow)
# =========================================================================

@register
class DepthVariation4(GaussianWellIndepRotation):
    """4 wells with gradient of depths and a widened shallow basin."""
    name = "var_depth_gradient_4"
    category = "B"
    description = "4 wells with depth gradient testing occupancy balance"

    def __init__(self, **kw):
        wells = [
            (-1.3, 1.3, 2.2, 0.55),   # shallowest and slightly wider
            (1.3, 1.3, 2.5, 0.5),
            (1.3, -1.3, 3.0, 0.5),
            (-1.3, -1.3, 3.5, 0.5),   # deepest
        ]
        super().__init__(wells=wells, omega=1.3, conf=0.03, dt=0.03,
                         ic_box=[(-3.0, 3.0), (-3.0, 3.0)], **kw)


@register
class MixedWidths5(GaussianWellIndepRotation):
    """5 wells with different widths (test basin size variation)."""
    name = "var_mixed_widths_5"
    category = "B"
    description = "5 wells with varying widths (narrow/wide basins)"

    def __init__(self, **kw):
        r = 1.8
        widths = [0.35, 0.45, 0.55, 0.65, 0.4]
        wells = [(r*math.cos(2*math.pi*i/5 + math.pi/2),
                  r*math.sin(2*math.pi*i/5 + math.pi/2), 2.0, widths[i])
                 for i in range(5)]
        super().__init__(wells=wells, omega=1.0, conf=0.03, dt=0.03,
                         ic_box=[(-3.3, 3.3), (-3.3, 3.3)], **kw)


# =========================================================================
# Rotation strength variations (crossing fraction variety)
# =========================================================================

@register
class MildRotation5(GaussianWellIndepRotation):
    """5 wells with mild rotation — targets 30% crossing."""
    name = "var_mild_rot_5"
    category = "B"
    description = "5 wells with mild rotation (30% crossing target)"

    def __init__(self, **kw):
        r = 1.8
        wells = [(r*math.cos(2*math.pi*i/5 + math.pi/2),
                  r*math.sin(2*math.pi*i/5 + math.pi/2), 3.0, 0.5)
                 for i in range(5)]
        super().__init__(wells=wells, omega=1.0, conf=0.03, dt=0.03,
                         ic_box=[(-3.3, 3.3), (-3.3, 3.3)], **kw)


@register
class StrongRotation3(GaussianWellIndepRotation):
    """3 wells with strong rotation — targets 60% crossing."""
    name = "var_strong_rot_3"
    category = "B"
    description = "3 wells with strong rotation (60% crossing target)"

    def __init__(self, **kw):
        r = 1.8
        wells = [(r*math.cos(2*math.pi*i/3 + math.pi/2),
                  r*math.sin(2*math.pi*i/3 + math.pi/2), 3.0, 0.5)
                 for i in range(3)]
        super().__init__(wells=wells, omega=2.5, conf=0.04, dt=0.03,
                         ic_box=[(-3.3, 3.3), (-3.3, 3.3)], **kw)


# =========================================================================
# Special geometries
# =========================================================================

@register
class DiamondWells4(GaussianWellIndepRotation):
    """4 wells in diamond (rotated square)."""
    name = "var_diamond_4"
    category = "B"
    description = "4 wells in diamond pattern with rotation"

    def __init__(self, **kw):
        wells = [
            (0.0, 2.2, 2.5, 0.5),   # top
            (2.2, 0.0, 2.5, 0.5),   # right
            (0.0, -2.2, 2.5, 0.5),  # bottom
            (-2.2, 0.0, 2.5, 0.5),  # left
        ]
        super().__init__(wells=wells, omega=1.0, conf=0.02, dt=0.03,
                         ic_box=[(-3.5, 3.5), (-3.5, 3.5)], **kw)


@register
class LShapeWells5(GaussianWellIndepRotation):
    """5 wells in L-shape."""
    name = "var_l_shape_5"
    category = "B"
    description = "5 wells in L-shape (asymmetric topology)"

    def __init__(self, **kw):
        wells = [
            (-1.5, 1.5, 2.5, 0.5),
            (-1.5, 0.0, 2.5, 0.5),
            (-1.5, -1.5, 2.5, 0.5),
            (0.0, -1.5, 2.5, 0.5),
            (1.5, -1.5, 2.5, 0.5),
        ]
        super().__init__(wells=wells, omega=1.0, conf=0.03, dt=0.03,
                         ic_box=[(-3.5, 3.5), (-3.5, 3.5)], **kw)


@register
class TShapeWells5(GaussianWellIndepRotation):
    """5 wells in T-shape."""
    name = "var_t_shape_5"
    category = "B"
    description = "5 wells in T-shape topology"

    def __init__(self, **kw):
        wells = [
            (-2.0, 1.5, 2.5, 0.5),
            (0.0, 1.5, 2.5, 0.5),
            (2.0, 1.5, 2.5, 0.5),
            (0.0, 0.0, 2.5, 0.5),
            (0.0, -1.5, 2.5, 0.5),
        ]
        super().__init__(wells=wells, omega=1.0, conf=0.03, dt=0.03,
                         ic_box=[(-3.5, 3.5), (-3.0, 3.0)], **kw)


@register
class ZigzagWells6(GaussianWellIndepRotation):
    """6 wells in zigzag pattern."""
    name = "var_zigzag_6"
    category = "B"
    description = "6 wells in zigzag pattern"

    def __init__(self, **kw):
        wells = [
            (-2.5, 1.0, 2.5, 0.5),
            (-1.5, -1.0, 2.5, 0.5),
            (-0.5, 1.0, 2.5, 0.5),
            (0.5, -1.0, 2.5, 0.5),
            (1.5, 1.0, 2.5, 0.5),
            (2.5, -1.0, 2.5, 0.5),
        ]
        super().__init__(wells=wells, omega=1.0, conf=0.02, dt=0.03,
                         ic_box=[(-4.0, 4.0), (-2.5, 2.5)], **kw)


@register
class GridWells4(GaussianWellIndepRotation):
    """4 wells on 2x2 grid."""
    name = "var_grid_2x2"
    category = "B"
    description = "4 wells on 2×2 grid with rotation"

    def __init__(self, **kw):
        wells = [
            (-1.5, 1.5, 2.5, 0.5),
            (1.5, 1.5, 2.5, 0.5),
            (-1.5, -1.5, 2.5, 0.5),
            (1.5, -1.5, 2.5, 0.5),
        ]
        super().__init__(wells=wells, omega=1.0, conf=0.03, dt=0.03,
                         ic_box=[(-3.0, 3.0), (-3.0, 3.0)], **kw)


@register
class GridWells9(GaussianWellIndepRotation):
    """9 wells on 3x3 grid (test 9 basins)."""
    name = "var_grid_3x3"
    category = "B"
    description = "9 wells on 3×3 grid with rotation"

    def __init__(self, **kw):
        wells = []
        for i in range(3):
            for j in range(3):
                wells.append((-2.0 + 2.0*i, -2.0 + 2.0*j, 3.0, 0.45))
        super().__init__(wells=wells, omega=1.0, conf=0.02, dt=0.03,
                         ic_box=[(-4.0, 4.0), (-4.0, 4.0)], **kw)


# =========================================================================
# Systems with perturbations beyond simple rotation
# =========================================================================

@register
class VortexTriple(CatalogSystem):
    """3 wells with vortex-like flow between them.

    Each well has a different vorticity, creating asymmetric
    transitions. More interesting than simple uniform rotation.
    """
    name = "var_vortex_triple"
    category = "H"
    description = "3 wells with vortex flow creating asymmetric transitions"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-3.3, 3.3), (-3.3, 3.3)], **kw)
        self.wells = [
            (0.0, 1.8, 2.0, 0.5),
            (-1.5, -0.9, 2.0, 0.5),
            (1.5, -0.9, 2.0, 0.5),
        ]
        self.vorticities = [1.5, -0.8, 2.0]  # different rotation per well

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        dVdx, dVdy = torch.zeros_like(x), torch.zeros_like(y)
        # Gradient
        for cx, cy, amp, sigma in self.wells:
            dx, dy = x-cx, y-cy
            g = amp*torch.exp(-(dx**2+dy**2)/(2*sigma**2))
            dVdx += g*dx/(sigma**2)
            dVdy += g*dy/(sigma**2)
        dVdx += 0.12*x**3
        dVdy += 0.12*y**3
        # Position-dependent rotation
        omega = torch.zeros_like(x)
        total_w = torch.zeros_like(x)
        for (cx, cy, _, sigma), vort in zip(self.wells, self.vorticities):
            d2 = (x-cx)**2 + (y-cy)**2
            w = torch.exp(-d2/(2*sigma**2))
            omega += vort * w
            total_w += w
        omega = omega / (total_w + 0.1)
        omega = omega + 0.5  # base rotation
        dxdt = -dVdx + omega*y
        dydt = -dVdy - omega*x
        return torch.stack([dxdt, dydt])


@register
class ShearQuad(CatalogSystem):
    """4 wells with shear flow (not circular rotation).

    The shear creates directional transition bias:
    left→right transitions are easier than right→left.
    """
    name = "var_shear_quad"
    category = "H"
    description = "4 wells with shear flow creating directional transition bias"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-3.3, 3.3), (-3.3, 3.3)], **kw)
        self.wells = [
            (-1.5, 1.5, 2.5, 0.5),
            (1.5, 1.5, 2.5, 0.5),
            (-1.5, -1.5, 2.5, 0.5),
            (1.5, -1.5, 2.5, 0.5),
        ]

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        dVdx, dVdy = torch.zeros_like(x), torch.zeros_like(y)
        for cx, cy, amp, sigma in self.wells:
            dx, dy = x-cx, y-cy
            g = amp*torch.exp(-(dx**2+dy**2)/(2*sigma**2))
            dVdx += g*dx/(sigma**2)
            dVdy += g*dy/(sigma**2)
        dVdx += 0.12*x**3
        dVdy += 0.12*y**3
        # Shear flow (x-direction push proportional to y)
        dxdt = -dVdx + 1.2*y
        dydt = -dVdy + 0.3*x  # slight return
        return torch.stack([dxdt, dydt])


@register
class RadialModulated5(CatalogSystem):
    """5 wells with radius-modulated rotation creating spiral boundaries."""
    name = "var_radial_mod_5"
    category = "H"
    description = "5 wells with spiral basin boundaries from radial modulation"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-3.3, 3.3), (-3.3, 3.3)], **kw)
        r = 1.8
        self.wells = [(r*math.cos(2*math.pi*i/5 + math.pi/2),
                       r*math.sin(2*math.pi*i/5 + math.pi/2), 2.0, 0.5)
                      for i in range(5)]

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        dVdx, dVdy = torch.zeros_like(x), torch.zeros_like(y)
        for cx, cy, amp, sigma in self.wells:
            dx, dy = x-cx, y-cy
            g = amp*torch.exp(-(dx**2+dy**2)/(2*sigma**2))
            dVdx += g*dx/(sigma**2)
            dVdy += g*dy/(sigma**2)
        dVdx += 0.12*x**3
        dVdy += 0.12*y**3
        # Radially modulated rotation → spiral basin boundaries
        r = torch.sqrt(x**2 + y**2 + 1e-8)
        omega = 0.8 + 0.5*torch.sin(3.0*r)
        dxdt = -dVdx + omega*y
        dydt = -dVdy - omega*x
        return torch.stack([dxdt, dydt])


@register
class TimescaleSeparation4(CatalogSystem):
    """4 wells: 2 fast-settling + 2 slow-settling basins.

    Different decay rates per basin test whether Koopman representation
    adapts to different timescales.
    """
    name = "var_timescale_4"
    category = "H"
    description = "4 basins with 2 fast and 2 slow settling timescales"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-3.3, 3.3), (-3.3, 3.3)], **kw)
        self.centers = torch.tensor([
            [-1.5, 1.5], [1.5, 1.5], [-1.5, -1.5], [1.5, -1.5]
        ], dtype=torch.float64)
        self.rates = [2.0, 0.5, 0.5, 2.0]  # fast, slow, slow, fast

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        pos = state.unsqueeze(0)
        dists = torch.sum((pos - self.centers)**2, dim=1)
        weights = torch.softmax(-dists/2.0, dim=0)
        dxdt, dydt = torch.zeros_like(x), torch.zeros_like(y)
        for i in range(4):
            cx, cy = self.centers[i]
            dx, dy = x-cx, y-cy
            rate = self.rates[i]
            dxdt += weights[i] * (-rate*dx)
            dydt += weights[i] * (-rate*dy)
        # Independent rotation
        dxdt += 1.0*y
        dydt -= 1.0*x
        # Confinement
        dxdt -= 0.01*x**3
        dydt -= 0.01*y**3
        return torch.stack([dxdt, dydt])
