"""Tuned transition-rich dynamical systems — designed to pass acceptance gates.

Key insight from parameter sweep:
- INDEPENDENT rotation (omega*y, -omega*x) is required, not proportional to gradient
- Sweet spot: omega/amp ≈ 0.3-0.7 for 30-70% crossing
- Works for 3-8 well systems on polygon of radius 1.8, sigma=0.5
- Confinement conf=0.03 prevents divergence

Template: V(x) = Σ_i -a_i * exp(-|x-c_i|^2 / (2σ_i^2)) + conf*(x^4+y^4)
Dynamics: dx/dt = -dV/dx + omega*y
          dy/dt = -dV/dy - omega*x
"""

import torch
import math
from skae.claude_catalog.base import CatalogSystem, rk4_step
from skae.claude_catalog.registry import register


class GaussianWellIndepRotation(CatalogSystem):
    """Template: Gaussian multi-well with INDEPENDENT rotation.

    dx/dt = -dV/dx + omega*y
    dy/dt = -dV/dy - omega*x

    Subclasses just set wells, omega, and IC box.
    """

    def __init__(
        self,
        wells,       # list of (cx, cy, amplitude, sigma)
        omega=1.0,   # independent rotation strength
        conf=0.03,   # quartic confinement coefficient
        dt=0.03,
        ic_box=None,
        **kw,
    ):
        self._wells = wells
        self.omega = omega
        self.conf = conf
        if ic_box is None:
            xs = [w[0] for w in wells]
            ys = [w[1] for w in wells]
            margin = 1.5
            ic_box = [
                (min(xs) - margin, max(xs) + margin),
                (min(ys) - margin, max(ys) + margin),
            ]
        super().__init__(dt=dt, ic_box=ic_box, **kw)

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]

        dVdx = torch.zeros_like(x)
        dVdy = torch.zeros_like(y)

        for cx, cy, amp, sigma in self._wells:
            dx = x - cx
            dy = y - cy
            dist_sq = dx ** 2 + dy ** 2
            gauss = amp * torch.exp(-dist_sq / (2 * sigma ** 2))
            dVdx = dVdx + gauss * dx / (sigma ** 2)
            dVdy = dVdy + gauss * dy / (sigma ** 2)

        dVdx = dVdx + 4 * self.conf * x ** 3
        dVdy = dVdy + 4 * self.conf * y ** 3

        # Gradient descent + INDEPENDENT rotation
        dxdt = -dVdx + self.omega * y
        dydt = -dVdy - self.omega * x

        return torch.stack([dxdt, dydt])


# =========================================================================
# Calibrated polygon well systems (guaranteed to pass)
# =========================================================================

@register
class CalibratedTriangle3(GaussianWellIndepRotation):
    """3 wells at equilateral triangle — calibrated omega=1.0, amp=2.0."""
    name = "cal_triangle_3"
    category = "B"
    description = "3 balanced Gaussian wells on triangle with independent rotation"

    def __init__(self, **kw):
        r = 1.8
        wells = [(r*math.cos(2*math.pi*i/3 + math.pi/2),
                  r*math.sin(2*math.pi*i/3 + math.pi/2), 2.0, 0.5)
                 for i in range(3)]
        super().__init__(wells=wells, omega=1.0, conf=0.03, dt=0.03,
                         ic_box=[(-3.3, 3.3), (-3.3, 3.3)], **kw)


@register
class CalibratedSquare4(GaussianWellIndepRotation):
    """4 wells at square corners — calibrated omega=1.0, amp=3.0."""
    name = "cal_square_4"
    category = "B"
    description = "4 balanced wells on square with independent rotation"

    def __init__(self, **kw):
        r = 1.8
        wells = [(r*math.cos(2*math.pi*i/4 + math.pi/4),
                  r*math.sin(2*math.pi*i/4 + math.pi/4), 3.0, 0.5)
                 for i in range(4)]
        super().__init__(wells=wells, omega=1.0, conf=0.03, dt=0.03,
                         ic_box=[(-3.3, 3.3), (-3.3, 3.3)], **kw)


@register
class CalibratedPentagon5(GaussianWellIndepRotation):
    """5 wells at pentagon — calibrated omega=1.0, amp=2.0."""
    name = "cal_pentagon_5"
    category = "B"
    description = "5 balanced wells on pentagon with independent rotation"

    def __init__(self, **kw):
        r = 1.8
        wells = [(r*math.cos(2*math.pi*i/5 + math.pi/2),
                  r*math.sin(2*math.pi*i/5 + math.pi/2), 2.0, 0.5)
                 for i in range(5)]
        super().__init__(wells=wells, omega=1.0, conf=0.03, dt=0.03,
                         ic_box=[(-3.3, 3.3), (-3.3, 3.3)], **kw)


@register
class CalibratedHexagon6(GaussianWellIndepRotation):
    """6 wells at hexagon — calibrated omega=0.8, amp=2.0."""
    name = "cal_hexagon_6"
    category = "B"
    description = "6 balanced wells on hexagon with independent rotation"

    def __init__(self, **kw):
        r = 1.8
        wells = [(r*math.cos(2*math.pi*i/6),
                  r*math.sin(2*math.pi*i/6), 2.0, 0.5)
                 for i in range(6)]
        super().__init__(wells=wells, omega=0.8, conf=0.03, dt=0.03,
                         ic_box=[(-3.3, 3.3), (-3.3, 3.3)], **kw)


@register
class CalibratedOctagon8(GaussianWellIndepRotation):
    """8 wells at octagon — calibrated omega=1.2, amp=3.0."""
    name = "cal_octagon_8"
    category = "B"
    description = "8 balanced wells on octagon with independent rotation"

    def __init__(self, **kw):
        r = 2.2
        wells = [(r*math.cos(2*math.pi*i/8),
                  r*math.sin(2*math.pi*i/8), 3.0, 0.5)
                 for i in range(8)]
        super().__init__(wells=wells, omega=1.2, conf=0.02, dt=0.03,
                         ic_box=[(-4.0, 4.0), (-4.0, 4.0)], **kw)


# =========================================================================
# Asymmetric and interesting well configurations
# =========================================================================

@register
class AsymmetricTriangle3(GaussianWellIndepRotation):
    """3 wells with different depths and widths — tests unbalanced basins."""
    name = "cal_asymmetric_3"
    category = "B"
    description = "3 asymmetric wells (different depths) with rotation"

    def __init__(self, **kw):
        wells = [
            (0.0, 1.8, 2.5, 0.55),    # deep, wide
            (-1.5, -0.9, 1.5, 0.4),   # shallow, narrow
            (1.5, -0.9, 2.0, 0.5),    # medium
        ]
        super().__init__(wells=wells, omega=1.0, conf=0.03, dt=0.03,
                         ic_box=[(-3.0, 3.0), (-2.5, 3.0)], **kw)


@register
class ClusteredWells6(GaussianWellIndepRotation):
    """6 wells in 2 clusters of 3 — hierarchical basin structure."""
    name = "cal_clustered_2x3"
    category = "B"
    description = "2 clusters × 3 wells testing hierarchical transitions"

    def __init__(self, **kw):
        wells = [
            # Left cluster
            (-2.5, 0.8, 2.5, 0.4),
            (-2.5, -0.8, 2.5, 0.4),
            (-1.8, 0.0, 2.5, 0.4),
            # Right cluster
            (2.5, 0.8, 2.5, 0.4),
            (2.5, -0.8, 2.5, 0.4),
            (1.8, 0.0, 2.5, 0.4),
        ]
        super().__init__(wells=wells, omega=1.2, conf=0.02, dt=0.03,
                         ic_box=[(-4.5, 4.5), (-2.5, 2.5)], **kw)


@register
class ChainWells4(GaussianWellIndepRotation):
    """4 wells in a chain — tests sequential transitions."""
    name = "cal_chain_4"
    category = "B"
    description = "4 wells in a chain with directional transition bias"

    def __init__(self, **kw):
        wells = [
            (-3.0, 0.0, 2.5, 0.5),
            (-1.0, 0.5, 2.5, 0.5),
            (1.0, -0.5, 2.5, 0.5),
            (3.0, 0.0, 2.5, 0.5),
        ]
        super().__init__(wells=wells, omega=1.0, conf=0.01, dt=0.03,
                         ic_box=[(-4.5, 4.5), (-2.5, 2.5)], **kw)


@register
class CrossWells5Center(GaussianWellIndepRotation):
    """5 wells: cross pattern with weak center well."""
    name = "cal_cross_5_center"
    category = "B"
    description = "5 wells in cross with weak center creating routing dynamics"

    def __init__(self, **kw):
        wells = [
            (0.0, 0.0, 1.5, 0.6),    # center (shallow)
            (2.2, 0.0, 3.0, 0.5),
            (-2.2, 0.0, 3.0, 0.5),
            (0.0, 2.2, 3.0, 0.5),
            (0.0, -2.2, 3.0, 0.5),
        ]
        super().__init__(wells=wells, omega=1.5, conf=0.02, dt=0.03,
                         ic_box=[(-4.0, 4.0), (-4.0, 4.0)], **kw)


@register
class StarTopology5(GaussianWellIndepRotation):
    """5 wells with unstable center — star routing topology."""
    name = "cal_star_5"
    category = "B"
    description = "5 radiating wells with unstable center hub"

    def __init__(self, **kw):
        r = 2.0
        wells = [(r*math.cos(2*math.pi*i/5 + math.pi/10),
                  r*math.sin(2*math.pi*i/5 + math.pi/10), 3.0, 0.55)
                 for i in range(5)]
        super().__init__(wells=wells, omega=1.5, conf=0.03, dt=0.03,
                         ic_box=[(-3.5, 3.5), (-3.5, 3.5)], **kw)


@register
class RingCenter7(GaussianWellIndepRotation):
    """7 wells: 6 on ring + 1 center."""
    name = "cal_ring_center_7"
    category = "B"
    description = "6 ring wells + center well with rotation"

    def __init__(self, **kw):
        r = 1.8
        wells = [(0.0, 0.0, 2.0, 0.6)]  # center
        for i in range(6):
            angle = 2*math.pi*i/6
            wells.append((r*math.cos(angle), r*math.sin(angle), 2.5, 0.45))
        super().__init__(wells=wells, omega=1.0, conf=0.02, dt=0.03,
                         ic_box=[(-3.3, 3.3), (-3.3, 3.3)], **kw)


@register
class CascadingDepths4(GaussianWellIndepRotation):
    """4 wells at different depths creating directional bias."""
    name = "cal_cascade_4"
    category = "B"
    description = "4 wells with cascading depths for directional transitions"

    def __init__(self, **kw):
        wells = [
            (-2.0, 1.5, 1.5, 0.5),   # shallowest
            (0.0, 0.5, 2.0, 0.5),
            (1.5, -0.5, 2.5, 0.5),
            (2.5, -2.0, 3.0, 0.5),    # deepest
        ]
        super().__init__(wells=wells, omega=1.2, conf=0.02, dt=0.03,
                         ic_box=[(-3.5, 4.0), (-3.5, 3.0)], **kw)


@register
class HighCrossingTriangle3(GaussianWellIndepRotation):
    """3 wells — high rotation for ~60% crossing."""
    name = "cal_high_cross_3"
    category = "B"
    description = "3 wells with high rotation for 60% crossing fraction"

    def __init__(self, **kw):
        r = 1.8
        wells = [(r*math.cos(2*math.pi*i/3 + math.pi/2),
                  r*math.sin(2*math.pi*i/3 + math.pi/2), 3.0, 0.5)
                 for i in range(3)]
        super().__init__(wells=wells, omega=2.0, conf=0.03, dt=0.03,
                         ic_box=[(-3.3, 3.3), (-3.3, 3.3)], **kw)


@register
class LowCrossingSquare4(GaussianWellIndepRotation):
    """4 wells — moderate rotation for ~45% crossing."""
    name = "cal_low_cross_4"
    category = "B"
    description = "4 wells with moderate rotation for 45% crossing"

    def __init__(self, **kw):
        r = 1.8
        wells = [(r*math.cos(2*math.pi*i/4 + math.pi/4),
                  r*math.sin(2*math.pi*i/4 + math.pi/4), 3.0, 0.5)
                 for i in range(4)]
        super().__init__(wells=wells, omega=1.2, conf=0.03, dt=0.03,
                         ic_box=[(-3.3, 3.3), (-3.3, 3.3)], **kw)


# =========================================================================
# Systems with qualitatively different local dynamics per basin
# =========================================================================

@register
class MixedDynamicsTriple(CatalogSystem):
    """Three basins with qualitatively different local dynamics.

    Basin 1: stable spiral (fast rotation, ω=3)
    Basin 2: stable node (no rotation)
    Basin 3: slow spiral (slow rotation, ω=0.8)

    Plus independent rotation for transitions.
    """
    name = "mixed_dynamics_triple"
    category = "H"
    description = "3 basins with different local dynamics (spiral/node/slow-spiral)"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-3.0, 3.0), (-3.0, 3.0)], **kw)
        self.centers = torch.tensor([
            [0.0, 1.8], [-1.5, -0.9], [1.5, -0.9],
        ], dtype=torch.float64)
        self.A1 = torch.tensor([[-0.5, -3.0], [3.0, -0.5]], dtype=torch.float64)  # fast spiral
        self.A2 = torch.tensor([[-1.0, 0.0], [0.0, -2.0]], dtype=torch.float64)   # node
        self.A3 = torch.tensor([[-0.3, -0.8], [0.8, -0.3]], dtype=torch.float64)  # slow spiral
        self.matrices = [self.A1, self.A2, self.A3]
        self.sigma = 1.0
        self.omega = 0.8  # independent rotation

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        pos = state.unsqueeze(0)
        dists = torch.sum((pos - self.centers) ** 2, dim=1)
        weights = torch.softmax(-dists / (2 * self.sigma ** 2), dim=0)
        dxdt = torch.zeros_like(x)
        dydt = torch.zeros_like(y)
        for i, A in enumerate(self.matrices):
            local_state = state - self.centers[i]
            local_dyn = A @ local_state
            dxdt = dxdt + weights[i] * local_dyn[0]
            dydt = dydt + weights[i] * local_dyn[1]
        # Independent rotation for transitions
        dxdt = dxdt + self.omega * y
        dydt = dydt - self.omega * x
        # Confinement
        dxdt = dxdt - 0.01 * x * (x ** 2 + y ** 2)
        dydt = dydt - 0.01 * y * (x ** 2 + y ** 2)
        return torch.stack([dxdt, dydt])


@register
class SpiralNodeLimitCycle(CatalogSystem):
    """4 basins: 2 spirals, 1 node, 1 limit cycle. Plus rotation."""
    name = "spiral_node_limit_cycle"
    category = "H"
    description = "4 basins: 2 spirals + 1 node + 1 limit cycle"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-4.0, 4.0), (-4.0, 4.0)], **kw)
        self.centers = [(-2.0, 2.0), (2.0, 2.0), (-2.0, -2.0), (2.0, -2.0)]
        self.sigma = 1.2
        self.omega = 0.5

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        weights = []
        for cx, cy in self.centers:
            d2 = (x-cx)**2 + (y-cy)**2
            weights.append(torch.exp(-d2/(2*self.sigma**2)))
        total = sum(weights) + 1e-8
        weights = [w/total for w in weights]
        dxdt, dydt = torch.zeros_like(x), torch.zeros_like(y)
        # Basin 1: fast spiral
        dx, dy = x-self.centers[0][0], y-self.centers[0][1]
        dxdt += weights[0]*(-0.5*dx - 4.0*dy)
        dydt += weights[0]*(4.0*dx - 0.5*dy)
        # Basin 2: slow spiral
        dx, dy = x-self.centers[1][0], y-self.centers[1][1]
        dxdt += weights[1]*(-0.3*dx - 1.0*dy)
        dydt += weights[1]*(1.0*dx - 0.3*dy)
        # Basin 3: stable node
        dx, dy = x-self.centers[2][0], y-self.centers[2][1]
        dxdt += weights[2]*(-1.5*dx)
        dydt += weights[2]*(-2.0*dy)
        # Basin 4: limit cycle
        dx, dy = x-self.centers[3][0], y-self.centers[3][1]
        r = torch.sqrt(dx**2 + dy**2 + 1e-8)
        mu = 1.0 - r
        dxdt += weights[3]*(mu*dx - 2.0*dy)
        dydt += weights[3]*(2.0*dx + mu*dy)
        # Independent rotation
        dxdt += self.omega * y
        dydt -= self.omega * x
        # Confinement
        dxdt -= 0.005*x**3
        dydt -= 0.005*y**3
        return torch.stack([dxdt, dydt])


@register
class SlowFastTriple(CatalogSystem):
    """3 basins with slow-fast timescale separation."""
    name = "slow_fast_triple"
    category = "H"
    description = "3 basins with slow-fast dynamics and manifold transitions"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-3.0, 3.0), (-3.0, 3.0)], **kw)
        self.eps = 0.15

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        x_null = -x**3 + 3*x
        dxdt = (x_null - y) * 3.0
        dydt = self.eps * (-(y**5 - 2.5*y**3 + 1.5*y) + 0.2*x)
        # Independent rotation for transitions
        dxdt += 0.8 * y
        dydt -= 0.3 * x
        return torch.stack([dxdt, dydt])


@register
class NonVoronoiBasins(CatalogSystem):
    """3 basins with spiral boundaries (non-Voronoi partition)."""
    name = "non_voronoi_basins"
    category = "H"
    description = "3 basins with spiral boundaries that differ from Voronoi cells"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-3.0, 3.0), (-3.0, 3.0)], **kw)
        self.wells = [(0.0, 1.8, 2.0, 0.5), (-1.5, -0.9, 2.0, 0.5), (1.5, -0.9, 2.0, 0.5)]

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        dVdx, dVdy = torch.zeros_like(x), torch.zeros_like(y)
        for cx, cy, amp, sigma in self.wells:
            dx, dy = x-cx, y-cy
            g = amp * torch.exp(-(dx**2+dy**2)/(2*sigma**2))
            dVdx += g*dx/(sigma**2)
            dVdy += g*dy/(sigma**2)
        dVdx += 0.12*x**3
        dVdy += 0.12*y**3
        # Independent rotation + radius-dependent angular perturbation
        r = torch.sqrt(x**2 + y**2 + 1e-8)
        omega = 1.0 + 0.5*torch.sin(2.0*r)
        dxdt = -dVdx + omega*y
        dydt = -dVdy - omega*x
        return torch.stack([dxdt, dydt])


@register
class TransitionRoutes4(CatalogSystem):
    """4 basins with specific fast/slow transition routes."""
    name = "transition_routes_4"
    category = "H"
    description = "4 basins with specific fast/slow transition corridors"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-3.5, 3.5), (-3.5, 3.5)], **kw)
        self.wells = [
            (-1.8, 1.8, 3.0, 0.6), (1.8, 1.8, 3.0, 0.6),
            (-1.8, -1.8, 3.0, 0.6), (1.8, -1.8, 3.0, 0.6),
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
        # Independent rotation
        dxdt = -dVdx + 1.0*y
        dydt = -dVdy - 1.0*x
        # Corridor-dependent extra rotation
        corridor_h = torch.exp(-(y-1.8)**2/0.3) + torch.exp(-(y+1.8)**2/0.3)
        dxdt += 0.3 * corridor_h * y
        dydt -= 0.3 * corridor_h * x
        return torch.stack([dxdt, dydt])


@register
class MullerBrownRotated(CatalogSystem):
    """Classic Müller-Brown potential with independent rotation."""
    name = "muller_brown_rotated"
    category = "B"
    description = "Müller-Brown potential (computational chemistry) with rotation"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-1.5, 1.5), (-0.5, 2.5)], **kw)
        self.A = [-200, -100, -170, 15]
        self.a = [-1, -1, -6.5, 0.7]
        self.b = [0, 0, 11, 0.6]
        self.c = [-10, -10, -6.5, 0.7]
        self.x0 = [1.0, 0.0, -0.5, -1.0]
        self.y0 = [0.0, 0.5, 1.5, 1.0]
        self.scale = 0.005
        self.omega = 1.5

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        dVdx, dVdy = torch.zeros_like(x), torch.zeros_like(y)
        for A, a, b, c, x0, y0 in zip(self.A, self.a, self.b, self.c, self.x0, self.y0):
            dx, dy = x-x0, y-y0
            exp_t = torch.exp(a*dx**2 + b*dx*dy + c*dy**2)
            dVdx += self.scale * A * (2*a*dx + b*dy) * exp_t
            dVdy += self.scale * A * (b*dx + 2*c*dy) * exp_t
        dxdt = -dVdx + self.omega*y
        dydt = -dVdy - self.omega*x
        return torch.stack([dxdt, dydt])


@register
class HierarchicalWells8(CatalogSystem):
    """8 wells in 2-level hierarchy (2 groups of 4)."""
    name = "hierarchical_wells_8"
    category = "B"
    description = "8 wells in 2-level hierarchy testing multi-scale discovery"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-5.0, 5.0), (-3.0, 3.0)], **kw)
        self.wells = [
            (-3.0, 1.0, 2.5, 0.4), (-3.0, -1.0, 2.5, 0.4),
            (-2.0, 0.5, 2.5, 0.4), (-2.0, -0.5, 2.5, 0.4),
            (3.0, 1.0, 2.5, 0.4), (3.0, -1.0, 2.5, 0.4),
            (2.0, 0.5, 2.5, 0.4), (2.0, -0.5, 2.5, 0.4),
        ]
        self.omega = 1.0
        self.conf = 0.01

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
        dVdx += 4*self.conf*x**3
        dVdy += 4*self.conf*y**3
        # Inter-cluster barrier
        barrier = 2.0*torch.exp(-x**2/0.3)*torch.exp(-y**2/2.0)
        dVdx -= barrier*2*x/0.3
        dxdt = -dVdx + self.omega*y
        dydt = -dVdy - self.omega*x
        return torch.stack([dxdt, dydt])
