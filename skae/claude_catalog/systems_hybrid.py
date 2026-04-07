"""Hybrid systems: guaranteed-passing well templates + interesting dynamics.

These combine the calibrated Gaussian well + independent rotation formula
(which reliably passes acceptance gates) with creative dynamics modifications
that make each system unique and scientifically interesting.
"""

import torch
import math
from skae.claude_catalog.base import CatalogSystem, rk4_step
from skae.claude_catalog.registry import register


# ---------------------------------------------------------------------------
# System 1: Anisotropic wells — elliptical basins
# ---------------------------------------------------------------------------
@register
class AnisotropicWells4(CatalogSystem):
    """4 wells with different anisotropies (elliptical basins).

    Each well has a different aspect ratio (sigma_x != sigma_y).
    This means the local dynamics near each basin are qualitatively
    different — some attract faster in x, others in y.
    Tests whether Koopman learns distinct local linearizations.
    """
    name = "hybrid_anisotropic_4"
    category = "H"
    description = "4 wells with different elliptical shapes per basin"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-3.3, 3.3), (-3.3, 3.3)], **kw)
        self.wells = [
            # (cx, cy, amp, sigma_x, sigma_y)
            (-1.5, 1.5, 2.5, 0.4, 0.7),   # tall ellipse
            (1.5, 1.5, 2.5, 0.7, 0.4),    # wide ellipse
            (-1.5, -1.5, 2.5, 0.5, 0.5),  # circle
            (1.5, -1.5, 2.5, 0.3, 0.8),   # very tall ellipse
        ]
        self.omega = 1.0
        self.conf = 0.03

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        dVdx = torch.zeros_like(x)
        dVdy = torch.zeros_like(y)
        for cx, cy, amp, sx, sy in self.wells:
            dx, dy = x - cx, y - cy
            gauss = amp * torch.exp(-dx**2 / (2*sx**2) - dy**2 / (2*sy**2))
            dVdx = dVdx + gauss * dx / (sx**2)
            dVdy = dVdy + gauss * dy / (sy**2)
        dVdx = dVdx + 4 * self.conf * x**3
        dVdy = dVdy + 4 * self.conf * y**3
        dxdt = -dVdx + self.omega * y
        dydt = -dVdy - self.omega * x
        return torch.stack([dxdt, dydt])


# ---------------------------------------------------------------------------
# System 2: Coupled well + limit cycle
# ---------------------------------------------------------------------------
@register
class WellsPlusLimitCycle(CatalogSystem):
    """3 wells inside a stable limit cycle.

    The limit cycle acts as a 4th basin (for trajectories that spiral
    rather than settling into a well). Tests whether Koopman can
    distinguish point attractors from periodic attractors.
    """
    name = "hybrid_wells_limit_cycle"
    category = "H"
    description = "3 wells inside a limit cycle (4 basins: 3 points + 1 periodic)"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-3.5, 3.5), (-3.5, 3.5)], **kw)
        r = 1.2
        self.wells = [(r*math.cos(2*math.pi*i/3 + math.pi/2),
                       r*math.sin(2*math.pi*i/3 + math.pi/2), 3.0, 0.4)
                      for i in range(3)]
        self.lc_radius = 2.5
        self.lc_omega = 1.5

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        r = torch.sqrt(x**2 + y**2 + 1e-8)

        # Well gradient
        dVdx = torch.zeros_like(x)
        dVdy = torch.zeros_like(y)
        for cx, cy, amp, sigma in self.wells:
            dx, dy = x - cx, y - cy
            gauss = amp * torch.exp(-(dx**2 + dy**2) / (2*sigma**2))
            dVdx = dVdx + gauss * dx / (sigma**2)
            dVdy = dVdy + gauss * dy / (sigma**2)

        # Limit cycle (Stuart-Landau): dr/dt = mu*r*(1 - r^2/R^2), dtheta/dt = omega
        R = self.lc_radius
        mu = 0.3  # growth rate
        radial = mu * (1 - r**2 / R**2)

        # Combine: wells dominate inside, limit cycle dominates outside
        well_weight = torch.exp(-r**2 / 3.0)  # strong near origin
        lc_weight = 1 - well_weight

        dxdt = -well_weight * dVdx + lc_weight * (radial * x - self.lc_omega * y)
        dydt = -well_weight * dVdy + lc_weight * (radial * y + self.lc_omega * x)

        # Small independent rotation for well-region transitions
        dxdt = dxdt + 0.5 * well_weight * y
        dydt = dydt - 0.5 * well_weight * x

        return torch.stack([dxdt, dydt])


# ---------------------------------------------------------------------------
# System 3: State-dependent rotation strength
# ---------------------------------------------------------------------------
@register
class StateDepRotation5(CatalogSystem):
    """5 wells where rotation strength varies by position.

    Near the origin: strong rotation (high crossing).
    Near the edges: weak rotation (low crossing).
    Creates a gradient of crossing fraction across the state space.
    """
    name = "hybrid_state_dep_rot_5"
    category = "H"
    description = "5 wells with position-dependent rotation creating spatial crossing gradient"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-3.3, 3.3), (-3.3, 3.3)], **kw)
        r = 1.8
        self.wells = [(r*math.cos(2*math.pi*i/5 + math.pi/2),
                       r*math.sin(2*math.pi*i/5 + math.pi/2), 2.0, 0.5)
                      for i in range(5)]
        self.conf = 0.03

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        dVdx = torch.zeros_like(x)
        dVdy = torch.zeros_like(y)
        for cx, cy, amp, sigma in self.wells:
            dx, dy = x - cx, y - cy
            gauss = amp * torch.exp(-(dx**2 + dy**2) / (2*sigma**2))
            dVdx = dVdx + gauss * dx / (sigma**2)
            dVdy = dVdy + gauss * dy / (sigma**2)
        dVdx = dVdx + 4 * self.conf * x**3
        dVdy = dVdy + 4 * self.conf * y**3

        # State-dependent rotation: stronger near origin
        r_sq = x**2 + y**2
        omega = 1.5 * torch.exp(-r_sq / 8.0) + 0.3

        dxdt = -dVdx + omega * y
        dydt = -dVdy - omega * x
        return torch.stack([dxdt, dydt])


# ---------------------------------------------------------------------------
# System 4: Rotating well centers (quasi-periodic)
# ---------------------------------------------------------------------------
@register
class RotatingCenters3(CatalogSystem):
    """3 wells that effectively rotate due to angular potential gradient.

    The wells are on a rotating potential landscape, creating
    a system where the "preferred" basin changes depending on
    the angular position. Tests adaptation to rotating frames.
    """
    name = "hybrid_rotating_centers_3"
    category = "H"
    description = "3 wells with angular bias creating rotational preference"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-3.3, 3.3), (-3.3, 3.3)], **kw)
        r = 1.8
        self.wells = [(r*math.cos(2*math.pi*i/3 + math.pi/2),
                       r*math.sin(2*math.pi*i/3 + math.pi/2), 2.5, 0.5)
                      for i in range(3)]
        self.conf = 0.03
        self.omega = 1.0
        self.angular_bias = 0.3

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        theta = torch.atan2(y, x)

        dVdx = torch.zeros_like(x)
        dVdy = torch.zeros_like(y)
        for cx, cy, amp, sigma in self.wells:
            dx, dy = x - cx, y - cy
            # Angular-modulated well depth
            well_angle = math.atan2(cy, cx)
            angle_mod = 1.0 + self.angular_bias * torch.cos(theta - well_angle)
            gauss = amp * angle_mod * torch.exp(-(dx**2 + dy**2) / (2*sigma**2))
            dVdx = dVdx + gauss * dx / (sigma**2)
            dVdy = dVdy + gauss * dy / (sigma**2)
        dVdx = dVdx + 4 * self.conf * x**3
        dVdy = dVdy + 4 * self.conf * y**3
        dxdt = -dVdx + self.omega * y
        dydt = -dVdy - self.omega * x
        return torch.stack([dxdt, dydt])


# ---------------------------------------------------------------------------
# System 5: Saddle bridge between well clusters
# ---------------------------------------------------------------------------
@register
class SaddleBridge6(CatalogSystem):
    """6 wells in 2 clusters connected by a saddle bridge.

    Two groups of 3 wells separated by a potential ridge with a
    narrow saddle pass. Transitions within a cluster are easy,
    cross-cluster transitions are rare and route-specific.
    Tests hierarchical basin discovery.
    """
    name = "hybrid_saddle_bridge_6"
    category = "H"
    description = "2 clusters × 3 wells connected by a saddle bridge"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-4.5, 4.5), (-3.0, 3.0)], **kw)
        # Left cluster
        self.wells = [
            (-2.5, 1.0, 2.5, 0.45),
            (-2.5, -1.0, 2.5, 0.45),
            (-1.5, 0.0, 2.5, 0.45),
        ]
        # Right cluster
        self.wells += [
            (2.5, 1.0, 2.5, 0.45),
            (2.5, -1.0, 2.5, 0.45),
            (1.5, 0.0, 2.5, 0.45),
        ]
        self.omega = 1.0
        self.conf = 0.015

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        dVdx = torch.zeros_like(x)
        dVdy = torch.zeros_like(y)
        for cx, cy, amp, sigma in self.wells:
            dx, dy = x - cx, y - cy
            gauss = amp * torch.exp(-(dx**2 + dy**2) / (2*sigma**2))
            dVdx = dVdx + gauss * dx / (sigma**2)
            dVdy = dVdy + gauss * dy / (sigma**2)

        # Potential ridge between clusters (high barrier except at saddle)
        ridge = 3.0 * torch.exp(-x**2 / 0.2) * (1 - torch.exp(-y**2 / 0.3))
        dVdx = dVdx - ridge * 2 * x / 0.2

        # Saddle pass enhancer at y ≈ 0
        saddle = -1.5 * torch.exp(-x**2 / 0.3) * torch.exp(-y**2 / 0.3)
        dVdx = dVdx - saddle * 2 * x / 0.3
        dVdy = dVdy - saddle * 2 * y / 0.3

        dVdx = dVdx + 4 * self.conf * x**3
        dVdy = dVdy + 4 * self.conf * y**3

        dxdt = -dVdx + self.omega * y
        dydt = -dVdy - self.omega * x
        return torch.stack([dxdt, dydt])


# ---------------------------------------------------------------------------
# System 6: Multi-frequency wells
# ---------------------------------------------------------------------------
@register
class MultiFreqWells4(CatalogSystem):
    """4 wells where local dynamics have different oscillation frequencies.

    Near each well, the approach trajectory oscillates at a different
    frequency. This tests whether Koopman can represent multiple
    frequencies with a single linear operator.
    """
    name = "hybrid_multi_freq_4"
    category = "H"
    description = "4 wells with different local oscillation frequencies"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-3.3, 3.3), (-3.3, 3.3)], **kw)
        self.centers = torch.tensor([
            [-1.5, 1.5], [1.5, 1.5], [-1.5, -1.5], [1.5, -1.5]
        ], dtype=torch.float64)
        # Different frequencies per basin
        self.freqs = [1.0, 3.0, 0.5, 2.0]
        self.decay = 0.8
        self.omega_global = 0.8

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        pos = state.unsqueeze(0)

        # Compute blending weights
        dists = torch.sum((pos - self.centers)**2, dim=1)
        weights = torch.softmax(-dists / 2.0, dim=0)

        dxdt = torch.zeros_like(x)
        dydt = torch.zeros_like(y)

        for i in range(4):
            cx, cy = self.centers[i]
            dx, dy = x - cx, y - cy
            freq = self.freqs[i]
            # Local dynamics: spiral with frequency-dependent rotation
            dxdt = dxdt + weights[i] * (-self.decay * dx - freq * dy)
            dydt = dydt + weights[i] * (freq * dx - self.decay * dy)

        # Global independent rotation for transitions
        dxdt = dxdt + self.omega_global * y
        dydt = dydt - self.omega_global * x

        # Confinement
        dxdt = dxdt - 0.01 * x**3
        dydt = dydt - 0.01 * y**3
        return torch.stack([dxdt, dydt])
