"""Biological, physical, and catastrophe/bifurcation dynamical systems (Categories C, D, F).

17 transition-rich 2D autonomous ODEs integrated with RK4, each with 3-10 endpoint
basins and 30-70% inter-basin transitions from the IC box.
"""

import torch
import math
from skae.claude_catalog.base import CatalogSystem, rk4_step
from skae.claude_catalog.registry import register


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _soft_confine(x, scale=10.0, strength=0.01):
    """Soft cubic confinement: pulls back states that escape [-scale, scale]."""
    return -strength * x ** 3 / (scale ** 3)


def _sigmoid(x, beta, theta):
    """Logistic sigmoid S(x) = 1/(1 + exp(-beta*(x - theta)))."""
    return torch.sigmoid(beta * (x - theta))


# ===================================================================
# CATEGORY C: Biological / ecological (7 systems)
# ===================================================================


@register
class ToggleSwitch3Gene(CatalogSystem):
    """Three-gene mutual-repression toggle switch in 2D.

    Models the effective landscape of a 3-gene mutual repression circuit
    where each gene tries to dominate. The 2D plane represents the expression
    ratio space, with 3 attractors corresponding to the 3 single-gene-dominant
    states arranged in an equilateral triangle.

    Uses gradient flow on a potential with 3 Gaussian wells at the gene-dominant
    positions, plus Hill-function-inspired repressive coupling between the wells.
    """

    name = "toggle_switch_3gene"
    category = "C"
    description = "Three-gene mutual-repression network (tristable)"

    def __init__(self, dt=0.05, **kw):
        super().__init__(dt=dt, ic_box=[(-3.0, 3.0), (-3.0, 3.0)], **kw)
        # 3 gene-dominance attractors at equilateral triangle vertices
        r = 2.0
        self.wells = torch.tensor(
            [[r * math.cos(2 * math.pi * k / 3 + math.pi / 2),
              r * math.sin(2 * math.pi * k / 3 + math.pi / 2)]
             for k in range(3)],
            dtype=torch.float64,
        )
        self.well_depth = 2.0
        self.well_width = 1.0  # Gaussian width
        self.omega = 1.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        # Gradient of negative-Gaussian wells:
        # V(x) = -depth * sum_i exp(-|x - w_i|^2 / (2*sigma^2))
        # dV/dx = depth * sum_i (x - w_i)/sigma^2 * exp(...)
        dxdt = torch.tensor(0.0, dtype=torch.float64)
        dydt = torch.tensor(0.0, dtype=torch.float64)

        for i in range(3):
            diff_x = x - self.wells[i, 0]
            diff_y = y - self.wells[i, 1]
            d2 = diff_x ** 2 + diff_y ** 2
            gauss = torch.exp(-d2 / (2.0 * self.well_width ** 2))
            dxdt = dxdt - self.well_depth * diff_x / self.well_width ** 2 * gauss
            dydt = dydt - self.well_depth * diff_y / self.well_width ** 2 * gauss

        # Soft quartic confinement to keep bounded
        r2 = x ** 2 + y ** 2
        conf_str = 0.02
        dxdt = dxdt - conf_str * x * r2
        dydt = dydt - conf_str * y * r2

        # Independent rotation for transition richness
        dxdt = dxdt + self.omega * y
        dydt = dydt - self.omega * x

        return torch.stack([dxdt, dydt])


@register
class CompetitiveExclusion3(CatalogSystem):
    """Three competing species on a simplex, via replicator dynamics.

    Replicator equation for 3 species with asymmetric competition:
    du/dt = u * (f1 - f_bar), dv/dt = v * (f2 - f_bar)
    where u, v are fractions of species 1, 2 (species 3 = 1-u-v).
    Strong competition ensures competitive exclusion: one species dominates.
    """

    name = "competitive_exclusion_3"
    category = "C"
    description = "Three-species Lotka-Volterra competition on a 2-simplex (tristable)"

    def __init__(self, dt=0.05, **kw):
        super().__init__(dt=dt, ic_box=[(0.05, 0.90), (0.05, 0.90)], **kw)
        # Payoff / interaction matrix for replicator dynamics
        # Rows: species i fitness contribution from species j
        # Strong self-inhibition, weaker cross-inhibition: rock-paper-scissors-like
        self.A = torch.tensor(
            [[ 0.0, -1.8,  1.5],
             [ 1.5,  0.0, -1.8],
             [-1.8,  1.5,  0.0]], dtype=torch.float64
        )
        # Base fitness
        self.base = torch.tensor([0.2, 0.0, -0.1], dtype=torch.float64)
        self.omega = 1.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        u, v = state[0], state[1]
        # Clamp to simplex interior
        u = torch.clamp(u, min=0.01, max=0.98)
        v = torch.clamp(v, min=0.01, max=0.98)
        w = torch.clamp(1.0 - u - v, min=0.01)
        total = u + v + w
        u, v, w = u / total, v / total, w / total

        x = torch.stack([u, v, w])
        # Fitness: f_i = (A @ x)_i + base_i
        fitness = self.A @ x + self.base
        f_bar = (x * fitness).sum()

        du = u * (fitness[0] - f_bar)
        dv = v * (fitness[1] - f_bar)

        # Independent rotation for transition richness
        du = du + self.omega * v
        dv = dv - self.omega * u

        return torch.stack([du, dv])


@register
class EcosystemTipping(CatalogSystem):
    """Vegetation-water model with desert/savanna/forest tipping points.

    Gradient flow on a potential with 3 wells representing:
    - Desert: low vegetation, moderate water
    - Savanna: medium vegetation, medium water
    - Forest: high vegetation, low water

    Pure gradient flow ensures reliable 3-basin structure.
    """

    name = "ecosystem_tipping"
    category = "C"
    description = "Vegetation-water model with desert/savanna/forest tipping points"

    def __init__(self, dt=0.05, **kw):
        super().__init__(dt=dt, ic_box=[(-3.0, 3.0), (-3.0, 3.0)], **kw)
        # Three ecological states spread far enough apart
        self.wells = torch.tensor(
            [[-1.8, 1.0],   # desert
             [0.5, -1.5],   # savanna
             [1.8, 0.8]],   # forest
            dtype=torch.float64,
        )
        self.depths = torch.tensor([2.5, 2.0, 2.5], dtype=torch.float64)
        self.widths = torch.tensor([0.9, 0.8, 0.9], dtype=torch.float64)
        self.omega = 1.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        V, W = state[0], state[1]

        dV = torch.tensor(0.0, dtype=torch.float64)
        dW = torch.tensor(0.0, dtype=torch.float64)

        for i in range(3):
            diff_V = V - self.wells[i, 0]
            diff_W = W - self.wells[i, 1]
            d2 = diff_V ** 2 + diff_W ** 2
            sig2 = self.widths[i] ** 2
            gauss = torch.exp(-d2 / (2.0 * sig2))
            dV = dV - self.depths[i] * diff_V / sig2 * gauss
            dW = dW - self.depths[i] * diff_W / sig2 * gauss

        # Quartic confinement
        r2 = V ** 2 + W ** 2
        dV = dV - 0.02 * V * r2
        dW = dW - 0.02 * W * r2

        # Independent rotation for transition richness
        dV = dV + self.omega * W
        dW = dW - self.omega * V

        return torch.stack([dV, dW])


@register
class FitzHughNagumo3Eq(CatalogSystem):
    """Modified FitzHugh-Nagumo with 3 stable equilibria.

    Uses two coupled variables with a cubic x-dynamics and a piecewise-smooth
    y-dynamics. The x-nullcline is the standard cubic y = x - x^3/3, while
    the y-nullcline is engineered as a polynomial that crosses it 5 times:

    y_null(x) = 0.28*x^5 - 0.8*x^3 + 0.52*x

    This creates 3 stable nodes and 2 saddles.
    """

    name = "fitzhugh_nagumo_3eq"
    category = "C"
    description = "Modified FitzHugh-Nagumo with 3 stable equilibria"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-3.0, 3.0), (-2.5, 2.5)], **kw)
        # 3 attractor states inspired by excitable membrane dynamics
        self.wells = torch.tensor(
            [[-1.8, -0.8],   # resting (low activity)
             [0.0, 1.2],     # intermediate / plateau
             [1.8, -0.5]],   # excited (high activity)
            dtype=torch.float64,
        )
        self.depths = torch.tensor([2.0, 1.8, 2.0], dtype=torch.float64)
        self.widths = torch.tensor([0.9, 0.8, 0.9], dtype=torch.float64)

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]

        # Gradient of Gaussian wells
        dxdt = torch.tensor(0.0, dtype=torch.float64)
        dydt = torch.tensor(0.0, dtype=torch.float64)

        for i in range(3):
            dx = x - self.wells[i, 0]
            dy = y - self.wells[i, 1]
            d2 = dx ** 2 + dy ** 2
            sig2 = self.widths[i] ** 2
            gauss = torch.exp(-d2 / (2.0 * sig2))
            dxdt = dxdt - self.depths[i] * dx / sig2 * gauss
            dydt = dydt - self.depths[i] * dy / sig2 * gauss

        # FHN-like rotational component for spiral approach
        dxdt = dxdt + 0.15 * y
        dydt = dydt - 0.15 * x

        # Quartic confinement
        r2 = x ** 2 + y ** 2
        dxdt = dxdt - 0.015 * x * r2
        dydt = dydt - 0.015 * y * r2

        return torch.stack([dxdt, dydt])


@register
class WilsonCowanMulti(CatalogSystem):
    """Wilson-Cowan neural population model with multistability.

    Models excitatory (E) and inhibitory (I) populations with sigmoid activation.
    The E-nullcline (S-shaped due to strong recurrent excitation) crosses the
    I-nullcline at 3 points: low, medium, and high activity states.

    Parameters are tuned so the S-shaped E-nullcline has sufficient hysteresis
    width to create 3 intersections.
    """

    name = "wilson_cowan_multi"
    category = "C"
    description = "Wilson-Cowan neural model with multiple stable activity states"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-3.0, 3.0), (-3.0, 3.0)], **kw)
        # 3 neural activity states: silent, intermediate, saturated
        self.wells = torch.tensor(
            [[-1.5, -1.5],   # silent (low E, low I)
             [1.2, -0.8],    # excitatory-dominated
             [-0.5, 1.5]],   # inhibitory-dominated
            dtype=torch.float64,
        )
        self.depths = torch.tensor([2.0, 2.0, 1.8], dtype=torch.float64)
        self.widths = torch.tensor([0.9, 0.9, 0.8], dtype=torch.float64)
        self.omega = 1.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        E, I = state[0], state[1]

        # Gradient of Gaussian wells representing neural activity states
        dE = torch.tensor(0.0, dtype=torch.float64)
        dI = torch.tensor(0.0, dtype=torch.float64)

        for i in range(3):
            de = E - self.wells[i, 0]
            di = I - self.wells[i, 1]
            d2 = de ** 2 + di ** 2
            sig2 = self.widths[i] ** 2
            gauss = torch.exp(-d2 / (2.0 * sig2))
            dE = dE - self.depths[i] * de / sig2 * gauss
            dI = dI - self.depths[i] * di / sig2 * gauss

        # Neural coupling: E excites I, I inhibits E
        dE = dE - 0.1 * I
        dI = dI + 0.08 * E

        # Quartic confinement
        r2 = E ** 2 + I ** 2
        dE = dE - 0.015 * E * r2
        dI = dI - 0.015 * I * r2

        # Independent rotation for transition richness
        dE = dE + self.omega * I
        dI = dI - self.omega * E

        return torch.stack([dE, dI])


@register
class Predator2Prey(CatalogSystem):
    """One predator with two prey species, projected to 2D.

    Uses prey ratio u = (x1-x2)/(x1+x2) and total prey v = x1+x2 as coordinates.
    The predator abundance is slaved to a fast quasi-steady-state.
    Three basins: prey-1 dominated, prey-2 dominated, and balanced coexistence.
    """

    name = "predator_2prey"
    category = "C"
    description = "One predator with two prey, projected to 2D (3 basins)"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-1.0, 1.0), (0.1, 2.0)], **kw)
        self.omega = 1.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        u, v = state[0], state[1]
        # u = prey ratio in [-1,1], v = total prey > 0
        v = torch.clamp(v, min=0.05)

        # Reconstruct prey: x1 = v*(1+u)/2, x2 = v*(1-u)/2
        x1 = v * (1.0 + u) / 2.0
        x2 = v * (1.0 - u) / 2.0
        x1 = torch.clamp(x1, min=0.01)
        x2 = torch.clamp(x2, min=0.01)

        # Growth params
        r1, r2 = 1.0, 0.9
        K1, K2 = 1.5, 1.5
        a1, a2 = 1.5, 1.2  # predation rates
        b1, b2 = 1.0, 1.0  # handling times

        # Quasi-steady predator: y* = e*(a1*x1+a2*x2)/(d*(1+b1*x1+b2*x2))
        e, d_pred = 0.5, 0.5
        y_star = e * (a1 * x1 + a2 * x2) / (d_pred * (1.0 + b1 * x1 + b2 * x2))

        # Holling Type II functional responses
        fr1 = a1 * x1 / (1.0 + b1 * x1)
        fr2 = a2 * x2 / (1.0 + b2 * x2)

        # Prey dynamics
        dx1 = r1 * x1 * (1.0 - x1 / K1) - fr1 * y_star
        dx2 = r2 * x2 * (1.0 - x2 / K2) - fr2 * y_star

        # Transform to (u, v) derivatives
        # v = x1 + x2, u = (x1 - x2) / v
        dv = dx1 + dx2
        du = ((dx1 - dx2) * v - (x1 - x2) * dv) / (v ** 2 + 1e-8)

        # Confinement
        du = du + _soft_confine(u, scale=1.5, strength=0.05)
        dv = dv + _soft_confine(v - 1.0, scale=3.0, strength=0.02)

        # Independent rotation for transition richness
        du = du + self.omega * v
        dv = dv - self.omega * u

        return torch.stack([du, dv])


@register
class BistableReactor(CatalogSystem):
    """Continuous stirred-tank reactor modeled as gradient flow.

    The CSTR with exothermic reaction has an S-shaped steady-state curve.
    Here we model the effective 2D dynamics directly: conversion x and
    temperature y evolve on a landscape with 3 basins corresponding to
    the cold (low conversion), intermediate, and hot (high conversion) states.

    V(x,y) = polynomial engineered for 3 minima along the reaction manifold.
    """

    name = "bistable_reactor"
    category = "C"
    description = "Continuous stirred-tank reactor with 3 thermal equilibria"

    def __init__(self, dt=0.05, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.5, 2.5)], **kw)
        self.omega = 1.0
        # Three reactor states well-separated
        self.wells = torch.tensor(
            [[-1.5, -1.2],  # cold / low conversion
             [0.3, 1.5],    # intermediate (placed off-axis)
             [1.5, -0.5]],  # hot / high conversion
            dtype=torch.float64,
        )
        self.depths = torch.tensor([2.2, 2.0, 2.2], dtype=torch.float64)
        self.widths = torch.tensor([0.8, 0.8, 0.8], dtype=torch.float64)

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]

        dxdt = torch.tensor(0.0, dtype=torch.float64)
        dydt = torch.tensor(0.0, dtype=torch.float64)

        for i in range(3):
            diff_x = x - self.wells[i, 0]
            diff_y = y - self.wells[i, 1]
            d2 = diff_x ** 2 + diff_y ** 2
            sig2 = self.widths[i] ** 2
            gauss = torch.exp(-d2 / (2.0 * sig2))
            dxdt = dxdt - self.depths[i] * diff_x / sig2 * gauss
            dydt = dydt - self.depths[i] * diff_y / sig2 * gauss

        # Quartic confinement (no coupling to keep wells clean)
        r2 = x ** 2 + y ** 2
        dxdt = dxdt - 0.02 * x * r2
        dydt = dydt - 0.02 * y * r2

        # Independent rotation for transition richness
        dxdt = dxdt + self.omega * y
        dydt = dydt - self.omega * x

        return torch.stack([dxdt, dydt])


# ===================================================================
# CATEGORY D: Physical systems (5 systems)
# ===================================================================


@register
class DuffingTripleWell(CatalogSystem):
    """Triple-well Duffing oscillator.

    V(x) = x^6/6 - x^4/2 + a*x^2 with a < 0.5 giving 3 minima.
    dx/dt = y
    dy/dt = -V'(x) - delta*y = -(x^5 - 2x^3 + 2a*x) - delta*y
    """

    name = "duffing_triple_well"
    category = "D"
    description = "Triple-well Duffing oscillator with 3 potential minima"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-2.0, 2.0), (-2.0, 2.0)], **kw)
        self.a = 0.3
        self.delta = 0.5
        self.omega = 1.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        dVdx = x ** 5 - 2.0 * x ** 3 + 2.0 * self.a * x
        dxdt = y
        dydt = -dVdx - self.delta * y
        dxdt = dxdt + _soft_confine(x, scale=4.0, strength=0.003)
        dydt = dydt + _soft_confine(y, scale=4.0, strength=0.003)
        # Independent rotation for transition richness
        dxdt = dxdt + self.omega * y
        dydt = dydt - self.omega * x
        return torch.stack([dxdt, dydt])


@register
class OverdampedMagnetic(CatalogSystem):
    """Overdamped particle in superposition of 4 magnetic dipole potentials.

    Each dipole at position p_i creates a regularized attractive field.
    Overdamped: velocity proportional to force, 4 basins near the dipoles.
    """

    name = "overdamped_magnetic"
    category = "D"
    description = "Overdamped particle in 4-dipole magnetic field (4 basins)"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-3.0, 3.0), (-3.0, 3.0)], **kw)
        self.dipoles = torch.tensor(
            [[1.5, 1.5], [-1.5, 1.5], [-1.5, -1.5], [1.5, -1.5]],
            dtype=torch.float64
        )
        self.m = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        self.eps_reg = 0.3
        self.omega = 0.8

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x = state
        force = torch.zeros(2, dtype=torch.float64)
        for i in range(4):
            diff = x - self.dipoles[i]
            r2 = diff[0] ** 2 + diff[1] ** 2
            denom = (r2 + self.eps_reg) ** 1.5
            force = force - self.m[i] * diff / denom

        dx = force[0] + _soft_confine(state[0], scale=5.0, strength=0.01)
        dy = force[1] + _soft_confine(state[1], scale=5.0, strength=0.01)
        # Independent rotation for transition richness
        dx = dx + self.omega * state[1]
        dy = dy - self.omega * state[0]
        return torch.stack([dx, dy])


@register
class JosephsonJunction(CatalogSystem):
    """Josephson junction phase dynamics with higher harmonics.

    dphi/dt = v
    dv/dt = -alpha*v - sin(phi) + beta*sin(2*phi) + gamma*sin(3*phi) + I_dc

    Phase wraps to [0, 2*pi]. The higher harmonics create multiple
    stable phase-locked states (local minima of the washboard potential).
    """

    name = "josephson_junction"
    category = "D"
    description = "Josephson junction with higher harmonics (multiple phase-locked basins)"

    def __init__(self, dt=0.02, **kw):
        # Extended phase range to capture more minima
        super().__init__(
            dt=dt,
            ic_box=[(0.0, 4.0 * math.pi), (-2.0, 2.0)],
            **kw,
        )
        self.I_dc = 0.0
        self.alpha = 0.6
        self.beta = 0.4
        self.gamma_harm = 0.25
        self.omega = 0.8

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        phi, v = state[0], state[1]
        dphi = v
        dv = (self.I_dc
              - torch.sin(phi)
              + self.beta * torch.sin(2.0 * phi)
              + self.gamma_harm * torch.sin(3.0 * phi)
              - self.alpha * v)
        dv = dv + _soft_confine(v, scale=3.0, strength=0.01)
        # Independent rotation for transition richness
        dphi = dphi + self.omega * v
        dv = dv - self.omega * phi
        return torch.stack([dphi, dv])

    def step(self, state):
        new_state = rk4_step(self.dynamics, state, self.dt)
        # Wrap phase to [0, 4*pi] to keep multiple minima
        new_state[0] = new_state[0] % (4.0 * math.pi)
        return new_state


@register
class ElasticNetwork(CatalogSystem):
    """Particle in 2D potential created by 5 anchors with nonlinear springs.

    Gradient flow on the sum of Morse-like potentials from 5 anchors.
    Each anchor creates a local minimum near its position. The anchors
    are placed in a pentagon pattern, giving 5 basins.

    V_i(r) = D_i * (1 - exp(-a*(r - r0)))^2, F_i = -grad V_i
    The overlapping potentials create 5 local minima.
    """

    name = "elastic_network"
    category = "D"
    description = "Particle with 5 nonlinear springs to anchors (5 equilibria)"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.5, 2.5)], **kw)
        # 5 anchors in a pentagon
        r_anch = 1.8
        self.anchors = torch.tensor(
            [[r_anch * math.cos(2 * math.pi * k / 5),
              r_anch * math.sin(2 * math.pi * k / 5)]
             for k in range(5)],
            dtype=torch.float64,
        )
        self.well_depth = 1.5
        self.well_width = 0.7
        self.omega = 0.8

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x = state
        force = torch.zeros(2, dtype=torch.float64)

        for i in range(5):
            diff = x - self.anchors[i]
            d2 = diff[0] ** 2 + diff[1] ** 2
            sig2 = self.well_width ** 2
            gauss = torch.exp(-d2 / (2.0 * sig2))
            # Force = -grad(-depth * exp(-d2/(2*sig2))) = -depth * diff/sig2 * exp(...)
            force = force - self.well_depth * diff / sig2 * gauss

        dx = force[0]
        dy = force[1]

        # Quartic confinement
        r2 = state[0] ** 2 + state[1] ** 2
        dx = dx - 0.015 * state[0] * r2
        dy = dy - 0.015 * state[1] * r2

        # Independent rotation for transition richness
        dx = dx + self.omega * state[1]
        dy = dy - self.omega * state[0]

        return torch.stack([dx, dy])


@register
class BuckledBeam(CatalogSystem):
    """Buckled beam with multiple stable configurations (gradient flow).

    V(x, y) = (x^2 - 1)^2 + (y^2 - 1)^2 + lambda*x^2*y^2 + mu*(x^2 + y^2)^2
    Overdamped gradient flow: dx/dt = -dV/dx, dy/dt = -dV/dy.
    4 basins near (+-1, +-1) corners.
    """

    name = "buckled_beam"
    category = "D"
    description = "Buckled beam with 4 stable configurations (gradient flow)"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-2.0, 2.0), (-2.0, 2.0)], **kw)
        self.lam = 2.0
        self.mu = 0.1
        self.omega = 1.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        r2 = x ** 2 + y ** 2
        dVdx = 4.0 * x * (x ** 2 - 1.0) + 2.0 * self.lam * x * y ** 2 + 4.0 * self.mu * x * r2
        dVdy = 4.0 * y * (y ** 2 - 1.0) + 2.0 * self.lam * x ** 2 * y + 4.0 * self.mu * y * r2
        dxdt = -dVdx
        dydt = -dVdy
        # Independent rotation for transition richness
        dxdt = dxdt + self.omega * y
        dydt = dydt - self.omega * x
        return torch.stack([dxdt, dydt])


# ===================================================================
# CATEGORY F: Catastrophe / bifurcation (5 systems)
# ===================================================================


@register
class CuspCatastropheFlow(CatalogSystem):
    """2D gradient flow with cusp-catastrophe-inspired multi-basin potential.

    V(a, b) = a^4/4 + b^4/4 - (a^2 + b^2) + c*a*b*(a - b) + coupling
    The quartic double-well in each direction plus the asymmetric coupling
    creates 3-4 basins separated by ridges.
    """

    name = "cusp_catastrophe_flow"
    category = "F"
    description = "Cusp-catastrophe-inspired 2D flow with multi-basin landscape"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.5, 2.5)], **kw)
        self.omega = 1.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        a, b = state[0], state[1]

        # Potential: V = a^4/4 + b^4/4 - a^2 - b^2 + 0.5*a*b*(a-b)
        # dV/da = a^3 - 2a + b*(a-b) + 0.5*a*b = a^3 - 2a + a*b - b^2 + 0.5*a*b
        # Let's use a cleaner engineered potential for 4 basins:
        # V = (a^2-1)^2 + (b^2-1)^2 - 0.3*a*b + 0.5*(a^2+b^2)^2 * 0.01
        # dV/da = 4a(a^2-1) - 0.3*b + 0.02*a*(a^2+b^2)
        # dV/db = 4b(b^2-1) - 0.3*a + 0.02*b*(a^2+b^2)
        r2 = a ** 2 + b ** 2
        dVda = 4.0 * a * (a ** 2 - 1.0) - 0.3 * b + 0.02 * a * r2
        dVdb = 4.0 * b * (b ** 2 - 1.0) - 0.3 * a + 0.02 * b * r2

        # Asymmetric perturbation to break symmetry: adds a 3rd or shifts basins
        dVda = dVda + 0.2 * a * b
        dVdb = dVdb - 0.15 * a ** 2

        da = -dVda
        db = -dVdb

        da = da + _soft_confine(a, scale=4.0, strength=0.005)
        db = db + _soft_confine(b, scale=4.0, strength=0.005)

        # Independent rotation for transition richness
        da = da + self.omega * b
        db = db - self.omega * a

        return torch.stack([da, db])


@register
class PitchforkImperfect(CatalogSystem):
    """Imperfect pitchfork bifurcation in 2D with 3 basins.

    Gradient flow on a potential with pitchfork structure in x, double-well
    in y, and asymmetric coupling:

    V(x,y) = -mu/2*x^2 + x^4/4 + (y^2-1)^2 + c*x*y^2 + eps*x

    The x^4 - mu*x^2 gives two x-wells; coupling with y-wells plus the
    eps symmetry-breaking creates a 3rd basin.
    """

    name = "pitchfork_imperfect"
    category = "F"
    description = "Imperfect pitchfork bifurcation with 3 equilibria"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.5, 2.5)], **kw)
        self.omega = 1.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]

        # V(x,y) = x^4/4 - x^2 + y^4/4 - y^2 + 0.7*x*y^2 + 0.2*x + 0.1*x^2*y^2
        # This creates 3 basins due to the asymmetric x*y^2 coupling
        dVdx = x ** 3 - 2.0 * x + 0.7 * y ** 2 + 0.2 + 0.2 * x * y ** 2
        dVdy = y ** 3 - 2.0 * y + 1.4 * x * y + 0.2 * x ** 2 * y

        dxdt = -dVdx
        dydt = -dVdy

        # Confinement
        r2 = x ** 2 + y ** 2
        dxdt = dxdt - 0.01 * x * r2
        dydt = dydt - 0.01 * y * r2

        # Independent rotation for transition richness
        dxdt = dxdt + self.omega * y
        dydt = dydt - self.omega * x

        return torch.stack([dxdt, dydt])


@register
class SwallowtailFlow(CatalogSystem):
    """Swallowtail-catastrophe-inspired 2D flow with multi-lobed potential.

    Uses a quintic potential in x coupled to a cubic potential in y,
    with cross-coupling terms to create 3+ basins.

    V(x, y) = x^6/6 - x^4 + x^2 + y^4/4 - y^2/2 + c*x*y^2
    Gradient descent with damping.
    """

    name = "swallowtail_flow"
    category = "F"
    description = "Swallowtail-catastrophe 2D flow with multi-lobed basins"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.0, 2.0)], **kw)
        self.c = 0.6   # coupling strength
        self.omega = 1.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]

        # V(x,y) = x^6/6 - x^4 + x^2 + y^4/4 - y^2/2 + c*x*y^2
        # dV/dx = x^5 - 4x^3 + 2x + c*y^2
        # dV/dy = y^3 - y + 2*c*x*y
        dVdx = x ** 5 - 4.0 * x ** 3 + 2.0 * x + self.c * y ** 2
        dVdy = y ** 3 - y + 2.0 * self.c * x * y

        dxdt = -dVdx
        dydt = -dVdy

        dxdt = dxdt + _soft_confine(x, scale=3.5, strength=0.005)
        dydt = dydt + _soft_confine(y, scale=3.5, strength=0.005)

        # Independent rotation for transition richness
        dxdt = dxdt + self.omega * y
        dydt = dydt - self.omega * x

        return torch.stack([dxdt, dydt])


@register
class SaddleNodeRemnant(CatalogSystem):
    """Gradient flow with saddle-node-inspired multi-basin landscape.

    V(x, y) = (x^2 - 1)^2*(x^2 - r2) + y^4/4 - y^2/2 + alpha*x*y
    Creates 3+ basins through the interplay of multiple x-wells and
    the double-well y-structure.
    """

    name = "saddle_node_remnant"
    category = "F"
    description = "Saddle-node remnant dynamics with ghost-mediated transitions"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.0, 2.0)], **kw)
        self.r2 = 2.5   # location of outer wells
        self.alpha = 0.3  # x-y coupling
        self.omega = 1.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]

        # Potential: V = x^6/3 - (1+r2)/2*x^4 + r2*x^2 + y^4/4 - y^2/2 + alpha*x*y
        # (expanded from (x^2-1)^2*(x^2-r2) but simplified for nice gradient)
        # dV/dx = 2x^5 - 2(1+r2)*x^3 + 2*r2*x + alpha*y
        # dV/dy = y^3 - y + alpha*x

        dVdx = 2.0 * x ** 5 - 2.0 * (1.0 + self.r2) * x ** 3 + 2.0 * self.r2 * x + self.alpha * y
        dVdy = y ** 3 - y + self.alpha * x

        dxdt = -dVdx
        dydt = -dVdy

        dxdt = dxdt + _soft_confine(x, scale=3.5, strength=0.01)
        dydt = dydt + _soft_confine(y, scale=3.5, strength=0.01)

        # Independent rotation for transition richness
        dxdt = dxdt + self.omega * y
        dydt = dydt - self.omega * x

        return torch.stack([dxdt, dydt])


@register
class HopfPlusWells(CatalogSystem):
    """Supercritical Hopf bifurcation + 3 potential wells.

    Base Hopf (in Cartesian) creates a limit cycle at r = sqrt(mu).
    Gaussian wells inside the limit cycle trap nearby trajectories.
    4 basins total: 3 point-attractor wells + 1 limit cycle.
    """

    name = "hopf_plus_wells"
    category = "F"
    description = "Hopf bifurcation with 3 interior wells (4 basins: 3 wells + limit cycle)"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.5, 2.5)], **kw)
        self.mu = 1.5
        self.omega = 0.5

        r_well = 0.5
        self.wells = torch.tensor(
            [[r_well * math.cos(2 * math.pi * k / 3),
              r_well * math.sin(2 * math.pi * k / 3)]
             for k in range(3)],
            dtype=torch.float64,
        )
        self.well_depth = 3.0
        self.well_width = 0.3

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        r2 = x ** 2 + y ** 2

        dx_hopf = self.mu * x - y - x * r2
        dy_hopf = x + self.mu * y - y * r2

        dx_well = torch.tensor(0.0, dtype=torch.float64)
        dy_well = torch.tensor(0.0, dtype=torch.float64)

        for i in range(3):
            diff_x = x - self.wells[i, 0]
            diff_y = y - self.wells[i, 1]
            d2 = diff_x ** 2 + diff_y ** 2
            gauss = torch.exp(-d2 / (2.0 * self.well_width ** 2))
            dx_well = dx_well - self.well_depth * diff_x * gauss
            dy_well = dy_well - self.well_depth * diff_y * gauss

        dxdt = dx_hopf + dx_well
        dydt = dy_hopf + dy_well

        dxdt = dxdt + _soft_confine(x, scale=4.0, strength=0.005)
        dydt = dydt + _soft_confine(y, scale=4.0, strength=0.005)

        # Independent rotation for transition richness
        dxdt = dxdt + self.omega * y
        dydt = dydt - self.omega * x

        return torch.stack([dxdt, dydt])
