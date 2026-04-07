"""Novel and creative transition-rich dynamical systems — Batch 4.

These systems focus on unique dynamics that test specific aspects
of basin-aware Koopman learning:
- Qualitatively different local dynamics across basins
- Hierarchical basin structure
- Non-trivial transition routes
- Real-world inspiration with clean 2D dynamics
"""

import torch
import math
from skae.claude_catalog.base import CatalogSystem, rk4_step
from skae.claude_catalog.registry import register


# ---------------------------------------------------------------------------
# System 1: Waddington landscape — epigenetic differentiation
# ---------------------------------------------------------------------------
@register
class WaddingtonLandscape(CatalogSystem):
    """Epigenetic landscape with branching valleys.

    Inspired by Waddington's landscape metaphor for cell differentiation.
    A cell (ball) rolls down a landscape with branching valleys, choosing
    between fates. The landscape has a stem-cell hilltop, two progenitor
    valleys that each branch into two terminal fates → 4 endpoint basins
    arranged in a tree topology.
    """

    name = "waddington_landscape"
    category = "H"
    description = "Epigenetic landscape with 4 terminal fates in branching valleys"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-3.0, 3.0), (-1.0, 4.0)], **kw)
        # Terminal fate positions (bottom of landscape)
        self.fates = [(-2.5, -0.5), (-0.8, -0.5), (0.8, -0.5), (2.5, -0.5)]
        # Branching points
        self.branch1 = (0.0, 2.5)  # first branch (stem → left/right progenitor)
        self.branch2_l = (-1.6, 1.0)  # left progenitor → two fates
        self.branch2_r = (1.6, 1.0)  # right progenitor → two fates
        self.omega = 0.8

    @property
    def dim(self):
        return 2

    def _potential(self, x, y):
        """Landscape potential with branching valleys."""
        # Downhill drift (y decreases = differentiation)
        V = -0.5 * y

        # Ridge at top creating first bifurcation
        ridge1 = 2.0 * torch.exp(-x ** 2 / 0.3) * torch.exp(-(y - 2.5) ** 2 / 0.5)

        # Ridges creating second bifurcation (left and right)
        ridge2_l = 1.5 * torch.exp(-(x + 1.6) ** 2 / 0.2) * torch.exp(-(y - 1.0) ** 2 / 0.4)
        ridge2_r = 1.5 * torch.exp(-(x - 1.6) ** 2 / 0.2) * torch.exp(-(y - 1.0) ** 2 / 0.4)

        # Terminal fate wells
        wells = torch.zeros_like(x)
        for fx, fy in self.fates:
            wells = wells - 3.0 * torch.exp(-((x - fx) ** 2 + (y - fy) ** 2) / 0.4)

        # Lateral confinement
        conf = 0.05 * x ** 4 + 0.1 * (y + 1) ** 2 * (y < -0.5).float()

        V = V + ridge1 + ridge2_l + ridge2_r + wells + conf
        return V

    def dynamics(self, state):
        x, y = state[0], state[1]
        eps = 1e-6
        # Numerical gradient of potential
        Vx_p = self._potential(x + eps, y)
        Vx_m = self._potential(x - eps, y)
        Vy_p = self._potential(x, y + eps)
        Vy_m = self._potential(x, y - eps)
        dVdx = (Vx_p - Vx_m) / (2 * eps)
        dVdy = (Vy_p - Vy_m) / (2 * eps)

        # Gradient descent with slight damping
        dxdt = -dVdx - 0.1 * state[0]
        dydt = -dVdy

        # Independent rotation for transition richness
        dxdt = dxdt + self.omega * y
        dydt = dydt - self.omega * x
        # Confinement to prevent divergence from rotation
        dxdt = dxdt - 0.01 * x ** 3
        dydt = dydt - 0.01 * y ** 3

        return torch.stack([dxdt, dydt])


# ---------------------------------------------------------------------------
# System 2: Neural decision landscape — evidence accumulation with 3 choices
# ---------------------------------------------------------------------------
@register
class NeuralDecision3Choice(CatalogSystem):
    """Neural decision-making model with 3 choice basins.

    Based on attractor dynamics for perceptual decision-making.
    Three competing neural populations, projected to 2D.
    The state represents relative evidence for each choice.
    Transitions occur when evidence crosses decision boundaries.
    """

    name = "neural_decision_3choice"
    category = "H"
    description = "3-alternative decision-making via competing attractor dynamics"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-2.0, 2.0), (-2.0, 2.0)], **kw)
        self.tau = 1.0  # time constant
        self.beta = 2.0  # gain of sigmoid
        self.theta = 0.5  # threshold
        self.w_self = 4.0  # self-excitation
        self.w_inh = 2.5  # mutual inhibition
        self.I = [0.8, 0.85, 0.75]  # external inputs (slight bias)
        self.omega = 1.0

    @property
    def dim(self):
        return 2

    def _sigmoid(self, x):
        return 1.0 / (1.0 + torch.exp(-self.beta * (x - self.theta)))

    def dynamics(self, state):
        # State in 2D simplex-like coordinates
        # x = r1 - r2, y = (2*r3 - r1 - r2)/sqrt(3)
        # Convert back to firing rates (approximately)
        x, y = state[0], state[1]
        r1 = 0.5 + x / 2.0 - y / (2.0 * math.sqrt(3))
        r2 = 0.5 - x / 2.0 - y / (2.0 * math.sqrt(3))
        r3 = 0.5 + y / math.sqrt(3)

        # Clamp rates
        r1 = torch.clamp(r1, 0.0, 2.0)
        r2 = torch.clamp(r2, 0.0, 2.0)
        r3 = torch.clamp(r3, 0.0, 2.0)

        # Wilson-Cowan style dynamics
        input1 = self.w_self * r1 - self.w_inh * (r2 + r3) + self.I[0]
        input2 = self.w_self * r2 - self.w_inh * (r1 + r3) + self.I[1]
        input3 = self.w_self * r3 - self.w_inh * (r1 + r2) + self.I[2]

        dr1 = (-r1 + self._sigmoid(input1)) / self.tau
        dr2 = (-r2 + self._sigmoid(input2)) / self.tau
        dr3 = (-r3 + self._sigmoid(input3)) / self.tau

        # Convert back to 2D
        dxdt = (dr1 - dr2)
        dydt = (2 * dr3 - dr1 - dr2) / math.sqrt(3)

        # Add soft confinement
        r_sq = x ** 2 + y ** 2
        conf = -0.02 * r_sq
        dxdt = dxdt + conf * x
        dydt = dydt + conf * y

        # Independent rotation for transition richness
        dxdt = dxdt + self.omega * y
        dydt = dydt - self.omega * x
        # Confinement to prevent divergence from rotation
        dxdt = dxdt - 0.01 * x ** 3
        dydt = dydt - 0.01 * y ** 3

        return torch.stack([dxdt, dydt])


# ---------------------------------------------------------------------------
# System 3: Climate tipping cascade — ice-albedo-CO2 model
# ---------------------------------------------------------------------------
@register
class ClimateTippingCascade(CatalogSystem):
    """Simplified climate model with multiple tipping points.

    Two variables: temperature anomaly T and ice coverage I.
    Multiple stable states: glacial, interglacial, hothouse.
    Transitions are triggered by ice-albedo feedback.
    """

    name = "climate_tipping_cascade"
    category = "H"
    description = "Climate tipping points with ice-albedo feedback (3 stable states)"

    def __init__(self, dt=0.05, **kw):
        super().__init__(dt=dt, ic_box=[(-2.0, 3.0), (0.0, 1.0)], **kw)
        self.S = 1.0  # solar constant (normalized)
        self.eps = 0.6  # emissivity
        self.alpha_ice = 0.7  # ice albedo
        self.alpha_land = 0.3  # land albedo
        self.T_melt = 0.0  # melting threshold
        self.k_melt = 5.0  # sharpness of melt transition
        self.omega = 1.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        T, I = state[0], state[1]

        # Albedo depends on ice coverage
        alpha = self.alpha_land + (self.alpha_ice - self.alpha_land) * I

        # Temperature dynamics: absorbed solar - outgoing radiation + greenhouse
        Q_in = self.S * (1 - alpha)
        Q_out = self.eps * (0.5 + 0.3 * T + 0.05 * T ** 2)
        greenhouse = 0.1 * torch.tanh(T)  # positive feedback

        dTdt = Q_in - Q_out + greenhouse

        # Ice dynamics: growth when cold, melt when warm
        ice_growth = 0.5 * (1 - I) * torch.sigmoid(-self.k_melt * (T - self.T_melt))
        ice_melt = 0.8 * I * torch.sigmoid(self.k_melt * (T - self.T_melt - 0.5))
        dIdt = ice_growth - ice_melt

        # Add multi-stability through nonlinear ice-albedo feedback
        dTdt = dTdt + 0.3 * I * (1 - I) * (0.5 - I)  # triple-well in I direction

        # Independent rotation for transition richness
        dTdt = dTdt + self.omega * I
        dIdt = dIdt - self.omega * T
        # Confinement to prevent divergence from rotation
        dTdt = dTdt - 0.01 * T ** 3
        dIdt = dIdt - 0.01 * I ** 3

        return torch.stack([dTdt, dIdt])


# ---------------------------------------------------------------------------
# System 4: Morphogenesis — Turing pattern mode competition
# ---------------------------------------------------------------------------
@register
class MorphogenesisModeSel(CatalogSystem):
    """Mode competition in Turing pattern formation.

    Two Fourier amplitudes of a Turing pattern compete.
    Which mode wins determines the spatial pattern.
    Near onset, multiple modes can be stable → 3-4 basins
    (mode 1 wins, mode 2 wins, mixed state, trivial state).
    """

    name = "morphogenesis_mode_selection"
    category = "H"
    description = "Turing pattern mode competition with 4 stable configurations"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-0.5, 2.0), (-0.5, 2.0)], **kw)
        self.mu = 0.5  # distance from onset
        self.g11 = 1.0  # self-saturation mode 1
        self.g22 = 1.0  # self-saturation mode 2
        self.g12 = 1.8  # cross-saturation (>1 for competition)
        self.g21 = 1.5  # cross-saturation (asymmetric)
        self.c = 0.3  # coupling creating mixed state
        self.omega = 1.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        A1, A2 = state[0], state[1]

        # Amplitude equations near Turing onset
        # dA_i/dt = μ*A_i - g_ii*A_i^3 - g_ij*A_i*A_j^2 + c*A_j*(1-A_i)
        dA1 = (
            self.mu * A1
            - self.g11 * A1 ** 3
            - self.g12 * A1 * A2 ** 2
            + self.c * A2 * torch.clamp(1 - A1, min=0)
        )
        dA2 = (
            self.mu * A2
            - self.g22 * A2 ** 3
            - self.g21 * A2 * A1 ** 2
            + self.c * A1 * torch.clamp(1 - A2, min=0)
        )

        # Keep amplitudes positive-ish with soft wall
        dA1 = dA1 - 0.5 * torch.clamp(-A1, min=0)
        dA2 = dA2 - 0.5 * torch.clamp(-A2, min=0)

        # Independent rotation for transition richness
        dA1 = dA1 + self.omega * A2
        dA2 = dA2 - self.omega * A1
        # Confinement to prevent divergence from rotation
        dA1 = dA1 - 0.01 * A1 ** 3
        dA2 = dA2 - 0.01 * A2 ** 3

        return torch.stack([dA1, dA2])


# ---------------------------------------------------------------------------
# System 5: River delta branching — directional flow with bifurcations
# ---------------------------------------------------------------------------
@register
class RiverDeltaBranching(CatalogSystem):
    """River delta-like branching flow system.

    Flow goes generally downward (y decreasing) but encounters
    bifurcation points where it must choose left or right.
    Multiple endpoint basins at the bottom represent different
    delta outlets. Interesting because transitions are unidirectional.
    """

    name = "river_delta_branching"
    category = "H"
    description = "Branching flow with 4 outlet basins (unidirectional transitions)"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-3.0, 3.0), (-0.5, 4.0)], **kw)
        # Outlet positions (bottom)
        self.outlets = [(-2.5, -0.5), (-0.8, -0.5), (0.8, -0.5), (2.5, -0.5)]
        # Bifurcation ridges
        self.ridges = [
            (0.0, 2.8, 0.0, 0.3, 2.5),   # (cx, cy, angle, sigma_short, height)
            (-1.5, 1.2, 0.3, 0.25, 2.0),
            (1.5, 1.2, -0.3, 0.25, 2.0),
        ]
        self.omega = 0.8

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]

        # Base downhill flow
        dxdt = torch.zeros_like(x)
        dydt = -0.8 * torch.ones_like(y)

        # Channel guidance: attract toward nearest outlet's x-position weighted by y
        for ox, oy in self.outlets:
            dist_sq = (x - ox) ** 2 + (y - oy) ** 2
            weight = torch.exp(-dist_sq / 2.0) * torch.sigmoid(-(y - 1.0))
            dxdt = dxdt + weight * (ox - x) * 2.0
            dydt = dydt + weight * (oy - y) * 0.5

        # Bifurcation ridges push flow sideways
        for cx, cy, angle, sigma, height in self.ridges:
            # Ridge perpendicular to flow creates bifurcation
            dx_r = x - cx
            dy_r = y - cy
            along = dx_r * math.cos(angle) + dy_r * math.sin(angle)
            across = -dx_r * math.sin(angle) + dy_r * math.cos(angle)
            ridge_force = height * along * torch.exp(
                -along ** 2 / (2 * sigma ** 2) - across ** 2 / 1.0
            )
            dxdt = dxdt + ridge_force * math.cos(angle + math.pi / 2)
            dydt = dydt + ridge_force * math.sin(angle + math.pi / 2)

        # Terminal well attraction
        for ox, oy in self.outlets:
            dist_sq = (x - ox) ** 2 + (y - oy) ** 2
            well = -3.0 * torch.exp(-dist_sq / 0.3)
            dxdt = dxdt - well * 2 * (x - ox) / 0.3
            dydt = dydt - well * 2 * (y - oy) / 0.3

        # Lateral confinement
        dxdt = dxdt - 0.02 * x ** 3

        # Independent rotation for transition richness
        dxdt = dxdt + self.omega * y
        dydt = dydt - self.omega * x
        # Confinement to prevent divergence from rotation
        dxdt = dxdt - 0.01 * x ** 3
        dydt = dydt - 0.01 * y ** 3

        return torch.stack([dxdt, dydt])


# ---------------------------------------------------------------------------
# System 6: Volcanic eruption analogy — slow buildup, fast transition
# ---------------------------------------------------------------------------
@register
class VolcanicEruption(CatalogSystem):
    """Volcanic eruption analogy with slow-fast dynamics.

    Pressure builds slowly, then erupts rapidly to a new state.
    Multiple chambers (basins) with different eruption thresholds.
    The slow-fast structure creates interesting transition dynamics.
    """

    name = "volcanic_eruption"
    category = "H"
    description = "Slow-fast dynamics with 3 basins and eruption-like transitions"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.0, 2.0)], **kw)
        self.eps = 0.1  # slow-fast separation
        # Three chambers
        self.chambers = [(-1.5, 0.0), (0.0, 0.0), (1.5, 0.0)]
        self.thresholds = [1.2, 1.0, 1.4]
        self.omega = 1.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]

        # x is "location" (which chamber), y is "pressure"
        # Fast dynamics (x): attracted to nearest chamber when pressure is low
        x_force = torch.zeros_like(x)
        for (cx, _), threshold in zip(self.chambers, self.thresholds):
            dist = x - cx
            # Attractive when below threshold pressure
            attraction = -2.0 * dist * torch.exp(-dist ** 2 / 0.5)
            suppression = torch.sigmoid(-5.0 * (y - threshold))
            x_force = x_force + attraction * suppression

        # When pressure exceeds threshold, eruptive lateral force
        eruptive = 0.5 * y * torch.sign(x + 0.01) * torch.sigmoid(3.0 * (torch.abs(y) - 0.8))

        dxdt = x_force + eruptive - 0.1 * x ** 3  # confinement

        # Slow dynamics (y): pressure builds, then releases
        # Cubic nullcline creates relaxation oscillation
        y_nullcline = x ** 3 - 3.0 * x  # S-shaped curve
        dydt = self.eps * (y_nullcline - y + 0.5 * torch.sin(2 * x))

        # Independent rotation for transition richness
        dxdt = dxdt + self.omega * y
        dydt = dydt - self.omega * x
        # Confinement to prevent divergence from rotation
        dxdt = dxdt - 0.01 * x ** 3
        dydt = dydt - 0.01 * y ** 3

        return torch.stack([dxdt, dydt])


# ---------------------------------------------------------------------------
# System 7: DNA regulatory switch — 4-state epigenetic model
# ---------------------------------------------------------------------------
@register
class DNARegulatorySwitch(CatalogSystem):
    """DNA methylation / histone modification regulatory switch.

    Two regulatory marks (e.g., methylation level, acetylation level)
    with cooperative positive feedback creating 4 stable epigenetic states:
    - Both low (silenced)
    - Mark 1 high, mark 2 low
    - Mark 1 low, mark 2 high
    - Both high (active)
    """

    name = "dna_regulatory_switch"
    category = "H"
    description = "4-state epigenetic switch with cooperative regulation"

    def __init__(self, dt=0.05, **kw):
        super().__init__(dt=dt, ic_box=[(-0.5, 2.5), (-0.5, 2.5)], **kw)
        self.n = 4  # Hill coefficient (cooperativity)
        self.K = 1.0  # half-max
        self.alpha1 = 2.0  # max production rate for mark 1
        self.alpha2 = 1.8  # max production rate for mark 2
        self.d1 = 1.0  # degradation rate
        self.d2 = 1.0
        self.cross_act = 0.3  # cross-activation strength
        self.cross_inh = 0.2  # cross-inhibition strength
        self.omega = 1.0

    @property
    def dim(self):
        return 2

    def _hill(self, x, K=None):
        if K is None:
            K = self.K
        return x ** self.n / (K ** self.n + x ** self.n + 1e-10)

    def dynamics(self, state):
        m1, m2 = state[0], state[1]
        m1 = torch.clamp(m1, min=0.0)
        m2 = torch.clamp(m2, min=0.0)

        # Self-activation with cooperative feedback
        # Cross-activation at lower levels, cross-inhibition at higher levels
        self_act_1 = self.alpha1 * self._hill(m1)
        self_act_2 = self.alpha2 * self._hill(m2)

        cross_1 = self.cross_act * self._hill(m2, K=0.5) - self.cross_inh * self._hill(m2, K=1.5)
        cross_2 = self.cross_act * self._hill(m1, K=0.5) - self.cross_inh * self._hill(m1, K=1.5)

        # Basal production
        basal = 0.1

        dm1dt = basal + self_act_1 + cross_1 - self.d1 * m1
        dm2dt = basal + self_act_2 + cross_2 - self.d2 * m2

        # Independent rotation for transition richness
        dm1dt = dm1dt + self.omega * m2
        dm2dt = dm2dt - self.omega * m1
        # Confinement to prevent divergence from rotation
        dm1dt = dm1dt - 0.01 * m1 ** 3
        dm2dt = dm2dt - 0.01 * m2 ** 3

        return torch.stack([dm1dt, dm2dt])


# ---------------------------------------------------------------------------
# System 8: Predator foraging switch — optimal foraging with 3 patches
# ---------------------------------------------------------------------------
@register
class PredatorForagingSwitch(CatalogSystem):
    """Optimal foraging theory with 3 patch types.

    A predator allocates effort between 3 prey patches.
    Two state variables: relative effort on patch 1 vs 2 (patch 3 = 1-p1-p2).
    Nonlinear returns create multiple optimal strategies (basins).
    """

    name = "predator_foraging_switch"
    category = "H"
    description = "Optimal foraging with 3 patches creating specialist/generalist basins"

    def __init__(self, dt=0.05, **kw):
        super().__init__(dt=dt, ic_box=[(0.01, 0.98), (0.01, 0.98)], **kw)
        self.r = [1.0, 1.2, 0.8]  # patch renewal rates
        self.a = [2.0, 1.8, 2.2]  # attack rates
        self.h = [0.5, 0.6, 0.4]  # handling times
        self.speed = 2.0  # adaptation speed
        self.omega = 1.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        p1, p2 = state[0], state[1]
        p1 = torch.clamp(p1, 0.01, 0.98)
        p2 = torch.clamp(p2, 0.01, 0.98)
        p3 = torch.clamp(1.0 - p1 - p2, 0.01, 0.98)

        # Functional response (Holling Type II) for each patch
        f1 = self.a[0] * p1 / (1 + self.a[0] * self.h[0] * p1)
        f2 = self.a[1] * p2 / (1 + self.a[1] * self.h[1] * p2)
        f3 = self.a[2] * p3 / (1 + self.a[2] * self.h[2] * p3)

        # Marginal value of each patch
        mv1 = self.r[0] * f1 / (p1 + 0.01)
        mv2 = self.r[1] * f2 / (p2 + 0.01)
        mv3 = self.r[2] * f3 / (p3 + 0.01)

        # Replicator dynamics: increase effort on above-average patches
        avg_mv = p1 * mv1 + p2 * mv2 + p3 * mv3

        dp1dt = self.speed * p1 * (mv1 - avg_mv)
        dp2dt = self.speed * p2 * (mv2 - avg_mv)

        # Add small diffusion to prevent getting stuck on boundary
        dp1dt = dp1dt + 0.01 * (0.33 - p1)
        dp2dt = dp2dt + 0.01 * (0.33 - p2)

        # Independent rotation for transition richness
        dp1dt = dp1dt + self.omega * p2
        dp2dt = dp2dt - self.omega * p1
        # Confinement to prevent divergence from rotation
        dp1dt = dp1dt - 0.01 * p1 ** 3
        dp2dt = dp2dt - 0.01 * p2 ** 3

        return torch.stack([dp1dt, dp2dt])


# ---------------------------------------------------------------------------
# System 9: Gear mesh dynamics — multiple mesh positions
# ---------------------------------------------------------------------------
@register
class GearMeshDynamics(CatalogSystem):
    """Simplified gear meshing with multiple stable positions.

    Two gears can mesh in different relative positions (teeth alignment).
    The potential has N wells corresponding to tooth-on-tooth alignments.
    Adding rotation creates interesting transition dynamics.
    """

    name = "gear_mesh_dynamics"
    category = "H"
    description = "Gear meshing with 6 stable tooth-alignment positions"

    def __init__(self, dt=0.03, n_teeth=6, **kw):
        super().__init__(dt=dt, ic_box=[(-3.5, 3.5), (-3.5, 3.5)], **kw)
        self.n_teeth = n_teeth
        self.damping = 0.5
        self.spring = 1.0
        self.omega = 0.5

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        n = self.n_teeth

        # Periodic potential from gear teeth (cosine wells)
        # V = -cos(n*θ) where θ = atan2(y, x) → wells at θ = 2πk/n
        # Plus radial potential to keep on a ring
        r = torch.sqrt(x ** 2 + y ** 2 + 1e-8)
        theta = torch.atan2(y, x)

        # Radial: attract to unit circle
        dVdr = self.spring * 2.0 * (r - 1.5)

        # Angular: periodic wells
        dVdtheta = n * torch.sin(n * theta)

        # Add asymmetric perturbation (different well depths)
        dVdtheta = dVdtheta + 0.3 * 2 * torch.sin(2 * theta)

        # Convert to Cartesian
        cos_t = x / (r + 1e-8)
        sin_t = y / (r + 1e-8)

        dxdt = -(dVdr * cos_t - dVdtheta * sin_t / (r + 1e-8))
        dydt = -(dVdr * sin_t + dVdtheta * cos_t / (r + 1e-8))

        # Add damping and slight rotation
        dxdt = dxdt - self.damping * x + 0.15 * y
        dydt = dydt - self.damping * y - 0.15 * x

        # Independent rotation for transition richness
        dxdt = dxdt + self.omega * y
        dydt = dydt - self.omega * x
        # Confinement to prevent divergence from rotation
        dxdt = dxdt - 0.01 * x ** 3
        dydt = dydt - 0.01 * y ** 3

        return torch.stack([dxdt, dydt])


# ---------------------------------------------------------------------------
# System 10: Hourglass bottleneck — two regions connected by narrow passage
# ---------------------------------------------------------------------------
@register
class HourglassBottleneck(CatalogSystem):
    """Two large basins connected by a narrow bottleneck.

    Each side has 2 sub-basins, creating 4 total basins with
    hierarchical structure: transitions within a side are easy,
    transitions across the bottleneck are hard.
    Tests hierarchical basin partitioning.
    """

    name = "hourglass_bottleneck"
    category = "H"
    description = "4 basins with hierarchical bottleneck structure"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-3.5, 3.5), (-2.5, 2.5)], **kw)
        # Basin centers
        self.basins = [
            (-2.0, 1.0),   # top-left
            (-2.0, -1.0),  # bottom-left
            (2.0, 1.0),    # top-right
            (2.0, -1.0),   # bottom-right
        ]
        self.omega = 0.8

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]

        # Four Gaussian wells
        dxdt = torch.zeros_like(x)
        dydt = torch.zeros_like(y)
        for cx, cy in self.basins:
            dx = x - cx
            dy = y - cy
            dist_sq = dx ** 2 + dy ** 2
            force = 3.0 * torch.exp(-dist_sq / 0.8)
            dxdt = dxdt - force * 2 * dx / 0.8
            dydt = dydt - force * 2 * dy / 0.8

        # Hourglass barrier: high walls at x=0, except near y=0
        # Barrier = tall ridge along x=0, with a gap at y=0
        barrier_height = 4.0 * torch.exp(-x ** 2 / 0.15)
        gap = 1.0 - torch.exp(-y ** 2 / 0.3)  # gap at y=0
        barrier_force = barrier_height * gap * (-2 * x / 0.15)
        dxdt = dxdt - barrier_force

        # Horizontal barriers separating top/bottom within each side
        for side_x in [-2.0, 2.0]:
            side_barrier = 1.5 * torch.exp(-(x - side_x) ** 2 / 1.0) * torch.exp(-y ** 2 / 0.2)
            dydt = dydt + side_barrier * 2 * y / 0.2

        # Rotation through bottleneck (makes transitions more interesting)
        bottleneck_region = torch.exp(-x ** 2 / 0.5) * torch.exp(-y ** 2 / 0.5)
        dxdt = dxdt + 0.8 * y * bottleneck_region
        dydt = dydt - 0.8 * x * bottleneck_region

        # Confinement
        dxdt = dxdt - 0.02 * x ** 3
        dydt = dydt - 0.05 * y ** 3

        # Independent rotation for transition richness
        dxdt = dxdt + self.omega * y
        dydt = dydt - self.omega * x
        # Confinement to prevent divergence from rotation
        dxdt = dxdt - 0.01 * x ** 3
        dydt = dydt - 0.01 * y ** 3

        return torch.stack([dxdt, dydt])


# ---------------------------------------------------------------------------
# System 11: Opinion polarization — multi-party opinion dynamics
# ---------------------------------------------------------------------------
@register
class OpinionPolarization(CatalogSystem):
    """Opinion dynamics with 4 ideological basins.

    Based on bounded confidence model with social pressure.
    Two opinion dimensions (e.g., economic left-right, social liberal-conservative).
    Four stable ideological clusters create basins.
    """

    name = "opinion_polarization"
    category = "H"
    description = "4-party opinion dynamics with social pressure attractors"

    def __init__(self, dt=0.05, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.5, 2.5)], **kw)
        # Four ideological positions
        self.positions = [
            (1.5, 1.5),    # right-conservative
            (-1.5, 1.5),   # left-conservative
            (-1.5, -1.5),  # left-liberal
            (1.5, -1.5),   # right-liberal
        ]
        self.influence_radius = 1.5
        self.pressure_strength = 2.0
        self.omega = 0.8

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]

        dxdt = torch.zeros_like(x)
        dydt = torch.zeros_like(y)

        # Each ideological position exerts attractive force
        total_weight = torch.zeros_like(x)
        for px, py in self.positions:
            dx = x - px
            dy = y - py
            dist_sq = dx ** 2 + dy ** 2
            # Bounded confidence: only influenced if close enough
            influence = torch.exp(-dist_sq / (self.influence_radius ** 2))
            total_weight = total_weight + influence
            dxdt = dxdt - self.pressure_strength * influence * dx
            dydt = dydt - self.pressure_strength * influence * dy

        # Normalize by total influence
        dxdt = dxdt / (total_weight + 0.1)
        dydt = dydt / (total_weight + 0.1)

        # Moderate centrist pull (prevents extreme polarization)
        dxdt = dxdt - 0.05 * x
        dydt = dydt - 0.05 * y

        # Social media amplification near boundaries (pushes toward extremes)
        # This creates interesting transition dynamics
        dxdt = dxdt + 0.3 * x * (1 - x ** 2 / 4)
        dydt = dydt + 0.3 * y * (1 - y ** 2 / 4)

        # Independent rotation for transition richness
        dxdt = dxdt + self.omega * y
        dydt = dydt - self.omega * x
        # Confinement to prevent divergence from rotation
        dxdt = dxdt - 0.01 * x ** 3
        dydt = dydt - 0.01 * y ** 3

        return torch.stack([dxdt, dydt])


# ---------------------------------------------------------------------------
# System 12: Protein folding landscape — simplified 2D free energy surface
# ---------------------------------------------------------------------------
@register
class ProteinFoldingLandscape(CatalogSystem):
    """Simplified protein folding free energy landscape.

    Two reaction coordinates: end-to-end distance and radius of gyration.
    Multiple metastable states: unfolded, misfolded intermediates, native state.
    Complex transitions through intermediate states.
    """

    name = "protein_folding_landscape"
    category = "H"
    description = "Protein folding with native, misfolded, and unfolded basins (5 states)"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-0.5, 4.0), (-0.5, 3.5)], **kw)
        # States: (end-to-end distance, radius of gyration)
        self.states = {
            'native': (0.5, 0.5),
            'intermediate_1': (1.5, 1.0),
            'intermediate_2': (1.0, 2.0),
            'misfolded': (2.5, 1.5),
            'unfolded': (3.0, 3.0),
        }
        self.depths = [4.0, 2.0, 1.5, 2.5, 1.0]  # well depths
        self.widths = [0.3, 0.4, 0.4, 0.5, 0.8]  # well widths
        self.omega = 0.8

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]

        dxdt = torch.zeros_like(x)
        dydt = torch.zeros_like(y)

        # Gradient from each metastable well
        for (sx, sy), depth, width in zip(
            self.states.values(), self.depths, self.widths
        ):
            dx = x - sx
            dy = y - sy
            dist_sq = dx ** 2 + dy ** 2
            force = depth * torch.exp(-dist_sq / (2 * width ** 2))
            dxdt = dxdt - force * dx / (width ** 2)
            dydt = dydt - force * dy / (width ** 2)

        # Entropic bias: unfolded states have higher entropy → drift toward extended
        dxdt = dxdt + 0.1 * torch.sigmoid(x - 1.0)
        dydt = dydt + 0.08 * torch.sigmoid(y - 1.0)

        # Folding funnel: general tendency toward compact states (opposing entropy)
        dxdt = dxdt - 0.15 * x
        dydt = dydt - 0.12 * y

        # Energy barrier between native and misfolded (prevents easy interconversion)
        barrier_x = 1.5
        barrier_y = 1.0
        barrier = 2.0 * torch.exp(-((x - barrier_x) ** 2 + (y - barrier_y) ** 2) / 0.3)
        dxdt = dxdt + barrier * 2 * (x - barrier_x) / 0.3
        dydt = dydt + barrier * 2 * (y - barrier_y) / 0.3

        # Confine to positive quadrant
        dxdt = dxdt - 2.0 * torch.clamp(-x, min=0)
        dydt = dydt - 2.0 * torch.clamp(-y, min=0)

        # Independent rotation for transition richness
        dxdt = dxdt + self.omega * y
        dydt = dydt - self.omega * x
        # Confinement to prevent divergence from rotation
        dxdt = dxdt - 0.01 * x ** 3
        dydt = dydt - 0.01 * y ** 3

        return torch.stack([dxdt, dydt])


# ---------------------------------------------------------------------------
# System 13: Power grid frequency — multi-machine synchronization
# ---------------------------------------------------------------------------
@register
class PowerGridFrequency(CatalogSystem):
    """Simplified power grid with 2 generators and multiple sync states.

    Phase difference and frequency difference between two generators.
    Multiple stable synchronization modes depending on power transfer.
    """

    name = "power_grid_frequency"
    category = "D"
    description = "Power grid synchronization with multiple phase-locked states"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-4.0, 4.0), (-2.0, 2.0)], **kw)
        self.P = 1.0  # power transfer
        self.D = 0.5  # damping
        self.H = 1.0  # inertia
        self.K2 = 0.3  # second harmonic coupling
        self.omega = 0.8

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        delta, omega = state[0], state[1]  # phase diff, freq diff

        # Swing equation with higher harmonics
        ddelta = omega
        domega = (
            self.P
            - self.D * omega
            - torch.sin(delta)
            - self.K2 * torch.sin(2 * delta)
            - 0.1 * torch.sin(3 * delta)
        ) / self.H

        # Independent rotation for transition richness
        ddelta = ddelta + self.omega * omega
        domega = domega - self.omega * delta
        # Confinement to prevent divergence from rotation
        ddelta = ddelta - 0.01 * delta ** 3
        domega = domega - 0.01 * omega ** 3

        return torch.stack([ddelta, domega])


# ---------------------------------------------------------------------------
# System 14: Alluvial fan — sediment transport with channel avulsion
# ---------------------------------------------------------------------------
@register
class AlluvialFan(CatalogSystem):
    """Alluvial fan model with channel avulsion.

    Two variables: channel position (x) and aggradation level (y).
    When aggradation exceeds threshold, channel avulses (sudden shift).
    Multiple stable channel positions create basins.
    """

    name = "alluvial_fan"
    category = "H"
    description = "Sediment transport with 5 stable channel positions"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-3.0, 3.0), (-1.0, 3.0)], **kw)
        # Stable channel positions (topographic lows)
        self.channels = [-2.0, -1.0, 0.0, 1.0, 2.0]
        self.avulsion_threshold = 2.0
        self.omega = 0.8

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]

        # Channel attracted to nearest topographic low
        dxdt = torch.zeros_like(x)
        for cx in self.channels:
            dx = x - cx
            force = torch.exp(-dx ** 2 / 0.3)
            dxdt = dxdt - 2.0 * force * dx

        # Aggradation: builds up when channel is stable
        stability = torch.exp(-dxdt ** 2 / 0.1)  # more stable → more aggradation
        dydt = 0.5 * stability - 0.3 * y

        # Avulsion trigger: when y exceeds threshold, destabilize channel
        avulsion_factor = torch.sigmoid(5.0 * (y - self.avulsion_threshold))
        # During avulsion, add lateral force proportional to aggradation
        dxdt = dxdt + 1.5 * avulsion_factor * torch.sin(3.0 * x)
        # Aggradation drops during avulsion
        dydt = dydt - 2.0 * avulsion_factor * y

        # Confinement
        dxdt = dxdt - 0.02 * x ** 3

        # Independent rotation for transition richness
        dxdt = dxdt + self.omega * y
        dydt = dydt - self.omega * x
        # Confinement to prevent divergence from rotation
        dxdt = dxdt - 0.01 * x ** 3
        dydt = dydt - 0.01 * y ** 3

        return torch.stack([dxdt, dydt])


# ---------------------------------------------------------------------------
# System 15: Spin glass 2D — simplified frustrated magnet
# ---------------------------------------------------------------------------
@register
class SpinGlass2D(CatalogSystem):
    """Simplified spin glass with frustrated interactions.

    Two continuous spin variables with competing ferromagnetic
    and antiferromagnetic couplings creating multiple metastable states.
    """

    name = "spin_glass_2d"
    category = "D"
    description = "Frustrated spin system with 5 metastable spin configurations"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.5, 2.5)], **kw)
        self.J_ferro = 1.0  # ferromagnetic coupling
        self.J_anti = -0.6  # antiferromagnetic coupling
        self.h1 = 0.1  # external field on spin 1
        self.h2 = -0.05  # external field on spin 2
        self.T = 0.3  # effective temperature (noise-like but deterministic)
        self.omega = 1.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        s1, s2 = state[0], state[1]

        # Mean-field dynamics with double-well potential per spin
        # V(s) = -s^2/2 + s^4/4 (double well at ±1)
        dV1 = -s1 + s1 ** 3
        dV2 = -s2 + s2 ** 3

        # Coupling terms
        coupling_1 = -self.J_ferro * torch.tanh(s2) + self.J_anti * torch.tanh(2 * s2)
        coupling_2 = -self.J_ferro * torch.tanh(s1) + self.J_anti * torch.tanh(2 * s1)

        # External fields
        ds1dt = -dV1 + coupling_1 + self.h1 - 0.05 * s1 ** 5  # confinement
        ds2dt = -dV2 + coupling_2 + self.h2 - 0.05 * s2 ** 5

        # Frustrated coupling creates additional wells
        ds1dt = ds1dt + 0.4 * torch.sin(math.pi * s1) * torch.cos(math.pi * s2)
        ds2dt = ds2dt + 0.4 * torch.cos(math.pi * s1) * torch.sin(math.pi * s2)

        # Independent rotation for transition richness
        ds1dt = ds1dt + self.omega * s2
        ds2dt = ds2dt - self.omega * s1
        # Confinement to prevent divergence from rotation
        ds1dt = ds1dt - 0.01 * s1 ** 3
        ds2dt = ds2dt - 0.01 * s2 ** 3

        return torch.stack([ds1dt, ds2dt])


# ---------------------------------------------------------------------------
# System 16: Origami fold — flat sheet with multiple fold configurations
# ---------------------------------------------------------------------------
@register
class OrigamiFold(CatalogSystem):
    """Origami fold configurations as dynamical system.

    A flat sheet can fold into multiple stable configurations.
    Two fold angles determine the configuration. The elastic
    energy landscape has multiple wells corresponding to different folds.
    Interesting because transitions require coordinated angle changes.
    """

    name = "origami_fold"
    category = "H"
    description = "Origami with 4 stable fold configurations and coordinated transitions"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.5, 2.5)], **kw)
        self.k_fold = 2.0  # fold stiffness
        self.k_flat = 0.5  # flat stiffness
        self.coupling = 1.0  # coupling between folds
        self.omega = 1.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        theta1, theta2 = state[0], state[1]

        # Each fold has bistable potential: V(θ) = -cos(θ) + a*cos(2θ)
        # Minima at θ = ±θ_fold and θ = 0 (flat)
        dV1 = self.k_fold * torch.sin(theta1) - 0.8 * torch.sin(2 * theta1)
        dV2 = self.k_fold * torch.sin(theta2) - 0.8 * torch.sin(2 * theta2)

        # Coupling: certain fold combinations are energetically preferred
        # Parallel folds (θ1 ≈ θ2) preferred over anti-parallel
        coupling_force_1 = -self.coupling * (theta1 - theta2) * torch.exp(
            -(theta1 - theta2) ** 2 / 2.0
        )
        coupling_force_2 = -self.coupling * (theta2 - theta1) * torch.exp(
            -(theta1 - theta2) ** 2 / 2.0
        )

        # Anti-parallel coupling (θ1 ≈ -θ2) also somewhat stable
        coupling_force_1 = coupling_force_1 - 0.5 * self.coupling * (
            theta1 + theta2
        ) * torch.exp(-(theta1 + theta2) ** 2 / 2.0)
        coupling_force_2 = coupling_force_2 - 0.5 * self.coupling * (
            theta2 + theta1
        ) * torch.exp(-(theta1 + theta2) ** 2 / 2.0)

        dtheta1 = -dV1 + coupling_force_1 - 0.3 * theta1  # damping
        dtheta2 = -dV2 + coupling_force_2 - 0.3 * theta2

        # Soft confinement
        dtheta1 = dtheta1 - 0.01 * theta1 ** 3
        dtheta2 = dtheta2 - 0.01 * theta2 ** 3

        # Independent rotation for transition richness
        dtheta1 = dtheta1 + self.omega * theta2
        dtheta2 = dtheta2 - self.omega * theta1
        # Confinement to prevent divergence from rotation
        dtheta1 = dtheta1 - 0.01 * theta1 ** 3
        dtheta2 = dtheta2 - 0.01 * theta2 ** 3

        return torch.stack([dtheta1, dtheta2])


# ---------------------------------------------------------------------------
# System 17: Attention switching — cognitive model with 3 focus targets
# ---------------------------------------------------------------------------
@register
class AttentionSwitching(CatalogSystem):
    """Cognitive attention model with 3 focus targets.

    Neural activity settles to one of 3 attention states.
    Transitions occur when a stimulus exceeds threshold.
    Based on biased competition model of selective attention.
    """

    name = "attention_switching"
    category = "H"
    description = "Biased competition attention model with 3 focus states"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-2.0, 2.0), (-2.0, 2.0)], **kw)
        self.tau = 1.0
        self.w_self = 3.0  # self-excitation
        self.w_cross = -2.0  # mutual inhibition
        # Three attention targets at 120° apart in 2D activity space
        self.targets = [
            (0, 1.5),
            (-1.3, -0.75),
            (1.3, -0.75),
        ]
        self.bias = [0.1, 0.15, 0.05]  # bottom-up salience
        self.omega = 1.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]

        dxdt = torch.zeros_like(x)
        dydt = torch.zeros_like(y)

        # Each target creates attractive dynamics in its region
        activities = []
        for (tx, ty), bias in zip(self.targets, self.bias):
            dx = x - tx
            dy = y - ty
            dist_sq = dx ** 2 + dy ** 2
            # Activity proportional to proximity
            act = torch.exp(-dist_sq / 1.0) + bias
            activities.append(act)

        # Winner-take-all competition
        total_act = sum(activities) + 1e-8
        for i, ((tx, ty), act) in enumerate(zip(self.targets, activities)):
            # Normalized attraction
            weight = act ** 2 / (total_act ** 2 + 0.1)
            dx = x - tx
            dy = y - ty
            dxdt = dxdt - self.w_self * weight * dx
            dydt = dydt - self.w_self * weight * dy

        # Global inhibition
        dxdt = dxdt - 0.2 * x
        dydt = dydt - 0.2 * y

        # Habituation-like adaptation (slight instability at attractors)
        for tx, ty in self.targets:
            dx = x - tx
            dy = y - ty
            dist_sq = dx ** 2 + dy ** 2
            # Very close to attractor: slight repulsion (habituation)
            hab = 0.3 * torch.exp(-dist_sq / 0.1)
            dxdt = dxdt + hab * dx / (dist_sq + 0.01)
            dydt = dydt + hab * dy / (dist_sq + 0.01)

        # Confinement
        dxdt = dxdt - 0.02 * x ** 3
        dydt = dydt - 0.02 * y ** 3

        # Independent rotation for transition richness
        dxdt = dxdt + self.omega * y
        dydt = dydt - self.omega * x
        # Confinement to prevent divergence from rotation
        dxdt = dxdt - 0.01 * x ** 3
        dydt = dydt - 0.01 * y ** 3

        return torch.stack([dxdt, dydt]) / self.tau
