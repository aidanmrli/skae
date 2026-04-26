"""Gradient and gradient-with-rotation dynamical systems.

Category A: Multi-well gradient flows  (dx/dt = -grad V)
Category B: Gradient + rotation         (dx/dt = -grad V + alpha * R @ grad V)

All systems are 2D autonomous ODEs integrated with RK4, designed so that
30-70% of trajectories from the IC box transition between basins before
settling into a stable equilibrium.
"""

import torch
import math
from skae.claude_catalog.base import CatalogSystem, rk4_step
from skae.claude_catalog.registry import register


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_well_grad(x, y, cx, cy, amp, sigma):
    """Return (dV/dx, dV/dy) for a single Gaussian well V = -amp * exp(...)."""
    dx = x - cx
    dy = y - cy
    s2 = sigma * sigma
    exp_term = amp * torch.exp(-(dx * dx + dy * dy) / (2.0 * s2))
    # V = -amp * exp(-r^2 / 2s^2), so  dV/dx = amp * (x-cx)/s^2 * exp(...)
    #   but we want the negative sign on V, so dV/dx = (x-cx)/s^2 * exp_term
    #   with exp_term = amp * exp(...)
    grad_x = (dx / s2) * exp_term
    grad_y = (dy / s2) * exp_term
    return grad_x, grad_y


def _quartic_confinement_grad(x, y, strength):
    """Gradient of strength*(x^4 + y^4)."""
    return 4.0 * strength * x ** 3, 4.0 * strength * y ** 3


# ===================================================================
# Category A: Multi-well gradient flows
# ===================================================================


@register
class TripleWellTriangle(CatalogSystem):
    name = "triple_well_triangle"
    category = "A"
    description = "Three Gaussian wells at equilateral triangle vertices with quartic confinement"

    def __init__(self, dt=0.05, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.5, 2.5)], **kw)
        self.centers = [(0.0, 1.5), (-1.3, -0.75), (1.3, -0.75)]
        self.amp = 3.0
        self.sigma = 0.6
        self.quartic = 0.05

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        gx = torch.zeros_like(x)
        gy = torch.zeros_like(y)
        for cx, cy in self.centers:
            wx, wy = _gaussian_well_grad(x, y, cx, cy, self.amp, self.sigma)
            gx = gx + wx
            gy = gy + wy
        qx, qy = _quartic_confinement_grad(x, y, self.quartic)
        gx = gx + qx
        gy = gy + qy
        return torch.stack([-gx, -gy])


@register
class FourWellSquare(CatalogSystem):
    name = "four_well_square"
    category = "A"
    description = "Four Gaussian wells at square vertices with quartic confinement"

    def __init__(self, dt=0.05, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.5, 2.5)], **kw)
        self.centers = [(1.2, 1.2), (-1.2, 1.2), (-1.2, -1.2), (1.2, -1.2)]
        self.amp = 2.5
        self.sigma = 0.55
        self.quartic = 0.03

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        gx = torch.zeros_like(x)
        gy = torch.zeros_like(y)
        for cx, cy in self.centers:
            wx, wy = _gaussian_well_grad(x, y, cx, cy, self.amp, self.sigma)
            gx = gx + wx
            gy = gy + wy
        qx, qy = _quartic_confinement_grad(x, y, self.quartic)
        gx = gx + qx
        gy = gy + qy
        return torch.stack([-gx, -gy])


@register
class FiveWellCross(CatalogSystem):
    name = "five_well_cross"
    category = "A"
    description = "Five Gaussian wells in cross pattern (center plus cardinal directions)"

    def __init__(self, dt=0.05, **kw):
        super().__init__(dt=dt, ic_box=[(-3.5, 3.5), (-3.5, 3.5)], **kw)
        self.centers = [(0.0, 0.0), (2.0, 0.0), (-2.0, 0.0), (0.0, 2.0), (0.0, -2.0)]
        self.amps = [2.0, 2.5, 2.5, 2.5, 2.5]
        self.sigma = 0.65
        self.quartic = 0.02

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        gx = torch.zeros_like(x)
        gy = torch.zeros_like(y)
        for (cx, cy), amp in zip(self.centers, self.amps):
            wx, wy = _gaussian_well_grad(x, y, cx, cy, amp, self.sigma)
            gx = gx + wx
            gy = gy + wy
        qx, qy = _quartic_confinement_grad(x, y, self.quartic)
        gx = gx + qx
        gy = gy + qy
        return torch.stack([-gx, -gy])


@register
class AsymmetricTripleWell(CatalogSystem):
    name = "asymmetric_triple_well"
    category = "A"
    description = "Three wells with different depths and widths creating asymmetric transitions"

    def __init__(self, dt=0.05, **kw):
        super().__init__(dt=dt, ic_box=[(-3.0, 3.0), (-3.0, 3.0)], **kw)
        self.centers = [(-1.5, 0.0), (1.0, 1.2), (0.5, -1.5)]
        self.amps = [2.5, 3.5, 1.8]
        self.sigmas = [0.5, 0.7, 0.4]
        self.quartic = 0.03

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        gx = torch.zeros_like(x)
        gy = torch.zeros_like(y)
        for (cx, cy), amp, sig in zip(self.centers, self.amps, self.sigmas):
            wx, wy = _gaussian_well_grad(x, y, cx, cy, amp, sig)
            gx = gx + wx
            gy = gy + wy
        qx, qy = _quartic_confinement_grad(x, y, self.quartic)
        gx = gx + qx
        gy = gy + qy
        return torch.stack([-gx, -gy])


@register
class MullerBrown(CatalogSystem):
    name = "muller_brown"
    category = "A"
    description = "Classic Muller-Brown potential from computational chemistry (3 minima)"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-1.5, 1.5), (-0.5, 2.5)], **kw)
        # Standard Muller-Brown parameters
        self.A = [-200.0, -100.0, -170.0, 15.0]
        self.a = [-1.0, -1.0, -6.5, 0.7]
        self.b = [0.0, 0.0, 11.0, 0.6]
        self.c = [-10.0, -10.0, -6.5, 0.7]
        self.x0 = [1.0, 0.0, -0.5, -1.0]
        self.y0 = [0.0, 0.5, 1.5, 1.0]
        self.scale = 0.005

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        gx = torch.zeros_like(x)
        gy = torch.zeros_like(y)
        for i in range(4):
            dx = x - self.x0[i]
            dy = y - self.y0[i]
            exponent = self.a[i] * dx * dx + self.b[i] * dx * dy + self.c[i] * dy * dy
            exp_val = self.A[i] * torch.exp(exponent)
            # dV/dx = A * exp(...) * (2*a*(x-x0) + b*(y-y0))
            gx = gx + exp_val * (2.0 * self.a[i] * dx + self.b[i] * dy)
            # dV/dy = A * exp(...) * (b*(x-x0) + 2*c*(y-y0))
            gy = gy + exp_val * (self.b[i] * dx + 2.0 * self.c[i] * dy)
        # Scale the gradient
        gx = self.scale * gx
        gy = self.scale * gy
        # dx/dt = -dV/dx
        return torch.stack([-gx, -gy])


@register
class SixWellHexagon(CatalogSystem):
    name = "six_well_hexagon"
    category = "A"
    description = "Six Gaussian wells at hexagonal vertices with quartic confinement"

    def __init__(self, dt=0.05, **kw):
        super().__init__(dt=dt, ic_box=[(-3.5, 3.5), (-3.5, 3.5)], **kw)
        radius = 2.0
        self.centers = [
            (radius * math.cos(math.radians(60 * k)),
             radius * math.sin(math.radians(60 * k)))
            for k in range(6)
        ]
        self.amp = 3.0
        self.sigma = 0.5
        self.quartic = 0.02

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        gx = torch.zeros_like(x)
        gy = torch.zeros_like(y)
        for cx, cy in self.centers:
            wx, wy = _gaussian_well_grad(x, y, cx, cy, self.amp, self.sigma)
            gx = gx + wx
            gy = gy + wy
        qx, qy = _quartic_confinement_grad(x, y, self.quartic)
        gx = gx + qx
        gy = gy + qy
        return torch.stack([-gx, -gy])


@register
class GaussianMixture7(CatalogSystem):
    name = "gaussian_mixture_7"
    category = "A"
    description = "Seven Gaussian wells at hand-chosen positions with varying amplitudes"

    def __init__(self, dt=0.05, **kw):
        super().__init__(dt=dt, ic_box=[(-3.5, 3.5), (-3.5, 3.5)], **kw)
        self.centers = [
            (0.0, 0.0), (2.0, 1.0), (-1.0, 2.0), (-2.0, -0.5),
            (1.0, -2.0), (0.5, 2.5), (-1.5, 1.5),
        ]
        self.amps = [2.0, 2.5, 3.0, 2.0, 2.5, 1.8, 2.2]
        self.sigma = 0.55
        self.quartic = 0.02

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        gx = torch.zeros_like(x)
        gy = torch.zeros_like(y)
        for (cx, cy), amp in zip(self.centers, self.amps):
            wx, wy = _gaussian_well_grad(x, y, cx, cy, amp, self.sigma)
            gx = gx + wx
            gy = gy + wy
        qx, qy = _quartic_confinement_grad(x, y, self.quartic)
        gx = gx + qx
        gy = gy + qy
        return torch.stack([-gx, -gy])


@register
class CascadingTerraces(CatalogSystem):
    name = "cascading_terraces"
    category = "A"
    description = "Four wells at different energy offsets creating directional flow preference"

    def __init__(self, dt=0.05, **kw):
        super().__init__(dt=dt, ic_box=[(-3.5, 3.5), (-3.5, 3.5)], **kw)
        self.centers = [(-2.0, 0.0), (0.0, 1.0), (1.0, -1.0), (2.5, 0.0)]
        self.depth = 3.0
        self.sigma = 0.6
        # Base energy offsets: wells descend left-to-right, creating a
        # preferred cascade direction.  The offset only affects V (not
        # the Gaussian shape), so we add a linear tilt to the potential.
        # V_tilt = slope * x  => dV_tilt/dx = slope,  dV_tilt/dy = 0
        self.slope = -0.25  # tilts potential toward +x
        self.quartic = 0.02

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        gx = torch.zeros_like(x)
        gy = torch.zeros_like(y)
        for cx, cy in self.centers:
            wx, wy = _gaussian_well_grad(x, y, cx, cy, self.depth, self.sigma)
            gx = gx + wx
            gy = gy + wy
        qx, qy = _quartic_confinement_grad(x, y, self.quartic)
        gx = gx + qx
        gy = gy + qy
        # Add linear tilt gradient
        gx = gx + self.slope
        return torch.stack([-gx, -gy])


@register
class RingOfWells(CatalogSystem):
    name = "ring_of_wells"
    category = "A"
    description = "Five wells on a ring plus one central well"

    def __init__(self, dt=0.05, **kw):
        super().__init__(dt=dt, ic_box=[(-3.0, 3.0), (-3.0, 3.0)], **kw)
        ring_radius = 1.8
        self.ring_centers = [
            (ring_radius * math.cos(2.0 * math.pi * k / 5),
             ring_radius * math.sin(2.0 * math.pi * k / 5))
            for k in range(5)
        ]
        self.center = (0.0, 0.0)
        self.ring_amp = 2.5
        self.ring_sigma = 0.45
        self.center_amp = 2.5
        self.center_sigma = 0.5
        self.quartic = 0.02

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        gx = torch.zeros_like(x)
        gy = torch.zeros_like(y)
        # Ring wells
        for cx, cy in self.ring_centers:
            wx, wy = _gaussian_well_grad(x, y, cx, cy, self.ring_amp, self.ring_sigma)
            gx = gx + wx
            gy = gy + wy
        # Central well
        wx, wy = _gaussian_well_grad(
            x, y, self.center[0], self.center[1],
            self.center_amp, self.center_sigma,
        )
        gx = gx + wx
        gy = gy + wy
        qx, qy = _quartic_confinement_grad(x, y, self.quartic)
        gx = gx + qx
        gy = gy + qy
        return torch.stack([-gx, -gy])


@register
class DoubleFunnel(CatalogSystem):
    name = "double_funnel"
    category = "A"
    description = "Two deep wells connected by a narrow channel through a ridge"

    def __init__(self, dt=0.05, **kw):
        super().__init__(dt=dt, ic_box=[(-3.5, 3.5), (-2.5, 2.5)], **kw)
        # Two deep attracting wells
        self.well_centers = [(-2.0, 0.0), (2.0, 0.0)]
        self.well_amp = 4.0
        self.well_sigma = 0.6
        # Ridge along the y-axis: a positive Gaussian ridge at x=0
        # V_ridge = +ridge_amp * exp(-x^2/(2*ridge_sx^2)) * exp(-y^2/(2*ridge_sy^2))
        # We make ridge_sy large so the ridge extends in y, but put a gap
        # at y=0 by *subtracting* a narrow Gaussian notch.
        self.ridge_amp = 3.0
        self.ridge_sx = 0.4   # narrow in x
        self.ridge_sy = 2.0   # long in y
        self.notch_amp = 3.5  # slightly deeper than ridge to create a gap
        self.notch_sx = 0.5
        self.notch_sy = 0.35  # narrow in y => passage near y=0
        self.quartic = 0.03

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        gx = torch.zeros_like(x)
        gy = torch.zeros_like(y)

        # Deep wells (negative potential => attract)
        for cx, cy in self.well_centers:
            wx, wy = _gaussian_well_grad(x, y, cx, cy, self.well_amp, self.well_sigma)
            gx = gx + wx
            gy = gy + wy

        # Ridge (positive potential => repel): V_ridge = +amp * exp(...)
        # dV/dx for a positive Gaussian centered at (0,0)
        sx2 = self.ridge_sx ** 2
        sy2 = self.ridge_sy ** 2
        exp_ridge = self.ridge_amp * torch.exp(-x * x / (2.0 * sx2) - y * y / (2.0 * sy2))
        gx = gx - (x / sx2) * exp_ridge   # negative because V is positive
        gy = gy - (y / sy2) * exp_ridge

        # Notch (negative potential = cut into the ridge): V_notch = -amp*exp(...)
        nsx2 = self.notch_sx ** 2
        nsy2 = self.notch_sy ** 2
        exp_notch = self.notch_amp * torch.exp(-x * x / (2.0 * nsx2) - y * y / (2.0 * nsy2))
        gx = gx + (x / nsx2) * exp_notch
        gy = gy + (y / nsy2) * exp_notch

        qx, qy = _quartic_confinement_grad(x, y, self.quartic)
        gx = gx + qx
        gy = gy + qy
        return torch.stack([-gx, -gy])


# ===================================================================
# Category B: Gradient + rotation
# ===================================================================


@register
class TripleWellSpiral(CatalogSystem):
    name = "triple_well_spiral"
    category = "B"
    description = "Triple-well triangle with rotational component creating spiraling approach"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.5, 2.5)], **kw)
        self.centers = [(0.0, 1.5), (-1.3, -0.75), (1.3, -0.75)]
        self.amp = 3.0
        self.sigma = 0.6
        self.quartic = 0.05
        self.alpha = 0.5

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        gx = torch.zeros_like(x)
        gy = torch.zeros_like(y)
        for cx, cy in self.centers:
            wx, wy = _gaussian_well_grad(x, y, cx, cy, self.amp, self.sigma)
            gx = gx + wx
            gy = gy + wy
        qx, qy = _quartic_confinement_grad(x, y, self.quartic)
        gx = gx + qx
        gy = gy + qy
        # dx/dt = -dV/dx + alpha * dV/dy
        # dy/dt = -dV/dy - alpha * dV/dx
        dxdt = -gx + self.alpha * gy
        dydt = -gy - self.alpha * gx
        return torch.stack([dxdt, dydt])


@register
class FourWellVortex(CatalogSystem):
    name = "four_well_vortex"
    category = "B"
    description = "Four-well square with rotational component creating vortex-like flow"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.5, 2.5)], **kw)
        self.centers = [(1.2, 1.2), (-1.2, 1.2), (-1.2, -1.2), (1.2, -1.2)]
        self.amp = 2.5
        self.sigma = 0.55
        self.quartic = 0.03
        self.alpha = 0.6

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        gx = torch.zeros_like(x)
        gy = torch.zeros_like(y)
        for cx, cy in self.centers:
            wx, wy = _gaussian_well_grad(x, y, cx, cy, self.amp, self.sigma)
            gx = gx + wx
            gy = gy + wy
        qx, qy = _quartic_confinement_grad(x, y, self.quartic)
        gx = gx + qx
        gy = gy + qy
        dxdt = -gx + self.alpha * gy
        dydt = -gy - self.alpha * gx
        return torch.stack([dxdt, dydt])


@register
class CompetingSpirals3(CatalogSystem):
    name = "competing_spirals_3"
    category = "B"
    description = "Three point attractors with different local rotation rates blended by distance"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.5, 2.5)], **kw)
        # Attractor centers (equilateral triangle)
        self.centers = [(0.0, 1.5), (-1.3, -0.75), (1.3, -0.75)]
        # Decay rate (real part of eigenvalue) -- negative for stability
        self.decay = -1.0
        # Rotation rates (imaginary part of eigenvalue)
        self.omegas = [2.0, 5.0, 0.5]
        # Softmax temperature for blending
        self.temperature = 0.8
        # Quartic confinement to keep trajectories bounded
        self.quartic = 0.04

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]

        # Compute softmax weights based on negative distance to each center
        dists_sq = []
        for cx, cy in self.centers:
            dsq = (x - cx) ** 2 + (y - cy) ** 2
            dists_sq.append(dsq)

        # Softmax over -dist/temperature
        log_weights = torch.stack(
            [-d / self.temperature for d in dists_sq]
        )
        max_lw = torch.max(log_weights)
        exp_weights = torch.exp(log_weights - max_lw)
        weights = exp_weights / torch.sum(exp_weights)

        # Blend local linear dynamics from each attractor
        dxdt = torch.zeros_like(x)
        dydt = torch.zeros_like(y)
        for i, (cx, cy) in enumerate(self.centers):
            dx_local = x - cx
            dy_local = y - cy
            omega = self.omegas[i]
            # Local dynamics: A_i @ [dx, dy]  where A_i = [[decay, -omega], [omega, decay]]
            fx = self.decay * dx_local - omega * dy_local
            fy = omega * dx_local + self.decay * dy_local
            dxdt = dxdt + weights[i] * fx
            dydt = dydt + weights[i] * fy

        # Quartic confinement
        dxdt = dxdt - 4.0 * self.quartic * x ** 3
        dydt = dydt - 4.0 * self.quartic * y ** 3
        return torch.stack([dxdt, dydt])


@register
class ShearTripleWell(CatalogSystem):
    name = "shear_triple_well"
    category = "B"
    description = "Triple-well triangle with constant shear creating asymmetric transitions"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-3.0, 3.0), (-3.0, 3.0)], **kw)
        self.centers = [(0.0, 1.5), (-1.3, -0.75), (1.3, -0.75)]
        self.amp = 3.0
        self.sigma = 0.6
        self.quartic = 0.05
        self.gamma = 0.3  # shear strength

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        gx = torch.zeros_like(x)
        gy = torch.zeros_like(y)
        for cx, cy in self.centers:
            wx, wy = _gaussian_well_grad(x, y, cx, cy, self.amp, self.sigma)
            gx = gx + wx
            gy = gy + wy
        qx, qy = _quartic_confinement_grad(x, y, self.quartic)
        gx = gx + qx
        gy = gy + qy
        # Gradient flow + constant shear in x-dynamics
        dxdt = -gx + self.gamma * y
        dydt = -gy
        return torch.stack([dxdt, dydt])


@register
class FiveWellGyroscopic(CatalogSystem):
    name = "five_well_gyroscopic"
    category = "B"
    description = "Five-well cross with gyroscopic rotation creating spiraling inter-well paths"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-3.5, 3.5), (-3.5, 3.5)], **kw)
        self.centers = [(0.0, 0.0), (2.0, 0.0), (-2.0, 0.0), (0.0, 2.0), (0.0, -2.0)]
        self.amps = [2.0, 2.5, 2.5, 2.5, 2.5]
        self.sigma = 0.65
        self.quartic = 0.02
        self.alpha = 0.4

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        gx = torch.zeros_like(x)
        gy = torch.zeros_like(y)
        for (cx, cy), amp in zip(self.centers, self.amps):
            wx, wy = _gaussian_well_grad(x, y, cx, cy, amp, self.sigma)
            gx = gx + wx
            gy = gy + wy
        qx, qy = _quartic_confinement_grad(x, y, self.quartic)
        gx = gx + qx
        gy = gy + qy
        dxdt = -gx + self.alpha * gy
        dydt = -gy - self.alpha * gx
        return torch.stack([dxdt, dydt])


@register
class RotatingAsymmetric4(CatalogSystem):
    name = "rotating_asymmetric_4"
    category = "B"
    description = "Four wells with spatially varying rotation strength (tanh-modulated)"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-3.0, 3.0), (-2.5, 2.5)], **kw)
        self.centers = [(-1.5, 1.0), (-1.5, -1.0), (1.5, 1.0), (1.5, -1.0)]
        self.amp = 2.5
        self.sigma = 0.55
        self.quartic = 0.03
        # Rotation varies with x: alpha(x) = alpha_base + alpha_range * tanh(x)
        self.alpha_base = 0.3
        self.alpha_range = 0.5

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        gx = torch.zeros_like(x)
        gy = torch.zeros_like(y)
        for cx, cy in self.centers:
            wx, wy = _gaussian_well_grad(x, y, cx, cy, self.amp, self.sigma)
            gx = gx + wx
            gy = gy + wy
        qx, qy = _quartic_confinement_grad(x, y, self.quartic)
        gx = gx + qx
        gy = gy + qy
        alpha = self.alpha_base + self.alpha_range * torch.tanh(x)
        dxdt = -gx + alpha * gy
        dydt = -gy - alpha * gx
        return torch.stack([dxdt, dydt])


@register
class StuartLandauTriple(CatalogSystem):
    name = "stuart_landau_triple"
    category = "B"
    description = "Three Stuart-Landau oscillators blended by region via softmax weights"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.5, 2.5)], **kw)
        self.centers = [(0.0, 1.5), (-1.3, -0.75), (1.3, -0.75)]
        self.mu = [1.0, 1.0, 1.0]
        self.omega = [1.0, 3.0, -2.0]
        self.temperature = 0.8
        # Confinement damping to prevent divergence far from attractors
        self.quartic = 0.03

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]

        # Softmax weights based on negative squared distance to centers
        dists_sq = []
        for cx, cy in self.centers:
            dsq = (x - cx) ** 2 + (y - cy) ** 2
            dists_sq.append(dsq)

        log_weights = torch.stack([-d / self.temperature for d in dists_sq])
        max_lw = torch.max(log_weights)
        exp_weights = torch.exp(log_weights - max_lw)
        weights = exp_weights / torch.sum(exp_weights)

        # Each Stuart-Landau oscillator in local coordinates:
        #   dr/dt = mu*r - r^3,  dtheta/dt = omega
        # In Cartesian relative to center (cx, cy):
        #   dx_local/dt = (mu - r^2)*dx - omega*dy
        #   dy_local/dt =  omega*dx + (mu - r^2)*dy
        dxdt = torch.zeros_like(x)
        dydt = torch.zeros_like(y)
        for i, (cx, cy) in enumerate(self.centers):
            dx_local = x - cx
            dy_local = y - cy
            r2 = dx_local ** 2 + dy_local ** 2
            radial = self.mu[i] - r2
            fx = radial * dx_local - self.omega[i] * dy_local
            fy = self.omega[i] * dx_local + radial * dy_local
            dxdt = dxdt + weights[i] * fx
            dydt = dydt + weights[i] * fy

        # Quartic confinement far from all attractors
        dxdt = dxdt - 4.0 * self.quartic * x ** 3
        dydt = dydt - 4.0 * self.quartic * y ** 3
        return torch.stack([dxdt, dydt])
