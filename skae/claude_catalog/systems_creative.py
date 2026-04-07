"""Creative transition-rich dynamical systems for Koopman autoencoder benchmark.

Contains 17 systems across categories E (phase/oscillator), G (abstract/mathematical),
and H (creative/novel). Each system is a 2D or 3D autonomous ODE integrated with RK4,
designed so that 30-70% of trajectories transition between basins before settling into
one of 3-10 endpoint basins.
"""

import torch
import math
from skae.claude_catalog.base import CatalogSystem, rk4_step
from skae.claude_catalog.registry import register

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPS = 1e-8


def _soft_abs(x, k=10.0):
    """Smooth approximation to |x| using x*tanh(k*x)."""
    return x * torch.tanh(k * x)


def _soft_confine(x, scale=0.01, power=3):
    """Soft cubic confinement to prevent divergence."""
    return -scale * x ** power


# ===================================================================
# Category E: Phase / Oscillator (3 systems)
# ===================================================================


@register
class KuramotoFrustrated2D(CatalogSystem):
    name = "kuramoto_frustrated_2d"
    category = "E"
    description = "Two effective phase variables from frustrated Kuramoto oscillators with multiple phase-locked states"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-math.pi, math.pi), (-math.pi, math.pi)], **kw)
        self.omega1 = 0.5
        self.omega2 = -0.3
        self.K = 2.0
        self.alpha = math.pi / 4.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        theta1 = state[0]
        theta2 = state[1]
        # theta3 derived from constraint: theta3 = -(theta1 + theta2)
        # (zero-mean constraint on three oscillators)
        theta3 = -(theta1 + theta2)

        d1 = self.omega1 + self.K * torch.sin(theta2 - theta1 - self.alpha) + self.K * torch.sin(theta3 - theta1 - self.alpha)
        d2 = self.omega2 + self.K * torch.sin(theta1 - theta2 - self.alpha) + self.K * torch.sin(theta3 - theta2 - self.alpha)

        # Soft confinement to keep phases from growing unboundedly
        # (wrap-around is natural for phases, but confinement avoids numerical drift)
        confine1 = -0.005 * theta1 * (theta1 / math.pi) ** 2
        confine2 = -0.005 * theta2 * (theta2 / math.pi) ** 2

        return torch.stack([d1 + confine1, d2 + confine2])


@register
class SNICMulti(CatalogSystem):
    name = "snic_multi"
    category = "E"
    description = "Multiple saddle-node on invariant circle bifurcation with 3 fixed points"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-2.0, 2.0), (-2.0, 2.0)], **kw)
        self.a = 1.2
        self.eps = 0.3

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x = state[0]
        y = state[1]
        r2 = x ** 2 + y ** 2 + _EPS
        r = torch.sqrt(r2)
        # theta = atan2(y, x)
        theta = torch.atan2(y, x)

        # Polar dynamics:
        # dr/dt = r*(1 - r^2) - eps*cos(3*theta)
        # dtheta/dt = 1 - a*cos(3*theta)
        rdot = r * (1.0 - r2) - self.eps * torch.cos(3.0 * theta)
        thetadot = 1.0 - self.a * torch.cos(3.0 * theta)

        # Convert to Cartesian:
        # dx/dt = rdot * cos(theta) - r * thetadot * sin(theta)
        # dy/dt = rdot * sin(theta) + r * thetadot * cos(theta)
        cos_t = x / (r + _EPS)
        sin_t = y / (r + _EPS)
        dxdt = rdot * cos_t - r * thetadot * sin_t
        dydt = rdot * sin_t + r * thetadot * cos_t

        # Soft confinement
        dxdt = dxdt - 0.01 * x * (x ** 2 + y ** 2)
        dydt = dydt - 0.01 * y * (x ** 2 + y ** 2)

        return torch.stack([dxdt, dydt])


@register
class PhaseBistableRing(CatalogSystem):
    name = "phase_bistable_ring"
    category = "E"
    description = "Overdamped phase dynamics on a ring with 4 wells on a limit cycle"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-2.0, 2.0), (-2.0, 2.0)], **kw)
        self.eps = 0.3
        self.mu = 2.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x = state[0]
        y = state[1]
        r2 = x ** 2 + y ** 2 + _EPS
        r = torch.sqrt(r2)
        theta = torch.atan2(y, x)

        # Radial: dr/dt = mu*(1-r)*r  ->  stable limit cycle at r=1
        rdot = self.mu * (1.0 - r) * r

        # Phase: dtheta/dt = -sin(4*theta) + eps*sin(2*theta)
        # This creates 4 wells on the circle with symmetry-breaking from the eps term
        thetadot = -torch.sin(4.0 * theta) + self.eps * torch.sin(2.0 * theta)

        # Add a deterministic "noise-like" perturbation from the radius deviation
        # to create transitions between wells
        thetadot = thetadot + 0.3 * (r - 1.0) * torch.sin(2.0 * theta)

        # Convert to Cartesian
        cos_t = x / (r + _EPS)
        sin_t = y / (r + _EPS)
        dxdt = rdot * cos_t - r * thetadot * sin_t
        dydt = rdot * sin_t + r * thetadot * cos_t

        return torch.stack([dxdt, dydt])


# ===================================================================
# Category G: Abstract / Mathematical (7 systems)
# ===================================================================


@register
class Heteroclinic3Node(CatalogSystem):
    name = "heteroclinic_3node"
    category = "G"
    description = "Three-node heteroclinic cycle with cyclic dominance (3D Lotka-Volterra)"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(0.05, 1.0), (0.05, 1.0), (0.05, 1.0)], **kw)
        self.a = 1.5  # strong suppression of successor
        self.b = 0.5  # weak suppression of predecessor

    @property
    def dim(self):
        return 3

    def dynamics(self, state):
        x1 = torch.clamp(state[0], min=0.0)
        x2 = torch.clamp(state[1], min=0.0)
        x3 = torch.clamp(state[2], min=0.0)

        # dx_i/dt = x_i * (1 - x_i - a*x_{i+1} - b*x_{i-1})
        # Cyclic: 1->2->3->1
        dx1 = x1 * (1.0 - x1 - self.a * x2 - self.b * x3)
        dx2 = x2 * (1.0 - x2 - self.a * x3 - self.b * x1)
        dx3 = x3 * (1.0 - x3 - self.a * x1 - self.b * x2)

        # Small perturbation to break exact heteroclinic connections
        # and create definite basin outcomes
        dx1 = dx1 + 0.005 * (0.3 - x1)
        dx2 = dx2 + 0.005 * (0.3 - x2)
        dx3 = dx3 + 0.005 * (0.3 - x3)

        return torch.stack([dx1, dx2, dx3])


@register
class WinnerlessCompetition(CatalogSystem):
    name = "winnerless_competition"
    category = "G"
    description = "Rabinovich-type winnerless competition with asymmetric coupling and switching dynamics"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(0.01, 1.5), (0.01, 1.5), (0.01, 1.5)], **kw)
        self.sigma = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        self.rho = torch.tensor(
            [[1.0, 1.5, 0.5],
             [0.5, 1.0, 1.5],
             [1.5, 0.5, 1.0]],
            dtype=torch.float64,
        )
        self.eps = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)

    @property
    def dim(self):
        return 3

    def dynamics(self, state):
        # Clamp to avoid negative values (population model)
        x = torch.clamp(state, min=0.0)

        # dx_i/dt = x_i * (sigma_i - sum_j rho_ij * x_j) + eps_i
        interaction = self.rho @ x  # [3]
        dxdt = x * (self.sigma - interaction) + self.eps

        # Soft confinement to prevent divergence
        dxdt = dxdt - 0.005 * x ** 3

        return dxdt


@register
class PolynomialDesigned(CatalogSystem):
    name = "polynomial_designed"
    category = "G"
    description = "Hand-designed polynomial vector field with 5 stable equilibria from product terms"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-3.0, 3.0), (-3.0, 3.0)], **kw)
        self.alpha = 0.3
        self.beta = -0.3

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x = state[0]
        y = state[1]

        # Product terms create zeros at x = 0, +/-1, +/-2 and similarly for y
        # The coupling terms break the grid symmetry to select 5 stable equilibria
        dxdt = -x * (x ** 2 - 1.0) * (x ** 2 - 4.0) + self.alpha * y
        dydt = -y * (y ** 2 - 1.0) * (y ** 2 - 4.0) + self.beta * x

        # Scale down to avoid very steep gradients far from origin
        # The polynomial grows as x^5 so we dampen slightly at large |x|
        scale_x = 1.0 / (1.0 + 0.02 * x ** 2)
        scale_y = 1.0 / (1.0 + 0.02 * y ** 2)
        dxdt = dxdt * scale_x
        dydt = dydt * scale_y

        return torch.stack([dxdt, dydt])


@register
class RationalField(CatalogSystem):
    name = "rational_field"
    category = "G"
    description = "Rational vector field with regularized singularities and spiral coupling"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.5, 2.5)], **kw)
        self.alpha = 0.2

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x = state[0]
        y = state[1]

        denom = 1.0 + x ** 2 + y ** 2  # always >= 1, no blowup

        dxdt = (x ** 3 - 3.0 * x) / denom - self.alpha * y
        dydt = (y ** 3 - 3.0 * y) / denom + self.alpha * x

        # Soft confinement for large states
        dxdt = dxdt - 0.01 * x ** 3 / (1.0 + x ** 2)
        dydt = dydt - 0.01 * y ** 3 / (1.0 + y ** 2)

        return torch.stack([dxdt, dydt])


@register
class SelkovModified(CatalogSystem):
    name = "selkov_modified"
    category = "G"
    description = "Modified Selkov glycolysis model with logistic self-activation creating 3 steady states"

    def __init__(self, dt=0.05, **kw):
        super().__init__(dt=dt, ic_box=[(0.01, 2.0), (0.01, 4.0)], **kw)
        self.a = 0.08
        self.b = 0.6
        self.c = 0.15
        self.K = 1.5

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x = torch.clamp(state[0], min=0.0)
        y = torch.clamp(state[1], min=0.0)

        # Original Selkov: dx/dt = -x + a*y + x^2*y
        #                  dy/dt = b - a*y - x^2*y
        # Additional logistic self-activation: c*x*(1 - x/K)
        dxdt = -x + self.a * y + x ** 2 * y + self.c * x * (1.0 - x / self.K)
        dydt = self.b - self.a * y - x ** 2 * y

        # Soft confinement to keep within reasonable bounds
        dxdt = dxdt - 0.001 * x ** 3
        dydt = dydt - 0.001 * y ** 3

        return torch.stack([dxdt, dydt])


@register
class BrusselatorMulti(CatalogSystem):
    name = "brusselator_multi"
    category = "G"
    description = "Modified Brusselator with spatial coupling term creating multiple attracting modes"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(0.01, 4.0), (0.01, 4.0)], **kw)
        self.A = 1.0
        self.B = 3.0  # beyond Hopf bifurcation
        self.D = 0.5
        self.L = 2.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x = torch.clamp(state[0], min=_EPS)
        y = torch.clamp(state[1], min=_EPS)

        # Spatial coupling term adds multiple modes
        spatial = self.D * (
            torch.cos(math.pi * x / self.L) + torch.cos(math.pi * y / self.L)
        )

        dxdt = self.A - (self.B + 1.0) * x + x ** 2 * y + spatial
        dydt = self.B * x - x ** 2 * y

        # Soft confinement
        dxdt = dxdt - 0.005 * torch.clamp(x - 5.0, min=0.0) ** 2
        dydt = dydt - 0.005 * torch.clamp(y - 5.0, min=0.0) ** 2

        return torch.stack([dxdt, dydt])


@register
class LimitCycleWellsHybrid(CatalogSystem):
    name = "limit_cycle_wells_hybrid"
    category = "G"
    description = "Hybrid system with competing limit cycles and Gaussian wells on the unit circle"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-2.0, 2.0), (-2.0, 2.0)], **kw)
        # Three Gaussian wells placed on the unit circle at 120-degree intervals
        angles = [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0]
        self.well_x = torch.tensor([math.cos(a) for a in angles], dtype=torch.float64)
        self.well_y = torch.tensor([math.sin(a) for a in angles], dtype=torch.float64)
        self.well_depth = 2.5  # strength of each well
        self.well_sigma2 = 0.15  # width (variance) of each well

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x = state[0]
        y = state[1]
        r2 = x ** 2 + y ** 2 + _EPS
        r = torch.sqrt(r2)

        # Radial bistability: r_dot = r*(1-r)*(r-0.5)
        # Stable at r=0 (origin) and r=1 (limit cycle), unstable at r=0.5
        rdot = r * (1.0 - r) * (r - 0.5)

        # Angular: slow rotation to move trajectories along the cycle
        omega = 0.5
        theta_dot = omega

        # Base flow in Cartesian
        cos_t = x / (r + _EPS)
        sin_t = y / (r + _EPS)
        dxdt = rdot * cos_t - r * theta_dot * sin_t
        dydt = rdot * sin_t + r * theta_dot * cos_t

        # Add Gaussian well attraction (gradient descent on -sum of Gaussians)
        for i in range(3):
            dx = x - self.well_x[i]
            dy = y - self.well_y[i]
            dist2 = dx ** 2 + dy ** 2
            gauss = torch.exp(-dist2 / (2.0 * self.well_sigma2))
            # Gradient of Gaussian well: pushes toward center
            dxdt = dxdt - self.well_depth * gauss * dx / self.well_sigma2
            dydt = dydt - self.well_depth * gauss * dy / self.well_sigma2

        # Soft confinement beyond r=2
        dxdt = dxdt - 0.05 * x * torch.clamp(r - 2.0, min=0.0)
        dydt = dydt - 0.05 * y * torch.clamp(r - 2.0, min=0.0)

        return torch.stack([dxdt, dydt])


# ===================================================================
# Category H: Creative / Novel (7 systems)
# ===================================================================


@register
class MazePotential(CatalogSystem):
    name = "maze_potential"
    category = "H"
    description = "Maze-like potential landscape with Gaussian ridges forming corridors between 4 room wells"

    def __init__(self, dt=0.05, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.5, 2.5)], **kw)

        # 4 room wells at quadrant centers
        self.room_centers = torch.tensor(
            [[1.2, 1.2], [-1.2, 1.2], [-1.2, -1.2], [1.2, -1.2]],
            dtype=torch.float64,
        )
        self.room_depth = 3.0
        self.room_sigma2 = 0.5

        # Gaussian ridges (walls) defined by center, angle, height, long/short widths
        # Format: (cx, cy, angle, height, sigma_long, sigma_short)
        self.ridges = [
            # Horizontal wall, top-left to top-right with gap in middle
            (0.0, 1.8, 0.0, 5.0, 2.5, 0.15),
            # Horizontal wall, bottom
            (0.0, -1.8, 0.0, 5.0, 2.5, 0.15),
            # Vertical wall, left
            (-1.8, 0.0, math.pi / 2, 5.0, 2.5, 0.15),
            # Vertical wall, right
            (1.8, 0.0, math.pi / 2, 5.0, 2.5, 0.15),
            # Internal cross walls with gaps for passages
            # Vertical middle wall (upper half, gap at center)
            (0.0, 0.9, math.pi / 2, 3.0, 0.7, 0.15),
            # Vertical middle wall (lower half, gap at center)
            (0.0, -0.9, math.pi / 2, 3.0, 0.7, 0.15),
            # Horizontal middle wall (left half, gap at center)
            (-0.9, 0.0, 0.0, 3.0, 0.7, 0.15),
            # Horizontal middle wall (right half, gap at center)
            (0.9, 0.0, 0.0, 3.0, 0.7, 0.15),
        ]

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x = state[0]
        y = state[1]

        # Gradient of room wells (attractive)
        dxdt = torch.tensor(0.0, dtype=torch.float64)
        dydt = torch.tensor(0.0, dtype=torch.float64)

        for i in range(4):
            cx = self.room_centers[i, 0]
            cy = self.room_centers[i, 1]
            dx_well = x - cx
            dy_well = y - cy
            dist2 = dx_well ** 2 + dy_well ** 2
            gauss = torch.exp(-dist2 / (2.0 * self.room_sigma2))
            dxdt = dxdt - self.room_depth * gauss * dx_well / self.room_sigma2
            dydt = dydt - self.room_depth * gauss * dy_well / self.room_sigma2

        # Gradient of ridge walls (repulsive, gradient of potential)
        for cx, cy, angle, h, s_long, s_short in self.ridges:
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            dx_r = x - cx
            dy_r = y - cy
            # Rotate into ridge frame
            u = cos_a * dx_r + sin_a * dy_r  # along ridge
            v = -sin_a * dx_r + cos_a * dy_r  # across ridge

            exp_u = torch.exp(-u ** 2 / (2.0 * s_long ** 2))
            exp_v = torch.exp(-v ** 2 / (2.0 * s_short ** 2))
            ridge_val = h * exp_u * exp_v

            # Gradient in ridge frame
            du = -ridge_val * u / (s_long ** 2)
            dv = -ridge_val * v / (s_short ** 2)

            # Rotate gradient back; we want to go DOWN the potential so
            # force = -grad(V); ridges are positive potential -> repulsive
            # Actually we want gradient ascent on ridges (repulsion) mixed
            # with gradient descent on wells. The ridge IS the potential,
            # so force = +grad(V_ridge) pushes away from ridge.
            fx = cos_a * du - sin_a * dv
            fy = sin_a * du + cos_a * dv

            # Repel from ridges: go up the gradient of ridge potential
            dxdt = dxdt - fx
            dydt = dydt - fy

        # Quadratic confinement
        dxdt = dxdt - 0.1 * x * torch.clamp(x ** 2 + y ** 2 - 4.0, min=0.0)
        dydt = dydt - 0.1 * y * torch.clamp(x ** 2 + y ** 2 - 4.0, min=0.0)

        return torch.stack([dxdt, dydt])


@register
class ArrestedSpiral(CatalogSystem):
    name = "arrested_spiral"
    category = "H"
    description = "Global spiral inflow with local Gaussian wells that capture trajectories at different radii"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-3.0, 3.0), (-3.0, 3.0)], **kw)
        self.alpha_damp = 0.3
        self.omega = 2.0

        # 4 wells at different radii and angles, plus origin
        self.well_centers = torch.tensor(
            [
                [1.5, 0.0],    # radius ~1.5
                [0.0, 1.8],    # radius ~1.8
                [-1.0, -1.0],  # radius ~1.41
                [0.8, -1.5],   # radius ~1.7
            ],
            dtype=torch.float64,
        )
        self.well_depth = 4.0
        self.well_sigma2 = 0.25

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x = state[0]
        y = state[1]

        # Background spiral toward origin
        dxdt = -self.alpha_damp * x + self.omega * y
        dydt = -self.alpha_damp * y - self.omega * x

        # Gaussian well attractions
        for i in range(self.well_centers.shape[0]):
            cx = self.well_centers[i, 0]
            cy = self.well_centers[i, 1]
            dx_w = x - cx
            dy_w = y - cy
            dist2 = dx_w ** 2 + dy_w ** 2
            gauss = torch.exp(-dist2 / (2.0 * self.well_sigma2))
            dxdt = dxdt - self.well_depth * gauss * dx_w / self.well_sigma2
            dydt = dydt - self.well_depth * gauss * dy_w / self.well_sigma2

        # Origin also acts as a well (natural attractor of the spiral)
        # Already handled by the spiral inflow, but we strengthen it slightly
        r2 = x ** 2 + y ** 2
        origin_gauss = torch.exp(-r2 / 0.3)
        dxdt = dxdt - 1.5 * origin_gauss * x / 0.15
        dydt = dydt - 1.5 * origin_gauss * y / 0.15

        # Soft confinement
        dxdt = dxdt - 0.02 * x ** 3
        dydt = dydt - 0.02 * y ** 3

        return torch.stack([dxdt, dydt])


@register
class SoftXORLandscape(CatalogSystem):
    name = "soft_xor_landscape"
    category = "H"
    description = "Four-well XOR-like potential with asymmetric transition barriers and rotation"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.5, 2.5)], **kw)
        self.alpha = 0.4

        # Wells: same-sign corners are deep, cross-sign are shallow
        # (1,1) and (-1,-1) deep; (1,-1) and (-1,1) shallow
        self.wells = torch.tensor(
            [[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]],
            dtype=torch.float64,
        )
        self.depths = torch.tensor([3.0, 3.0, 1.5, 1.5], dtype=torch.float64)
        self.sigma2 = 0.4

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x = state[0]
        y = state[1]

        # Gradient descent on sum of negative Gaussians
        dxdt = torch.tensor(0.0, dtype=torch.float64)
        dydt = torch.tensor(0.0, dtype=torch.float64)

        for i in range(4):
            cx = self.wells[i, 0]
            cy = self.wells[i, 1]
            dx_w = x - cx
            dy_w = y - cy
            dist2 = dx_w ** 2 + dy_w ** 2
            gauss = torch.exp(-dist2 / (2.0 * self.sigma2))
            dxdt = dxdt - self.depths[i] * gauss * dx_w / self.sigma2
            dydt = dydt - self.depths[i] * gauss * dy_w / self.sigma2

        # Add saddle structure at origin to create XOR-like barrier geometry
        # V_saddle = s * x * y  ->  dV/dx = s*y, dV/dy = s*x
        saddle_strength = 0.5
        dxdt = dxdt - saddle_strength * y
        dydt = dydt - saddle_strength * x

        # Rotation to create asymmetric transitions
        dxdt_rot = dxdt + self.alpha * y
        dydt_rot = dydt - self.alpha * x

        # Soft confinement
        dxdt_rot = dxdt_rot - 0.05 * x * torch.clamp(x ** 2 + y ** 2 - 4.0, min=0.0)
        dydt_rot = dydt_rot - 0.05 * y * torch.clamp(x ** 2 + y ** 2 - 4.0, min=0.0)

        return torch.stack([dxdt_rot, dydt_rot])


@register
class CompetitiveMemory(CatalogSystem):
    name = "competitive_memory"
    category = "H"
    description = "Competitive memory retrieval network with 5 stored patterns and lateral inhibition"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.5, 2.5)], **kw)
        self.sigma = 1.0
        # 5 patterns: 4 corners + center
        self.patterns = torch.tensor(
            [
                [1.5, 1.5],
                [-1.5, 1.5],
                [-1.5, -1.5],
                [1.5, -1.5],
                [0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        self.n_patterns = 5
        # Lateral inhibition exponent (higher = more winner-take-all)
        self.inhibition_exp = 3.0

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x = state[0]
        y = state[1]
        xy = torch.stack([x, y])  # [2]

        # Compute distances to each pattern
        diffs = self.patterns - xy.unsqueeze(0)  # [5, 2]
        dist2 = (diffs ** 2).sum(dim=1)  # [5]

        # Softmax weights with lateral inhibition
        # w_i = exp(-|x-p_i|^2 / sigma^2)
        raw_weights = torch.exp(-dist2 / (self.sigma ** 2))

        # Raise to power for winner-take-all inhibition
        inhibited = raw_weights ** self.inhibition_exp
        total = inhibited.sum() + _EPS
        weights = inhibited / total  # [5]

        # Weighted pull toward patterns
        # dx/dt = sum_i w_i * (p_i - x)
        pull = (weights.unsqueeze(1) * diffs).sum(dim=0)  # [2]

        dxdt = pull[0]
        dydt = pull[1]

        # Soft confinement
        dxdt = dxdt - 0.01 * x ** 3
        dydt = dydt - 0.01 * y ** 3

        return torch.stack([dxdt, dydt])


@register
class IntermittentBurst(CatalogSystem):
    name = "intermittent_burst"
    category = "H"
    description = "Near-homoclinic bifurcation system with intermittent bursting near basin boundaries"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.5, 2.5)], **kw)
        self.mu = 0.3
        self.eps = 0.5

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x = state[0]
        y = state[1]

        # Near-homoclinic: multiple equilibria from cubic x term
        # dx/dt = y
        # dy/dt = mu*y - x^3 + x + eps*y*(1 - x^2 - y^2)
        dxdt = y
        dydt = self.mu * y - x ** 3 + x + self.eps * y * (1.0 - x ** 2 - y ** 2)

        # Add Gaussian wells at the equilibria to create definite basins
        # Equilibria of x^3 - x = 0 are x = -1, 0, 1
        eq_x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        well_depth = 1.0
        well_sigma2 = 0.3
        for i in range(3):
            dx_w = x - eq_x[i]
            dist2 = dx_w ** 2 + y ** 2
            gauss = torch.exp(-dist2 / (2.0 * well_sigma2))
            dxdt = dxdt - well_depth * gauss * dx_w / well_sigma2
            dydt = dydt - well_depth * gauss * y / well_sigma2

        # Soft confinement
        r2 = x ** 2 + y ** 2
        conf = 0.02 * torch.clamp(r2 - 4.0, min=0.0)
        dxdt = dxdt - conf * x
        dydt = dydt - conf * y

        return torch.stack([dxdt, dydt])


@register
class MultiScroll2D(CatalogSystem):
    name = "multi_scroll_2d"
    category = "H"
    description = "2D multi-scroll system inspired by Chua's circuit with smooth piecewise-linear nonlinearity"

    def __init__(self, dt=0.01, **kw):
        super().__init__(dt=dt, ic_box=[(-3.0, 3.0), (-3.0, 3.0)], **kw)
        self.alpha_chua = 9.0
        self.m0 = -1.0 / 7.0
        self.m1 = 2.0 / 7.0
        self.beta = 0.0
        # Breakpoints for multi-scroll
        self.breakpoints = torch.tensor([1.0], dtype=torch.float64)
        # Smoothness parameter for tanh approximation of |x|
        self.k = 10.0

    @property
    def dim(self):
        return 2

    def _smooth_abs(self, x):
        """Smooth |x| approximation: x * tanh(k*x)."""
        return x * torch.tanh(self.k * x)

    def _h(self, x):
        """Smooth piecewise-linear function with breakpoints.

        h(x) = m1*x + sum_i (m0-m1)/2 * (|x-b_i| - |x+b_i|)
        """
        result = self.m1 * x
        for b in self.breakpoints:
            result = result + (self.m0 - self.m1) / 2.0 * (
                self._smooth_abs(x - b) - self._smooth_abs(x + b)
            )
        return result

    def dynamics(self, state):
        x = state[0]
        y = state[1]

        dxdt = self.alpha_chua * (y - self._h(x))
        dydt = x - y + self.beta

        # Add weak wells at the scroll centers to create definite basins
        # Scroll centers are approximately at x = 0, +/-2 (between breakpoints)
        centers_x = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float64)
        well_depth = 0.3
        well_sigma2 = 0.8
        for cx in centers_x:
            dx_w = x - cx
            dist2 = dx_w ** 2 + y ** 2
            gauss = torch.exp(-dist2 / (2.0 * well_sigma2))
            dxdt = dxdt - well_depth * gauss * dx_w / well_sigma2
            dydt = dydt - well_depth * gauss * y / well_sigma2

        # Soft confinement
        dxdt = dxdt - 0.01 * x ** 3
        dydt = dydt - 0.01 * y ** 3

        return torch.stack([dxdt, dydt])


@register
class CheckerboardPotential(CatalogSystem):
    name = "checkerboard_potential"
    category = "H"
    description = "Checkerboard of alternating cosine wells and saddles with rotation and confinement"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-2.5, 2.5), (-2.5, 2.5)], **kw)
        self.A = 1.5  # checkerboard amplitude
        self.alpha = 0.3  # rotation strength
        self.confinement = 0.15  # quadratic confinement

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x = state[0]
        y = state[1]

        # Checkerboard potential: V = -A * cos(pi*x) * cos(pi*y) + confinement
        # Wells at integer points where cos(pi*x)*cos(pi*y) = 1, i.e. (even, even) points
        # -dV/dx:
        grad_x = -self.A * math.pi * torch.sin(math.pi * x) * torch.cos(math.pi * y)
        grad_y = -self.A * math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y)

        # Confinement gradient: V_conf = confinement * (x^2 + y^2) / 2
        grad_x = grad_x + self.confinement * x
        grad_y = grad_y + self.confinement * y

        # Gradient descent: dx/dt = -dV/dx
        dxdt = -grad_x
        dydt = -grad_y

        # Add rotation (non-conservative)
        dxdt = dxdt + self.alpha * y
        dydt = dydt - self.alpha * x

        # Stronger confinement far from origin to limit to ~4 wells
        r2 = x ** 2 + y ** 2
        strong_conf = 0.05 * torch.clamp(r2 - 3.0, min=0.0)
        dxdt = dxdt - strong_conf * x
        dydt = dydt - strong_conf * y

        return torch.stack([dxdt, dydt])
