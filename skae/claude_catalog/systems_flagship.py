"""Flagship systems: the most scientifically interesting designs.

Each system here has a clear one-sentence story that would interest
a NeurIPS reviewer. They combine the calibrated well+rotation formula
with creative dynamics that test specific aspects of Koopman learning.
"""

import torch
import math
from skae.claude_catalog.base import CatalogSystem, rk4_step
from skae.claude_catalog.registry import register


@register
class VoronoiMismatch4(CatalogSystem):
    """4 wells where the TRUE basin map maximally disagrees with Voronoi.

    Story: "The geometric nearest-center partition is wrong for 40% of
    the state space — can Koopman learning discover the true partition?"

    This is achieved by placing wells asymmetrically and using strong
    rotation that creates large spiral arms crossing Voronoi boundaries.
    """
    name = "flagship_voronoi_mismatch"
    category = "H"
    description = "4 wells where true basins maximally disagree with Voronoi cells"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-3.5, 3.5), (-3.5, 3.5)], **kw)
        # Place wells so rotation creates maximum Voronoi mismatch
        self.wells = [
            (-1.2, 1.8, 2.5, 0.5),   # close together but different basins
            (1.2, 1.8, 2.5, 0.5),
            (-1.5, -1.2, 2.5, 0.5),
            (1.5, -1.2, 2.5, 0.5),
        ]
        self.omega = 1.0  # moderate rotation for good Voronoi mismatch
        self.conf = 0.04

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        dVdx = torch.zeros_like(x)
        dVdy = torch.zeros_like(y)
        for cx, cy, amp, sigma in self.wells:
            dx, dy = x - cx, y - cy
            gauss = amp * torch.exp(-(dx**2 + dy**2) / (2 * sigma**2))
            dVdx = dVdx + gauss * dx / (sigma**2)
            dVdy = dVdy + gauss * dy / (sigma**2)
        dVdx = dVdx + 4 * self.conf * x**3
        dVdy = dVdy + 4 * self.conf * y**3
        dxdt = -dVdx + self.omega * y
        dydt = -dVdy - self.omega * x
        return torch.stack([dxdt, dydt])


@register
class FrequencyDiscrimination5(CatalogSystem):
    """5 basins where each has a different damped oscillation frequency.

    Story: "Each basin has distinct temporal dynamics — can a single
    Koopman operator represent all five natural frequencies?"

    Near each attractor, the trajectory spirals inward at a unique
    frequency. The global rotation handles transitions.
    """
    name = "flagship_freq_discrimination"
    category = "H"
    description = "5 basins with different damped oscillation frequencies per basin"

    def __init__(self, dt=0.02, **kw):
        super().__init__(dt=dt, ic_box=[(-3.5, 3.5), (-3.5, 3.5)], **kw)
        r = 1.8
        self.centers = torch.tensor([
            [r * math.cos(2 * math.pi * i / 5 + math.pi / 2),
             r * math.sin(2 * math.pi * i / 5 + math.pi / 2)]
            for i in range(5)
        ], dtype=torch.float64)
        self.freqs = [0.5, 1.5, 3.0, 5.0, 8.0]  # Hz-like
        self.decay = 0.6
        self.sigma = 1.0
        self.omega_global = 0.8

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        pos = state.unsqueeze(0)
        dists = torch.sum((pos - self.centers)**2, dim=1)
        weights = torch.softmax(-dists / (2 * self.sigma**2), dim=0)
        dxdt = torch.zeros_like(x)
        dydt = torch.zeros_like(y)
        for i in range(5):
            cx, cy = self.centers[i]
            dx, dy = x - cx, y - cy
            freq = self.freqs[i]
            dxdt = dxdt + weights[i] * (-self.decay * dx - freq * dy)
            dydt = dydt + weights[i] * (freq * dx - self.decay * dy)
        dxdt = dxdt + self.omega_global * y
        dydt = dydt - self.omega_global * x
        dxdt = dxdt - 0.01 * x**3
        dydt = dydt - 0.01 * y**3
        return torch.stack([dxdt, dydt])


@register
class DampingGradient4(CatalogSystem):
    """4 basins with different damping rates (settling timescales).

    Story: "Two basins settle in 10 steps, two in 100 — does the Koopman
    spectral radius adapt to match local settling timescales?"

    Near each attractor, the decay rate differs by 10x.
    """
    name = "flagship_damping_gradient"
    category = "H"
    description = "4 basins with 10x difference in settling timescale"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-3.3, 3.3), (-3.3, 3.3)], **kw)
        self.centers = torch.tensor([
            [-1.5, 1.5], [1.5, 1.5], [-1.5, -1.5], [1.5, -1.5]
        ], dtype=torch.float64)
        self.rates = [2.0, 0.2, 0.5, 1.5]  # damping rates (10x range)
        self.sigma = 1.2
        self.omega = 0.8

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        pos = state.unsqueeze(0)
        dists = torch.sum((pos - self.centers)**2, dim=1)
        weights = torch.softmax(-dists / (2 * self.sigma**2), dim=0)
        dxdt = torch.zeros_like(x)
        dydt = torch.zeros_like(y)
        for i in range(4):
            cx, cy = self.centers[i]
            dx, dy = x - cx, y - cy
            rate = self.rates[i]
            dxdt = dxdt + weights[i] * (-rate * dx - 1.0 * dy)
            dydt = dydt + weights[i] * (1.0 * dx - rate * dy)
        dxdt = dxdt + self.omega * y
        dydt = dydt - self.omega * x
        dxdt = dxdt - 0.01 * x**3
        dydt = dydt - 0.01 * y**3
        return torch.stack([dxdt, dydt])


@register
class RotationReversal3(CatalogSystem):
    """3 basins: one clockwise spiral, one counterclockwise, one no rotation.

    Story: "Local dynamics rotate in opposite directions across basins —
    a single global Koopman operator must embed both chiralities."

    This is a challenging test because a global linear model can only
    represent one rotation direction.
    """
    name = "flagship_rotation_reversal"
    category = "H"
    description = "3 basins with CW, CCW, and non-rotational local dynamics"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-3.3, 3.3), (-3.3, 3.3)], **kw)
        self.centers = torch.tensor([
            [0.0, 1.8], [-1.5, -0.9], [1.5, -0.9]
        ], dtype=torch.float64)
        # CW rotation, CCW rotation, pure node
        self.A = [
            torch.tensor([[-0.5, -2.0], [2.0, -0.5]], dtype=torch.float64),   # CW spiral
            torch.tensor([[-0.5, 2.0], [-2.0, -0.5]], dtype=torch.float64),   # CCW spiral
            torch.tensor([[-1.0, 0.0], [0.0, -1.5]], dtype=torch.float64),    # pure node
        ]
        self.sigma = 1.0
        self.omega = 0.8

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        pos = state.unsqueeze(0)
        dists = torch.sum((pos - self.centers)**2, dim=1)
        weights = torch.softmax(-dists / (2 * self.sigma**2), dim=0)
        dxdt = torch.zeros_like(x)
        dydt = torch.zeros_like(y)
        for i in range(3):
            local_state = state - self.centers[i]
            dyn = self.A[i] @ local_state
            dxdt = dxdt + weights[i] * dyn[0]
            dydt = dydt + weights[i] * dyn[1]
        dxdt = dxdt + self.omega * y
        dydt = dydt - self.omega * x
        dxdt = dxdt - 0.01 * x**3
        dydt = dydt - 0.01 * y**3
        return torch.stack([dxdt, dydt])


@register
class NestedBasins6(CatalogSystem):
    """6 wells in nested rings: 3 inner + 3 outer.

    Story: "Basins are organized in concentric rings — does the encoder
    learn radial vs angular structure?"

    The inner ring has different dynamics than the outer ring.
    """
    name = "flagship_nested_rings"
    category = "H"
    description = "6 wells in 2 concentric rings (3 inner + 3 outer)"

    def __init__(self, dt=0.03, **kw):
        super().__init__(dt=dt, ic_box=[(-4.0, 4.0), (-4.0, 4.0)], **kw)
        # Inner ring
        r_in = 1.0
        self.wells_inner = [(r_in * math.cos(2*math.pi*i/3 + math.pi/6),
                             r_in * math.sin(2*math.pi*i/3 + math.pi/6), 3.0, 0.4)
                            for i in range(3)]
        # Outer ring (offset by 60 degrees)
        r_out = 2.8
        self.wells_outer = [(r_out * math.cos(2*math.pi*i/3),
                             r_out * math.sin(2*math.pi*i/3), 2.5, 0.5)
                            for i in range(3)]
        self.omega = 1.0
        self.conf = 0.02

    @property
    def dim(self):
        return 2

    def dynamics(self, state):
        x, y = state[0], state[1]
        dVdx = torch.zeros_like(x)
        dVdy = torch.zeros_like(y)
        for wells in [self.wells_inner, self.wells_outer]:
            for cx, cy, amp, sigma in wells:
                dx, dy = x - cx, y - cy
                gauss = amp * torch.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                dVdx = dVdx + gauss * dx / (sigma**2)
                dVdy = dVdy + gauss * dy / (sigma**2)
        dVdx = dVdx + 4 * self.conf * x**3
        dVdy = dVdy + 4 * self.conf * y**3
        dxdt = -dVdx + self.omega * y
        dydt = -dVdy - self.omega * x
        return torch.stack([dxdt, dydt])
