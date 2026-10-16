"""Thirteen analytic two-dimensional multibasin systems.

Each system defines a vector field ``dynamics(state)`` on one state ``[2]``
and advances one stored observation with one fixed-step RK4 step.
``multibasin.MultibasinSystem`` integrates them in float64 and stores float32 states.
The equations and parameters are those of the paper's appendix "Multibasin
benchmark inventory and system definitions"; below, ``R(x, y) = (y, -x)``.
"""

from __future__ import annotations

import math
from functools import partial

import torch


def rk4_step(dynamics, state: torch.Tensor, dt: float) -> torch.Tensor:
    """One fourth-order Runge--Kutta step of size ``dt`` of ``dx/dt = dynamics(x)``."""
    k1 = dynamics(state)
    k2 = dynamics(state + 0.5 * dt * k1)
    k3 = dynamics(state + 0.5 * dt * k2)
    k4 = dynamics(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class AnalyticSystem:
    """An autonomous ODE whose initial conditions are uniform in the box ``ic_box``."""

    def __init__(self, dt: float, ic_box):
        self.dt = dt
        self.ic_box = ic_box

    def step(self, state: torch.Tensor) -> torch.Tensor:
        return rk4_step(self.dynamics, state, self.dt)

    def reset(self, generator: torch.Generator) -> torch.Tensor:
        """A float64 initial condition, one uniform draw per coordinate."""
        return torch.tensor(
            [lo + (hi - lo) * torch.rand(1, generator=generator, dtype=torch.float64).item() for lo, hi in self.ic_box],
            dtype=torch.float64,
        )


class GaussianWells(AnalyticSystem):
    """Gaussian wells, quartic confinement and a rotational component (the paper's eq. app_gaussian_well).

    The field is ``(-dV/dx + omega*y, -dV/dy - omega*x)`` with
    ``V = -sum_k a_k exp(-|x-c_k|^2 / 2 s_k^2) + conf * (x^4 + y^4)``;
    each well is ``(c_x, c_y, a, s)``. ``GAUSSIAN_WELLS`` lists the nine systems.
    """

    def __init__(self, wells, omega, conf, ic_box, dt):
        super().__init__(dt, ic_box)
        self.wells = wells
        self.omega = omega
        self.conf = conf

    def dynamics(self, state):
        x, y = state[0], state[1]
        d_v_dx = torch.zeros_like(x)
        d_v_dy = torch.zeros_like(y)
        for center_x, center_y, amplitude, sigma in self.wells:
            delta_x = x - center_x
            delta_y = y - center_y
            gaussian = amplitude * torch.exp(-(delta_x**2 + delta_y**2) / (2 * sigma**2))
            d_v_dx = d_v_dx + gaussian * delta_x / sigma**2
            d_v_dy = d_v_dy + gaussian * delta_y / sigma**2
        d_v_dx = d_v_dx + 4 * self.conf * x**3
        d_v_dy = d_v_dy + 4 * self.conf * y**3
        return torch.stack([-d_v_dx + self.omega * y, -d_v_dy - self.omega * x])


def _polygon(count, radius, amplitude, phase=0.0):
    """``count`` wells of width 0.5 evenly spaced on a circle."""
    return [
        (radius * math.cos(2 * math.pi * k / count + phase), radius * math.sin(2 * math.pi * k / count + phase),
         amplitude, 0.5)
        for k in range(count)
    ]


def _box(half_width):
    return [(-half_width, half_width), (-half_width, half_width)]


# name: (wells, omega, conf, ic_box)
GAUSSIAN_WELLS = {
    "cal_square_4": (_polygon(4, 1.8, 3.0, math.pi / 4), 1.0, 0.03, _box(3.3)),
    "cal_pentagon_5": (_polygon(5, 1.8, 2.0, math.pi / 2), 1.1, 0.03, _box(3.3)),
    "cal_hexagon_6": (_polygon(6, 1.7, 2.0), 1.0, 0.03, _box(3.3)),
    "cal_octagon_8": (_polygon(8, 2.2, 3.0), 0.90, 0.02, _box(4.0)),
    "cal_asymmetric_3": ([(0.0, 1.8, 2.5, 0.55), (-1.5, -0.9, 1.5, 0.4), (1.5, -0.9, 2.0, 0.5)],
                         1.0, 0.03, [(-3.0, 3.0), (-2.5, 3.0)]),
    "cal_high_cross_3": (_polygon(3, 1.8, 3.0, math.pi / 2), 2.0, 0.03, _box(3.3)),
    "var_depth_gradient_4": ([(-1.3, 1.3, 2.2, 0.55), (1.3, 1.3, 2.5, 0.5), (1.3, -1.3, 3.0, 0.5),
                              (-1.3, -1.3, 3.5, 0.5)], 1.3, 0.03, _box(3.0)),
    "var_diamond_4": ([(0.0, 2.2, 2.5, 0.5), (2.2, 0.0, 2.5, 0.5), (0.0, -2.2, 2.5, 0.5), (-2.2, 0.0, 2.5, 0.5)],
                      1.0, 0.02, _box(3.5)),
    "var_l_shape_5": ([(-1.5, 1.5, 2.5, 0.5), (-1.5, 0.0, 2.5, 0.5), (-1.5, -1.5, 2.5, 0.5), (0.0, -1.5, 2.5, 0.5),
                       (1.5, -1.5, 2.5, 0.5)], 1.0, 0.03, _box(3.5)),
}


class TransitionRoutes4(AnalyticSystem):
    """Four Gaussian wells with rotational corridors along ``y = +-1.8`` (the paper's eq. app_transition_routes).

    ``-grad V + (1 + 0.3 C(y)) R x`` with wells at ``(+-1.8, +-1.8)``,
    ``a = 3.0``, ``s = 0.6``, confinement ``0.03 (x^4 + y^4)`` and
    ``C(y) = exp(-(y - 1.8)^2 / 0.3) + exp(-(y + 1.8)^2 / 0.3)``.
    """

    def __init__(self, dt):
        super().__init__(dt, _box(3.5))
        self.wells = [(-1.8, 1.8, 3.0, 0.6), (1.8, 1.8, 3.0, 0.6), (-1.8, -1.8, 3.0, 0.6), (1.8, -1.8, 3.0, 0.6)]

    def dynamics(self, state):
        x, y = state[0], state[1]
        d_v_dx = torch.zeros_like(x)
        d_v_dy = torch.zeros_like(y)
        for center_x, center_y, amplitude, sigma in self.wells:
            delta_x = x - center_x
            delta_y = y - center_y
            gaussian = amplitude * torch.exp(-(delta_x**2 + delta_y**2) / (2 * sigma**2))
            d_v_dx += gaussian * delta_x / sigma**2
            d_v_dy += gaussian * delta_y / sigma**2
        d_v_dx += 0.12 * x**3
        d_v_dy += 0.12 * y**3
        dx_dt = -d_v_dx + y
        dy_dt = -d_v_dy - x
        corridor = torch.exp(-(y - 1.8) ** 2 / 0.3) + torch.exp(-(y + 1.8) ** 2 / 0.3)
        dx_dt += 0.3 * corridor * y
        dy_dt -= 0.3 * corridor * x
        return torch.stack([dx_dt, dy_dt])


def _soft_confine(value, scale, strength):
    return -strength * value**3 / scale**3


class DuffingTripleWell(AnalyticSystem):
    """Damped triple-well Duffing oscillator (the paper's eq. app_duffing_triple).

    ``dq/dt = p - eps q^3/s^3``, ``dp/dt = -V'(q) - delta p - eps p^3/s^3`` with
    ``V'(q) = q^5 - 2q^3 + 2aq``, ``a = 0.3``, ``delta = 0.5``, ``eps = 0.003``,
    ``s = 4``. The rotation strength ``omega`` is zero in the paper; its terms
    are kept so the floating-point operations are exactly those that generated
    the data.
    """

    def __init__(self, dt):
        super().__init__(dt, _box(2.0))
        self.a = 0.3
        self.delta = 0.5
        self.omega = 0.0

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


class SNICMulti(AnalyticSystem):
    """Three-fold symmetric field near a saddle-node on an invariant circle (SNIC).

    The polar field ``dr/dt = r (1 - r^2) - 0.3 cos 3theta``,
    ``dtheta/dt = 1 - 1.2 cos 3theta`` in Cartesian coordinates, plus the
    confinement ``-0.01 x |x|^2`` and the rotation ``0.5 R x``.
    """

    def __init__(self, dt):
        super().__init__(dt, _box(2.0))
        self.a = 1.2
        self.eps = 0.3
        self.omega = 0.5

    def dynamics(self, state):
        x, y = state[0], state[1]
        radius_squared = x**2 + y**2 + _EPS
        radius = torch.sqrt(radius_squared)
        theta = torch.atan2(y, x)
        radial_rate = radius * (1.0 - radius_squared) - self.eps * torch.cos(3.0 * theta)
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


class ArrestedSpiral(AnalyticSystem):
    """A damped spiral arrested by four Gaussian traps and an origin trap (its fifth basin).

    ``-0.3 x + 2 R x - 4 sum_i exp(-|x - c_i|^2 / 0.5) (x - c_i) / 0.25
    - 1.5 exp(-|x|^2 / 0.3) x / 0.15 - 0.02 (x^3, y^3)``.
    """

    def __init__(self, dt):
        super().__init__(dt, _box(3.0))
        self.alpha_damp = 0.3
        self.omega = 2.0
        self.well_centers = torch.tensor([[1.5, 0.0], [0.0, 1.8], [-1.0, -1.0], [0.8, -1.5]], dtype=torch.float64)
        self.well_depth = 4.0
        self.well_sigma2 = 0.25

    def dynamics(self, state):
        x, y = state[0], state[1]
        dx_dt = -self.alpha_damp * x + self.omega * y
        dy_dt = -self.alpha_damp * y - self.omega * x
        for center in self.well_centers:
            delta_x = x - center[0]
            delta_y = y - center[1]
            gaussian = torch.exp(-(delta_x**2 + delta_y**2) / (2.0 * self.well_sigma2))
            dx_dt = dx_dt - self.well_depth * gaussian * delta_x / self.well_sigma2
            dy_dt = dy_dt - self.well_depth * gaussian * delta_y / self.well_sigma2
        origin_gaussian = torch.exp(-(x**2 + y**2) / 0.3)
        dx_dt = dx_dt - 1.5 * origin_gaussian * x / 0.15
        dy_dt = dy_dt - 1.5 * origin_gaussian * y / 0.15
        dx_dt = dx_dt - 0.02 * x**3
        dy_dt = dy_dt - 0.02 * y**3
        return torch.stack([dx_dt, dy_dt])


# name -> constructor taking the stored step ``dt``
ANALYTIC_SYSTEMS = {
    **{name: partial(GaussianWells, *params) for name, params in GAUSSIAN_WELLS.items()},
    "transition_routes_4": TransitionRoutes4,
    "duffing_triple_well_no_rotation": DuffingTripleWell,
    "snic_multi": SNICMulti,
    "arrested_spiral": ArrestedSpiral,
}
