"""Compatibility namespace for the renamed analytic multibasin systems.

New code should import :mod:`skae.dynamics.analytic`.
"""

from skae.dynamics.analytic import *  # noqa: F401,F403
from skae.dynamics.analytic import __all__
