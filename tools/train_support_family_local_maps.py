"""Compatibility imports for the historical paper support-routing API.

This file is intentionally not a command-line trainer.
"""

from experiments.neurips_2026.local_operators.contract import *  # noqa: F401,F403
from experiments.neurips_2026.local_operators.routing import (
    _build_route_codebook,
    _generate_source_route_fit_batches,
    _route_indices_np,
    _validate_protocol,
)
from skae.support.routing import *  # noqa: F401,F403
from skae.support.routing import (
    _assign_family_ids_np,
    _binary_jaccard,
    _prototype_masks,
    _step_routes_for_torch,
    _support_family_labels,
    _support_keys,
    _support_mask,
)
