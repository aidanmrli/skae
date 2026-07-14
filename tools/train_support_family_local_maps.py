"""Compatibility imports for :mod:`skae.support.routing`.

This file is intentionally not a command-line trainer.
"""

from skae.support.routing import *  # noqa: F401,F403
from skae.support.routing import (
    _assign_family_ids_np,
    _binary_jaccard,
    _build_route_codebook,
    _generate_source_route_fit_batches,
    _prototype_masks,
    _route_indices_np,
    _step_routes_for_torch,
    _support_family_labels,
    _support_keys,
    _support_mask,
    _validate_protocol,
)
