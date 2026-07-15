"""Compatibility imports for :mod:`skae.support.local_operator`."""

from skae.support.local_operator import *  # noqa: F401,F403
from skae.support.local_operator import (
    _encode_sequence_batches,
    _freeze_autoencoder,
    _local_train_step,
    _make_wrapped_model,
    _route_indices_for_torch,
    _target_centers_from_global,
)
