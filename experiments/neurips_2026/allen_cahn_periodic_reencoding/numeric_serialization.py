"""JSON-safe serialization for extended-real inferential statistics."""

from __future__ import annotations

import math
from typing import Any


def json_safe_statistic(value: float, *, name: str) -> dict[str, Any]:
    """Encode a finite or infinite scalar without nonstandard JSON numbers."""

    scalar = float(value)
    if math.isnan(scalar):
        raise FloatingPointError(f"{name} is NaN")
    if math.isinf(scalar):
        status = "positive_infinity" if scalar > 0.0 else "negative_infinity"
        return {name: None, f"{name}_status": status}
    return {name: scalar, f"{name}_status": "finite"}
