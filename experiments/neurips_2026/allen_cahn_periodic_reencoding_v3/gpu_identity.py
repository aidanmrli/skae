"""Fail-closed CUDA UUID normalization shared by smoke and lineage."""

from __future__ import annotations

import subprocess
import uuid
from typing import Any


def _canonical_bare_uuid(value: str, *, source: str) -> uuid.UUID:
    """Accept only a canonical hyphenated UUID in uniformly lower/upper case."""

    text = value.strip()
    try:
        parsed = uuid.UUID(text)
    except (AttributeError, ValueError) as error:
        raise RuntimeError(f"{source} UUID is malformed") from error
    canonical = str(parsed)
    if text not in {canonical, canonical.upper()}:
        raise RuntimeError(f"{source} UUID is not canonical hyphenated text")
    return parsed


def _pytorch_uuid(raw_uuid: object) -> tuple[uuid.UUID, str]:
    """Derive identity from the exact 16 bytes exposed by torch._C._CUuuid."""

    raw_type = type(raw_uuid).__name__
    if raw_type != "_CUuuid":
        raise RuntimeError(f"Expected PyTorch _CUuuid, observed {raw_type}")
    try:
        raw_bytes = bytes(getattr(raw_uuid, "bytes"))
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("PyTorch _CUuuid.bytes is not a byte sequence") from error
    if len(raw_bytes) != 16:
        raise RuntimeError(
            f"PyTorch _CUuuid.bytes must contain 16 bytes, observed {len(raw_bytes)}"
        )
    from_bytes = uuid.UUID(bytes=raw_bytes)
    raw_text = str(raw_uuid).strip()
    from_text = _canonical_bare_uuid(raw_text, source="PyTorch")
    if from_text != from_bytes:
        raise RuntimeError("PyTorch _CUuuid bytes and string disagree")
    return from_bytes, raw_text


def _query_nvidia_smi_uuid() -> str:
    """Query the single visible NVIDIA device exactly once without a shell."""

    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise RuntimeError("nvidia-smi UUID query failed") from error
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(
            f"Expected exactly one nvidia-smi GPU UUID, observed {len(lines)}"
        )
    return lines[0]


def verified_cuda_uuid_record(raw_uuid: object) -> dict[str, Any]:
    """Cross-check PyTorch's byte identity against one independent NVIDIA query."""

    pytorch_uuid, pytorch_text = _pytorch_uuid(raw_uuid)
    nvidia_raw = _query_nvidia_smi_uuid()
    if not nvidia_raw.startswith("GPU-"):
        raise RuntimeError("nvidia-smi UUID must have the GPU- prefix")
    nvidia_uuid = _canonical_bare_uuid(
        nvidia_raw.removeprefix("GPU-"), source="nvidia-smi"
    )
    if nvidia_uuid != pytorch_uuid:
        raise RuntimeError("PyTorch and nvidia-smi CUDA UUIDs disagree")
    canonical_bare = str(pytorch_uuid)
    canonical_gpu = f"GPU-{canonical_bare}"
    return {
        "gpu_uuid": canonical_gpu,
        "raw_uuid_type": type(raw_uuid).__name__,
        "pytorch_uuid_raw_text": pytorch_text,
        "pytorch_uuid_canonical": canonical_bare,
        "nvidia_smi_uuid_raw_text": nvidia_raw,
        "nvidia_smi_uuid_canonical": canonical_gpu,
        "nvidia_smi_visible_gpu_count": 1,
        "uuid_sources_match": True,
    }
