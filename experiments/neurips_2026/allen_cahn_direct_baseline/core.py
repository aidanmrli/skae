"""Frozen, label-free direct-model and metric primitives.

This is an exact local port of the residual convolutional control used in the
representation-focused Allen--Cahn packet, with generic training/evaluation
helpers added for the matched full-horizon protocol.  Dataset consumers touch
only ``fields`` and ``split_indices``; basin metadata is never returned.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duplicate_safe_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            payload[key] = value
        return payload

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def write_json_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)


def torch_load(path: Path, *, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_source_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"Malformed source-manifest line {line_number}")
        digest, source = parts
        source = source.strip()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Malformed SHA-256 on line {line_number}")
        if source in entries:
            raise ValueError(f"Duplicate source path {source}")
        entries[source] = digest
    return entries


def verify_source_manifest(repo_root: Path, path: Path) -> str:
    entries = parse_source_manifest(path)
    for source, expected in entries.items():
        observed = sha256_path(repo_root / source)
        if observed != expected:
            raise RuntimeError(f"Source drift: {source}: {observed} != {expected}")
    return sha256_path(path)


@dataclass(frozen=True)
class DirectConfig:
    grid_size: int = 16
    channels: int = 2
    hidden_channels: int = 344
    num_blocks: int = 4
    activation: str = "tanh"
    padding_mode: str = "circular"
    residual_scale: float = 0.1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "DirectConfig":
        valid = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in values.items() if key in valid})


def _activation(name: str) -> nn.Module:
    if name.lower() == "tanh":
        return nn.Tanh()
    raise ValueError("The frozen direct baseline permits only tanh activations")


class ConvBlock(nn.Module):
    def __init__(self, width: int, *, activation: str, padding_mode: str):
        super().__init__()
        groups = max(1, min(8, int(width) // 8))
        if int(width) % groups:
            raise ValueError("hidden_channels must be divisible by GroupNorm groups")
        layers: list[nn.Module] = []
        for _ in range(2):
            layers.extend(
                [
                    nn.Conv2d(
                        int(width),
                        int(width),
                        kernel_size=3,
                        padding=1,
                        padding_mode=padding_mode,
                    ),
                    nn.GroupNorm(groups, int(width)),
                    _activation(activation),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DirectResidualConv(nn.Module):
    """Autonomous observation-space stepper with periodic convolution."""

    def __init__(self, cfg: DirectConfig):
        super().__init__()
        self.cfg = cfg
        width = int(cfg.hidden_channels)
        self.input_conv = nn.Conv2d(
            int(cfg.channels),
            width,
            kernel_size=3,
            padding=1,
            padding_mode=str(cfg.padding_mode),
        )
        self.input_activation = _activation(str(cfg.activation))
        self.blocks = nn.Sequential(
            *[
                ConvBlock(
                    width,
                    activation=str(cfg.activation),
                    padding_mode=str(cfg.padding_mode),
                )
                for _ in range(int(cfg.num_blocks))
            ]
        )
        self.output_conv = nn.Conv2d(
            width,
            int(cfg.channels),
            kernel_size=3,
            padding=1,
            padding_mode=str(cfg.padding_mode),
        )
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    @property
    def observation_size(self) -> int:
        return int(self.cfg.channels) * int(self.cfg.grid_size) ** 2

    def _unflatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(
            x.shape[0],
            int(self.cfg.grid_size),
            int(self.cfg.grid_size),
            int(self.cfg.channels),
        ).permute(0, 3, 1, 2)

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)

    def step_observation(self, x: torch.Tensor) -> torch.Tensor:
        field = self._unflatten(x)
        hidden = self.input_activation(self.input_conv(field))
        update = self.output_conv(self.blocks(hidden))
        return self._flatten(field + float(self.cfg.residual_scale) * update)

    def rollout(self, x0: torch.Tensor, *, horizon: int) -> torch.Tensor:
        current = x0
        predictions: list[torch.Tensor] = []
        for _ in range(int(horizon)):
            current = self.step_observation(current)
            predictions.append(current)
        return torch.stack(predictions, dim=1)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def load_field_splits(
    path: Path,
    *,
    expected_sha256: str,
    expected_total_shape: tuple[int, ...],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Load only the two training-authorized dataset fields."""

    observed = sha256_path(path)
    if observed != expected_sha256:
        raise RuntimeError(f"Dataset drift: {path}: {observed} != {expected_sha256}")
    payload = torch_load(path)
    if not isinstance(payload, dict):
        raise TypeError("Dataset must be a mapping")
    fields = payload["fields"]
    raw_splits = payload["split_indices"]
    if not isinstance(fields, torch.Tensor) or tuple(fields.shape) != expected_total_shape:
        raise ValueError(f"Unexpected field shape: {getattr(fields, 'shape', None)}")
    if fields.dtype != torch.float32 or not bool(torch.isfinite(fields).all()):
        raise FloatingPointError("Dataset fields must be entirely finite float32")
    if not isinstance(raw_splits, dict):
        raise TypeError("split_indices must be a mapping")
    splits = {
        str(key): value.detach().cpu().to(dtype=torch.int64).contiguous()
        for key, value in raw_splits.items()
        if str(key) in {"train", "val", "test"}
    }
    return fields, splits


def flatten_fields(fields: torch.Tensor) -> torch.Tensor:
    if fields.ndim != 5 or tuple(fields.shape[-3:]) != (16, 16, 2):
        raise ValueError(f"Expected [N,T,16,16,2], got {tuple(fields.shape)}")
    return fields.reshape(fields.shape[0], fields.shape[1], 512).contiguous()


def select_split(
    fields: torch.Tensor,
    splits: dict[str, torch.Tensor],
    split: str,
) -> torch.Tensor:
    if split not in splits:
        raise KeyError(f"Missing split {split!r}")
    return flatten_fields(fields[splits[split]])


def sample_sequence_batch(
    fields: torch.Tensor,
    *,
    batch_size: int,
    window_length: int,
    generator: torch.Generator,
) -> torch.Tensor:
    trajectory_count, time_count, _ = fields.shape
    max_start = time_count - (int(window_length) + 1)
    if max_start < 0:
        raise ValueError("Training trajectory is shorter than the requested window")
    trajectory_index = torch.randint(
        0, trajectory_count, (int(batch_size),), generator=generator
    )
    time_index = torch.randint(
        0, max_start + 1, (int(batch_size),), generator=generator
    )
    offsets = torch.arange(int(window_length) + 1)
    return fields[trajectory_index[:, None], time_index[:, None] + offsets[None, :]]


def augment_periodic_symmetries(
    sequence: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    batch = sequence.reshape(sequence.shape[0], sequence.shape[1], 16, 16, 2)
    augmented: list[torch.Tensor] = []
    for sample in batch:
        quarter_turns = int(torch.randint(0, 4, (1,), generator=generator).item())
        transformed = torch.rot90(sample, k=quarter_turns, dims=(1, 2))
        if bool(torch.randint(0, 2, (1,), generator=generator).item()):
            transformed = torch.flip(transformed, dims=(1,))
        if bool(torch.randint(0, 2, (1,), generator=generator).item()):
            transformed = torch.flip(transformed, dims=(2,))
        shift_x = int(torch.randint(0, 16, (1,), generator=generator).item())
        shift_y = int(torch.randint(0, 16, (1,), generator=generator).item())
        transformed = torch.roll(
            transformed, shifts=(shift_x, shift_y), dims=(1, 2)
        )
        augmented.append(transformed)
    return torch.stack(augmented).reshape_as(sequence)


def periodic_gradient_mse(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    pred = prediction.reshape(*prediction.shape[:-1], 16, 16, 2)
    target = truth.reshape(*truth.shape[:-1], 16, 16, 2)
    pred_x = torch.roll(pred, shifts=-1, dims=-3) - pred
    pred_y = torch.roll(pred, shifts=-1, dims=-2) - pred
    true_x = torch.roll(target, shifts=-1, dims=-3) - target
    true_y = torch.roll(target, shifts=-1, dims=-2) - target
    return 0.5 * (F.mse_loss(pred_x, true_x) + F.mse_loss(pred_y, true_y))


def joint_endpoint_metrics(
    model: DirectResidualConv,
    fields: torch.Tensor,
    *,
    horizons: Iterable[int],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, dict[str, float]], float]:
    """Exact forecast-packet selector: mean of four persistence-normalized cells."""

    selected = tuple(sorted({int(value) for value in horizons}))
    if not selected or selected[0] < 1 or selected[-1] >= fields.shape[1]:
        raise ValueError("Invalid checkpoint horizon")
    totals = {
        horizon: {
            "field_sse": 0.0,
            "field_count": 0,
            "final_sse": 0.0,
            "final_count": 0,
            "persistence_field_sse": 0.0,
            "persistence_final_sse": 0.0,
        }
        for horizon in selected
    }
    model.eval()
    with torch.inference_mode():
        for start in range(0, fields.shape[0], int(batch_size)):
            batch = fields[start : start + int(batch_size)].to(device)
            truth = batch[:, 1 : selected[-1] + 1]
            prediction = model.rollout(batch[:, 0], horizon=selected[-1])
            if not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError("Nonfinite validation rollout")
            for horizon in selected:
                difference = prediction[:, :horizon] - truth[:, :horizon]
                final_difference = difference[:, -1]
                persistence = batch[:, :1] - truth[:, :horizon]
                row = totals[horizon]
                row["field_sse"] += float(difference.square().sum())
                row["field_count"] += int(difference.numel())
                row["final_sse"] += float(final_difference.square().sum())
                row["final_count"] += int(final_difference.numel())
                row["persistence_field_sse"] += float(persistence.square().sum())
                row["persistence_final_sse"] += float(
                    persistence[:, -1].square().sum()
                )
    output: dict[str, dict[str, float]] = {}
    normalized: list[float] = []
    for horizon in selected:
        row = totals[horizon]
        field_mse = row["field_sse"] / max(1, row["field_count"])
        terminal_mse = row["final_sse"] / max(1, row["final_count"])
        persistence_field = row["persistence_field_sse"] / max(
            1, row["field_count"]
        )
        persistence_terminal = row["persistence_final_sse"] / max(
            1, row["final_count"]
        )
        output[str(horizon)] = {
            "field_mse": field_mse,
            "final_field_mse": terminal_mse,
            "persistence_field_mse": persistence_field,
            "persistence_final_field_mse": persistence_terminal,
        }
        normalized.extend(
            [
                field_mse / max(persistence_field, 1e-12),
                terminal_mse / max(persistence_terminal, 1e-12),
            ]
        )
    return output, float(sum(normalized) / len(normalized))
