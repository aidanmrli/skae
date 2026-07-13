"""Convolutional Koopman models for spatialized reaction-diffusion fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SpatialConvKoopmanConfig:
    grid_size: int
    channels: int = 2
    z_dim: int = 512
    hidden_channels: int = 64
    num_blocks: int = 3
    encoder_kind: str = "lista"
    lista_loops: int = 2
    lista_alpha: float = 1e-3
    decoder_kind: str = "upsample"
    k_init_scale: float = 1e-2
    dense_activation: str = "tanh"
    conv_activation: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "SpatialConvKoopmanConfig":
        valid = set(cls.__dataclass_fields__.keys())
        return cls(**{key: value for key, value in dict(values).items() if key in valid})


@dataclass(frozen=True)
class SpatialConvLossWeights:
    prediction: float = 1.0
    reconstruction: float = 0.25
    latent: float = 0.1
    sparsity: float = 0.0
    k_stability: float = 1e-4
    gradient: float = 0.05

    def to_dict(self) -> Dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "SpatialConvLossWeights":
        valid = set(cls.__dataclass_fields__.keys())
        return cls(**{key: float(value) for key, value in dict(values).items() if key in valid})


def _activation_module(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown convolutional activation '{name}'.")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, activation: str = "gelu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, min(8, out_channels // 8)), num_channels=out_channels),
            _activation_module(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, min(8, out_channels // 8)), num_channels=out_channels),
            _activation_module(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialConvKoopman(nn.Module):
    """Small convolutional Koopman autoencoder with optional LISTA shrinkage.

    The public API consumes and returns flattened fields with shape ``[B, 2*N*N]``
    so it can share the existing PDE evaluator and support-family machinery.
    """

    def __init__(self, cfg: SpatialConvKoopmanConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder_kind = cfg.encoder_kind.lower()
        if self.encoder_kind not in {"lista", "dense", "sparse_mlp"}:
            raise ValueError("encoder_kind must be one of: lista, dense, sparse_mlp")
        if cfg.decoder_kind.lower() != "upsample":
            raise ValueError("Only decoder_kind='upsample' is currently supported.")
        conv_activation = str(cfg.conv_activation).strip().lower()
        if not conv_activation:
            conv_activation = cfg.dense_activation if self.encoder_kind == "dense" else "gelu"

        channels = int(cfg.channels)
        width = int(cfg.hidden_channels)
        blocks: List[nn.Module] = []
        in_channels = channels
        current_grid = int(cfg.grid_size)
        for block_index in range(int(cfg.num_blocks)):
            out_channels = width * (2 ** block_index)
            blocks.append(ConvBlock(in_channels, out_channels, activation=conv_activation))
            if current_grid > 4:
                blocks.append(nn.AvgPool2d(kernel_size=2))
                current_grid //= 2
            in_channels = out_channels
        self.encoder_conv = nn.Sequential(*blocks)
        self.encoded_grid_size = current_grid
        self.encoded_channels = in_channels
        encoded_width = self.encoded_channels * self.encoded_grid_size * self.encoded_grid_size
        self.pre_code = nn.Linear(encoded_width, int(cfg.z_dim))
        self.lista_s = nn.Linear(int(cfg.z_dim), int(cfg.z_dim), bias=False)
        nn.init.zeros_(self.lista_s.weight)

        self.kmat = nn.Parameter(torch.eye(int(cfg.z_dim)) + float(cfg.k_init_scale) * torch.randn(int(cfg.z_dim), int(cfg.z_dim)))
        self.decoder_fc = nn.Linear(int(cfg.z_dim), encoded_width)
        decoder_blocks: List[nn.Module] = []
        dec_channels = self.encoded_channels
        dec_grid = self.encoded_grid_size
        while dec_grid < int(cfg.grid_size):
            next_channels = max(width, dec_channels // 2)
            decoder_blocks.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            decoder_blocks.append(ConvBlock(dec_channels, next_channels, activation=conv_activation))
            dec_channels = next_channels
            dec_grid *= 2
        decoder_blocks.append(nn.Conv2d(dec_channels, channels, kernel_size=3, padding=1))
        self.decoder_conv = nn.Sequential(*decoder_blocks)

    @property
    def observation_size(self) -> int:
        return int(self.cfg.channels) * int(self.cfg.grid_size) * int(self.cfg.grid_size)

    def _unflatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], int(self.cfg.grid_size), int(self.cfg.grid_size), int(self.cfg.channels)).permute(0, 3, 1, 2)

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self.observation_size:
            raise ValueError(f"Expected flat fields [B, {self.observation_size}], got {tuple(x.shape)}")
        h = self.encoder_conv(self._unflatten(x)).reshape(x.shape[0], -1)
        code = self.pre_code(h)
        if self.encoder_kind == "dense":
            return code
        if self.encoder_kind == "sparse_mlp":
            return F.softshrink(code, lambd=float(self.cfg.lista_alpha))
        z = torch.zeros_like(code)
        threshold = float(self.cfg.lista_alpha)
        for _ in range(max(1, int(self.cfg.lista_loops))):
            z = F.softshrink(code + self.lista_s(z), lambd=threshold)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z).reshape(
            z.shape[0],
            self.encoded_channels,
            self.encoded_grid_size,
            self.encoded_grid_size,
        )
        out = self.decoder_conv(h)
        if out.shape[-2:] != (int(self.cfg.grid_size), int(self.cfg.grid_size)):
            out = F.interpolate(out, size=(int(self.cfg.grid_size), int(self.cfg.grid_size)), mode="bilinear", align_corners=False)
        return self._flatten(out)

    def step_latent(self, z: torch.Tensor) -> torch.Tensor:
        return F.linear(z, self.kmat)

    def rollout_latent_discrete(self, z0: torch.Tensor, *, horizon: int) -> torch.Tensor:
        z = z0
        latents: List[torch.Tensor] = []
        for _ in range(int(horizon)):
            z = self.step_latent(z)
            latents.append(z)
        return torch.stack(latents, dim=1)

    def rollout_observation_discrete(self, x0: torch.Tensor, *, horizon: int) -> Tuple[torch.Tensor, torch.Tensor]:
        z0 = self.encode(x0)
        z_roll = self.rollout_latent_discrete(z0, horizon=horizon)
        pred = self.decode(z_roll.reshape(-1, z_roll.shape[-1])).reshape(x0.shape[0], int(horizon), -1)
        return z_roll, pred

    def rollout_observation_periodic_reencode(
        self,
        x0: torch.Tensor,
        *,
        horizon: int,
        period: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Roll out while periodically re-encoding decoded predictions.

        This never uses ground-truth future states. At each refresh boundary the
        model decodes its own current prediction and encodes that prediction to
        continue the latent rollout.
        """

        period = max(1, int(period))
        remaining = int(horizon)
        current = x0
        latent_chunks: List[torch.Tensor] = []
        pred_chunks: List[torch.Tensor] = []
        while remaining > 0:
            chunk = min(period, remaining)
            z_chunk, pred_chunk = self.rollout_observation_discrete(current, horizon=chunk)
            latent_chunks.append(z_chunk)
            pred_chunks.append(pred_chunk)
            current = pred_chunk[:, -1, :]
            remaining -= chunk
        return torch.cat(latent_chunks, dim=1), torch.cat(pred_chunks, dim=1)


def periodic_gradient_mse(pred_flat: torch.Tensor, true_flat: torch.Tensor, *, grid_size: int, channels: int) -> torch.Tensor:
    pred = pred_flat.reshape(*pred_flat.shape[:-1], int(grid_size), int(grid_size), int(channels))
    truth = true_flat.reshape(*true_flat.shape[:-1], int(grid_size), int(grid_size), int(channels))
    pred_gx = torch.roll(pred, shifts=-1, dims=-3) - pred
    pred_gy = torch.roll(pred, shifts=-1, dims=-2) - pred
    true_gx = torch.roll(truth, shifts=-1, dims=-3) - truth
    true_gy = torch.roll(truth, shifts=-1, dims=-2) - truth
    return 0.5 * (F.mse_loss(pred_gx, true_gx) + F.mse_loss(pred_gy, true_gy))
