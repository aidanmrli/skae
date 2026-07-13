"""Small benchmark-local models used when the main SKAE factory is insufficient."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
    *,
    activation: str = "tanh",
) -> nn.Sequential:
    modules: List[nn.Module] = []
    current = int(input_dim)
    act = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[activation]
    for _ in range(int(num_layers)):
        modules.append(nn.Linear(current, int(hidden_dim)))
        modules.append(act())
        current = int(hidden_dim)
    modules.append(nn.Linear(current, int(output_dim)))
    return nn.Sequential(*modules)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def koopman_diagnostics(kmat: torch.Tensor) -> Dict[str, object]:
    k = kmat.detach().float().cpu().numpy()
    max_abs = float(np.max(np.abs(k))) if k.size else 0.0
    eigvals = np.linalg.eigvals(k) if k.size else np.asarray([], dtype=np.complex64)
    out: Dict[str, object] = {
        "k_l1": float(np.sum(np.abs(k))),
        "k_l0_exact": int(np.count_nonzero(k == 0.0)),
        "k_max_abs": max_abs,
        "spectral_radius": float(np.max(np.abs(eigvals))) if eigvals.size else float("nan"),
        "eig_abs": np.abs(eigvals).astype(float).tolist(),
    }
    for rel, tag in ((1e-4, "1e4"), (1e-3, "1e3"), (1e-2, "1e2")):
        threshold = rel * max_abs
        density = float(np.mean(np.abs(k) > threshold)) if k.size else 0.0
        out[f"effective_density_{tag}"] = density
        out[f"avg_active_per_coord_{tag}"] = float(density * k.shape[1]) if k.ndim == 2 else 0.0
    return out


class ControlledKoopmanAE(nn.Module):
    """Controlled latent Koopman model implementing z[t+1] = z[t] K + B u[t]."""

    def __init__(
        self,
        *,
        history_dim: int,
        output_dim: int,
        action_dim: int,
        z_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = "tanh",
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.action_dim = int(action_dim)
        self.z_dim = int(z_dim)
        self.encoder = make_mlp(history_dim, z_dim, hidden_dim, num_layers, activation=activation)
        self.decoder = make_mlp(z_dim, output_dim, hidden_dim, num_layers, activation=activation)
        self.kmat = nn.Parameter(torch.eye(z_dim) + 0.01 * torch.randn(z_dim, z_dim))
        self.action_linear = nn.Linear(action_dim, z_dim, bias=False)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        return self.encoder(history)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def step_latent(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return z @ self.kmat + self.action_linear(action)

    def rollout(self, history: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(history)
        latents: List[torch.Tensor] = []
        preds: List[torch.Tensor] = []
        for t in range(actions.shape[1]):
            z = self.step_latent(z, actions[:, t])
            latents.append(z)
            preds.append(self.decode(z))
        return torch.stack(latents, dim=1), torch.stack(preds, dim=1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(max(1, min(8, out_channels // 8)), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(max(1, min(8, out_channels // 8)), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class ConvKoopmanConfig:
    grid_size: int
    channels: int
    history: int
    z_dim: int
    hidden_channels: int = 16
    num_blocks: int = 2


class ConvKoopmanAE(nn.Module):
    """Compact convolutional Koopman model for PDE fields."""

    def __init__(self, cfg: ConvKoopmanConfig):
        super().__init__()
        self.cfg = cfg
        in_channels = int(cfg.channels) * int(cfg.history)
        width = int(cfg.hidden_channels)
        grid = int(cfg.grid_size)
        modules: List[nn.Module] = []
        current_channels = in_channels
        current_grid = grid
        for block_idx in range(int(cfg.num_blocks)):
            out_channels = width * (2**block_idx)
            modules.append(ConvBlock(current_channels, out_channels))
            if current_grid > 8:
                modules.append(nn.AvgPool2d(2))
                current_grid //= 2
            current_channels = out_channels
        self.encoder_conv = nn.Sequential(*modules)
        self.encoded_channels = current_channels
        self.encoded_grid = current_grid
        encoded_dim = current_channels * current_grid * current_grid
        self.to_z = nn.Linear(encoded_dim, int(cfg.z_dim))
        self.kmat = nn.Parameter(torch.eye(int(cfg.z_dim)) + 0.01 * torch.randn(int(cfg.z_dim), int(cfg.z_dim)))
        self.from_z = nn.Linear(int(cfg.z_dim), encoded_dim)
        dec_modules: List[nn.Module] = []
        dec_channels = current_channels
        dec_grid = current_grid
        while dec_grid < grid:
            next_channels = max(width, dec_channels // 2)
            dec_modules.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            dec_modules.append(ConvBlock(dec_channels, next_channels))
            dec_channels = next_channels
            dec_grid *= 2
        dec_modules.append(nn.Conv2d(dec_channels, int(cfg.channels), 3, padding=1))
        self.decoder_conv = nn.Sequential(*dec_modules)

    @property
    def frame_dim(self) -> int:
        return int(self.cfg.channels) * int(self.cfg.grid_size) * int(self.cfg.grid_size)

    def _context_to_image(self, context: torch.Tensor) -> torch.Tensor:
        b = context.shape[0]
        return context.reshape(
            b,
            int(self.cfg.history),
            int(self.cfg.grid_size),
            int(self.cfg.grid_size),
            int(self.cfg.channels),
        ).permute(0, 1, 4, 2, 3).reshape(
            b,
            int(self.cfg.history) * int(self.cfg.channels),
            int(self.cfg.grid_size),
            int(self.cfg.grid_size),
        )

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(self._context_to_image(context)).reshape(context.shape[0], -1)
        return self.to_z(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_z(z).reshape(z.shape[0], self.encoded_channels, self.encoded_grid, self.encoded_grid)
        out = self.decoder_conv(h)
        if out.shape[-2:] != (int(self.cfg.grid_size), int(self.cfg.grid_size)):
            out = F.interpolate(out, size=(int(self.cfg.grid_size), int(self.cfg.grid_size)), mode="bilinear", align_corners=False)
        return out.permute(0, 2, 3, 1).reshape(z.shape[0], -1)

    def rollout(self, context: torch.Tensor, horizon: int) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode_context(context)
        zs: List[torch.Tensor] = []
        preds: List[torch.Tensor] = []
        for _ in range(int(horizon)):
            z = z @ self.kmat
            zs.append(z)
            preds.append(self.decode(z))
        return torch.stack(zs, dim=1), torch.stack(preds, dim=1)


class ConvARBaseline(nn.Module):
    """Small recursive convolutional baseline for PDE smoke comparisons."""

    def __init__(self, *, grid_size: int, channels: int, history: int, hidden_channels: int = 32):
        super().__init__()
        self.grid_size = int(grid_size)
        self.channels = int(channels)
        self.history = int(history)
        in_channels = self.channels * self.history
        self.net = nn.Sequential(
            ConvBlock(in_channels, hidden_channels),
            ConvBlock(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, self.channels, 3, padding=1),
        )

    @property
    def frame_dim(self) -> int:
        return self.channels * self.grid_size * self.grid_size

    def _context_to_image(self, context: torch.Tensor) -> torch.Tensor:
        return context.reshape(
            context.shape[0], self.history, self.grid_size, self.grid_size, self.channels
        ).permute(0, 1, 4, 2, 3).reshape(context.shape[0], self.history * self.channels, self.grid_size, self.grid_size)

    def one_step(self, context: torch.Tensor) -> torch.Tensor:
        out = self.net(self._context_to_image(context))
        return out.permute(0, 2, 3, 1).reshape(context.shape[0], -1)

    def rollout(self, context: torch.Tensor, horizon: int) -> torch.Tensor:
        frames = context.reshape(context.shape[0], self.history, self.frame_dim)
        preds: List[torch.Tensor] = []
        for _ in range(int(horizon)):
            pred = self.one_step(frames.reshape(context.shape[0], -1))
            preds.append(pred)
            frames = torch.cat([frames[:, 1:], pred[:, None]], dim=1)
        return torch.stack(preds, dim=1)
