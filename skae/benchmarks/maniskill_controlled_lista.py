"""Minimal controlled LISTA/SKAE path for state-only ManiSkill insertion."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skae.benchmarks.maniskill_insertion_dataset import CompactManiSkillDataset


@dataclass(frozen=True)
class ControlledLISTAConfig:
    """Small model config for the one-seed ManiSkill insertion path."""

    obs_dim: int
    action_dim: int
    z_dim: int = 128
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    encoder_kind: str = "lista"
    lista_loops: int = 2
    lista_alpha: float = 0.05
    action_hidden_dim: int = 64
    decoder_hidden_dim: int = 256
    activation: str = "auto"


@dataclass(frozen=True)
class LossWeights:
    prediction: float = 1.0
    reconstruction: float = 0.1
    latent: float = 0.1
    sparsity: float = 1e-3
    k_stability: float = 1e-4


@dataclass(frozen=True)
class NormalizationStats:
    obs_mean: np.ndarray
    obs_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def to_jsonable(self) -> Dict[str, List[float]]:
        return {
            "obs_mean": self.obs_mean.astype(float).tolist(),
            "obs_std": self.obs_std.astype(float).tolist(),
            "action_mean": self.action_mean.astype(float).tolist(),
            "action_std": self.action_std.astype(float).tolist(),
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Sequence[float]]) -> "NormalizationStats":
        return cls(
            obs_mean=np.asarray(values["obs_mean"], dtype=np.float32),
            obs_std=np.asarray(values["obs_std"], dtype=np.float32),
            action_mean=np.asarray(values["action_mean"], dtype=np.float32),
            action_std=np.asarray(values["action_std"], dtype=np.float32),
        )


VALID_ACTIVATIONS = {"relu", "tanh", "gelu"}


def resolve_controlled_activation(encoder_kind: str, activation: object = "auto") -> str:
    """Resolve model-level activation defaults.

    Dense model configs resolve to tanh by default. The training CLI uses
    ``resolve_controlled_training_activation`` to keep nonzero-sparsity dense
    encoders on the sparse/LISTA candidate default unless overridden.
    """

    normalized = str(activation).strip().lower()
    encoder = str(encoder_kind).strip().lower()
    if normalized in {"", "auto", "default"}:
        return "tanh" if encoder == "dense" else "relu"
    if normalized not in VALID_ACTIVATIONS:
        raise ValueError("activation must be one of: auto, relu, tanh, gelu")
    return normalized


def is_dense_no_sparsity_baseline(encoder_kind: str, sparsity_weight: float) -> bool:
    return str(encoder_kind).strip().lower() == "dense" and float(sparsity_weight) == 0.0


def resolve_controlled_training_activation(
    encoder_kind: str,
    activation: object,
    sparsity_weight: float,
) -> str:
    """Resolve CLI training activation with the dense-baseline tanh rule."""

    normalized = str(activation).strip().lower()
    is_dense_baseline = is_dense_no_sparsity_baseline(encoder_kind, sparsity_weight)
    if normalized in {"", "auto", "default"}:
        return "tanh" if is_dense_baseline else "relu"
    resolved = resolve_controlled_activation(encoder_kind, normalized)
    if is_dense_baseline and resolved != "tanh":
        raise ValueError(
            "Dense no-sparsity baselines must use tanh activations. "
            "Do not launch or report dense-ReLU as the dense baseline."
        )
    return resolved


def make_activation(name: str) -> nn.Module:
    """Return an activation module for controlled insertion MLPs."""

    normalized = str(name).lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError("activation must be one of: relu, tanh, gelu")


class MLP(nn.Module):
    """Small MLP used by the controlled insertion model."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        *,
        activation: str = "relu",
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = int(input_dim)
        for _ in range(int(num_hidden_layers)):
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(make_activation(activation))
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, int(output_dim)))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ControlledLISTAKoopman(nn.Module):
    """Controlled Koopman autoencoder with a LISTA-style sparse encoder.

    The dynamics are ``z_{t+1} = K z_t + B phi(a_t)``. Support masks are
    extracted only from ``z_t``.
    """

    def __init__(self, cfg: ControlledLISTAConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder_kind = cfg.encoder_kind.lower()
        if self.encoder_kind not in {"lista", "dense"}:
            raise ValueError("encoder_kind must be 'lista' or 'dense'")
        self.activation = resolve_controlled_activation(self.encoder_kind, cfg.activation)

        self.pre_code = MLP(
            cfg.obs_dim,
            cfg.z_dim,
            cfg.hidden_dim,
            cfg.num_hidden_layers,
            activation=self.activation,
        )
        if self.encoder_kind == "lista":
            self.lista_s = nn.Linear(cfg.z_dim, cfg.z_dim, bias=False)
            nn.init.zeros_(self.lista_s.weight)
        else:
            self.lista_s = None

        self.action_features = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.action_hidden_dim),
            make_activation(self.activation),
            nn.Linear(cfg.action_hidden_dim, cfg.z_dim),
        )
        self.kmat = nn.Parameter(torch.eye(cfg.z_dim) + 0.01 * torch.randn(cfg.z_dim, cfg.z_dim))
        self.decoder = MLP(
            cfg.z_dim,
            cfg.obs_dim,
            cfg.decoder_hidden_dim,
            cfg.num_hidden_layers,
            activation=self.activation,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        code = self.pre_code(x)
        if self.encoder_kind == "dense":
            return code

        z = torch.zeros_like(code)
        threshold = float(self.cfg.lista_alpha)
        for _ in range(max(1, int(self.cfg.lista_loops))):
            residual = code if self.lista_s is None else code + self.lista_s(z)
            z = F.softshrink(residual, lambd=threshold)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def step_latent(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return F.linear(z, self.kmat) + self.action_features(action)

    def rollout_latent(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Roll out latents for ``actions`` with shape ``[batch, horizon, action_dim]``."""

        z = z0
        latents: List[torch.Tensor] = []
        for time_index in range(actions.shape[1]):
            z = self.step_latent(z, actions[:, time_index])
            latents.append(z)
        return torch.stack(latents, dim=1)

    def rollout_observations(self, x0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        z0 = self.encode(x0)
        z_roll = self.rollout_latent(z0, actions)
        flat = z_roll.reshape(-1, z_roll.shape[-1])
        return self.decode(flat).reshape(z_roll.shape[0], z_roll.shape[1], -1)

    def rollout_observations_periodic_reencode(
        self,
        x0: torch.Tensor,
        actions: torch.Tensor,
        *,
        period: int,
    ) -> torch.Tensor:
        """Roll out actions and periodically re-encode decoded predictions."""

        period = int(period)
        if period <= 0:
            raise ValueError("period must be a positive integer")

        z = self.encode(x0)
        predictions: List[torch.Tensor] = []
        for time_index in range(actions.shape[1]):
            predicted_z = self.step_latent(z, actions[:, time_index])
            predicted_x = self.decode(predicted_z)
            predictions.append(predicted_x)
            if (time_index + 1) % period == 0:
                z = self.encode(predicted_x)
            else:
                z = predicted_z
        return torch.stack(predictions, dim=1)

    def support_mask(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        return self.encode(x).abs() > float(threshold)


class WindowSampler:
    """Random valid fixed-length windows from trajectory-level arrays."""

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        valid: np.ndarray,
        episode_indices: Sequence[int],
        *,
        sequence_length: int,
        rng: np.random.Generator,
    ):
        self.observations = observations
        self.actions = actions
        self.valid = valid
        self.sequence_length = int(sequence_length)
        self.rng = rng
        self.windows: List[Tuple[int, int]] = []
        for episode_index in episode_indices:
            episode_index = int(episode_index)
            valid_steps = np.asarray(valid[episode_index], dtype=bool)
            max_start = len(valid_steps) - self.sequence_length
            for start in range(max_start + 1):
                if bool(np.all(valid_steps[start : start + self.sequence_length])):
                    self.windows.append((episode_index, start))
        if not self.windows:
            raise ValueError(
                f"No valid windows of length {self.sequence_length} found for selected episodes"
            )

    def sample(self, batch_size: int, *, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        chosen = self.rng.integers(0, len(self.windows), size=int(batch_size))
        x_batch = []
        a_batch = []
        for window_index in chosen.tolist():
            episode_index, start = self.windows[window_index]
            stop = start + self.sequence_length
            x_batch.append(self.observations[episode_index, start : stop + 1])
            a_batch.append(self.actions[episode_index, start:stop])
        x = torch.as_tensor(np.stack(x_batch, axis=0), dtype=torch.float32, device=device)
        a = torch.as_tensor(np.stack(a_batch, axis=0), dtype=torch.float32, device=device)
        return x, a


def compute_normalization(dataset: CompactManiSkillDataset, train_indices: Sequence[int]) -> NormalizationStats:
    """Compute train-only state/action normalization statistics."""

    train_indices = np.asarray(train_indices, dtype=np.int64)
    if train_indices.size == 0:
        raise ValueError("At least one train episode is required")

    valid = dataset.valid[train_indices]
    state_valid = np.zeros((len(train_indices), dataset.max_transitions + 1), dtype=bool)
    state_valid[:, :-1] |= valid
    state_valid[:, 1:] |= valid

    train_obs = dataset.observations[train_indices][state_valid]
    train_actions = dataset.actions[train_indices][valid]
    obs_mean, obs_std = _safe_mean_std(train_obs, dataset.obs_dim)
    action_mean, action_std = _safe_mean_std(train_actions, dataset.action_dim)
    return NormalizationStats(
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
    )


def normalize_observations(observations: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return ((observations - stats.obs_mean) / stats.obs_std).astype(np.float32)


def normalize_actions(actions: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return ((actions - stats.action_mean) / stats.action_std).astype(np.float32)


def denormalize_observations(observations: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (observations * stats.obs_std + stats.obs_mean).astype(np.float32)


def train_step(
    model: ControlledLISTAKoopman,
    optimizer: torch.optim.Optimizer,
    x_seq: torch.Tensor,
    action_seq: torch.Tensor,
    weights: LossWeights,
) -> Dict[str, float]:
    """One optimization step on normalized sequence windows."""

    model.train()
    optimizer.zero_grad(set_to_none=True)

    batch_size, seq_plus_one, obs_dim = x_seq.shape
    horizon = seq_plus_one - 1
    z_all = model.encode(x_seq.reshape(batch_size * seq_plus_one, obs_dim)).reshape(
        batch_size,
        seq_plus_one,
        -1,
    )
    z_roll = model.rollout_latent(z_all[:, 0], action_seq)
    pred = model.decode(z_roll.reshape(batch_size * horizon, -1)).reshape(batch_size, horizon, obs_dim)
    recon = model.decode(z_all.reshape(batch_size * seq_plus_one, -1)).reshape(
        batch_size,
        seq_plus_one,
        obs_dim,
    )

    prediction_loss = F.mse_loss(pred, x_seq[:, 1:])
    reconstruction_loss = F.mse_loss(recon, x_seq)
    latent_loss = F.mse_loss(z_roll, z_all[:, 1:].detach())
    sparsity_loss = z_all.abs().mean()
    spectral_norm_proxy = torch.linalg.matrix_norm(model.kmat, ord="fro") / math.sqrt(model.kmat.shape[0])
    k_stability_loss = F.relu(spectral_norm_proxy - 1.25).square()

    loss = (
        float(weights.prediction) * prediction_loss
        + float(weights.reconstruction) * reconstruction_loss
        + float(weights.latent) * latent_loss
        + float(weights.sparsity) * sparsity_loss
        + float(weights.k_stability) * k_stability_loss
    )
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        sparsity_ratio = (z_all.abs() <= 1e-4).float().mean()
    return {
        "loss": float(loss.detach().cpu()),
        "prediction_loss": float(prediction_loss.detach().cpu()),
        "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
        "latent_loss": float(latent_loss.detach().cpu()),
        "sparsity_loss": float(sparsity_loss.detach().cpu()),
        "k_stability_loss": float(k_stability_loss.detach().cpu()),
        "sparsity_ratio_1e-4": float(sparsity_ratio.detach().cpu()),
    }


@torch.no_grad()
def validation_rollout_mse(
    model: ControlledLISTAKoopman,
    sampler: WindowSampler,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    x_seq, action_seq = sampler.sample(batch_size, device=device)
    pred = model.rollout_observations(x_seq[:, 0], action_seq)
    return float(F.mse_loss(pred, x_seq[:, 1:]).detach().cpu())


def save_checkpoint(
    path: str | Path,
    *,
    model: ControlledLISTAKoopman,
    optimizer: torch.optim.Optimizer,
    model_config: ControlledLISTAConfig,
    loss_weights: LossWeights,
    normalization: NormalizationStats,
    step: int,
    metrics: Mapping[str, float],
    metadata: Mapping[str, object],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "loss_weights": asdict(loss_weights),
            "normalization": normalization.to_jsonable(),
            "step": int(step),
            "metrics": dict(metrics),
            "metadata": dict(metadata),
        },
        output_path,
    )


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> Tuple[ControlledLISTAKoopman, NormalizationStats, Dict[str, object]]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    model_config = controlled_lista_config_from_mapping(checkpoint["model_config"])
    model = ControlledLISTAKoopman(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stats = NormalizationStats.from_mapping(checkpoint["normalization"])
    return model, stats, checkpoint


def controlled_lista_config_from_mapping(values: Mapping[str, object]) -> ControlledLISTAConfig:
    valid_names = {field.name for field in fields(ControlledLISTAConfig)}
    filtered = {key: value for key, value in dict(values).items() if key in valid_names}
    if "activation" not in filtered:
        # Legacy controlled-ManiSkill checkpoints predate explicit activations
        # and used ReLU hidden blocks. Preserve their behavior when loading.
        filtered["activation"] = "relu"
    return ControlledLISTAConfig(**filtered)


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _safe_mean_std(values: np.ndarray, width: int) -> Tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.zeros((width,), dtype=np.float32), np.ones((width,), dtype=np.float32)
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std
