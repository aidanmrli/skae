"""Action-conditioned sparse Koopman world-model utilities.

This module is intentionally separate from the autonomous SKAE trainer.  It
supports state-observation control datasets, additive and bilinear Koopman
latent transitions, and a residual MLP latent transition baseline.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DMC_TASKS: Mapping[str, Tuple[str, str]] = {
    "cartpole_swingup": ("cartpole", "swingup"),
    "finger_spin": ("finger", "spin"),
    "cheetah_run": ("cheetah", "run"),
    "walker_walk": ("walker", "walk"),
}

VALID_TRANSITIONS = {"additive", "bilinear", "mlp"}
VALID_ACTIVATIONS = {"relu", "tanh", "gelu"}


@dataclass(frozen=True)
class ControlTrajectoryDataset:
    """Compact offline control dataset.

    Observations have shape ``[episode, time + 1, obs_dim]``.  Actions,
    rewards, continuations, and valid masks have shape ``[episode, time, ...]``.
    Splits are stored per episode.  Labels or task internals are not required
    for training.
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    continuations: np.ndarray
    valid: np.ndarray
    split: np.ndarray
    episode_ids: np.ndarray
    feature_names: Tuple[str, ...]
    action_names: Tuple[str, ...]
    metadata: Dict[str, Any]

    @property
    def num_episodes(self) -> int:
        return int(self.observations.shape[0])

    @property
    def max_transitions(self) -> int:
        return int(self.actions.shape[1])

    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[-1])

    def indices_for_split(self, split_name: str) -> np.ndarray:
        return np.nonzero(self.split.astype(str) == str(split_name))[0].astype(np.int64)


@dataclass(frozen=True)
class NormalizationStats:
    obs_mean: np.ndarray
    obs_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    reward_mean: float
    reward_std: float

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "obs_mean": self.obs_mean.astype(float).tolist(),
            "obs_std": self.obs_std.astype(float).tolist(),
            "action_mean": self.action_mean.astype(float).tolist(),
            "action_std": self.action_std.astype(float).tolist(),
            "reward_mean": float(self.reward_mean),
            "reward_std": float(self.reward_std),
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "NormalizationStats":
        return cls(
            obs_mean=np.asarray(values["obs_mean"], dtype=np.float32),
            obs_std=np.asarray(values["obs_std"], dtype=np.float32),
            action_mean=np.asarray(values["action_mean"], dtype=np.float32),
            action_std=np.asarray(values["action_std"], dtype=np.float32),
            reward_mean=float(values["reward_mean"]),
            reward_std=float(values["reward_std"]),
        )


@dataclass(frozen=True)
class ControlWorldModelConfig:
    obs_dim: int
    action_dim: int
    z_dim: int = 128
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    transition_kind: str = "additive"
    sparse_matrices: bool = True
    activation: str = "auto"
    decoder_hidden_dim: int = 256
    head_hidden_dim: int = 128
    mlp_predict_delta: bool = True


@dataclass(frozen=True)
class LossWeights:
    prediction: float = 1.0
    reconstruction: float = 0.1
    latent: float = 0.1
    reward: float = 1.0
    continuation: float = 0.1
    k_sparsity: float = 1e-4
    k_stability: float = 1e-4


def make_activation(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError("activation must be one of: relu, tanh, gelu")


def resolve_world_model_activation(
    transition_kind: str,
    sparse_matrices: bool,
    activation: object = "auto",
) -> str:
    """Resolve activation defaults while enforcing the dense-baseline rule."""

    normalized = str(activation).strip().lower()
    transition = str(transition_kind).strip().lower()
    if transition not in VALID_TRANSITIONS:
        raise ValueError(f"transition_kind must be one of {sorted(VALID_TRANSITIONS)}")

    dense_no_sparsity = (transition == "mlp") or (not bool(sparse_matrices))
    if normalized in {"", "auto", "default"}:
        return "tanh" if dense_no_sparsity else "relu"
    if normalized not in VALID_ACTIVATIONS:
        raise ValueError("activation must be one of: auto, relu, tanh, gelu")
    if dense_no_sparsity and normalized != "tanh":
        raise ValueError(
            "Dense no-sparsity world-model baselines must use tanh activations. "
            "A dense ReLU run should be labeled as a ReLU ablation, not a baseline."
        )
    return normalized


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        *,
        activation: str,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = int(input_dim)
        for _ in range(int(num_hidden_layers)):
            layers.append(nn.Linear(current_dim, int(hidden_dim)))
            layers.append(make_activation(activation))
            current_dim = int(hidden_dim)
        layers.append(nn.Linear(current_dim, int(output_dim)))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ControlWorldModel(nn.Module):
    """Latent world model with action-conditioned dynamics.

    ``transition_kind='additive'`` implements ``z' = K z + B a``.
    ``transition_kind='bilinear'`` implements
    ``z' = K0 z + sum_j a_j K_j z + B a``.
    ``transition_kind='mlp'`` is a residual MLP transition baseline.
    """

    def __init__(self, cfg: ControlWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.transition_kind = str(cfg.transition_kind).lower()
        if self.transition_kind not in VALID_TRANSITIONS:
            raise ValueError(f"transition_kind must be one of {sorted(VALID_TRANSITIONS)}")
        self.activation = resolve_world_model_activation(
            self.transition_kind,
            cfg.sparse_matrices,
            cfg.activation,
        )

        self.encoder = MLP(
            cfg.obs_dim,
            cfg.z_dim,
            cfg.hidden_dim,
            cfg.num_hidden_layers,
            activation=self.activation,
        )
        self.decoder = MLP(
            cfg.z_dim,
            cfg.obs_dim,
            cfg.decoder_hidden_dim,
            cfg.num_hidden_layers,
            activation=self.activation,
        )

        if self.transition_kind in {"additive", "bilinear"}:
            self.k0 = nn.Parameter(torch.eye(cfg.z_dim) + 0.01 * torch.randn(cfg.z_dim, cfg.z_dim))
            self.action_linear = nn.Linear(cfg.action_dim, cfg.z_dim, bias=False)
        else:
            self.k0 = None
            self.action_linear = None

        if self.transition_kind == "bilinear":
            self.k_action = nn.Parameter(
                0.001 * torch.randn(cfg.action_dim, cfg.z_dim, cfg.z_dim)
            )
        else:
            self.k_action = None

        if self.transition_kind == "mlp":
            self.latent_mlp = MLP(
                cfg.z_dim + cfg.action_dim,
                cfg.z_dim,
                cfg.hidden_dim,
                cfg.num_hidden_layers,
                activation=self.activation,
            )
        else:
            self.latent_mlp = None

        head_input_dim = cfg.z_dim + cfg.action_dim
        self.reward_head = MLP(
            head_input_dim,
            1,
            cfg.head_hidden_dim,
            1,
            activation=self.activation,
        )
        self.continuation_head = MLP(
            head_input_dim,
            1,
            cfg.head_hidden_dim,
            1,
            activation=self.activation,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def step_latent(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.transition_kind == "additive":
            assert self.k0 is not None and self.action_linear is not None
            return F.linear(z, self.k0) + self.action_linear(action)
        if self.transition_kind == "bilinear":
            assert self.k0 is not None and self.k_action is not None and self.action_linear is not None
            base = F.linear(z, self.k0) + self.action_linear(action)
            bilinear = torch.einsum("ba,aoi,bi->bo", action, self.k_action, z)
            return base + bilinear
        assert self.latent_mlp is not None
        delta_or_next = self.latent_mlp(torch.cat([z, action], dim=-1))
        return z + delta_or_next if self.cfg.mlp_predict_delta else delta_or_next

    def predict_reward_and_continuation(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat([z, action], dim=-1)
        reward = self.reward_head(features).squeeze(-1)
        continuation_logits = self.continuation_head(features).squeeze(-1)
        return reward, continuation_logits

    def rollout(
        self,
        z0: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Roll out latents, rewards, and continuation logits.

        Args:
            z0: Initial latent state, shape ``[batch, z_dim]``.
            actions: Action sequence, shape ``[batch, horizon, action_dim]``.
        """

        z = z0
        latents: List[torch.Tensor] = []
        rewards: List[torch.Tensor] = []
        continuation_logits: List[torch.Tensor] = []
        for time_index in range(actions.shape[1]):
            action = actions[:, time_index]
            reward, logits = self.predict_reward_and_continuation(z, action)
            z = self.step_latent(z, action)
            rewards.append(reward)
            continuation_logits.append(logits)
            latents.append(z)
        return (
            torch.stack(latents, dim=1),
            torch.stack(rewards, dim=1),
            torch.stack(continuation_logits, dim=1),
        )

    def rollout_observations(self, x0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        z0 = self.encode(x0)
        z_roll, _, _ = self.rollout(z0, actions)
        flat = z_roll.reshape(-1, z_roll.shape[-1])
        return self.decode(flat).reshape(z_roll.shape[0], z_roll.shape[1], -1)

    def transition_matrix_parameters(self) -> Dict[str, torch.Tensor]:
        matrices: Dict[str, torch.Tensor] = {}
        if self.k0 is not None:
            matrices["K0"] = self.k0
        if self.k_action is not None:
            for action_index in range(self.k_action.shape[0]):
                matrices[f"K_action_{action_index}"] = self.k_action[action_index]
        return matrices

    def sparsified_matrix_parameters(self) -> Dict[str, torch.Tensor]:
        if not bool(self.cfg.sparse_matrices):
            return {}
        return self.transition_matrix_parameters()

    def matrix_density(self, threshold: float = 1e-4) -> Dict[str, float]:
        densities: Dict[str, float] = {}
        for name, matrix in self.transition_matrix_parameters().items():
            densities[f"density/{name}"] = float((matrix.detach().abs() > threshold).float().mean().cpu())
        if self.action_linear is not None:
            weight = self.action_linear.weight.detach()
            densities["density_unpenalized/B"] = float((weight.abs() > threshold).float().mean().cpu())
        return densities


class WindowSampler:
    """Random fixed-length windows from compact control trajectories."""

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        continuations: np.ndarray,
        valid: np.ndarray,
        episode_indices: Sequence[int],
        *,
        sequence_length: int,
        rng: np.random.Generator,
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.continuations = continuations
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

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen = self.rng.integers(0, len(self.windows), size=int(batch_size))
        x_batch = []
        a_batch = []
        r_batch = []
        c_batch = []
        for window_index in chosen.tolist():
            episode_index, start = self.windows[window_index]
            stop = start + self.sequence_length
            x_batch.append(self.observations[episode_index, start : stop + 1])
            a_batch.append(self.actions[episode_index, start:stop])
            r_batch.append(self.rewards[episode_index, start:stop])
            c_batch.append(self.continuations[episode_index, start:stop])
        x = torch.as_tensor(np.stack(x_batch, axis=0), dtype=torch.float32, device=device)
        a = torch.as_tensor(np.stack(a_batch, axis=0), dtype=torch.float32, device=device)
        r = torch.as_tensor(np.stack(r_batch, axis=0), dtype=torch.float32, device=device)
        c = torch.as_tensor(np.stack(c_batch, axis=0), dtype=torch.float32, device=device)
        return x, a, r, c


def save_control_dataset(path: str | Path, dataset: ControlTrajectoryDataset) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=np.asarray(dataset.observations, dtype=np.float32),
        actions=np.asarray(dataset.actions, dtype=np.float32),
        rewards=np.asarray(dataset.rewards, dtype=np.float32),
        continuations=np.asarray(dataset.continuations, dtype=np.float32),
        valid=np.asarray(dataset.valid, dtype=bool),
        split=np.asarray(dataset.split, dtype=str),
        episode_ids=np.asarray(dataset.episode_ids, dtype=np.int64),
        feature_names=np.asarray(dataset.feature_names, dtype=str),
        action_names=np.asarray(dataset.action_names, dtype=str),
        metadata_json=np.asarray(json.dumps(dataset.metadata, sort_keys=True), dtype=str),
    )


def load_control_dataset(path: str | Path) -> ControlTrajectoryDataset:
    dataset_path = Path(path)
    with np.load(dataset_path, allow_pickle=False) as data:
        return ControlTrajectoryDataset(
            observations=np.asarray(data["observations"], dtype=np.float32),
            actions=np.asarray(data["actions"], dtype=np.float32),
            rewards=np.asarray(data["rewards"], dtype=np.float32),
            continuations=np.asarray(data["continuations"], dtype=np.float32),
            valid=np.asarray(data["valid"], dtype=bool),
            split=np.asarray(data["split"]).astype(str),
            episode_ids=np.asarray(data["episode_ids"], dtype=np.int64),
            feature_names=tuple(np.asarray(data["feature_names"]).astype(str).tolist()),
            action_names=tuple(np.asarray(data["action_names"]).astype(str).tolist()),
            metadata=json.loads(_scalar_string(data["metadata_json"])),
        )


def generate_dm_control_dataset(
    *,
    task: str,
    output_path: str | Path,
    num_episodes: int,
    episode_length: int,
    seed: int,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    policy: str = "random",
) -> ControlTrajectoryDataset:
    """Generate a state-observation DeepMind Control Suite dataset.

    The dependency is optional.  Install the ``control`` extra or otherwise
    provide ``dm_control`` in the environment before running this function.
    """

    try:
        from dm_control import suite
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "dm_control is required to generate DeepMind Control Suite datasets. "
            "Install the optional control extra before running this experiment."
        ) from exc

    task_key = str(task).strip().lower()
    if task_key not in DMC_TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(DMC_TASKS)}")
    if policy != "random":
        raise ValueError("Only policy='random' is currently implemented for dataset generation")

    domain_name, task_name = DMC_TASKS[task_key]
    rng = np.random.default_rng(int(seed))
    observations: Optional[np.ndarray] = None
    actions: Optional[np.ndarray] = None
    rewards = np.zeros((num_episodes, episode_length), dtype=np.float32)
    continuations = np.zeros((num_episodes, episode_length), dtype=np.float32)
    valid = np.zeros((num_episodes, episode_length), dtype=bool)
    feature_names: Optional[Tuple[str, ...]] = None
    action_names: Optional[Tuple[str, ...]] = None

    for episode_index in range(int(num_episodes)):
        env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs={"random": int(seed) + episode_index},
        )
        action_spec = env.action_spec()
        action_min = np.asarray(action_spec.minimum, dtype=np.float32)
        action_max = np.asarray(action_spec.maximum, dtype=np.float32)
        if action_names is None:
            action_names = tuple(f"action/{index}" for index in range(action_min.size))

        time_step = env.reset()
        obs0, names = flatten_dm_control_observation(time_step.observation)
        if observations is None:
            feature_names = names
            observations = np.zeros((num_episodes, episode_length + 1, obs0.size), dtype=np.float32)
            actions = np.zeros((num_episodes, episode_length, action_min.size), dtype=np.float32)
        assert observations is not None and actions is not None and feature_names is not None
        if names != feature_names:
            raise ValueError(f"Observation feature names changed for task {task_key}")
        observations[episode_index, 0] = obs0

        for step_index in range(int(episode_length)):
            action = rng.uniform(action_min, action_max).astype(np.float32)
            time_step = env.step(action)
            obs, names = flatten_dm_control_observation(time_step.observation)
            if names != feature_names:
                raise ValueError(f"Observation feature names changed for task {task_key}")
            actions[episode_index, step_index] = action
            rewards[episode_index, step_index] = 0.0 if time_step.reward is None else float(time_step.reward)
            discount = 0.0 if time_step.discount is None else float(time_step.discount)
            continuations[episode_index, step_index] = 0.0 if time_step.last() else discount
            valid[episode_index, step_index] = True
            observations[episode_index, step_index + 1] = obs
            if time_step.last():
                break

    if observations is None or actions is None or feature_names is None or action_names is None:
        raise RuntimeError("No DeepMind Control trajectories were generated")

    split = make_episode_split(
        int(num_episodes),
        seed=int(seed),
        train_fraction=float(train_fraction),
        val_fraction=float(val_fraction),
    )
    dataset = ControlTrajectoryDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        continuations=continuations,
        valid=valid,
        split=split,
        episode_ids=np.arange(int(num_episodes), dtype=np.int64),
        feature_names=feature_names,
        action_names=action_names,
        metadata={
            "source": "dm_control",
            "task": task_key,
            "domain_name": domain_name,
            "task_name": task_name,
            "num_episodes": int(num_episodes),
            "episode_length": int(episode_length),
            "seed": int(seed),
            "policy": policy,
            "labels_used_for_training": False,
        },
    )
    save_control_dataset(output_path, dataset)
    return dataset


def flatten_dm_control_observation(observation: Mapping[str, Any]) -> Tuple[np.ndarray, Tuple[str, ...]]:
    parts: List[np.ndarray] = []
    names: List[str] = []
    for key in sorted(observation):
        value = np.asarray(observation[key], dtype=np.float32).reshape(-1)
        parts.append(value)
        if value.size == 1:
            names.append(str(key))
        else:
            names.extend(f"{key}/{index}" for index in range(value.size))
    if not parts:
        raise ValueError("DeepMind Control observation is empty")
    return np.concatenate(parts, axis=0).astype(np.float32), tuple(names)


def make_episode_split(
    num_episodes: int,
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> np.ndarray:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be in (0, 1)")
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must be in [0, 1)")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1")

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(int(num_episodes))
    split = np.full((num_episodes,), "test", dtype="<U5")
    train_count = max(1, int(round(num_episodes * train_fraction)))
    val_count = 0 if val_fraction == 0.0 else max(1, int(round(num_episodes * val_fraction)))
    if train_count + val_count >= num_episodes:
        val_count = max(0, num_episodes - train_count - 1)
    split[order[:train_count]] = "train"
    split[order[train_count : train_count + val_count]] = "val"
    return split


def compute_normalization(
    dataset: ControlTrajectoryDataset,
    train_indices: Sequence[int],
) -> NormalizationStats:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if train_indices.size == 0:
        raise ValueError("At least one train episode is required")
    valid = dataset.valid[train_indices]
    state_valid = np.zeros((len(train_indices), dataset.max_transitions + 1), dtype=bool)
    state_valid[:, :-1] |= valid
    state_valid[:, 1:] |= valid
    train_obs = dataset.observations[train_indices][state_valid]
    train_actions = dataset.actions[train_indices][valid]
    train_rewards = dataset.rewards[train_indices][valid]
    obs_mean, obs_std = _safe_mean_std(train_obs, dataset.obs_dim)
    action_mean, action_std = _safe_mean_std(train_actions, dataset.action_dim)
    reward_mean = float(train_rewards.mean()) if train_rewards.size else 0.0
    reward_std = float(train_rewards.std()) if train_rewards.size else 1.0
    if reward_std < 1e-6:
        reward_std = 1.0
    return NormalizationStats(
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
        reward_mean=reward_mean,
        reward_std=reward_std,
    )


def normalize_observations(observations: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return ((observations - stats.obs_mean) / stats.obs_std).astype(np.float32)


def normalize_actions(actions: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return ((actions - stats.action_mean) / stats.action_std).astype(np.float32)


def normalize_rewards(rewards: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return ((rewards - float(stats.reward_mean)) / float(stats.reward_std)).astype(np.float32)


def select_fraction(indices: Sequence[int], fraction: float, *, seed: int) -> np.ndarray:
    indices_array = np.asarray(indices, dtype=np.int64)
    if not (0.0 < float(fraction) <= 1.0):
        raise ValueError("data fraction must be in (0, 1]")
    if fraction >= 1.0 or indices_array.size <= 1:
        return indices_array
    rng = np.random.default_rng(int(seed))
    count = max(1, int(math.ceil(indices_array.size * float(fraction))))
    chosen = rng.choice(indices_array, size=count, replace=False)
    return np.sort(chosen.astype(np.int64))


def control_world_model_config_from_mapping(values: Mapping[str, Any]) -> ControlWorldModelConfig:
    valid_names = {field.name for field in fields(ControlWorldModelConfig)}
    filtered = {key: value for key, value in dict(values).items() if key in valid_names}
    return ControlWorldModelConfig(**filtered)


def train_step(
    model: ControlWorldModel,
    optimizer: torch.optim.Optimizer,
    x_seq: torch.Tensor,
    action_seq: torch.Tensor,
    reward_seq: torch.Tensor,
    continuation_seq: torch.Tensor,
    weights: LossWeights,
    *,
    density_threshold: float = 1e-4,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    batch_size, seq_plus_one, obs_dim = x_seq.shape
    horizon = seq_plus_one - 1
    z_all = model.encode(x_seq.reshape(batch_size * seq_plus_one, obs_dim)).reshape(
        batch_size,
        seq_plus_one,
        -1,
    )
    z_roll, reward_pred, continuation_logits = model.rollout(z_all[:, 0], action_seq)
    pred = model.decode(z_roll.reshape(batch_size * horizon, -1)).reshape(batch_size, horizon, obs_dim)
    recon = model.decode(z_all.reshape(batch_size * seq_plus_one, -1)).reshape(
        batch_size,
        seq_plus_one,
        obs_dim,
    )

    prediction_loss = F.mse_loss(pred, x_seq[:, 1:])
    reconstruction_loss = F.mse_loss(recon, x_seq)
    latent_loss = F.mse_loss(z_roll, z_all[:, 1:].detach())
    reward_loss = F.mse_loss(reward_pred, reward_seq)
    continuation_loss = F.binary_cross_entropy_with_logits(continuation_logits, continuation_seq)
    sparsity_loss = matrix_l1_loss(
        model.sparsified_matrix_parameters().values(),
        device=x_seq.device,
    )
    k_stability_loss = stability_loss(
        model.transition_matrix_parameters().values(),
        device=x_seq.device,
    )

    loss = (
        float(weights.prediction) * prediction_loss
        + float(weights.reconstruction) * reconstruction_loss
        + float(weights.latent) * latent_loss
        + float(weights.reward) * reward_loss
        + float(weights.continuation) * continuation_loss
        + float(weights.k_sparsity) * sparsity_loss
        + float(weights.k_stability) * k_stability_loss
    )
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        metrics = {
            "loss": float(loss.detach().cpu()),
            "prediction_loss": float(prediction_loss.detach().cpu()),
            "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
            "latent_loss": float(latent_loss.detach().cpu()),
            "reward_loss": float(reward_loss.detach().cpu()),
            "continuation_loss": float(continuation_loss.detach().cpu()),
            "k_sparsity_loss": float(sparsity_loss.detach().cpu()),
            "k_stability_loss": float(k_stability_loss.detach().cpu()),
        }
        metrics.update(model.matrix_density(threshold=density_threshold))
    return metrics


@torch.no_grad()
def evaluate_world_model(
    model: ControlWorldModel,
    sampler: WindowSampler,
    *,
    batch_size: int,
    device: torch.device,
    horizons: Sequence[int],
    instability_norm_threshold: float = 1e4,
    timing_repeats: int = 3,
    planning_candidates: int = 256,
    density_threshold: float = 1e-4,
) -> Dict[str, float]:
    model.eval()
    max_horizon = max(int(h) for h in horizons)
    x_seq, action_seq, reward_seq, continuation_seq = sampler.sample(batch_size, device=device)
    if action_seq.shape[1] < max_horizon:
        raise ValueError(
            f"Sampler sequence_length={action_seq.shape[1]} is shorter than max horizon {max_horizon}"
        )
    x_seq = x_seq[:, : max_horizon + 1]
    action_seq = action_seq[:, :max_horizon]
    reward_seq = reward_seq[:, :max_horizon]
    continuation_seq = continuation_seq[:, :max_horizon]

    z_all = model.encode(x_seq.reshape(-1, x_seq.shape[-1])).reshape(x_seq.shape[0], x_seq.shape[1], -1)
    z_roll, reward_pred, continuation_logits = model.rollout(z_all[:, 0], action_seq)
    pred = model.decode(z_roll.reshape(-1, z_roll.shape[-1])).reshape(x_seq.shape[0], max_horizon, -1)

    metrics: Dict[str, float] = {}
    for horizon in horizons:
        horizon = int(horizon)
        metrics[f"open_loop_mse_h{horizon}"] = float(F.mse_loss(pred[:, :horizon], x_seq[:, 1 : horizon + 1]).cpu())
        metrics[f"latent_mse_h{horizon}"] = float(F.mse_loss(z_roll[:, :horizon], z_all[:, 1 : horizon + 1]).cpu())
    metrics["one_step_latent_mse"] = metrics.get("latent_mse_h1", float(F.mse_loss(z_roll[:, :1], z_all[:, 1:2]).cpu()))
    metrics["reward_mse"] = float(F.mse_loss(reward_pred, reward_seq).cpu())
    metrics["continuation_bce"] = float(
        F.binary_cross_entropy_with_logits(continuation_logits, continuation_seq).cpu()
    )
    finite_mask = torch.isfinite(z_roll).all(dim=-1)
    norm_mask = torch.linalg.vector_norm(torch.nan_to_num(z_roll, nan=0.0), dim=-1) < float(instability_norm_threshold)
    metrics["unstable_rollout_fraction"] = float((~(finite_mask & norm_mask)).float().mean().cpu())
    radius = spectral_radius(model)
    if radius is not None:
        metrics["spectral_radius_K0"] = radius
    metrics.update(model.matrix_density(threshold=density_threshold))
    metrics["rollouts_per_second"] = measure_rollouts_per_second(
        model,
        z_all[:, 0],
        action_seq,
        repeats=timing_repeats,
    )
    metrics["random_shooting_planning_latency_ms"] = measure_random_shooting_latency_ms(
        model,
        z_all[:1, 0],
        action_seq[:1],
        candidates=planning_candidates,
    )
    if device.type == "cuda":
        metrics["peak_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    else:
        metrics["peak_memory_mb"] = 0.0
    return metrics


def matrix_l1_loss(
    matrices: Iterable[torch.Tensor],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    tensors = list(matrices)
    if not tensors:
        return torch.tensor(0.0, device=device)
    total = sum(matrix.abs().mean() for matrix in tensors)
    return total / float(len(tensors))


def stability_loss(
    matrices: Iterable[torch.Tensor],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    tensors = list(matrices)
    if not tensors:
        return torch.tensor(0.0, device=device)
    losses = []
    for matrix in tensors:
        proxy = torch.linalg.matrix_norm(matrix, ord="fro") / math.sqrt(max(1, matrix.shape[0]))
        losses.append(F.relu(proxy - 1.25).square())
    return sum(losses) / float(len(losses))


def spectral_radius(model: ControlWorldModel) -> Optional[float]:
    if model.k0 is None:
        return None
    matrix = model.k0.detach()
    matrix_cpu = matrix.cpu() if matrix.device.type != "cpu" else matrix
    try:
        eigvals = torch.linalg.eigvals(matrix_cpu)
        return float(torch.max(torch.abs(eigvals)).item())
    except RuntimeError:
        return None


def measure_rollouts_per_second(
    model: ControlWorldModel,
    z0: torch.Tensor,
    actions: torch.Tensor,
    *,
    repeats: int,
) -> float:
    repeats = max(1, int(repeats))
    device = z0.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        model.rollout(z0, actions)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - start, 1e-12)
    return float(repeats * z0.shape[0] / elapsed)


def measure_random_shooting_latency_ms(
    model: ControlWorldModel,
    z0: torch.Tensor,
    reference_actions: torch.Tensor,
    *,
    candidates: int,
) -> float:
    candidates = max(1, int(candidates))
    action_scale = reference_actions.std(dim=(0, 1), keepdim=True, correction=0).clamp(min=1e-3)
    sampled_actions = torch.randn(
        candidates,
        reference_actions.shape[1],
        reference_actions.shape[2],
        device=reference_actions.device,
        dtype=reference_actions.dtype,
    ) * action_scale
    z0_repeated = z0.expand(candidates, -1)
    device = z0.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    _, reward_pred, _ = model.rollout(z0_repeated, sampled_actions)
    _ = reward_pred.sum(dim=1).argmax()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return float(1000.0 * (time.perf_counter() - start))


def save_checkpoint(
    path: str | Path,
    *,
    model: ControlWorldModel,
    optimizer: torch.optim.Optimizer,
    model_config: ControlWorldModelConfig,
    loss_weights: LossWeights,
    normalization: NormalizationStats,
    step: int,
    metrics: Mapping[str, float],
    metadata: Mapping[str, Any],
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
) -> Tuple[ControlWorldModel, NormalizationStats, Dict[str, Any]]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    model_config = control_world_model_config_from_mapping(checkpoint["model_config"])
    model = ControlWorldModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stats = NormalizationStats.from_mapping(checkpoint["normalization"])
    return model, stats, checkpoint


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
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


def _scalar_string(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(array.reshape(-1)[0])
