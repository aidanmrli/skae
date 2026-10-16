"""The sparse Koopman autoencoder and its training loss.

An observation ``x`` (shape ``[..., obs_dim]``) is encoded to a code
``z = E(x)`` (shape ``[..., latent_dim]``). The code evolves linearly,
``z_{t+h} = z_t K^h`` with a learned ``[latent_dim, latent_dim]`` matrix ``K``,
and a linear decoder with unit-norm atoms maps codes back to observations.
A sparse encoder sets most code entries to exactly zero, so each state uses a
few active coordinates (its support), and states in different basins of
attraction can use different supports.

Codes are row vectors: one step is ``z @ K``, the transpose of the paper's
``K z``, so the paper's ``K^h E(x)`` is ``E(x) K^h`` here (and its ``K`` and
LISTA ``S`` are the transposes of the matrices stored here). The active
coordinates of a code select rows of ``K``.

Encoders (``ModelConfig.encoder``):

* ``"lista"``: an MLP pre-code refined by unrolled, learned ISTA iterations
  (thresholding), whose decoder atoms start as a random dictionary;
* ``"mlp"``: an MLP whose code is sparse with a final ReLU (Sparse MLP) or
  dense without it (Dense MLP).
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from skae.config import ModelConfig

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}
_CHOICES = {
    "encoder": ("lista", "mlp"),
    "activation": tuple(_ACTIVATIONS),
    "lista_output": ("sign_split", "none"),
    "lista_threshold": ("soft", "nonnegative"),
    "koopman": ("dense", "block_diagonal"),
    "koopman_rollout": ("per_block", "assembled"),
    "loss_time_reduction": ("divide_horizon", "mean"),
}


def shrink(x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """Soft thresholding ``sign(x) * max(|x| - threshold, 0)``."""
    return torch.sign(x) * torch.maximum(torch.abs(x) - threshold, torch.zeros_like(x))


def split_sizes(total: int, parts: int) -> List[int]:
    """Split ``total`` into ``parts`` near-equal sizes, larger first: ``(256, 3) -> [86, 85, 85]``."""
    if not 0 < parts <= total:
        raise ValueError(f"cannot split {total} into {parts} blocks")
    base, remainder = divmod(total, parts)
    return [base + 1 if index < remainder else base for index in range(parts)]


def orthogonal_dictionary(obs_dim: int, code_dim: int) -> torch.Tensor:
    """``[obs_dim, code_dim]`` unit-norm atoms: the columns of stacked random orthonormal bases."""
    atoms = torch.empty(obs_dim, code_dim)
    for start in range(0, code_dim, obs_dim):
        q, _ = torch.linalg.qr(torch.randn(obs_dim, obs_dim))
        take = min(obs_dim, code_dim - start)
        atoms[:, start:start + take] = q[:, :take]
    return atoms / torch.norm(atoms, dim=0, keepdim=True).clamp(min=1e-8)


def lipschitz_constant(atoms: torch.Tensor) -> float:
    """Power-iteration estimate of ``||W^T W||_2`` for atoms ``W``, plus a 5% margin."""
    with torch.no_grad():
        gram = atoms.T @ atoms
        v = torch.randn(gram.shape[0], 1)
        v = v / torch.norm(v).clamp(min=1e-8)
        for _ in range(10):
            v = gram @ v
            v = v / torch.norm(v).clamp(min=1e-8)
        estimate = torch.norm(gram @ v) / torch.norm(v).clamp(min=1e-8)
    return estimate.item() * 1.05


class MLP(nn.Module):
    """Linear layers with a hidden activation and an optional final ReLU."""

    def __init__(self, in_dim: int, out_dim: int, hidden: Sequence[int], activation: str,
                 output_relu: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        for width in hidden:
            layers += [nn.Linear(in_dim, width), _ACTIVATIONS[activation]()]
            in_dim = width
        layers.append(nn.Linear(in_dim, out_dim))
        if output_relu:
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LISTAEncoder(nn.Module):
    """Learned ISTA: ``c = MLP(x)``, ``z = T(c)``, then ``lista_loops`` times ``z = T(z S + c)``.

    ``T`` soft-thresholds at ``t = lista_alpha / L`` (``relu(u - t)`` with
    ``lista_threshold="nonnegative"``), where ``L`` bounds ``||W^T W||`` for the initial
    dictionary ``W``, and ``S`` starts at the ISTA matrix ``I - W^T W / L``.
    With ``lista_output="sign_split"`` the output is ``[relu(z), relu(-z)]``,
    twice as wide as ``z``.
    """

    def __init__(self, cfg: ModelConfig, obs_dim: int, atoms: torch.Tensor, lipschitz: float):
        super().__init__()
        code_dim = atoms.shape[1]
        self.loops = cfg.lista_loops
        self.sign_split = cfg.lista_output == "sign_split"
        self.nonnegative = cfg.lista_threshold == "nonnegative"
        self.learn_thresholds = cfg.lista_learn_thresholds
        self.precode = MLP(obs_dim, code_dim, cfg.hidden, cfg.activation)
        self.S = nn.Parameter(torch.eye(code_dim) - (1.0 / lipschitz) * (atoms.T @ atoms))
        threshold = float(cfg.lista_alpha / lipschitz)
        if self.learn_thresholds:
            self.log_thresholds = nn.Parameter(torch.full((code_dim,), math.log(threshold), dtype=atoms.dtype))
        else:
            self.register_buffer("threshold", torch.tensor(threshold))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.precode(x)
        threshold = torch.exp(self.log_thresholds) if self.learn_thresholds else self.threshold
        threshold = threshold.expand_as(c)

        def step(u: torch.Tensor) -> torch.Tensor:
            return F.relu(u - threshold) if self.nonnegative else shrink(u, threshold)

        z = step(c)
        for _ in range(self.loops):
            z = step(z @ self.S + c)
        return torch.cat([F.relu(z), F.relu(-z)], dim=-1) if self.sign_split else z


class KoopmanAutoencoder(nn.Module):
    """Encoder ``E``, Koopman matrix ``K`` and linear decoder ``D``: ``x -> E(x) K^h -> D``.

    * ``encode``: observations ``[..., obs_dim]`` -> codes ``[..., latent_dim]``;
    * ``rollout_latent``: codes ``[..., latent_dim]`` -> ``[..., horizon, latent_dim]``;
    * ``decode``: codes ``[..., latent_dim]`` -> observations ``[..., obs_dim]``.

    With ``input_normalization`` the model stores a per-coordinate mean and
    scale (:meth:`set_input_normalization`); ``encode`` and ``decode`` then take
    and return original coordinates, and the loss is computed in normalized
    coordinates. Without it, ``normalize_observations`` is the identity.

    Parameters (``state_dict`` names) by variant:

    * LISTA: ``encoder`` (:class:`LISTAEncoder`: ``precode`` MLP, ``S``, and
      ``log_thresholds`` or the buffer ``threshold``) and ``dictionary``
      ``[latent_dim, obs_dim]``, one decoder atom per row;
    * MLP encoder: ``encoder.network`` and ``decoder`` (``nn.Linear`` without
      bias; its weight ``[obs_dim, latent_dim]`` holds one atom per column);
    * dense ``K``: ``K`` ``[latent_dim, latent_dim]``; block-diagonal ``K``:
      ``K_blocks``, one square block per entry of ``block_sizes``;
    * buffers ``_input_mean`` and ``_input_scale`` ``[obs_dim]`` with
      ``input_normalization``.
    """

    def __init__(self, cfg: ModelConfig, obs_dim: int):
        super().__init__()
        for name, choices in _CHOICES.items():
            if getattr(cfg, name) not in choices:
                raise ValueError(f"ModelConfig.{name} must be one of {choices}, got {getattr(cfg, name)!r}")
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.latent_dim = cfg.latent_dim
        if cfg.input_normalization:
            self.register_buffer("_input_mean", torch.zeros(obs_dim))
            self.register_buffer("_input_scale", torch.ones(obs_dim))
            self._normalization_checked = False  # see normalize_observations

        if cfg.encoder == "lista":
            sign_split = cfg.lista_output == "sign_split"
            if sign_split and cfg.latent_dim % 2:
                raise ValueError(f"lista_output='sign_split' needs an even latent_dim, got {cfg.latent_dim}")
            atoms = orthogonal_dictionary(obs_dim, cfg.latent_dim // 2 if sign_split else cfg.latent_dim)
            # One decoder atom per code coordinate (row); the sign split decodes relu(-z) with -W.
            decoder_atoms = torch.cat([atoms, -atoms], dim=1) if sign_split else atoms
            self.dictionary = nn.Parameter(decoder_atoms.T.clone())  # [latent_dim, obs_dim]
            self.encoder = LISTAEncoder(cfg, obs_dim, atoms, lipschitz_constant(atoms))
        else:
            self.encoder = MLP(obs_dim, cfg.latent_dim, cfg.hidden, cfg.activation, cfg.mlp_output_relu)
            self.decoder = nn.Linear(cfg.latent_dim, obs_dim, bias=False)  # atoms are the weight columns

        if cfg.koopman == "block_diagonal":
            self.block_sizes = split_sizes(cfg.latent_dim, cfg.num_blocks)
            self.K_blocks = nn.ParameterList([nn.Parameter(torch.eye(size)) for size in self.block_sizes])
        else:
            self.K = nn.Parameter(torch.eye(cfg.latent_dim))
        if cfg.soft_block_weight > 0.0:
            if cfg.koopman != "dense":
                raise ValueError("soft_block_weight > 0 needs koopman='dense'")
            blocks = [torch.ones(size, size) for size in split_sizes(cfg.latent_dim, cfg.num_blocks)]
            self.register_buffer("block_mask", torch.block_diag(*blocks).bool(), persistent=False)

    # -- observations <-> codes -----------------------------------------------------------
    def normalize_observations(self, x: torch.Tensor) -> torch.Tensor:
        """Original coordinates -> the normalized coordinates of the loss."""
        if not self.cfg.input_normalization:
            return x
        if not self._normalization_checked:  # once per model: warn if the mean and scale were never set
            self._normalization_checked = True
            if not bool(self._input_mean.any()) and bool((self._input_scale == 1).all()):
                warnings.warn("input_normalization=True, but the mean and scale are still 0 and 1; call "
                              "set_input_normalization with the training data's moments", stacklevel=2)
        return (x - self._input_mean) / self._input_scale

    def encode_normalized(self, x: torch.Tensor) -> torch.Tensor:
        """Normalized observations ``[..., obs_dim]`` -> codes ``[..., latent_dim]``."""
        return self.encoder(x)

    def decode_normalized(self, z: torch.Tensor) -> torch.Tensor:
        """Codes ``[..., latent_dim]`` -> normalized observations: ``z`` times the unit-norm atoms."""
        if self.cfg.encoder == "lista":
            atoms = self.dictionary / torch.norm(self.dictionary, dim=1, keepdim=True).clamp(min=1e-4)
            return z @ atoms
        weight = self.decoder.weight  # [obs_dim, latent_dim]
        return z @ (weight / torch.norm(weight, dim=0, keepdim=True).clamp(min=1e-4)).T

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Observations ``[..., obs_dim]`` -> codes ``[..., latent_dim]``."""
        return self.encode_normalized(self.normalize_observations(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Codes ``[..., latent_dim]`` -> observations ``[..., obs_dim]``."""
        x = self.decode_normalized(z)
        if self.cfg.input_normalization:
            return x * self._input_scale + self._input_mean
        return x

    def set_input_normalization(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        """Store the per-coordinate mean and scale (computed on training data only)."""
        if not self.cfg.input_normalization:
            raise RuntimeError("this model has input_normalization=False")
        mean, scale = torch.as_tensor(mean), torch.as_tensor(scale)
        if mean.shape != (self.obs_dim,) or scale.shape != (self.obs_dim,) or not bool((scale > 0).all()):
            raise ValueError(f"mean and scale must have shape ({self.obs_dim},) and the scale must be positive")
        with torch.no_grad():
            self._input_mean.copy_(mean)
            self._input_scale.copy_(scale)
        self._normalization_checked = True

    # -- linear latent dynamics -----------------------------------------------------------
    def kmatrix(self) -> torch.Tensor:
        """``K`` as one ``[latent_dim, latent_dim]`` matrix."""
        if self.cfg.koopman == "block_diagonal":
            return torch.block_diag(*self.K_blocks)
        return self.K

    def koopman_parameters(self) -> List[nn.Parameter]:
        """The parameters of ``K`` (the paper's trainer gives them their own learning rate)."""
        return list(self.K_blocks) if self.cfg.koopman == "block_diagonal" else [self.K]

    def step_latent(self, z: torch.Tensor) -> torch.Tensor:
        """One step ``z K``; a block-diagonal ``K`` multiplies each block's coordinates separately."""
        if self.cfg.koopman == "dense":
            return z @ self.K
        parts, start = [], 0
        for size, block in zip(self.block_sizes, self.K_blocks):
            parts.append(z[..., start:start + size] @ block)
            start += size
        return torch.cat(parts, dim=-1)

    def latent_stepper(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """``z -> z K`` for one rollout.

        With ``koopman_rollout="assembled"`` a block-diagonal ``K`` is assembled
        once and applied with one product per step, which is faster on a GPU.
        """
        if self.cfg.koopman == "block_diagonal" and self.cfg.koopman_rollout == "assembled":
            k = self.kmatrix()
            return lambda z: z @ k
        return self.step_latent

    def rollout_latent(self, z0: torch.Tensor, horizon: int) -> torch.Tensor:
        """Codes ``z0 K^h`` for ``h = 1..horizon``: ``[..., latent_dim] -> [..., horizon, latent_dim]``; ``horizon >= 1``."""
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")
        step, states, z = self.latent_stepper(), [], z0
        for _ in range(horizon):
            z = step(z)
            states.append(z)
        return torch.stack(states, dim=-2)

    def forecast(self, x0: torch.Tensor, horizon: int, reencode_every: Optional[int] = None) -> torch.Tensor:
        """Forecast ``[..., obs_dim] -> [..., horizon, obs_dim]`` from the code of ``x0``.

        ``horizon >= 1``. With ``reencode_every = m >= 1`` the decoded prediction is re-encoded after every
        ``m`` steps and the rollout continues from that code. Callers wrap this in ``torch.no_grad()``; there is
        no divergence fallback.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")
        if reencode_every is not None and reencode_every < 1:
            raise ValueError(f"reencode_every must be None or at least 1, got {reencode_every}")
        step, z, states = self.latent_stepper(), self.encode(x0), []
        for t in range(1, horizon + 1):
            z = step(z)
            x = self.decode(z)
            states.append(x)
            if reencode_every is not None and t % reencode_every == 0:
                z = self.encode(x)
        return torch.stack(states, dim=-2)

    def soft_block_penalty(self) -> torch.Tensor:
        """``sum |K_ij|`` over the entries of a dense ``K`` outside its diagonal blocks."""
        return self.K.masked_select(~self.block_mask).abs().sum()


def training_loss(model: KoopmanAutoencoder, windows: torch.Tensor):
    """The training loss on windows ``[B, L + 1, obs_dim]``; returns ``(loss, terms)``.

    A window holds a state ``x_t`` and the ``L`` states after it. With codes
    ``z_{t+h} = E(x_{t+h})`` and predicted codes ``p_{t+h} = E(x_t) K^h`` for
    ``h = 1..L``, the loss is the weighted sum of

    * prediction:      ``||D(p_{t+h}) - x_{t+h}|| / sqrt(obs_dim)``
    * alignment:       ``||p_{t+h} - z_{t+h}||``
    * reconstruction:  ``||D(z_{t+h}) - x_{t+h}|| / sqrt(obs_dim)``
    * sparsity:        ``||z_{t+h}||_1``

    each averaged over the batch and the ``L`` steps (and divided by ``L`` once
    more with ``loss_time_reduction="divide_horizon"``), plus
    ``soft_block_weight`` times :meth:`KoopmanAutoencoder.soft_block_penalty`.
    Observations enter in normalized coordinates. ``terms`` holds the detached
    terms and ``active_fraction``, the fraction of code entries with
    ``|z| > 1e-6``. The windows must be on the model's device and dtype and
    have ``L >= 1`` (:func:`skae.fit` checks and converts them).
    """
    cfg = model.cfg
    batch, length, obs_dim = windows.shape
    horizon = length - 1
    x = model.normalize_observations(windows)
    z = model.encode_normalized(x.reshape(batch * length, obs_dim)).reshape(batch, length, -1)
    x_true, z_true = x[:, 1:], z[:, 1:]
    z_pred = model.rollout_latent(z[:, 0], horizon)
    x_pred = model.decode_normalized(z_pred.reshape(batch * horizon, -1)).reshape(batch, horizon, obs_dim)
    x_recon = model.decode_normalized(z_true.reshape(batch * horizon, -1)).reshape(batch, horizon, obs_dim)

    dim_scale = torch.tensor(math.sqrt(obs_dim), device=windows.device, dtype=windows.dtype)
    scale = 1.0 / horizon if cfg.loss_time_reduction == "divide_horizon" else 1.0
    prediction = torch.norm(x_pred - x_true, dim=-1).mean() / dim_scale * scale
    alignment = torch.norm(z_pred - z_true, dim=-1).mean() * scale
    reconstruction = torch.norm(x_true - x_recon, dim=-1).mean() / dim_scale * scale
    sparsity = torch.norm(z_true, p=1, dim=-1).mean() * scale
    loss = (
        cfg.alignment_weight * alignment
        + cfg.reconstruction_weight * reconstruction
        + cfg.prediction_weight * prediction
        + cfg.sparsity_weight * sparsity
    )
    if cfg.soft_block_weight > 0.0:
        loss = loss + cfg.soft_block_weight * model.soft_block_penalty()
    terms = {
        "prediction": prediction.detach(),
        "alignment": alignment.detach(),
        "reconstruction": reconstruction.detach(),
        "sparsity": sparsity.detach(),
        "active_fraction": (z_true.detach().abs() > 1e-6).float().sum(dim=-1).mean() / model.latent_dim,
    }
    return loss, terms
