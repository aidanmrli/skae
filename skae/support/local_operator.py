"""Model and optimization mechanics for support-routed local operators."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from skae.support.routing import _step_routes_for_torch


def _freeze_autoencoder(model: nn.Module) -> None:
    """Freeze the complete stage-one model, including the shared global K."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _encode_sequence_batches(
    model: nn.Module,
    batches: Sequence[torch.Tensor],
    device: str,
) -> np.ndarray:
    if not batches:
        raise RuntimeError("No route-fit batches were provided")
    model.eval()
    latents: List[np.ndarray] = []
    model_device = next(model.parameters()).device
    with torch.no_grad():
        for batch in batches:
            x_seq = batch.to(model_device)
            batch_size, sequence_length, observation_dim = x_seq.shape
            z = model.encode(x_seq.reshape(batch_size * sequence_length, observation_dim))
            z = z.reshape(batch_size, sequence_length, -1)
            latents.append(z.detach().cpu().numpy().astype(np.float32, copy=False))
    del device
    return np.concatenate(latents, axis=0)


def _target_centers_from_global(
    centers: Dict[object, np.ndarray],
    family_ids: Sequence[object],
    global_k: np.ndarray,
) -> Dict[object, np.ndarray]:
    return {
        family_id: (
            np.asarray(centers[family_id], dtype=np.float32) @ global_k
        ).astype(np.float32, copy=False)
        for family_id in family_ids
    }


def _route_indices_for_torch(
    z: torch.Tensor,
    *,
    model: nn.Module,
    route_env: object,
    route_codebook: Dict[str, object],
    scheme: str,
    support_value: float,
    family_jaccard_threshold: float,
    family_to_index: Dict[str, int],
    family_cache: Dict[object, object],
    device: torch.device,
) -> torch.Tensor:
    # Kept in the signature for compatibility with frozen checkpoint loaders.
    del model, route_env
    return _step_routes_for_torch(
        z,
        scheme=scheme,
        value=support_value,
        family_jaccard_threshold=family_jaccard_threshold,
        support_key_to_family=route_codebook["support_key_to_family"],
        family_prototypes=route_codebook["family_prototypes"],
        family_to_index=family_to_index,
        family_cache=family_cache,
        device=device,
    )


class SourceTargetLocalMapBundle(nn.Module):
    """Affine local charts initialized to reproduce the frozen global map."""

    def __init__(
        self,
        *,
        family_ids: Sequence[object],
        source_centers: Dict[object, np.ndarray],
        target_centers: Dict[object, np.ndarray],
        global_k: np.ndarray,
        device: str,
        learn_target_centers: bool = False,
    ) -> None:
        super().__init__()
        self.family_ids = [str(item) for item in family_ids]
        self.family_to_index = {
            family_id: index for index, family_id in enumerate(self.family_ids)
        }
        source_array = np.stack(
            [source_centers[item] for item in family_ids], axis=0
        ).astype(np.float32, copy=False)
        target_array = np.stack(
            [target_centers[item] for item in family_ids], axis=0
        ).astype(np.float32, copy=False)
        self.register_buffer(
            "source_centers", torch.from_numpy(source_array).to(device=device)
        )
        target_tensor = torch.from_numpy(target_array).to(device=device)
        if learn_target_centers:
            self.target_centers = nn.Parameter(target_tensor)
        else:
            self.register_buffer("target_centers", target_tensor)
        initial_k = torch.from_numpy(
            global_k.astype(np.float32, copy=False)
        ).to(device=device)
        self.local_maps = nn.Parameter(
            initial_k.unsqueeze(0).repeat(len(self.family_ids), 1, 1)
        )
        self.register_buffer("global_k", initial_k)

    def forward(
        self,
        z: torch.Tensor,
        route_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = route_index >= 0
        output = z @ self.global_k
        if bool(valid.any()):
            selected = route_index[valid]
            output[valid] = self.target_centers[selected] + torch.bmm(
                (z[valid] - self.source_centers[selected]).unsqueeze(1),
                self.local_maps[selected],
            ).squeeze(1)
        return output, valid


class StagedLocalKoopmanWrapper(nn.Module):
    """Evaluation wrapper routing every latent step through a local map."""

    def __init__(
        self,
        *,
        base_model: nn.Module,
        local_bundle: SourceTargetLocalMapBundle,
        route_codebook: Dict[str, object],
        route_env: object,
        scheme: str,
        support_value: float,
        family_jaccard_threshold: float,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.local_bundle = local_bundle
        self.route_codebook = route_codebook
        self.route_env = route_env
        self.scheme = scheme
        self.support_value = float(support_value)
        self.family_jaccard_threshold = float(family_jaccard_threshold)
        self.family_cache: Dict[object, object] = {}
        self.cfg = getattr(base_model, "cfg")
        self.observation_size = int(getattr(base_model, "observation_size"))
        self.target_size = int(getattr(base_model, "target_size"))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model.encode(x)

    def encode_with_prior(
        self,
        x: torch.Tensor,
        latent_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.base_model.encode_with_prior(x, latent_prior=latent_prior)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.base_model.decode(z)

    def kmatrix(self) -> torch.Tensor:
        return self.local_bundle.global_k

    def step_latent(self, z: torch.Tensor) -> torch.Tensor:
        route_index = _route_indices_for_torch(
            z,
            model=self.base_model,
            route_env=self.route_env,
            route_codebook=self.route_codebook,
            scheme=self.scheme,
            support_value=self.support_value,
            family_jaccard_threshold=self.family_jaccard_threshold,
            family_to_index=self.local_bundle.family_to_index,
            family_cache=self.family_cache,
            device=z.device,
        )
        return self.local_bundle(z, route_index)[0]

    def step_env(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.step_latent(self.encode(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.step_env(x)


def _make_wrapped_model(
    model: nn.Module,
    bundle: SourceTargetLocalMapBundle,
    route_codebook: Dict[str, object],
    *,
    route_env: object,
    scheme: str,
    support_value: float,
    family_jaccard_threshold: float,
) -> StagedLocalKoopmanWrapper:
    wrapped = StagedLocalKoopmanWrapper(
        base_model=model,
        local_bundle=bundle,
        route_codebook=route_codebook,
        route_env=route_env,
        scheme=scheme,
        support_value=support_value,
        family_jaccard_threshold=family_jaccard_threshold,
    )
    return wrapped.to(next(model.parameters()).device)


def _local_train_step(
    *,
    model: nn.Module,
    bundle: SourceTargetLocalMapBundle,
    route_codebook: Dict[str, object],
    route_env: object,
    x_seq: torch.Tensor,
    scheme: str,
    support_value: float,
    family_jaccard_threshold: float,
    optimizer: torch.optim.Optimizer,
    family_cache: Dict[object, object],
    step: int,
) -> Dict[str, float]:
    model.eval()
    bundle.train()
    optimizer.zero_grad()
    if x_seq.ndim != 3 or x_seq.shape[1] < 2:
        raise ValueError("x_seq must have shape [batch, horizon+1, observation]")
    batch_size, sequence_length, observation_dim = x_seq.shape
    horizon = sequence_length - 1
    x_true = x_seq[:, 1:, :]
    with torch.no_grad():
        z_all = model.encode(
            x_seq.reshape(batch_size * sequence_length, observation_dim)
        ).reshape(batch_size, sequence_length, -1)
        z_true = z_all[:, 1:, :]
        x_reconstructed = model.decode(
            z_true.reshape(batch_size * horizon, -1)
        ).reshape(batch_size, horizon, observation_dim)
        reconstruction_error = torch.norm(
            x_true - x_reconstructed, dim=-1
        ).mean()

    z = z_all[:, 0, :]
    predictions: List[torch.Tensor] = []
    used_local: List[torch.Tensor] = []
    route_counts = torch.zeros(
        len(bundle.family_ids), dtype=torch.long, device=z.device
    )
    fallback_count = torch.zeros((), dtype=torch.long, device=z.device)
    for _ in range(horizon):
        route_index = _route_indices_for_torch(
            z,
            model=model,
            route_env=route_env,
            route_codebook=route_codebook,
            scheme=scheme,
            support_value=support_value,
            family_jaccard_threshold=family_jaccard_threshold,
            family_to_index=bundle.family_to_index,
            family_cache=family_cache,
            device=z.device,
        )
        valid = route_index >= 0
        if bool(valid.any()):
            route_counts += torch.bincount(
                route_index[valid], minlength=len(bundle.family_ids)
            )[: len(bundle.family_ids)]
        fallback_count += (~valid).sum()
        z, used = bundle(z, route_index)
        predictions.append(z)
        used_local.append(used.float())

    z_prediction = torch.stack(predictions, dim=1)
    x_prediction = model.decode(
        z_prediction.reshape(batch_size * horizon, -1)
    ).reshape(batch_size, horizon, observation_dim)
    loss, metrics = model.loss(
        x_pred=x_prediction,
        x_true=x_true,
        x0=x_seq[:, 0, :],
        z0=z_all[:, 0, :],
        z_pred=z_prediction,
        z_true=z_true,
        reconstruction_error=reconstruction_error,
        sparsity_latent=z_prediction,
        step=step,
    )
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        metrics["route_coverage"] = float(
            torch.stack(used_local, dim=1).mean().detach().cpu().item()
        )
        metrics["fallback_fraction"] = 1.0 - metrics["route_coverage"]
        metrics["route_total_count"] = float(batch_size * horizon)
        metrics["route_fallback_count"] = float(fallback_count.detach().cpu().item())
        for index, family_id in enumerate(bundle.family_ids):
            metrics[f"route_family_{family_id}_count"] = float(
                route_counts[index].detach().cpu().item()
            )
    return metrics
