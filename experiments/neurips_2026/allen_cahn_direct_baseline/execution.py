"""High-utilization, fail-fast CUDA execution for the matched direct model."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F

from experiments.neurips_2026.allen_cahn_direct_baseline.core import (
    DirectResidualConv,
    periodic_gradient_mse,
)


def loss_tensors(
    model: DirectResidualConv,
    sequence: torch.Tensor,
    *,
    gradient_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = model.rollout(sequence[:, 0], horizon=sequence.shape[1] - 1)
    truth = sequence[:, 1:]
    field_loss = F.mse_loss(prediction, truth)
    gradient_loss = periodic_gradient_mse(prediction, truth)
    loss = field_loss + float(gradient_weight) * gradient_loss
    return loss, field_loss, gradient_loss


def _assert_named_finite(named: Iterable[tuple[str, torch.Tensor]]) -> None:
    values = list(named)
    if not values:
        raise RuntimeError("Finiteness audit received no tensors")
    checks = torch.stack([torch.isfinite(tensor).all() for _, tensor in values])
    if bool(checks.all()):
        return
    failed = [
        name for name, tensor in values if not bool(torch.isfinite(tensor).all())
    ]
    raise FloatingPointError(f"Nonfinite CUDA training tensors: {failed}")


def _metrics(
    loss: torch.Tensor,
    field_loss: torch.Tensor,
    gradient_loss: torch.Tensor,
) -> dict[str, float]:
    values = torch.stack((loss.detach(), field_loss.detach(), gradient_loss.detach()))
    loss_value, field_value, gradient_value = values.cpu().tolist()
    return {
        "loss": float(loss_value),
        "field_mse": float(field_value),
        "gradient_mse": float(gradient_value),
    }


def eager_training_step(
    model: DirectResidualConv,
    optimizer: torch.optim.Optimizer,
    sequence: torch.Tensor,
    *,
    gradient_weight: float,
) -> dict[str, float]:
    """Run the first update and initialize Adam/CUDNN state before capture."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, field_loss, gradient_loss = loss_tensors(
        model, sequence, gradient_weight=gradient_weight
    )
    loss.backward()
    _assert_named_finite(
        [("loss", loss)]
        + [
            (f"gradient:{name}", parameter.grad)
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        ]
    )
    optimizer.step()
    _assert_named_finite(
        (f"parameter:{name}", parameter)
        for name, parameter in model.named_parameters()
    )
    return _metrics(loss, field_loss, gradient_loss)


class CudaGraphTrainingStepper:
    """Capture one fixed-shape H200 update and replay it without launch gaps."""

    def __init__(
        self,
        model: DirectResidualConv,
        optimizer: torch.optim.Optimizer,
        sequence: torch.Tensor,
        *,
        gradient_weight: float,
    ) -> None:
        if sequence.device.type != "cuda":
            raise ValueError("CUDA graph capture requires a CUDA sequence")
        if not bool(optimizer.defaults.get("capturable", False)):
            raise ValueError("CUDA graph capture requires capturable Adam")
        if any(parameter.grad is None for parameter in model.parameters()):
            raise RuntimeError("An eager update must initialize every gradient")
        self.model = model
        self.optimizer = optimizer
        self.static_sequence = sequence.detach().clone()
        optimizer.zero_grad(set_to_none=False)
        torch.cuda.synchronize(sequence.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            optimizer.zero_grad(set_to_none=False)
            self.loss, self.field_loss, self.gradient_loss = loss_tensors(
                model,
                self.static_sequence,
                gradient_weight=gradient_weight,
            )
            self.loss.backward()
            optimizer.step()
        # Capture records kernels but does not constitute an optimizer update.
        # Replay exactly once so this sampled batch contributes one update.
        self.graph.replay()
        self.last_metrics = self._audit_and_measure()

    def _audit_and_measure(self) -> dict[str, float]:
        _assert_named_finite(
            [("loss", self.loss)]
            + [
                (f"gradient:{name}", parameter.grad)
                for name, parameter in self.model.named_parameters()
                if parameter.grad is not None
            ]
            + [
                (f"parameter:{name}", parameter)
                for name, parameter in self.model.named_parameters()
            ]
        )
        return _metrics(self.loss, self.field_loss, self.gradient_loss)

    def step(self, sequence: torch.Tensor) -> dict[str, float]:
        if (
            sequence.shape != self.static_sequence.shape
            or sequence.dtype != self.static_sequence.dtype
            or sequence.device != self.static_sequence.device
        ):
            raise ValueError("CUDA graph input shape, dtype, or device drift")
        self.static_sequence.copy_(sequence)
        self.graph.replay()
        self.last_metrics = self._audit_and_measure()
        return self.last_metrics
