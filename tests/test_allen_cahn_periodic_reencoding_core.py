from __future__ import annotations

import pytest
import torch

from experiments.neurips_2026.allen_cahn_periodic_reencoding.core import (
    DIRECT_MODE,
    evaluate_model_packed,
    segmented_rollout,
    validate_period_candidates,
)


class CountingKoopman(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kmat = torch.nn.Parameter(
            torch.tensor([[0.8, 0.3], [-0.4, 1.1]], dtype=torch.float32)
        )
        self.encoded_inputs: list[torch.Tensor] = []
        self.autocast_states: list[bool] = []

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        self.encoded_inputs.append(values.detach().clone())
        self.autocast_states.append(torch.is_autocast_enabled(values.device.type))
        return values + 0.125

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        self.autocast_states.append(torch.is_autocast_enabled(values.device.type))
        return 1.25 * values - 0.05


def _synthetic_fields(
    *,
    datasets: int = 2,
    trajectories: int = 5,
    horizon: int = 5,
) -> torch.Tensor:
    initial_scalar = torch.linspace(-0.8, 0.9, trajectories, dtype=torch.float32)
    initial = torch.stack((initial_scalar, 0.4 * initial_scalar + 0.3), dim=-1)
    initial = torch.stack([initial + 0.07 * index for index in range(datasets)])
    time = torch.arange(horizon + 1, dtype=torch.float32).view(1, 1, -1, 1)
    velocity = torch.tensor([0.035, -0.02], dtype=torch.float32).view(1, 1, 1, 2)
    return initial[:, :, None, :] + time * velocity


def test_segmented_rollout_has_exact_orientation_and_boundary_encode_calls() -> None:
    model = CountingKoopman()
    initial = torch.tensor([[1.0, -0.5], [0.2, 0.7]], dtype=torch.float32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        predictions = segmented_rollout(model, initial, horizon=5, period=2)

    assert predictions.dtype == torch.float32
    assert len(model.encoded_inputs) == 3  # initial, h=2 boundary, h=4 boundary
    torch.testing.assert_close(model.encoded_inputs[0], initial, rtol=0, atol=0)
    torch.testing.assert_close(
        model.encoded_inputs[1], predictions[:, 1], rtol=0, atol=0
    )
    torch.testing.assert_close(
        model.encoded_inputs[2], predictions[:, 3], rtol=0, atol=0
    )
    assert not any(model.autocast_states)

    first_latent = (initial + 0.125) @ model.kmat.detach().T
    expected_first = 1.25 * first_latent - 0.05
    torch.testing.assert_close(predictions[:, 0], expected_first, rtol=0, atol=0)
    wrong_first = 1.25 * ((initial + 0.125) @ model.kmat.detach()) - 0.05
    assert not torch.allclose(predictions[:, 0], wrong_first)
    assert first_latent.dtype == torch.float32


def test_direct_mode_encodes_once_and_never_refreshes() -> None:
    initial = torch.tensor([[0.4, -0.2]], dtype=torch.float32)
    model = CountingKoopman()
    predictions = segmented_rollout(
        model,
        initial,
        horizon=5,
        period=None,
        max_decode_segment=2,
    )
    assert len(model.encoded_inputs) == 1

    unchunked_model = CountingKoopman()
    unchunked_model.load_state_dict(model.state_dict())
    unchunked = segmented_rollout(
        unchunked_model,
        initial,
        horizon=5,
        period=None,
        max_decode_segment=5,
    )
    assert len(unchunked_model.encoded_inputs) == 1
    torch.testing.assert_close(predictions, unchunked, rtol=0, atol=0)

    latent = initial + 0.125
    expected = []
    for _ in range(5):
        latent = latent @ model.kmat.detach().T
        expected.append(1.25 * latent - 0.05)
    torch.testing.assert_close(
        predictions, torch.stack(expected, dim=1), rtol=0, atol=0
    )


def test_divisible_period_does_not_encode_final_unused_boundary() -> None:
    model = CountingKoopman()
    predictions = segmented_rollout(
        model, torch.ones(3, 2, dtype=torch.float32), horizon=4, period=2
    )
    assert len(model.encoded_inputs) == 2  # initial and h=2, never final h=4
    torch.testing.assert_close(
        model.encoded_inputs[1], predictions[:, 1], rtol=0, atol=0
    )


def test_decode_memory_chunks_do_not_create_refresh_boundaries() -> None:
    initial = torch.tensor([[0.6, -0.3]], dtype=torch.float32)
    chunked_model = CountingKoopman()
    chunked = segmented_rollout(
        chunked_model,
        initial,
        horizon=5,
        period=4,
        max_decode_segment=2,
    )
    assert len(chunked_model.encoded_inputs) == 2  # initial and true h=4 boundary
    torch.testing.assert_close(
        chunked_model.encoded_inputs[1], chunked[:, 3], rtol=0, atol=0
    )

    period_sized_model = CountingKoopman()
    period_sized_model.load_state_dict(chunked_model.state_dict())
    period_sized = segmented_rollout(
        period_sized_model,
        initial,
        horizon=5,
        period=4,
        max_decode_segment=4,
    )
    torch.testing.assert_close(chunked, period_sized, rtol=0, atol=0)


def test_streaming_packed_matches_dataset_by_dataset_scoring() -> None:
    fields = _synthetic_fields(datasets=2, trajectories=5, horizon=5)
    packed_model = CountingKoopman()
    packed = evaluate_model_packed(
        packed_model,
        fields,
        horizon=5,
        period=2,
        batch_size=10,
    )
    # Both datasets share one rollout batch: initial plus the h=2 and h=4
    # predicted-boundary encodes.
    assert len(packed_model.encoded_inputs) == 3

    sequential = []
    sequential_model = CountingKoopman()
    sequential_model.load_state_dict(packed_model.state_dict())
    for dataset_index in range(fields.shape[0]):
        record = evaluate_model_packed(
            sequential_model,
            fields[dataset_index : dataset_index + 1],
            horizon=5,
            period=2,
            batch_size=5,
        )[0]
        record["dataset_index"] = dataset_index
        sequential.append(record)
    assert len(sequential_model.encoded_inputs) == 2 * 3
    assert packed == sequential
    assert all(record["rollout_mode"] == "periodic_reencode" for record in packed)
    assert all(record["period"] == 2 for record in packed)
    for record in packed:
        for key in (
            "instantaneous_field_sse",
            "instantaneous_field_mse",
            "cumulative_field_mse",
        ):
            assert len(record[key]) == 5
            assert torch.as_tensor(record[key]).isfinite().all()

    direct = evaluate_model_packed(
        CountingKoopman(), fields[:1], horizon=5, period=None, batch_size=5
    )[0]
    assert direct["rollout_mode"] == DIRECT_MODE
    assert direct["period"] is None


@pytest.mark.parametrize(
    "candidates",
    [(), (0,), (-1,), (True,), (2, 2), (1, 6)],
)
def test_period_candidate_validation_fails_closed(candidates: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        validate_period_candidates(candidates, horizon=5)  # type: ignore[arg-type]


def test_period_candidates_and_full_finiteness_are_strict() -> None:
    assert validate_period_candidates((1, 2, 5), horizon=5) == (1, 2, 5)
    with pytest.raises(ValueError, match="period must be a positive integer"):
        segmented_rollout(
            CountingKoopman(),
            torch.ones(1, 2, dtype=torch.float32),
            horizon=3,
            period=0,
        )
    with pytest.raises(
        ValueError, match="max_decode_segment must be a positive integer"
    ):
        segmented_rollout(
            CountingKoopman(),
            torch.ones(1, 2, dtype=torch.float32),
            horizon=3,
            max_decode_segment=0,
        )
    fields = _synthetic_fields(datasets=1, trajectories=2, horizon=3)
    fields[0, 0, 3, 1] = torch.nan
    model = CountingKoopman()
    with pytest.raises(FloatingPointError, match="packed fields"):
        evaluate_model_packed(model, fields, period=1, batch_size=2)
    assert not model.encoded_inputs
