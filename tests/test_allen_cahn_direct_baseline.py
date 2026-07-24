from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from copy import deepcopy

import pytest
import torch

from experiments.neurips_2026.allen_cahn_direct_baseline.core import (
    DirectConfig,
    DirectResidualConv,
    joint_endpoint_metrics,
    parameter_count,
    sha256_path,
)
from experiments.neurips_2026.allen_cahn_direct_baseline.summarize import (
    EXPECTED_CANDIDATES,
    authenticate_seed,
    exact_one_sided_sign_flip,
    validate_curves,
)
from experiments.neurips_2026.allen_cahn_direct_baseline.telemetry import (
    main as telemetry_main,
)
from experiments.neurips_2026.allen_cahn_direct_baseline.execution import (
    CudaGraphTrainingStepper,
    eager_training_step,
)


HISTORICAL_MODEL = Path(
    "/network/scratch/l/lia/skae-rebuttal/skae/benchmarks/"
    "spatialized_conv_koopman.py"
)
HISTORICAL_MODEL_SHA256 = (
    "4f71c674d1e82174f437365c728c8e3684617364b979d624ed35fbbba0371f3b"
)


def test_exact_parameter_budget_and_no_zero_inducing_activation() -> None:
    model = DirectResidualConv(DirectConfig())
    assert parameter_count(model) == 8_541_178
    forbidden = (
        torch.nn.ReLU,
        torch.nn.GELU,
        torch.nn.Softshrink,
        torch.nn.Dropout,
    )
    assert not any(isinstance(module, forbidden) for module in model.modules())
    assert sum(isinstance(module, torch.nn.Tanh) for module in model.modules()) == 9


def test_zero_initialized_direct_model_is_exact_persistence() -> None:
    model = DirectResidualConv(DirectConfig(hidden_channels=8, num_blocks=1))
    initial = torch.randn(3, 512)
    prediction = model.rollout(initial, horizon=5)
    assert torch.equal(prediction, initial[:, None].expand(-1, 5, -1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph needs a GPU")
def test_cuda_graph_updates_match_eager_adam() -> None:
    device = torch.device("cuda")
    torch.manual_seed(20260721)
    eager_model = DirectResidualConv(
        DirectConfig(hidden_channels=8, num_blocks=1)
    ).to(device)
    graph_model = deepcopy(eager_model)
    eager_optimizer = torch.optim.Adam(
        eager_model.parameters(), lr=3e-4, capturable=True
    )
    graph_optimizer = torch.optim.Adam(
        graph_model.parameters(), lr=3e-4, capturable=True
    )
    sequences = [torch.randn(2, 4, 512, device=device) for _ in range(3)]
    eager_training_step(
        eager_model, eager_optimizer, sequences[0], gradient_weight=0.05
    )
    eager_training_step(
        graph_model, graph_optimizer, sequences[0], gradient_weight=0.05
    )
    eager_second = eager_training_step(
        eager_model, eager_optimizer, sequences[1], gradient_weight=0.05
    )
    graph_stepper = CudaGraphTrainingStepper(
        graph_model, graph_optimizer, sequences[1], gradient_weight=0.05
    )
    assert graph_stepper.last_metrics == pytest.approx(eager_second)
    for eager_parameter, graph_parameter in zip(
        eager_model.parameters(), graph_model.parameters()
    ):
        torch.testing.assert_close(eager_parameter, graph_parameter)
    eager_third = eager_training_step(
        eager_model, eager_optimizer, sequences[2], gradient_weight=0.05
    )
    graph_third = graph_stepper.step(sequences[2])
    assert graph_third == pytest.approx(eager_third)
    for eager_parameter, graph_parameter in zip(
        eager_model.parameters(), graph_model.parameters()
    ):
        torch.testing.assert_close(eager_parameter, graph_parameter)
        eager_state = eager_optimizer.state[eager_parameter]
        graph_state = graph_optimizer.state[graph_parameter]
        assert eager_state.keys() == graph_state.keys()
        for key in eager_state:
            torch.testing.assert_close(eager_state[key], graph_state[key])


def test_local_port_matches_historical_direct_forward_path() -> None:
    assert sha256_path(HISTORICAL_MODEL) == HISTORICAL_MODEL_SHA256
    spec = importlib.util.spec_from_file_location("_historical_direct", HISTORICAL_MODEL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    local_config = DirectConfig(hidden_channels=8, num_blocks=1)
    historical_config = module.SpatialConvAutoregressiveConfig(
        grid_size=16,
        channels=2,
        hidden_channels=8,
        num_blocks=1,
        activation="tanh",
        padding_mode="circular",
        residual_scale=0.1,
    )
    torch.manual_seed(123)
    local = DirectResidualConv(local_config)
    torch.manual_seed(123)
    historical = module.SpatialConvAutoregressive(historical_config)
    assert local.state_dict().keys() == historical.state_dict().keys()
    for key, value in local.state_dict().items():
        assert torch.equal(value, historical.state_dict()[key])
    inputs = torch.randn(2, 512)
    assert torch.equal(
        local.rollout(inputs, horizon=3),
        historical.rollout_observation_discrete(inputs, horizon=3)[1],
    )


def test_joint_selector_is_one_for_exact_persistence() -> None:
    class Persistence(torch.nn.Module):
        def rollout(self, initial: torch.Tensor, *, horizon: int) -> torch.Tensor:
            return initial[:, None].expand(-1, horizon, -1)

        def eval(self) -> "Persistence":
            return self

    model = Persistence()
    fields = torch.randn(4, 201, 512)
    endpoints, score = joint_endpoint_metrics(
        model,
        fields,
        horizons=(160, 200),
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert score == pytest.approx(1.0, abs=1e-7)
    assert set(endpoints) == {"160", "200"}


def test_exact_sign_flip_is_literal_and_directional() -> None:
    import numpy as np

    positive = np.arange(1.0, 11.0)
    assert exact_one_sided_sign_flip(positive) == pytest.approx(1 / 1024)
    assert exact_one_sided_sign_flip(-positive) == pytest.approx(1.0)


def test_exact_historical_selector_cadence_is_locked() -> None:
    assert EXPECTED_CANDIDATES == [2000] + list(range(2251, 5252, 250)) + [5500]
    assert len(EXPECTED_CANDIDATES) == 15


def test_full_launcher_obeys_frozen_authorization_gate() -> None:
    repository = Path(__file__).resolve().parents[1]
    lock = json.loads(
        (repository / "experiments/neurips_2026/allen_cahn_direct_baseline/task_lock.json")
        .read_text(encoding="utf-8")
    )
    launcher = (
        repository / "scripts/neurips_2026/allen_cahn_direct_baseline/queue.sh"
    ).read_text(encoding="utf-8")
    assert lock["source_locked_command_graph"]["full_launch_authorized"] is False
    assert lock["telemetry"]["smoke_optimizer_updates"] == 80
    assert lock["telemetry"]["smoke_length_or_utilization_amendments_remaining"] == 0
    assert "FULL_LAUNCH_AUTHORIZED" in launcher
    assert '!= "true"' in launcher
    smoke = (
        repository / "scripts/neurips_2026/allen_cahn_direct_baseline/run_smoke.sh"
    ).read_text(encoding="utf-8")
    assert "--smoke-steps 80" in smoke


def test_seed_authentication_requires_complete_artifact_chain(tmp_path: Path) -> None:
    training = tmp_path / "training"
    evaluation = tmp_path / "evaluation"
    (training / "seed_64" / "model").mkdir(parents=True)
    (evaluation / "seed_64").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Missing seed-64 artifacts"):
        authenticate_seed(
            seed=64,
            training_root=training,
            evaluation_root=evaluation,
            lock={},
            task_lock_sha256="f" * 64,
        )


def _valid_curves() -> dict[str, object]:
    import numpy as np

    instant = np.linspace(0.001, 0.2, 200)
    persistence = np.linspace(0.01, 0.4, 200)
    cumulative = np.cumsum(instant) / np.arange(1.0, 201.0)
    persistence_cumulative = np.cumsum(persistence) / np.arange(1.0, 201.0)
    return {
        "trajectories": 256,
        "horizon_steps": 200,
        "instantaneous_field_mse": instant.tolist(),
        "through_horizon_field_mse": cumulative.tolist(),
        "persistence_instantaneous_field_mse": persistence.tolist(),
        "persistence_through_horizon_field_mse": persistence_cumulative.tolist(),
        "endpoints": {
            str(horizon): {
                "through_horizon_field_mse": float(cumulative[horizon - 1]),
                "terminal_field_mse": float(instant[horizon - 1]),
                "persistence_through_horizon_field_mse": float(
                    persistence_cumulative[horizon - 1]
                ),
                "persistence_terminal_field_mse": float(
                    persistence[horizon - 1]
                ),
            }
            for horizon in (80, 120, 160, 200)
        },
    }


def test_curve_guard_rejects_truncation_and_endpoint_forgery() -> None:
    curves = _valid_curves()
    validate_curves(curves)
    truncated = json.loads(json.dumps(curves))
    truncated["instantaneous_field_mse"] = truncated[
        "instantaneous_field_mse"
    ][:-1]
    with pytest.raises(RuntimeError, match="Invalid evaluation curve"):
        validate_curves(truncated)
    forged = json.loads(json.dumps(curves))
    forged["endpoints"]["200"]["terminal_field_mse"] += 1.0
    with pytest.raises(RuntimeError, match="Endpoint/curve identity"):
        validate_curves(forged)


def _write_telemetry_fixture(root: Path, timestamps: list[float]) -> tuple[Path, Path, Path]:
    telemetry = root / "raw.csv"
    header = (
        "unix_time_seconds,gpu_index,gpu_uuid,gpu_name,utilization_gpu_percent,"
        "utilization_memory_percent,memory_used_mib,memory_total_mib,power_draw_w,"
        "power_limit_w\n"
    )
    rows = [f"{value},0,GPU-test,A100,95,80,30000,40960,300,400\n" for value in timestamps]
    telemetry.write_text(header + "".join(rows), encoding="utf-8")
    start = root / "start.json"
    end = root / "end.json"
    start.write_text(
        json.dumps({"phase": "optimizer_loop_start", "unix_time_seconds": 1.5}),
        encoding="utf-8",
    )
    end.write_text(
        json.dumps({"phase": "optimizer_loop_end", "unix_time_seconds": 5.5}),
        encoding="utf-8",
    )
    return telemetry, start, end


def test_telemetry_keeps_boundaries_and_rejects_sampling_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    valid_root = tmp_path / "valid"
    valid_root.mkdir()
    telemetry, start, end = _write_telemetry_fixture(
        valid_root, [float(value) for value in range(8)]
    )
    output = valid_root / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "telemetry",
            "--telemetry",
            str(telemetry),
            "--phase-start",
            str(start),
            "--phase-end",
            str(end),
            "--output",
            str(output),
            "--minimum-core-samples",
            "4",
            "--task-lock-sha256",
            "f" * 64,
            "--seed",
            "64",
            "--artifact-role",
            "scientific_training",
            "--slurm-job-id",
            "123",
        ],
    )
    telemetry_main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["startup_before_optimizer_loop"]["samples"] == 2
    assert payload["tail_after_optimizer_loop"]["samples"] == 2

    bad_root = tmp_path / "bad"
    bad_root.mkdir()
    telemetry, start, end = _write_telemetry_fixture(
        bad_root, [0.0, 1.0, 2.0, 3.0, 7.0, 8.0]
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "telemetry",
            "--telemetry",
            str(telemetry),
            "--phase-start",
            str(start),
            "--phase-end",
            str(end),
            "--output",
            str(bad_root / "audit.json"),
            "--minimum-core-samples",
            "2",
            "--task-lock-sha256",
            "f" * 64,
            "--seed",
            "64",
            "--artifact-role",
            "scientific_training",
            "--slurm-job-id",
            "123",
        ],
    )
    with pytest.raises(SystemExit):
        telemetry_main()
