import json

import pytest
import torch

from experiments.neurips_2026.allen_cahn_lista_refinement_stable.validate_smoke import (
    validate_run,
)


def _write_run(tmp_path, *, loss=1.0, sparsity=0.2, refinements=2):
    run_dir = tmp_path / "model"
    run_dir.mkdir()
    (run_dir / "training_args.json").write_text(json.dumps({
        "freeze_lista_s_pretrain": True,
        "lista_s_lr": 1e-5,
        "lista_num_loops": refinements,
        "pretrain_steps": 0,
        "warm_start_pretrain_steps": 2000,
    }))
    (run_dir / "metrics_history.jsonl").write_text(json.dumps({
        "step": 499,
        "loss": loss,
        "latent_loss": loss,
        "sparsity_ratio_1e-4": sparsity,
        "val_mse": 0.4,
        "checkpoint_score": 0.8,
        "lista_s_gradient_norm": 0.1 if refinements else 0.0,
        "lista_s_update_norm": 0.01 if refinements else 0.0,
    }) + "\n")
    with (run_dir / "metrics_history.jsonl").open("a") as handle:
        handle.write(json.dumps({"step": -1, "checkpoint_score": 0.9}) + "\n")
    torch.save({
        "model_config": {"lista_loops": refinements},
        "model_state_dict": {
            "lista_s.weight": (
                torch.zeros(2, 2)
                if refinements == 0
                else torch.ones(2, 2) * 1e-3
            )
        },
        "training_generator_state": torch.arange(8, dtype=torch.uint8),
    }, run_dir / "last.pt")
    (run_dir / "warm_start_receipt.json").write_text(json.dumps({
        "branch_lista_refinements": refinements,
        "source_checkpoint_sha256": "checkpoint",
        "source_model_state_sha256": "model",
        "source_training_generator_sha256": "rng",
    }))
    (run_dir / "training_summary.json").write_text(json.dumps({
        "first_forecast_batch_sha256": "batch",
    }))
    return run_dir


def test_stability_gate_accepts_finite_sparse_run(tmp_path):
    row = validate_run(_write_run(tmp_path))
    assert row["max_forecast_step"] == 499
    assert len(row["final_training_generator_sha256"]) == 64


@pytest.mark.parametrize("loss,sparsity", [(101.0, 0.2), (1.0, 0.001)])
def test_stability_gate_rejects_divergence_or_collapse(tmp_path, loss, sparsity):
    with pytest.raises(ValueError):
        validate_run(_write_run(tmp_path, loss=loss, sparsity=sparsity))
