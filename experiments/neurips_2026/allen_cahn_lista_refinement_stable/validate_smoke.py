"""Fail closed on unstable learned-LISTA recurrence smoke runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def _sha256_tensor(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def validate_run(run_dir: Path) -> dict[str, float | int | str]:
    args = json.loads((run_dir / "training_args.json").read_text())
    if not bool(args.get("freeze_lista_s_pretrain")):
        raise ValueError(f"{run_dir}: S freeze provenance is missing")
    if int(args.get("pretrain_steps", -1)) != 0:
        raise ValueError(f"{run_dir}: branch reran pretraining")
    if int(args.get("warm_start_pretrain_steps", -1)) != 2000:
        raise ValueError(f"{run_dir}: wrong shared pretraining budget")
    if not math.isclose(float(args.get("lista_s_lr", -1)), 1e-5):
        raise ValueError(f"{run_dir}: unexpected lista_s_lr")

    rows = [
        json.loads(line)
        for line in (run_dir / "metrics_history.jsonl").read_text().splitlines()
        if line.strip()
    ]
    forecast = [row for row in rows if int(row.get("step", -1)) >= 0]
    if not forecast or max(int(row["step"]) for row in forecast) < 499:
        raise ValueError(f"{run_dir}: smoke did not reach forecast step 500")

    required = ("loss", "latent_loss", "sparsity_ratio_1e-4", "val_mse")
    for row in forecast:
        for key in required:
            value = float(row[key])
            if not math.isfinite(value):
                raise ValueError(f"{run_dir}: nonfinite {key} at step {row['step']}")

    max_loss = max(float(row["loss"]) for row in forecast)
    max_latent = max(float(row["latent_loss"]) for row in forecast)
    min_sparsity = min(float(row["sparsity_ratio_1e-4"]) for row in forecast)
    if max_loss > 100.0:
        raise ValueError(f"{run_dir}: unstable total loss {max_loss}")
    if max_latent > 1000.0:
        raise ValueError(f"{run_dir}: unstable latent loss {max_latent}")
    if min_sparsity < 0.01:
        raise ValueError(f"{run_dir}: sparse-code collapse {min_sparsity}")

    try:
        checkpoint = torch.load(
            run_dir / "last.pt", map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(run_dir / "last.pt", map_location="cpu")
    s_weight = checkpoint["model_state_dict"]["lista_s.weight"]
    s_fro = float(torch.linalg.vector_norm(s_weight).item())
    final_generator_state = checkpoint.get("training_generator_state")
    if not isinstance(final_generator_state, torch.Tensor):
        raise ValueError(f"{run_dir}: last checkpoint lacks training generator state")
    final_generator_sha256 = _sha256_tensor(final_generator_state)
    refinements = int(args["lista_num_loops"])
    if refinements == 0 and s_fro != 0.0:
        raise ValueError(f"{run_dir}: Q=0 control changed inert S ({s_fro})")
    if refinements > 0 and s_fro <= 1e-6:
        raise ValueError(f"{run_dir}: learned refinements left S inert ({s_fro})")
    max_gradient = max(float(row["lista_s_gradient_norm"]) for row in forecast)
    max_update = max(float(row["lista_s_update_norm"]) for row in forecast)
    if refinements == 0 and (max_gradient != 0.0 or max_update != 0.0):
        raise ValueError(f"{run_dir}: inert Q=0 routed S gradients or updates")
    if refinements > 0 and (max_gradient <= 0.0 or max_update <= 0.0):
        raise ValueError(f"{run_dir}: learned refinements did not train S")

    receipt = json.loads((run_dir / "warm_start_receipt.json").read_text())
    summary = json.loads((run_dir / "training_summary.json").read_text())
    if int(receipt["branch_lista_refinements"]) != refinements:
        raise ValueError(f"{run_dir}: warm-start receipt has wrong depth")
    if int(checkpoint["model_config"]["lista_loops"]) != refinements:
        raise ValueError(f"{run_dir}: last checkpoint resolves wrong depth")

    pretrain = [row for row in rows if int(row.get("step", -2)) == -1]
    if len(pretrain) != 1:
        raise ValueError(f"{run_dir}: missing unique pretrain checkpoint row")
    pretrain_score = float(pretrain[0]["checkpoint_score"])
    best_forecast_score = min(float(row["checkpoint_score"]) for row in forecast)
    if refinements > 0 and not best_forecast_score < pretrain_score:
        raise ValueError(f"{run_dir}: no forecast checkpoint improved over pretrain")

    return {
        "run_dir": str(run_dir),
        "refinements": refinements,
        "max_forecast_step": max(int(row["step"]) for row in forecast),
        "max_total_loss": max_loss,
        "max_latent_loss": max_latent,
        "min_sparsity_ratio_1e-4": min_sparsity,
        "last_val_mse": float(forecast[-1]["val_mse"]),
        "lista_s_frobenius_norm": s_fro,
        "max_lista_s_gradient_norm": max_gradient,
        "max_lista_s_update_norm": max_update,
        "source_checkpoint_sha256": receipt["source_checkpoint_sha256"],
        "source_model_state_sha256": receipt["source_model_state_sha256"],
        "source_training_generator_sha256": receipt[
            "source_training_generator_sha256"
        ],
        "first_forecast_batch_sha256": summary["first_forecast_batch_sha256"],
        "final_training_generator_sha256": final_generator_sha256,
        "pretrain_checkpoint_score": pretrain_score,
        "best_forecast_checkpoint_score": best_forecast_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run_dir) != 3:
        raise ValueError("exactly three smoke run directories are required")
    rows = [validate_run(path) for path in args.run_dir]
    if sorted(int(row["refinements"]) for row in rows) != [0, 2, 3]:
        raise ValueError("smoke must cover refinement depths 0, 2, and 3")
    for key in (
        "source_checkpoint_sha256",
        "source_model_state_sha256",
        "source_training_generator_sha256",
        "first_forecast_batch_sha256",
        "final_training_generator_sha256",
    ):
        if len({row[key] for row in rows}) != 1:
            raise ValueError(f"smoke branches do not share identical {key}")
    payload = {
        "schema_version": 1,
        "status": "passed_recurrence_stability_gate",
        "selected_lista_s_lr": 1e-5,
        "freeze_lista_s_pretrain": True,
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
