"""Stage-one, route-fit, and stage-two loops for the support-routed run."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from skae.data import VectorWrapper
from experiments.neurips_2026.local_operators.artifact_io import (
    _log_phase,
    _save_checkpoint,
    _write_json,
)
from skae.support.local_operator import (
    SourceTargetLocalMapBundle,
    _encode_sequence_batches,
    _freeze_autoencoder,
    _local_train_step,
    _make_wrapped_model,
    _target_centers_from_global,
)
from experiments.neurips_2026.local_operators.protocol import (
    STAGE1_TRAINING_STEPS,
    STAGE2_SELECTION_CANDIDATE_STEPS,
    TOTAL_TRAINING_STEPS,
    _quick_eval_best_periodic_horizon_mse,
    _strictly_improves,
    _validate_frozen_fabs_artifact,
)
from skae.training import MetricsLogger, generate_sequence_batch_for_device, train_step
from skae.support.routing import (
    FAMILY_JACCARD_THRESHOLD,
    FIT_CONFIGURED_ROWS,
    FIT_STATES,
    MIN_FAMILY_TRANSITIONS,
    _build_route_codebook,
    _generate_source_route_fit_batches,
)


def _run_stage_one(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_env: VectorWrapper,
    rngs: list[torch.Generator],
    cfg: object,
    start_step: int,
    logger: MetricsLogger,
    run_dir: Path,
    last_path: Path,
    save_last_checkpoint: bool,
    device: str,
) -> Dict[str, float]:
    last_metrics: Dict[str, float] = {}
    _log_phase(
        run_dir,
        "stage1_start",
        device=device,
        start_step=start_step,
        end_step=STAGE1_TRAINING_STEPS,
    )
    for step in range(start_step, STAGE1_TRAINING_STEPS):
        sequence = generate_sequence_batch_for_device(
            train_env,
            rngs[step % len(rngs)],
            window_length=cfg.TRAIN.SEQUENCE_LENGTH,
            device=device,
        )
        last_metrics = train_step(model, optimizer, sequence, step=step)
        logger.log_dict(last_metrics, step, prefix="stage1_train")
        if step % 100 == 0:
            print(
                f"Stage 1 step {step}/{STAGE1_TRAINING_STEPS} "
                f"loss={last_metrics['loss']:.6g}",
                flush=True,
            )
        if save_last_checkpoint and (
            (step > 0 and step % 500 == 0)
            or step == STAGE1_TRAINING_STEPS - 1
        ):
            _save_checkpoint(
                last_path,
                stage="stage1_joint",
                next_step=step + 1,
                model=model,
                optimizer=optimizer,
                bundle=None,
                local_optimizer=None,
                best_eval_final_error=float("inf"),
                metrics=last_metrics,
                cfg=cfg,
                training_generators=rngs,
                include_optimizer_state=True,
            )
    _log_phase(run_dir, "stage1_end", device=device, next_step=STAGE1_TRAINING_STEPS)
    return last_metrics


def _construct_route_bundle(
    *,
    model: torch.nn.Module,
    train_env: VectorWrapper,
    artifact: Optional[Dict[str, object]],
    fit_seed: int,
    scheme: str,
    support_value: float,
    device: str,
) -> Tuple[Dict[str, object], Dict[object, np.ndarray], SourceTargetLocalMapBundle]:
    global_k = model.kmatrix().detach().cpu().numpy().astype(np.float32, copy=False)
    if artifact is None:
        fit_batches = _generate_source_route_fit_batches(train_env, seed=fit_seed)
        fit_latents = _encode_sequence_batches(model, fit_batches, device)
        if fit_latents.shape[:2] != (FIT_CONFIGURED_ROWS, FIT_STATES):
            raise RuntimeError(
                "Route fit must contain 512 configured rows and 193 states; "
                f"got {fit_latents.shape[:2]}."
            )
        route_codebook = _build_route_codebook(
            fit_latents,
            scheme=scheme,
            value=support_value,
            min_operator_transitions=MIN_FAMILY_TRANSITIONS,
            family_jaccard_threshold=FAMILY_JACCARD_THRESHOLD,
        )
        target_centers = _target_centers_from_global(
            route_codebook["centers"],
            route_codebook["fitted_family_ids"],
            global_k,
        )
    else:
        route_codebook = artifact["route_codebook"]
        _validate_frozen_fabs_artifact(
            route_codebook,
            dict(artifact.get("route_metadata", {})),
            expected_fit_seed=fit_seed,
        )
        target_centers = artifact.get("target_centers")
        if target_centers is None:
            target_centers = _target_centers_from_global(
                route_codebook["centers"],
                route_codebook["fitted_family_ids"],
                global_k,
            )
    if not route_codebook["fitted_family_ids"]:
        raise RuntimeError("The route fit produced no support families.")
    bundle = SourceTargetLocalMapBundle(
        family_ids=route_codebook["fitted_family_ids"],
        source_centers=route_codebook["centers"],
        target_centers=target_centers,
        global_k=global_k,
        device=device,
        learn_target_centers=True,
    ).to(device)
    if artifact is not None and artifact.get("local_bundle_state_dict") is not None:
        bundle.load_state_dict(artifact["local_bundle_state_dict"])
    return route_codebook, target_centers, bundle


def _run_stage_two(
    *,
    model: torch.nn.Module,
    bundle: SourceTargetLocalMapBundle,
    route_codebook: Dict[str, object],
    route_metadata: Dict[str, object],
    target_centers: Dict[object, np.ndarray],
    base_env: object,
    train_env: VectorWrapper,
    selection_env: VectorWrapper,
    selection_starts: torch.Tensor,
    rngs: list[torch.Generator],
    cfg: object,
    start_step: int,
    resume_payload: Optional[Dict[str, object]],
    logger: MetricsLogger,
    run_dir: Path,
    checkpoint_path: Path,
    last_path: Path,
    local_lr: float,
    save_last_checkpoint: bool,
    scheme: str,
    support_value: float,
    device: str,
) -> Dict[str, float]:
    _freeze_autoencoder(model)
    optimizer = torch.optim.AdamW(bundle.parameters(), lr=local_lr, weight_decay=0.0)
    if resume_payload is not None and start_step >= STAGE1_TRAINING_STEPS:
        if resume_payload.get("local_bundle_state_dict") is not None:
            bundle.load_state_dict(resume_payload["local_bundle_state_dict"])
        if resume_payload.get("local_optimizer_state_dict") is not None:
            optimizer.load_state_dict(resume_payload["local_optimizer_state_dict"])
    best_score = float(
        resume_payload.get("best_eval_final_error", float("inf"))
        if resume_payload
        else float("inf")
    )
    family_cache: Dict[object, object] = {}
    last_metrics: Dict[str, float] = {}
    route_counts = {str(family_id): 0 for family_id in bundle.family_ids}
    total_count = 0
    fallback_count = 0
    counts_path = run_dir / "stage2_route_counts.json"
    _log_phase(
        run_dir,
        "stage2_start",
        device=device,
        start_step=max(start_step, STAGE1_TRAINING_STEPS),
        end_step=TOTAL_TRAINING_STEPS,
    )
    for step in range(max(start_step, STAGE1_TRAINING_STEPS), TOTAL_TRAINING_STEPS):
        sequence = generate_sequence_batch_for_device(
            train_env,
            rngs[step % len(rngs)],
            window_length=cfg.TRAIN.SEQUENCE_LENGTH,
            device=device,
        )
        last_metrics = _local_train_step(
            model=model,
            bundle=bundle,
            route_codebook=route_codebook,
            route_env=base_env,
            x_seq=sequence,
            scheme=scheme,
            support_value=support_value,
            family_jaccard_threshold=FAMILY_JACCARD_THRESHOLD,
            optimizer=optimizer,
            family_cache=family_cache,
            step=step,
        )
        logger.log_dict(last_metrics, step, prefix="stage2_local_train")
        total_count += int(last_metrics.get("route_total_count", 0))
        fallback_count += int(last_metrics.get("route_fallback_count", 0))
        for family_id in bundle.family_ids:
            route_counts[family_id] += int(
                last_metrics.get(f"route_family_{family_id}_count", 0)
            )
        if step % 100 == 0:
            print(
                f"Stage 2 step {step}/{TOTAL_TRAINING_STEPS} "
                f"loss={last_metrics['loss']:.6g} "
                f"coverage={last_metrics['route_coverage']:.3f}",
                flush=True,
            )
        if step not in STAGE2_SELECTION_CANDIDATE_STEPS:
            continue
        wrapped = _make_wrapped_model(
            model,
            bundle,
            route_codebook,
            route_env=base_env,
            scheme=scheme,
            support_value=support_value,
            family_jaccard_threshold=FAMILY_JACCARD_THRESHOLD,
        )
        score, best_by_horizon = _quick_eval_best_periodic_horizon_mse(
            wrapped,
            val_x=selection_starts,
            eval_env=selection_env,
        )
        logger.log_scalar("stage2_eval/best_periodic_horizon_mse", score, step)
        details = []
        for horizon, (value, period) in best_by_horizon.items():
            logger.log_scalar(f"stage2_eval/h{horizon}_best_periodic_mse", value, step)
            logger.log_scalar(
                f"stage2_eval/h{horizon}_best_periodic_period", float(period), step
            )
            details.append(f"H{horizon}={value:.6g}@{period}")
        print(f"  Selector: {score:.6g} ({', '.join(details)})", flush=True)
        improved = _strictly_improves(score, best_score)
        if improved:
            best_score = score
        counts = {
            "stage2_start_step": STAGE1_TRAINING_STEPS,
            "last_recorded_step": step,
            "route_total_count": total_count,
            "route_fallback_count": fallback_count,
            "route_counts_by_family": route_counts,
            "family_ids": list(bundle.family_ids),
        }
        _write_json(counts_path, counts)
        if save_last_checkpoint:
            _save_checkpoint(
                last_path,
                stage="stage2_local",
                next_step=step + 1,
                model=model,
                optimizer=None,
                bundle=bundle,
                local_optimizer=optimizer,
                best_eval_final_error=best_score,
                metrics=last_metrics,
                cfg=cfg,
                route_metadata=route_metadata,
                route_codebook=route_codebook,
                target_centers=target_centers,
                training_generators=rngs,
                include_optimizer_state=True,
            )
        if improved:
            _save_checkpoint(
                checkpoint_path,
                stage="stage2_local",
                next_step=step + 1,
                model=model,
                optimizer=None,
                bundle=bundle,
                local_optimizer=optimizer,
                best_eval_final_error=best_score,
                metrics=last_metrics,
                cfg=cfg,
                route_metadata=route_metadata,
                route_codebook=route_codebook,
                target_centers=target_centers,
                training_generators=rngs,
                include_optimizer_state=True,
            )
            counts["best_recorded_step"] = step
            _write_json(run_dir / "stage2_route_counts_best.json", counts)
    _log_phase(run_dir, "stage2_end", device=device, next_step=TOTAL_TRAINING_STEPS)
    if not checkpoint_path.is_file():
        raise RuntimeError("No finite positive staged checkpoint-selection score was produced.")
    return last_metrics
