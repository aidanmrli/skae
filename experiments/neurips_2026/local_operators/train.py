"""Train the frozen support-routed local affine model used by the paper.

Scientific constants, model mechanics, and artifact I/O live in focused
modules so this entry point reads as the experimental workflow.  Names imported
below are intentionally re-exported for frozen evaluation utilities.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch

from skae.data import VectorWrapper, make_env, wrap_training_env
from skae.evaluation import evaluate_model
from skae.model import make_model
from experiments.neurips_2026.local_operators.artifact_io import (
    _find_completed_run,
    _find_resume_run,
    _load_torch,
    _log_phase,
    _restore_training_rng_states,
    _save_checkpoint,
    _write_json,
    _write_torch,
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
    FINAL_EVALUATION_BATCH_SIZE,
    FINAL_EVALUATION_SEED_OFFSET,
    LOCAL_MAP_PARAMETERIZATION,
    PAPER_REENCODE_PERIODS,
    ROUTE_PROTOCOL,
    STAGE1_TRAINING_STEPS,
    STAGE2_SELECTION_BATCH_SIZE,
    STAGE2_SELECTION_CANDIDATE_STEPS,
    STAGE2_SELECTION_HORIZONS,
    STAGE2_SELECTION_SEED_OFFSET,
    STAGE2_TRAINING_STEPS,
    TARGET_CENTER_RULE,
    TOTAL_TRAINING_STEPS,
    _build_parser,
    _make_eval_settings,
    _make_stage2_selection_starts,
    _paper_route_metadata,
    _parse_args,
    _parse_int_csv,
    _quick_eval_best_periodic_horizon_mse,
    _strictly_improves,
    _support_definition,
    _validate_frozen_fabs_artifact,
)
from experiments.neurips_2026.local_operators.tasks import (
    _apply_task_row_to_config,
    _maybe_float,
    _read_task_row,
    _safe_str,
    _tagify,
)
from experiments.neurips_2026.local_operators.training import (
    _construct_route_bundle,
    _run_stage_one,
    _run_stage_two,
)
from experiments.neurips_2026.local_operators.contract import (
    FAMILY_JACCARD_THRESHOLD,
    FIT_CONFIGURED_ROWS,
    FIT_DUPLICATION_FACTOR,
    FIT_NUM_TRAJECTORIES,
    FIT_SEED_OFFSET,
    FIT_STATES,
    FIT_TRAJECTORY_LENGTH,
    FIT_TRANSITIONS,
    FIT_UNIQUE_TRAJECTORIES,
    SUPPORT_DEFINITION,
)
from experiments.neurips_2026.protocol import CONTROLLED_PAPER_PROTOCOL
from skae.training import (
    MetricsLogger,
    build_optimizer,
    get_device,
)
def main() -> None:
    args = _parse_args()
    row = _read_task_row(
        Path(args.task_tsv),
        array_index=args.array_index,
        array_offset=args.array_offset,
    )
    cfg = _apply_task_row_to_config(row)
    if int(cfg.TRAIN.NUM_STEPS) != TOTAL_TRAINING_STEPS:
        raise ValueError("The source protocol requires num_steps=200000.")
    if int(cfg.TRAIN.BATCH_SIZE) != FIT_UNIQUE_TRAJECTORIES:
        raise ValueError("The source protocol requires training batch_size=256.")
    scheme, support_value = _support_definition(SUPPORT_DEFINITION)
    device = get_device(args.device)
    phase = _safe_str(row.get("phase")) or CONTROLLED_PAPER_PROTOCOL.protocol_id
    variant = _safe_str(row.get("model_variant")) or "lista_fabs_local_k_staged"
    system_slug = _safe_str(row.get("system_slug")) or _safe_str(
        row.get("system_key")
    ).replace(":", "_")
    seed = int(cfg.SEED)
    dt = _maybe_float(row, "env_dt")
    seed_dir = (
        Path(args.base_out)
        / phase
        / variant
        / system_slug
        / f"dt_{_tagify(dt if dt is not None else 'default')}"
        / f"seed_{seed}"
    )
    seed_dir.mkdir(parents=True, exist_ok=True)
    completed = _find_completed_run(seed_dir)
    if args.skip_completed and completed is not None:
        print(f"Completed staged run already exists: {completed}", flush=True)
        return
    resume_run = _find_resume_run(seed_dir) if args.resume_from_latest else None
    run_dir = resume_run or seed_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(str(run_dir / "config.json"))
    fit_seed = seed + FIT_SEED_OFFSET
    _write_json(
        run_dir / "staged_local_k_config.json",
        {
            "task_tsv": str(args.task_tsv),
            "task_id": row.get("task_id"),
            "protocol": ROUTE_PROTOCOL,
            "training_steps": [STAGE1_TRAINING_STEPS, STAGE2_TRAINING_STEPS],
            "route_fit": {
                "support_definition": SUPPORT_DEFINITION,
                "family_jaccard_threshold": FAMILY_JACCARD_THRESHOLD,
                "configured_rows": FIT_CONFIGURED_ROWS,
                "unique_trajectories": FIT_UNIQUE_TRAJECTORIES,
                "duplication_factor": FIT_DUPLICATION_FACTOR,
                "transitions": FIT_TRANSITIONS,
                "states": FIT_STATES,
                "seed_offset": FIT_SEED_OFFSET,
            },
            "checkpoint_selection": {
                "candidate_steps": list(STAGE2_SELECTION_CANDIDATE_STEPS),
                "batch_size": STAGE2_SELECTION_BATCH_SIZE,
                "seed_offset": STAGE2_SELECTION_SEED_OFFSET,
                "horizons": list(STAGE2_SELECTION_HORIZONS),
                "periods": list(PAPER_REENCODE_PERIODS),
                "metric": "best_periodic_horizon_mse",
                "improvement": "strict_less_than",
            },
            "final_evaluation": {
                "batch_size": FINAL_EVALUATION_BATCH_SIZE,
                "seed_offset": FINAL_EVALUATION_SEED_OFFSET,
            },
            "local_map_parameterization": LOCAL_MAP_PARAMETERIZATION,
            "target_center_rule": TARGET_CENTER_RULE,
            "routing_cadence": "every_latent_transition_step",
            "reencoding_role": "periodic_decode_encode_refreshes_latent_before_next_route",
            "device": device,
        },
    )
    _log_phase(run_dir, "init", device=device, system=cfg.ENV.ENV_NAME, seed=seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    base_env = make_env(cfg)
    train_env = VectorWrapper(wrap_training_env(base_env, cfg), cfg.TRAIN.BATCH_SIZE)
    model = make_model(cfg, base_env.observation_size).to(device)
    stage1_optimizer = build_optimizer(model, cfg)
    logger = MetricsLogger(run_dir, save_history=bool(args.save_metrics_history))
    rngs = [
        torch.Generator().manual_seed(seed + index * cfg.TRAIN.BATCH_SIZE)
        for index in range(max(1, cfg.TRAIN.DATA_SIZE // cfg.TRAIN.BATCH_SIZE))
    ]
    last_path = run_dir / "last.pt"
    checkpoint_path = run_dir / "checkpoint.pt"
    resume_path = None
    if args.resume_from_latest:
        resume_path = last_path if last_path.is_file() else checkpoint_path if checkpoint_path.is_file() else None
    resume_payload = _load_torch(resume_path, map_location=device) if resume_path else None
    start_step = int(resume_payload.get("next_step", 0)) if resume_payload else 0
    if not 0 <= start_step <= TOTAL_TRAINING_STEPS:
        raise ValueError(f"Invalid checkpoint next_step={start_step}.")
    if resume_payload:
        model.load_state_dict(resume_payload["model_state_dict"])
        if start_step < STAGE1_TRAINING_STEPS and resume_payload.get("optimizer_state_dict"):
            stage1_optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
    if start_step < STAGE1_TRAINING_STEPS:
        if resume_payload and not _restore_training_rng_states(resume_payload, rngs):
            print(
                "Warning: legacy checkpoint has no RNG state; resumed stage one "
                "will replay its seeded data streams.",
                flush=True,
            )
        _run_stage_one(
            model=model,
            optimizer=stage1_optimizer,
            train_env=train_env,
            rngs=rngs,
            cfg=cfg,
            start_step=start_step,
            logger=logger,
            run_dir=run_dir,
            last_path=last_path,
            save_last_checkpoint=args.save_last_checkpoint,
            device=device,
        )

    artifact = resume_payload if resume_payload and resume_payload.get("route_codebook") else None
    artifact_path = run_dir / "stage2_artifacts.pt"
    if artifact is None and artifact_path.is_file():
        artifact = _load_torch(artifact_path, map_location=device)
    route_codebook, target_centers, bundle = _construct_route_bundle(
        model=model,
        train_env=train_env,
        artifact=artifact,
        fit_seed=fit_seed,
        scheme=scheme,
        support_value=support_value,
        device=device,
    )
    route_metadata = _paper_route_metadata(route_codebook, fit_seed=fit_seed)
    _write_json(run_dir / "route_codebook.json", route_metadata)
    if args.save_stage2_artifacts:
        _write_torch(
            artifact_path,
            {
                "route_codebook": route_codebook,
                "route_metadata": route_metadata,
                "target_centers": target_centers,
                "local_bundle_state_dict": bundle.state_dict(),
            },
        )
    selection_env, selection_starts = _make_stage2_selection_starts(
        base_env, seed=seed, device=device
    )
    if (
        resume_payload
        and start_step >= STAGE1_TRAINING_STEPS
        and not _restore_training_rng_states(resume_payload, rngs)
    ):
        print(
            "Warning: legacy checkpoint has no RNG state; resumed stage two "
            "will replay its seeded data streams.",
            flush=True,
        )
    last_metrics = _run_stage_two(
        model=model,
        bundle=bundle,
        route_codebook=route_codebook,
        route_metadata=route_metadata,
        target_centers=target_centers,
        base_env=base_env,
        train_env=train_env,
        selection_env=selection_env,
        selection_starts=selection_starts,
        rngs=rngs,
        cfg=cfg,
        start_step=start_step,
        resume_payload=resume_payload,
        logger=logger,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        last_path=last_path,
        local_lr=float(cfg.TRAIN.K_MATRIX_LR),
        save_last_checkpoint=args.save_last_checkpoint,
        scheme=scheme,
        support_value=support_value,
        device=device,
    )
    logger.close()
    _write_json(run_dir / "final_metrics.json", last_metrics)
    best_payload = _load_torch(checkpoint_path, map_location=device)
    model.load_state_dict(best_payload["model_state_dict"])
    bundle.load_state_dict(best_payload["local_bundle_state_dict"])
    wrapped = _make_wrapped_model(
        model,
        bundle,
        route_codebook,
        route_env=base_env,
        scheme=scheme,
        support_value=support_value,
        family_jaccard_threshold=FAMILY_JACCARD_THRESHOLD,
    )
    settings = _make_eval_settings(
        args.eval_profile,
        cfg,
        save_rollout_artifacts=args.save_eval_rollout_artifacts,
        save_plots=args.save_eval_plots,
        include_per_ic_values=args.save_eval_per_ic_values,
        include_error_curves=args.save_eval_error_curves,
    )
    results = evaluate_model(
        model=wrapped,
        cfg=cfg,
        device=device,
        settings=settings,
        output_dir=run_dir / "evaluation_best",
    )
    _write_json(run_dir / "evaluation_results_best.json", results)
    _write_json(
        run_dir / "evaluation_summary.json",
        {"best_checkpoint": True, "staged_local_k": True, **route_metadata},
    )
    print(f"Training complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
