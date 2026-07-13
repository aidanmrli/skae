"""Top-level benchmark runner for Lorenz-96, PDEBench SWE, and Silverbox."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from skae.config import get_config
from skae.model import make_model

from .baselines import fit_dmd, fit_truncated_dmd, persistence_rollout, select_arx
from .data import (
    TrajectoryDataset,
    build_lorenz96_dataset,
    compute_normalization,
    generate_pde_smoke_fixture,
    load_or_fixture_silverbox,
    load_pdebench_h5,
    pde_fields_to_dataset,
    write_json,
)
from .metrics import fit_percent, metric_rows_for_rollout, summarize_rows
from .models import (
    ConvKoopmanAE,
    ConvKoopmanConfig,
    ControlledKoopmanAE,
    count_parameters,
    koopman_diagnostics,
)


def parse_config(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
    except Exception:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config {path} must parse to a mapping.")
    return loaded


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def set_determinism(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def git_metadata() -> Dict[str, object]:
    def run(cmd: Sequence[str]) -> str:
        return subprocess.check_output(cmd, text=True).strip()

    try:
        commit = run(["git", "rev-parse", "HEAD"])
        status = run(["git", "status", "--short"])
    except Exception as exc:
        commit = "unknown"
        status = f"git metadata failed: {exc}"
    return {"commit": commit, "status_short": status}


def system_metadata() -> Dict[str, object]:
    cuda_version = torch.version.cuda if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": cuda_version,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu": platform.processor(),
    }


def sample_windows(data: np.ndarray, *, batch_size: int, horizon: int, rng: np.random.Generator) -> torch.Tensor:
    traj_count, time_count, dim = data.shape
    max_start = time_count - horizon - 1
    if max_start < 0:
        raise ValueError("Data too short for requested horizon.")
    traj_idx = rng.integers(0, traj_count, size=batch_size)
    start_idx = rng.integers(0, max_start + 1, size=batch_size)
    out = np.empty((batch_size, horizon + 1, dim), dtype=np.float32)
    for i, (traj, start) in enumerate(zip(traj_idx, start_idx)):
        out[i] = data[traj, start : start + horizon + 1]
    return torch.from_numpy(out)


def model_config_payload(label: str, cfg: Mapping[str, object], extra: Mapping[str, object]) -> Dict[str, object]:
    payload = {"label": label, **dict(extra)}
    payload["training"] = dict(cfg)
    return payload


def _apply_flat_model_variant(
    cfg: object,
    *,
    variant: Mapping[str, object],
    train_cfg: Mapping[str, object],
    latent_dim: int,
) -> None:
    cfg.MODEL.TARGET_SIZE = int(latent_dim)
    cfg.MODEL.ENCODER.LAYERS = [int(train_cfg.get("hidden_dim", 64))] * int(train_cfg.get("num_layers", 2))
    cfg.MODEL.DECODER.LAYERS = [int(train_cfg.get("hidden_dim", 64))] * max(0, int(train_cfg.get("decoder_layers", 1)))
    cfg.MODEL.ENCODER.ACTIVATION = str(variant.get("encoder_activation", train_cfg.get("encoder_activation", cfg.MODEL.ENCODER.ACTIVATION)))
    cfg.MODEL.DECODER.ACTIVATION = str(variant.get("decoder_activation", train_cfg.get("decoder_activation", cfg.MODEL.DECODER.ACTIVATION)))
    if "encoder_last_relu" in variant:
        cfg.MODEL.ENCODER.LAST_RELU = bool(variant["encoder_last_relu"])
    if "encoder_use_bias" in variant:
        cfg.MODEL.ENCODER.USE_BIAS = bool(variant["encoder_use_bias"])
    if "decoder_use_bias" in variant:
        cfg.MODEL.DECODER.USE_BIAS = bool(variant["decoder_use_bias"])
    if "decoder_affine_bias" in variant:
        cfg.MODEL.DECODER.AFFINE_BIAS = bool(variant["decoder_affine_bias"])
    if "use_homogeneous" in variant:
        cfg.MODEL.USE_HOMOGENEOUS = bool(variant["use_homogeneous"])
    if "k_structure" in variant:
        cfg.MODEL.K_STRUCTURE = str(variant["k_structure"])
    if "k_num_blocks" in variant:
        cfg.MODEL.K_NUM_BLOCKS = int(variant["k_num_blocks"])
    if "k_block_size" in variant:
        cfg.MODEL.K_BLOCK_SIZE = int(variant["k_block_size"])
    if "soft_block_weight" in variant:
        cfg.MODEL.SOFT_BLOCK.ENABLED = float(variant["soft_block_weight"]) > 0.0
        cfg.MODEL.SOFT_BLOCK.WEIGHT = float(variant["soft_block_weight"])
    if "soft_block_num_blocks" in variant:
        cfg.MODEL.SOFT_BLOCK.NUM_BLOCKS = int(variant["soft_block_num_blocks"])
    if "lista_final_op" in variant:
        cfg.MODEL.ENCODER.LISTA.FINAL_OP = str(variant["lista_final_op"])
    if "lista_alpha" in variant:
        cfg.MODEL.ENCODER.LISTA.ALPHA = float(variant["lista_alpha"])
    if "lista_l" in variant:
        cfg.MODEL.ENCODER.LISTA.L = float(variant["lista_l"])
    if "lista_num_loops" in variant:
        cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = int(variant["lista_num_loops"])
    if "lista_linear_encoder" in variant:
        cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER = bool(variant["lista_linear_encoder"])
    if "lista_precode_mode" in variant:
        cfg.MODEL.ENCODER.LISTA.PRECODE_MODE = str(variant["lista_precode_mode"])
    if "hyperlista_num_loops" in variant:
        cfg.MODEL.ENCODER.HYPERLISTA.NUM_LOOPS = int(variant["hyperlista_num_loops"])
    if "hyperlista_c_theta" in variant:
        cfg.MODEL.ENCODER.HYPERLISTA.C_THETA = float(variant["hyperlista_c_theta"])
    if "hyperlista_c_beta" in variant:
        cfg.MODEL.ENCODER.HYPERLISTA.C_BETA = float(variant["hyperlista_c_beta"])
    if "hyperlista_c_ss" in variant:
        cfg.MODEL.ENCODER.HYPERLISTA.C_SS = float(variant["hyperlista_c_ss"])


def train_flat_koopman(
    *,
    train_data: np.ndarray,
    val_data: np.ndarray,
    label: str,
    latent_dim: int,
    sparsity_coeff: float,
    train_cfg: Mapping[str, object],
    run_dir: Path,
    seed: int,
    device: str,
    variant: Mapping[str, object] | None = None,
) -> Tuple[torch.nn.Module, Dict[str, object], Path, Path]:
    set_determinism(seed)
    variant = dict(variant or {})
    preset = str(variant.get("preset", "generic_no_shrink"))
    cfg = get_config(preset)
    cfg.SEED = int(seed)
    _apply_flat_model_variant(cfg, variant=variant, train_cfg=train_cfg, latent_dim=latent_dim)
    model = make_model(cfg, train_data.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 3e-4)), weight_decay=float(train_cfg.get("weight_decay", 1e-6)))
    epochs = int(train_cfg.get("epochs", 5))
    patience = int(train_cfg.get("patience", 3))
    horizon = int(train_cfg.get("train_horizon", 10))
    batch_size = int(train_cfg.get("batch_size", 32))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    sparsity_target = str(train_cfg.get("sparsity_target", "k_l1")).lower()
    rng = np.random.default_rng(seed + 1000)
    best_val = float("inf")
    best_epoch = -1
    best_path = run_dir / "checkpoint.pt"
    final_path = run_dir / "final.pt"
    history: List[Dict[str, float]] = []
    start = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        batch = sample_windows(train_data, batch_size=batch_size, horizon=horizon, rng=rng).to(device)
        b, seq_len, obs_dim = batch.shape
        z_all = model.encode(batch.reshape(b * seq_len, obs_dim)).reshape(b, seq_len, -1)
        z_roll = model.rollout_latent_discrete(z_all[:, 0], horizon=horizon)
        pred = model.decode(z_roll.reshape(b * horizon, -1)).reshape(b, horizon, obs_dim)
        recon = model.decode(z_all.reshape(b * seq_len, -1)).reshape(b, seq_len, obs_dim)
        one = F.mse_loss(pred[:, 0], batch[:, 1])
        multi = F.mse_loss(pred, batch[:, 1:])
        recon_loss = F.mse_loss(recon, batch)
        k_l1 = model.kmatrix().abs().sum() / float(latent_dim * latent_dim)
        if sparsity_target in {"latent", "latent_rollout", "rollout"}:
            sparsity_loss = z_roll.abs().mean()
        elif sparsity_target in {"encoded", "latent_encoded"}:
            sparsity_loss = z_all.abs().mean()
        elif sparsity_target in {"k", "k_l1", "operator"}:
            sparsity_loss = k_l1
        else:
            raise ValueError(
                f"Unknown sparsity_target={sparsity_target!r}; expected latent_rollout, encoded, or k_l1."
            )
        loss = recon_loss + one + 0.5 * multi + float(sparsity_coeff) * sparsity_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        val = evaluate_flat_validation(model, val_data, horizon=min(horizon, val_data.shape[1] - 1), device=device)
        history.append({"epoch": float(epoch), "loss": float(loss.detach().cpu()), "val_nrmse": val})
        if val < best_val:
            best_val = val
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "config": cfg.to_dict(), "epoch": epoch}, best_path)
        if epoch - best_epoch >= patience:
            break

    torch.save({"model_state_dict": model.state_dict(), "config": cfg.to_dict(), "epoch": len(history) - 1}, final_path)
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    resolved_config = run_dir / "resolved_config.json"
    total, trainable = count_parameters(model)
    elapsed = time.perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    payload = model_config_payload(
        label,
        train_cfg,
        {
            "latent_dim": int(latent_dim),
            "sparsity_coeff": float(sparsity_coeff),
            "sparsity_target": sparsity_target,
            "model_preset": preset,
            "model_variant": dict(variant),
            "parameter_count": total,
            "trainable_parameter_count": trainable,
            "training_time_seconds": elapsed,
            "peak_gpu_memory_bytes": int(peak_memory),
            "best_val_nrmse": best_val,
            "history": history,
        },
    )
    write_json(resolved_config, payload)
    return model, payload, best_path, resolved_config


@torch.no_grad()
def evaluate_flat_validation(model: torch.nn.Module, data: np.ndarray, *, horizon: int, device: str) -> float:
    model.eval()
    truth = data[:, 1 : horizon + 1]
    x0 = torch.from_numpy(data[:, 0]).to(device)
    _z, pred = model.rollout_observation_discrete(x0, horizon=horizon)
    pred_np = pred.detach().cpu().numpy()
    return float(np.sqrt(np.sum((pred_np - truth) ** 2) / (np.sum((truth - truth.mean(axis=(0, 1))) ** 2) + 1e-8)))


def add_k_rows(rows: List[Dict[str, object]], *, base: Mapping[str, object], kmat: torch.Tensor) -> Dict[str, object]:
    diag = koopman_diagnostics(kmat)
    density = float(diag.get("effective_density_1e3", np.nan))
    for key, value in diag.items():
        if isinstance(value, list):
            continue
        rows.append(dict(base, horizon=0, threshold="", effective_density=density, metric_name=key, metric_value=float(value), trajectory_identifier="model"))
    return diag


def append_rollout_rows(
    rows: List[Dict[str, object]],
    *,
    base: Mapping[str, object],
    pred_by_traj: np.ndarray,
    truth_by_traj: np.ndarray,
    train_mean: np.ndarray,
    trajectory_ids: Sequence[str],
    horizons: Sequence[int],
    checkpoint_path: str,
    config_path: str,
    density: float,
) -> None:
    for i, traj_id in enumerate(trajectory_ids):
        for horizon, name, value in metric_rows_for_rollout(pred=pred_by_traj[i], truth=truth_by_traj[i], train_mean=train_mean, horizons=horizons):
            rows.append(
                dict(
                    base,
                    trajectory_identifier=str(traj_id),
                    horizon=int(horizon),
                    threshold="",
                    effective_density=float(density),
                    metric_name=name,
                    metric_value=float(value),
                    checkpoint_path=checkpoint_path,
                    configuration_path=config_path,
                    training_status="completed",
                )
            )


@torch.no_grad()
def rollout_flat_model(model: torch.nn.Module, data: np.ndarray, *, horizon: int, device: str) -> np.ndarray:
    model.eval()
    preds = []
    for start in range(0, data.shape[0], 64):
        x0 = torch.from_numpy(data[start : start + 64, 0]).to(device)
        _z, pred = model.rollout_observation_discrete(x0, horizon=horizon)
        preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def add_latent_sparsity_rows(
    rows: List[Dict[str, object]],
    *,
    base: Mapping[str, object],
    model: torch.nn.Module,
    data: np.ndarray,
) -> None:
    model.eval()
    device = next(model.parameters()).device
    flat = torch.from_numpy(data.reshape(-1, data.shape[-1])).to(device)
    chunks = []
    for start in range(0, flat.shape[0], 2048):
        chunks.append(model.encode(flat[start : start + 2048]).detach().cpu())
    z = torch.cat(chunks, dim=0).float()
    abs_z = z.abs()
    max_abs = float(abs_z.max().item()) if abs_z.numel() else 0.0
    thresholds = {
        "latent_active_density_abs_1e-4": 1e-4,
        "latent_active_density_abs_1e-3": 1e-3,
        "latent_active_density_abs_1e-2": 1e-2,
        "latent_active_density_rel_1e-3max": 1e-3 * max_abs,
    }
    metrics: Dict[str, float] = {
        "latent_mean_abs": float(abs_z.mean().item()) if abs_z.numel() else 0.0,
        "latent_max_abs": max_abs,
    }
    for name, threshold in thresholds.items():
        active = abs_z > threshold
        density = float(active.float().mean().item()) if active.numel() else 0.0
        metrics[name] = density
        metrics[name.replace("active_density", "active_coords")] = density * float(z.shape[-1])
    for name, value in metrics.items():
        rows.append(
            dict(
                base,
                trajectory_identifier="model",
                horizon=0,
                threshold="",
                effective_density=float(base.get("effective_density", np.nan)),
                metric_name=name,
                metric_value=float(value),
                checkpoint_path="",
                configuration_path="",
                training_status="completed",
            )
        )


def _default_lorenz_model_variants(cfg: Mapping[str, object], train_cfg: Mapping[str, object], sparsity_target: str) -> List[Dict[str, object]]:
    variants: List[Dict[str, object]] = []
    for latent_dim in cfg.get("latent_dims", [32]):
        for sparsity in cfg.get("sparsity_coefficients", [0.0, 1e-4]):
            if float(sparsity) == 0.0:
                model_label = "dense_kae"
                preset = "generic_no_shrink"
                encoder_activation = "tanh"
                encoder_last_relu = False
            elif sparsity_target in {"latent", "latent_rollout", "rollout", "encoded", "latent_encoded"}:
                model_label = "skae_latent_l1"
                preset = "generic_no_shrink"
                encoder_activation = "tanh"
                encoder_last_relu = False
            else:
                model_label = "skae_k_l1"
                preset = "generic_no_shrink"
                encoder_activation = "tanh"
                encoder_last_relu = False
            variants.append(
                {
                    "label": model_label,
                    "preset": preset,
                    "latent_dim": int(latent_dim),
                    "sparsity_coefficient": float(sparsity),
                    "sparsity_target": train_cfg.get("sparsity_target", "k_l1"),
                    "encoder_activation": encoder_activation,
                    "encoder_last_relu": encoder_last_relu,
                    "encoder_use_bias": True,
                }
            )
    return variants


def run_lorenz(
    config: Mapping[str, object],
    root: Path,
    rows: List[Dict[str, object]],
    failures: List[Dict[str, object]],
    device: str,
    flush_callback: Callable[[], None] | None = None,
) -> None:
    cfg = dict(config)
    if "conditions" in cfg:
        base = {key: value for key, value in cfg.items() if key != "conditions"}
        for condition_cfg in cfg.get("conditions", []):
            merged = dict(base)
            merged.update(dict(condition_cfg))
            run_lorenz(merged, root, rows, failures, device, flush_callback=flush_callback)
        return
    seeds = list(cfg.get("seeds", [0]))
    horizons = [int(h) for h in cfg.get("horizons", [1, 5, 10])]
    max_horizon = max(horizons)
    data_root = root / "data" / "lorenz96"
    for seed in seeds:
        try:
            dataset = build_lorenz96_dataset(
                dimension=int(cfg.get("dimension", 64)),
                forcing=float(cfg.get("forcing", 8.0)),
                n_train=int(cfg.get("n_train", 64)),
                n_val=int(cfg.get("n_val", 16)),
                n_test=int(cfg.get("n_test", 16)),
                time_points=max(int(cfg.get("time_points", 600)), max_horizon + 2),
                seed=int(seed),
                observed_fraction=float(cfg.get("observed_fraction", 1.0)),
                noise_fraction=float(cfg.get("noise_fraction", 0.0)),
                output_dir=data_root,
            )
            norm = compute_normalization(dataset.observations, dataset.indices("train"))
            obs = norm.apply(dataset.observations).astype(np.float32)
            val, test = obs[dataset.indices("val")], obs[dataset.indices("test")]
            train_indices_all = np.asarray(dataset.indices("train"), dtype=np.int64)
            train_sizes = [int(v) for v in cfg.get("train_sizes", [len(train_indices_all)])]
            for train_size in train_sizes:
                if train_size > len(train_indices_all):
                    raise ValueError(f"train_size={train_size} exceeds available train trajectories={len(train_indices_all)}")
                train_indices = train_indices_all[:train_size]
                train = obs[train_indices]
                train_mean = train.reshape(-1, train.shape[-1]).mean(axis=0)
                train_horizons = [int(v) for v in cfg.get("train_horizons", [dict(cfg.get("training", {})).get("train_horizon", 10)])]
                for train_horizon in train_horizons:
                    train_cfg = dict(cfg.get("training", {}))
                    train_cfg["train_horizon"] = int(train_horizon)
                    sparsity_target = str(train_cfg.get("sparsity_target", "k_l1")).lower()
                    condition = (
                        f"D{cfg.get('dimension', 64)}_F{cfg.get('forcing', 8.0)}"
                        f"_obs{cfg.get('observed_fraction', 1.0)}_noise{cfg.get('noise_fraction', 0.0)}"
                        f"_train{train_size}_Htrain{train_horizon}"
                    )
                    model_variants = [dict(v) for v in cfg.get("model_variants", [])]
                    if not model_variants:
                        model_variants = _default_lorenz_model_variants(cfg, train_cfg, sparsity_target)
                    for variant in model_variants:
                        model_label = str(variant.get("label", variant.get("preset", "model")))
                        latent_dim = int(variant.get("latent_dim", cfg.get("latent_dims", [32])[0]))
                        sparsity = float(variant.get("sparsity_coefficient", variant.get("sparsity_coeff", 0.0)))
                        variant_train_cfg = dict(train_cfg)
                        if "sparsity_target" in variant:
                            variant_train_cfg["sparsity_target"] = variant["sparsity_target"]
                        run_dir = root / "runs" / "lorenz96" / condition / f"{model_label}_seed{seed}_dz{latent_dim}_sp{sparsity:g}"
                        model, payload, ckpt, resolved = train_flat_koopman(
                            train_data=train,
                            val_data=val,
                            label=model_label,
                            latent_dim=int(latent_dim),
                            sparsity_coeff=float(sparsity),
                            train_cfg=variant_train_cfg,
                            run_dir=run_dir,
                            seed=int(seed),
                            device=device,
                            variant=variant,
                        )
                        base_common = {
                            "benchmark": "lorenz96",
                            "dataset_version": dataset.metadata["dataset_version"],
                            "condition": condition,
                            "model": model_label,
                            "seed": int(seed),
                            "latent_dim": int(latent_dim),
                            "sparsity_coefficient": float(sparsity),
                        }
                        base = dict(base_common, split="test")
                        diag = add_k_rows(rows, base=base, kmat=model.kmatrix())
                        density = float(diag.get("effective_density_1e3", np.nan))
                        add_latent_sparsity_rows(rows, base=dict(base, effective_density=density), model=model, data=test)
                        add_latent_sparsity_rows(rows, base=dict(base_common, split="val", effective_density=density), model=model, data=val)
                        val_pred = rollout_flat_model(model, val, horizon=max_horizon, device=device)
                        val_truth = val[:, 1 : max_horizon + 1]
                        append_rollout_rows(
                            rows,
                            base=dict(base_common, split="val"),
                            pred_by_traj=val_pred,
                            truth_by_traj=val_truth,
                            train_mean=train_mean,
                            trajectory_ids=dataset.trajectory_ids[dataset.indices("val")],
                            horizons=horizons,
                            checkpoint_path=str(ckpt),
                            config_path=str(resolved),
                            density=density,
                        )
                        infer_start = time.perf_counter()
                        pred = rollout_flat_model(model, test, horizon=max_horizon, device=device)
                        infer_time = time.perf_counter() - infer_start
                        truth = test[:, 1 : max_horizon + 1]
                        pred_path = run_dir / "predictions_test.npz"
                        np.savez_compressed(pred_path, pred=pred, truth=truth, trajectory_ids=dataset.trajectory_ids[dataset.indices("test")])
                        append_rollout_rows(
                            rows,
                            base=base,
                            pred_by_traj=pred,
                            truth_by_traj=truth,
                            train_mean=train_mean,
                            trajectory_ids=dataset.trajectory_ids[dataset.indices("test")],
                            horizons=horizons,
                            checkpoint_path=str(ckpt),
                            config_path=str(resolved),
                            density=density,
                        )
                        rows.append(dict(base, trajectory_identifier="model", horizon=0, threshold="", effective_density=density, metric_name="inference_time_seconds", metric_value=float(infer_time), checkpoint_path=str(ckpt), configuration_path=str(resolved), training_status="completed"))
                        if flush_callback is not None:
                            flush_callback()
                    run_lorenz_baselines(cfg, dataset, obs, norm, root, rows, condition, horizons, seed, train_indices=train_indices)
                    if flush_callback is not None:
                        flush_callback()
        except Exception as exc:
            failures.append({"benchmark": "lorenz96", "seed": int(seed), "status": "failed", "reason": repr(exc)})
            if flush_callback is not None:
                flush_callback()


def run_lorenz_baselines(
    cfg: Mapping[str, object],
    dataset: TrajectoryDataset,
    obs: np.ndarray,
    norm: object,
    root: Path,
    rows: List[Dict[str, object]],
    condition: str,
    horizons: Sequence[int],
    seed: int,
    train_indices: np.ndarray | None = None,
) -> None:
    if train_indices is None:
        train_indices = dataset.indices("train")
    train, test = obs[train_indices], obs[dataset.indices("test")]
    train_mean = train.reshape(-1, train.shape[-1]).mean(axis=0)
    models = [("persistence", None), ("dmd", fit_dmd(train))]
    if int(cfg.get("truncated_dmd_rank", 0)) > 0:
        models.append(("truncated_svd_dmd", fit_truncated_dmd(train, rank=int(cfg.get("truncated_dmd_rank")))))
    max_horizon = max(horizons)
    for label, model in models:
        preds = []
        for traj in test:
            preds.append(persistence_rollout(traj[0], max_horizon) if model is None else model.rollout(traj[0], max_horizon))
        pred = np.stack(preds, axis=0)
        truth = test[:, 1 : max_horizon + 1]
        pred_dir = root / "runs" / "lorenz96" / condition / f"{label}_seed{seed}"
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pred_dir / "predictions_test.npz", pred=pred, truth=truth)
        base = {
            "benchmark": "lorenz96",
            "dataset_version": dataset.metadata["dataset_version"],
            "condition": condition,
            "model": label,
            "seed": int(seed),
            "split": "test",
            "latent_dim": 0,
            "sparsity_coefficient": 0.0,
        }
        append_rollout_rows(
            rows,
            base=base,
            pred_by_traj=pred,
            truth_by_traj=truth,
            train_mean=train_mean,
            trajectory_ids=dataset.trajectory_ids[dataset.indices("test")],
            horizons=horizons,
            checkpoint_path="",
            config_path="",
            density=0.0,
        )


def sample_pde_batch(fields: np.ndarray, *, history: int, horizon: int, batch_size: int, rng: np.random.Generator) -> torch.Tensor:
    traj_count, time_count, dim = fields.shape
    max_start = time_count - history - horizon
    traj_idx = rng.integers(0, traj_count, size=batch_size)
    start_idx = rng.integers(0, max_start + 1, size=batch_size)
    batch = np.empty((batch_size, history + horizon, dim), dtype=np.float32)
    for i, (traj, start) in enumerate(zip(traj_idx, start_idx)):
        batch[i] = fields[traj, start : start + history + horizon]
    return torch.from_numpy(batch)


def train_pde_model(
    model: ConvKoopmanAE,
    *,
    train: np.ndarray,
    val: np.ndarray,
    sparsity_coeff: float,
    cfg: Mapping[str, object],
    seed: int,
    device: str,
    run_dir: Path,
) -> Tuple[ConvKoopmanAE, Path, Path]:
    start_time = time.perf_counter()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)), weight_decay=float(cfg.get("weight_decay", 1e-6)))
    rng = np.random.default_rng(seed + 700)
    epochs = int(cfg.get("epochs", 3))
    horizon = int(cfg.get("train_horizon", 5))
    batch_size = int(cfg.get("batch_size", 4))
    run_dir.mkdir(parents=True, exist_ok=True)
    best, best_path = float("inf"), run_dir / "checkpoint.pt"
    for epoch in range(epochs):
        batch = sample_pde_batch(train, history=model.cfg.history, horizon=horizon, batch_size=batch_size, rng=rng).to(device)
        context = batch[:, : model.cfg.history].reshape(batch.shape[0], -1)
        target = batch[:, model.cfg.history :]
        _z, pred = model.rollout(context, horizon)
        loss = F.mse_loss(pred, target) + float(sparsity_coeff) * model.kmat.abs().sum() / float(model.cfg.z_dim**2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
        opt.step()
        val_loss = float(loss.detach().cpu())
        if val_loss < best:
            best = val_loss
            torch.save({"model_state_dict": model.state_dict(), "model_config": model.cfg.__dict__, "epoch": epoch}, best_path)
    final = run_dir / "final.pt"
    torch.save({"model_state_dict": model.state_dict(), "model_config": model.cfg.__dict__, "epoch": epochs - 1}, final)
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    resolved = run_dir / "resolved_config.json"
    total, trainable = count_parameters(model)
    write_json(
        resolved,
        {
            "model": "conv_koopman",
            "model_config": model.cfg.__dict__,
            "training": dict(cfg),
            "sparsity_coeff": float(sparsity_coeff),
            "parameter_count": total,
            "trainable_parameter_count": trainable,
            "training_time_seconds": time.perf_counter() - start_time,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
        },
    )
    return model, best_path, resolved


@torch.no_grad()
def rollout_pde(model: ConvKoopmanAE, data: np.ndarray, *, horizon: int, device: str) -> np.ndarray:
    model.eval()
    preds = []
    hist = model.cfg.history
    for start in range(0, data.shape[0], 16):
        context = torch.from_numpy(data[start : start + 16, :hist].reshape(data[start : start + 16].shape[0], -1)).to(device)
        _z, pred = model.rollout(context, horizon)
        preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def run_pde(config: Mapping[str, object], root: Path, rows: List[Dict[str, object]], failures: List[Dict[str, object]], device: str) -> None:
    cfg = dict(config)
    seed = int(cfg.get("seed", 0))
    try:
        path = Path(str(cfg.get("data_path", "")))
        if path.exists():
            fields, metadata = load_pdebench_h5(path, max_trajectories=int(cfg.get("max_trajectories", 0)))
        elif bool(cfg.get("allow_smoke_fixture", False)):
            fields, metadata = generate_pde_smoke_fixture(seed=seed, n_trajectories=int(cfg.get("fixture_trajectories", 6)), time_points=int(cfg.get("fixture_time_points", 24)), grid_size=int(cfg.get("grid_size", 16)))
            failures.append({"benchmark": "pdebench_swe", "status": "official_data_omitted", "reason": f"Official file missing at {path}; ran explicit non-official smoke fixture."})
        else:
            failures.append({"benchmark": "pdebench_swe", "status": "omitted", "reason": f"Official PDEBench file missing at {path}."})
            return
        dataset = pde_fields_to_dataset(fields, metadata, seed=seed)
        norm = compute_normalization(dataset.observations, dataset.indices("train"))
        obs = norm.apply(dataset.observations).astype(np.float32)
        train, test = obs[dataset.indices("train")], obs[dataset.indices("test")]
        train_mean = train.reshape(-1, train.shape[-1]).mean(axis=0)
        horizons = [int(h) for h in cfg.get("horizons", [1, 5, 10])]
        max_h = max(horizons)
        grid = int(fields.shape[2])
        channels = int(fields.shape[-1])
        for sparsity in cfg.get("sparsity_coefficients", [0.0, 1e-4]):
            label = "dense_conv_kae" if float(sparsity) == 0.0 else "skae_conv_k_l1"
            model = ConvKoopmanAE(ConvKoopmanConfig(grid_size=grid, channels=channels, history=int(cfg.get("history", 4)), z_dim=int(cfg.get("latent_dim", 32)), hidden_channels=int(cfg.get("hidden_channels", 8))))
            run_dir = root / "runs" / "pdebench_swe" / f"{label}_seed{seed}_sp{sparsity:g}"
            model, ckpt, resolved = train_pde_model(model, train=train, val=test, sparsity_coeff=float(sparsity), cfg=cfg.get("training", {}), seed=seed, device=device, run_dir=run_dir)
            infer_start = time.perf_counter()
            pred = rollout_pde(model, test, horizon=max_h, device=device)
            infer_time = time.perf_counter() - infer_start
            truth = test[:, int(cfg.get("history", 4)) : int(cfg.get("history", 4)) + max_h]
            np.savez_compressed(run_dir / "predictions_test.npz", pred=pred, truth=truth)
            base = {"benchmark": "pdebench_swe", "dataset_version": metadata["dataset_version"], "condition": "clean_or_fixture", "model": label, "seed": seed, "split": "test", "latent_dim": int(cfg.get("latent_dim", 32)), "sparsity_coefficient": float(sparsity)}
            diag = add_k_rows(rows, base=base, kmat=model.kmat)
            rows.append(dict(base, trajectory_identifier="model", horizon=0, threshold="", effective_density=float(diag.get("effective_density_1e3", np.nan)), metric_name="inference_time_seconds", metric_value=float(infer_time), checkpoint_path=str(ckpt), configuration_path=str(resolved), training_status="completed"))
            density = float(diag.get("effective_density_1e3", np.nan))
            append_rollout_rows(rows, base=base, pred_by_traj=pred, truth_by_traj=truth, train_mean=train_mean, trajectory_ids=dataset.trajectory_ids[dataset.indices("test")], horizons=horizons, checkpoint_path=str(ckpt), config_path=str(resolved), density=density)
    except Exception as exc:
        failures.append({"benchmark": "pdebench_swe", "status": "failed", "reason": repr(exc)})


def run_silverbox(config: Mapping[str, object], root: Path, rows: List[Dict[str, object]], failures: List[Dict[str, object]], device: str) -> None:
    cfg = dict(config)
    seed = int(cfg.get("seed", 0))
    try:
        dataset = load_or_fixture_silverbox(allow_fixture=bool(cfg.get("allow_smoke_fixture", False)), seed=seed)
        if "not_official" in str(dataset.metadata.get("dataset_version", "")):
            failures.append({"benchmark": "silverbox", "status": "official_data_omitted", "reason": "Official nonlinear_benchmarks loader unavailable; ran explicit non-official smoke fixture."})
        series = dataset.observations[0]
        finite = np.isfinite(series).all(axis=1)
        series = series[finite].astype(np.float32)
        n = series.shape[0]
        history = int(cfg.get("history", 64))
        guard = int(cfg.get("guard", min(256, history)))
        train_end = int(0.6 * n)
        val_start = min(n - history - 2, train_end + guard)
        val_end = int(0.8 * n)
        train = series[:train_end]
        val = series[val_start:val_end]
        test = series[val_end - history :]
        mean = train.mean(axis=0)
        std = np.where(train.std(axis=0) < 1e-6, 1.0, train.std(axis=0))
        train_n, val_n, test_n = (train - mean) / std, (val - mean) / std, (test - mean) / std
        horizons = [int(h) for h in cfg.get("horizons", [10, 50])]
        max_h = min(max(horizons), test_n.shape[0] - history - 1)
        for sparsity in cfg.get("sparsity_coefficients", [0.0, 1e-4]):
            label = "dense_controlled_kae" if float(sparsity) == 0.0 else "skae_controlled_k_l1"
            model, ckpt, resolved = train_silver_model(train_n, val_n, cfg, seed, float(sparsity), device, root / "runs" / "silverbox" / f"{label}_seed{seed}_sp{sparsity:g}")
            infer_start = time.perf_counter()
            pred = rollout_silver_model(model, test_n, history=history, horizon=max_h, device=device)
            infer_time = time.perf_counter() - infer_start
            truth = test_n[history : history + max_h, 1][None, :, None]
            pred3 = pred[None, :, None]
            run_dir = root / "runs" / "silverbox" / f"{label}_seed{seed}_sp{sparsity:g}"
            np.savez_compressed(run_dir / "predictions_test.npz", pred=pred3, truth=truth)
            base = {"benchmark": "silverbox", "dataset_version": dataset.metadata["dataset_version"], "condition": "free_run", "model": label, "seed": seed, "split": "test", "latent_dim": int(cfg.get("latent_dim", 8)), "sparsity_coefficient": float(sparsity)}
            diag = add_k_rows(rows, base=base, kmat=model.kmat)
            density = float(diag.get("effective_density_1e3", np.nan))
            rows.append(dict(base, trajectory_identifier="model", horizon=0, threshold="", effective_density=density, metric_name="inference_time_seconds", metric_value=float(infer_time), checkpoint_path=str(ckpt), configuration_path=str(resolved), training_status="completed"))
            append_rollout_rows(rows, base=base, pred_by_traj=pred3, truth_by_traj=truth, train_mean=np.asarray([train_n[:, 1].mean()]), trajectory_ids=["silverbox_test"], horizons=horizons, checkpoint_path=str(ckpt), config_path=str(resolved), density=density)
            rows.append(dict(base, trajectory_identifier="silverbox_test", horizon=max_h, threshold="", effective_density=density, metric_name="fit_percent", metric_value=fit_percent(pred3.reshape(-1), truth.reshape(-1)), checkpoint_path=str(ckpt), configuration_path=str(resolved), training_status="completed"))
        run_arx_baseline(train_n, val_n, test_n, history, horizons, rows, dataset, seed)
    except Exception as exc:
        failures.append({"benchmark": "silverbox", "status": "failed", "reason": repr(exc)})


def sample_silver_batch(series: np.ndarray, *, history: int, horizon: int, batch: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_start = series.shape[0] - history - horizon
    starts = rng.integers(0, max_start + 1, size=batch)
    histories, actions, targets = [], [], []
    for start in starts:
        histories.append(series[start : start + history].reshape(-1))
        future = series[start + history : start + history + horizon]
        actions.append(future[:, :1])
        targets.append(future[:, 1:2])
    return torch.from_numpy(np.stack(histories).astype(np.float32)), torch.from_numpy(np.stack(actions).astype(np.float32)), torch.from_numpy(np.stack(targets).astype(np.float32))


def train_silver_model(series: np.ndarray, val: np.ndarray, cfg: Mapping[str, object], seed: int, sparsity: float, device: str, run_dir: Path) -> Tuple[ControlledKoopmanAE, Path, Path]:
    start_time = time.perf_counter()
    history = int(cfg.get("history", 64))
    model = ControlledKoopmanAE(history_dim=history * 2, output_dim=1, action_dim=1, z_dim=int(cfg.get("latent_dim", 8)), hidden_dim=int(cfg.get("hidden_dim", 64))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)), weight_decay=float(cfg.get("weight_decay", 1e-6)))
    rng = np.random.default_rng(seed + 900)
    run_dir.mkdir(parents=True, exist_ok=True)
    best, best_path = float("inf"), run_dir / "checkpoint.pt"
    for epoch in range(int(cfg.get("epochs", 5))):
        h, u, y = sample_silver_batch(series, history=history, horizon=int(cfg.get("train_horizon", 20)), batch=int(cfg.get("batch_size", 32)), rng=rng)
        h, u, y = h.to(device), u.to(device), y.to(device)
        _z, pred = model.rollout(h, u)
        loss = F.mse_loss(pred, y) + float(sparsity) * model.kmat.abs().sum() / float(model.z_dim**2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
        opt.step()
        val_loss = float(loss.detach().cpu())
        if val_loss < best:
            best = val_loss
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, best_path)
    final = run_dir / "final.pt"
    torch.save({"model_state_dict": model.state_dict()}, final)
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model_state_dict"])
    resolved = run_dir / "resolved_config.json"
    total, trainable = count_parameters(model)
    write_json(
        resolved,
        {
            "model": "controlled_koopman",
            "training": dict(cfg),
            "sparsity_coeff": sparsity,
            "parameter_count": total,
            "trainable_parameter_count": trainable,
            "training_time_seconds": time.perf_counter() - start_time,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
        },
    )
    return model, best_path, resolved


@torch.no_grad()
def rollout_silver_model(model: ControlledKoopmanAE, series: np.ndarray, *, history: int, horizon: int, device: str) -> np.ndarray:
    hist = torch.from_numpy(series[:history].reshape(1, -1).astype(np.float32)).to(device)
    actions = torch.from_numpy(series[history : history + horizon, :1][None].astype(np.float32)).to(device)
    _z, pred = model.rollout(hist, actions)
    return pred.detach().cpu().numpy()[0, :, 0]


def run_arx_baseline(train: np.ndarray, val: np.ndarray, test: np.ndarray, history: int, horizons: Sequence[int], rows: List[Dict[str, object]], dataset: TrajectoryDataset, seed: int) -> None:
    model, _mse = select_arx(train[:, 0], train[:, 1], val[:, 0], val[:, 1], orders=[min(o, max(2, history // 2)) for o in (5, 10, 20, 40, 80)])
    max_h = min(max(horizons), test.shape[0] - history - 1)
    pred = model.freerun(test[:, 0], test[:history, 1], horizon=max_h)
    truth = test[history : history + max_h, 1]
    base = {"benchmark": "silverbox", "dataset_version": dataset.metadata["dataset_version"], "condition": "free_run", "model": "linear_arx", "seed": seed, "split": "test", "latent_dim": 0, "sparsity_coefficient": 0.0}
    append_rollout_rows(rows, base=base, pred_by_traj=pred[None, :, None], truth_by_traj=truth[None, :, None], train_mean=np.asarray([train[:, 1].mean()]), trajectory_ids=["silverbox_test"], horizons=horizons, checkpoint_path="", config_path="", density=0.0)


def write_outputs(root: Path, rows: List[Dict[str, object]], failures: List[Dict[str, object]], config: Mapping[str, object], config_path: Path) -> None:
    results_dir = root / "results"
    reports_dir = root / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    if raw.empty:
        raw = pd.DataFrame(columns=["benchmark", "dataset_version", "condition", "model", "seed", "split", "trajectory_identifier", "horizon", "latent_dim", "sparsity_coefficient", "threshold", "effective_density", "metric_name", "metric_value", "checkpoint_path", "configuration_path", "training_status"])
    raw.to_parquet(results_dir / "raw_metrics.parquet", index=False)
    raw.to_csv(results_dir / "raw_metrics.csv", index=False)
    summary = pd.DataFrame(summarize_rows(rows, n_resamples=int(config.get("bootstrap_resamples", 2000))))
    summary.to_csv(results_dir / "summary_metrics.csv", index=False)
    write_json(results_dir / "failures.json", failures)
    manifest = {"git": git_metadata(), "system": system_metadata(), "config_path": str(config_path), "config": config, "raw_rows": int(len(rows)), "failures": failures}
    write_json(results_dir / "run_manifest.json", manifest)
    shutil.copy2(config_path, results_dir / "resolved_suite_config.yaml")
    make_figures(raw, reports_dir / "figures")
    write_reports(root, raw, summary, failures, manifest)


def write_partial_outputs(root: Path, rows: List[Dict[str, object]], failures: List[Dict[str, object]]) -> None:
    """Persist completed rows during long sweeps without running report generation."""
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    if raw.empty:
        raw = pd.DataFrame(columns=["benchmark", "dataset_version", "condition", "model", "seed", "split", "trajectory_identifier", "horizon", "latent_dim", "sparsity_coefficient", "threshold", "effective_density", "metric_name", "metric_value", "checkpoint_path", "configuration_path", "training_status"])
    raw.to_parquet(results_dir / "raw_metrics.partial.parquet", index=False)
    raw.to_csv(results_dir / "raw_metrics.partial.csv", index=False)
    write_json(results_dir / "failures.partial.json", failures)


def make_figures(raw: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    if raw.empty:
        return
    for metric, filename in [("nrmse", "rollout_nrmse"), ("spectral_radius", "spectral_radius"), ("effective_density_1e3", "density")]:
        data = raw[raw["metric_name"] == metric].copy()
        if data.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
        if metric == "nrmse":
            grouped = data.groupby(["benchmark", "model", "horizon"], as_index=False)["metric_value"].mean()
            for (bench, model), part in grouped.groupby(["benchmark", "model"]):
                ax.plot(part["horizon"], part["metric_value"], marker="o", label=f"{bench}:{model}")
            ax.set_xlabel("Rollout horizon")
            ax.set_ylabel("NRMSE")
        else:
            grouped = data.groupby(["benchmark", "model", "sparsity_coefficient"], as_index=False)["metric_value"].mean()
            for (bench, model), part in grouped.groupby(["benchmark", "model"]):
                ax.plot(part["sparsity_coefficient"], part["metric_value"], marker="o", label=f"{bench}:{model}")
            ax.set_xscale("symlog", linthresh=1e-8)
            ax.set_xlabel("Sparsity coefficient")
            ax.set_ylabel(metric.replace("_", " "))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"{filename}.{ext}", dpi=200)
        plt.close(fig)


def write_reports(root: Path, raw: pd.DataFrame, summary: pd.DataFrame, failures: List[Dict[str, object]], manifest: Mapping[str, object]) -> None:
    reports_dir = root / "reports"
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if not summary.empty:
        summary.to_csv(tables_dir / "summary_metrics.csv", index=False)
    key = summary[(summary.get("metric_name") == "nrmse") & (summary.get("split") == "test")] if not summary.empty else pd.DataFrame()
    completed = sorted(set(str(v) for v in raw.get("benchmark", pd.Series(dtype=str)).dropna().unique()))
    model_list = sorted(set(str(v) for v in raw.get("model", pd.Series(dtype=str)).dropna().unique()))
    cost = summary[summary["metric_name"].isin(["inference_time_seconds", "spectral_radius", "effective_density_1e3"])] if not summary.empty else pd.DataFrame()
    sparse_tradeoff = summary[(summary["metric_name"] == "nrmse") & (summary["horizon"] != 0)] if not summary.empty else pd.DataFrame()
    lines = [
        "# Final SKAE benchmark report",
        "",
        "## Executive summary",
        "",
        "This report contains completed smoke-scale benchmark evidence plus an explicit list of omitted full runs. Results from non-official fixtures are pipeline checks only and are not scientific evidence for the requested PDEBench or Silverbox benchmarks.",
        "",
        f"Completed benchmark pipelines in this run: {', '.join(completed) if completed else 'none'}. Models/baselines recorded: {', '.join(model_list) if model_list else 'none'}.",
        "",
        "## Repository and implementation audit",
        "",
        "See `reports/repository_audit.md`. The main repository `SparseKM` maps to `GenericKM`; its built-in sparsity penalty is latent-activation L1, while this suite labels operator-L1 runs as `skae_k_l1` or related benchmark adaptations.",
        "",
        "## Experimental protocol",
        "",
        "The checked-in config is `configs/skae_benchmark_suite.yaml`. It uses trajectory-level splits, train-only normalization, AdamW, gradient clipping, saved best/final checkpoints, raw per-trajectory metrics, and 2,000 bootstrap resamples for summary CIs. The profile is explicitly reduced for smoke execution.",
        "",
        "## Dataset and split details",
        "",
        f"Rows recorded: {len(raw)}. Failures or omissions recorded: {len(failures)}. Lorenz-96 data were generated by the RK4 generator in this suite. PDEBench and Silverbox official data were not available in this environment and are listed under limitations when fixtures were used.",
        "",
        "## Hyperparameter selection procedure",
        "",
        "The smoke run evaluates the configured sparsity coefficients and latent dimensions directly. It does not complete the full requested grid or the 5% validation-error sparse operating-point rule; that remains encoded as future configuration work rather than a supported conclusion.",
        "",
        "## Main quantitative results",
        "",
    ]
    if key.empty:
        lines.append("No test NRMSE rows were produced.")
    else:
        display_cols = ["benchmark", "condition", "model", "horizon", "mean", "std", "ci95_low", "ci95_high", "n"]
        lines.extend(markdown_table(key[display_cols]))
    lines.extend(["", "## Sparsity-accuracy tradeoffs", ""])
    if sparse_tradeoff.empty:
        lines.append("No sparsity-accuracy rows were produced.")
    else:
        lines.extend(markdown_table(sparse_tradeoff[["benchmark", "model", "horizon", "sparsity_coefficient", "mean", "ci95_low", "ci95_high", "n"]].head(60)))
    lines.extend(
        [
            "",
            "## Long-horizon stability analysis",
            "",
            "Long-horizon evidence is limited to the configured smoke horizons. Valid-prediction-time, spectral-radius, and instability-adjacent diagnostics are present in `results/raw_metrics.parquet`; no broad stability claim is supported.",
            "",
            "## Noise and partial-observability analysis",
            "",
            "The smoke config ran full-observation, clean Lorenz-96 only. The requested noise and partial-observation grids were not executed.",
            "",
            "## High-dimensional PDE results",
            "",
            "The PDE path completed on an explicit synthetic fixture because the official PDEBench `2D_rdb_NA_NA.h5` file was absent. These rows validate code paths and output schemas only.",
            "",
            "## Real-data Silverbox results",
            "",
            "The Silverbox path completed on an explicit synthetic fixture because the official `nonlinear_benchmarks` loader was unavailable. These rows are not evidence on real measured data.",
            "",
            "## Structural and spectral analysis",
            "",
            "The suite records raw K L1, exact zeros, effective densities at relative thresholds, average active entries per coordinate, spectral radius, and eigenvalue moduli. Latent support is not claimed to identify physical causal structure.",
            "",
            "## Computational-cost analysis",
            "",
        ]
    )
    if cost.empty:
        lines.append("No computational-cost rows were produced.")
    else:
        lines.extend(markdown_table(cost[["benchmark", "model", "metric_name", "mean", "n"]].head(80)))
    lines.extend(
        [
            "",
            "## Limitations and failed runs",
            "",
            "The following entries are machine-readable in `results/failures.json`:",
            "",
            "```json",
            json.dumps(failures, indent=2, sort_keys=True),
            "```",
            "",
            "## Reproduction commands",
            "",
            "Use the commands in `reports/reproduction_instructions.md`.",
            "",
            "## Direct answers to the central questions",
            "",
            "- Predictive accuracy lost with sparsity: not answered beyond smoke-scale configured comparisons.",
            "- Sparsity effects on stability, interpretability, sample efficiency, or robustness: not established.",
            "- Long free-running accuracy: measured only at smoke horizons.",
            "- Sparse structure stability across seeds: not established; only seed 0 completed.",
            "- Scaling to high-dimensional states: not established; PDE fixture and Lorenz smoke are too small.",
            "- Hidden Markov state and real input-output data: Silverbox official data were not run.",
            "- Real measured input-output performance: not supported by this run.",
        ]
    )
    report_text = "\n".join(lines) + "\n"
    (reports_dir / "final_report.md").write_text(report_text, encoding="utf-8")
    write_pdf_report(reports_dir / "final_report.pdf", report_text)
    (reports_dir / "reproduction_instructions.md").write_text(
        "# Reproduction instructions\n\n"
        "Run Python workloads on a compute node. Example:\n\n"
        "```bash\n"
        "salloc --mem=8G -c 4 --partition=long --time=01:00:00 \\\n"
        "  srun --cpu-bind=none bash -lc 'uv run python -m experiments.run_suite --config configs/skae_benchmark_suite.yaml'\n"
        "```\n\n"
        "Main tables are regenerated from `results/raw_metrics.parquet` by rerunning the same command.\n",
        encoding="utf-8",
    )
    write_json(reports_dir / "manifest_snapshot.json", dict(manifest))


def markdown_table(frame: pd.DataFrame) -> List[str]:
    cols = list(frame.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        out.append("| " + " | ".join(values) + " |")
    return out


def write_pdf_report(path: Path, text: str) -> None:
    import textwrap
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    lines: List[str] = []
    for raw_line in text.splitlines():
        if not raw_line:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw_line, width=96) or [""])
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for start in range(0, len(lines), 48):
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.06, 0.96, "\n".join(lines[start : start + 48]), va="top", family="monospace", fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the SKAE benchmark suite.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark", choices=["all", "lorenz96", "pdebench_swe", "silverbox"], default="all")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    config = parse_config(args.config)
    root = Path(str(config.get("output_root", "."))).resolve()
    device = resolve_device(str(config.get("device", "auto")))
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    selected = [args.benchmark] if args.benchmark != "all" else ["lorenz96", "silverbox", "pdebench_swe"]
    if args.seed is not None and "lorenz96" in config:
        config["lorenz96"] = dict(config["lorenz96"], seeds=[int(args.seed)])  # type: ignore[index]
    def flush_partial() -> None:
        write_partial_outputs(root, rows, failures)
    if "lorenz96" in selected:
        run_lorenz(config.get("lorenz96", {}), root, rows, failures, device, flush_callback=flush_partial)
    if "silverbox" in selected:
        run_silverbox(config.get("silverbox", {}), root, rows, failures, device)
    if "pdebench_swe" in selected:
        run_pde(config.get("pdebench_swe", {}), root, rows, failures, device)
    write_outputs(root, rows, failures, config, args.config)


if __name__ == "__main__":
    main()
