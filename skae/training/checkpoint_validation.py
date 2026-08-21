"""Strict validation for complete checkpoint payloads."""

from __future__ import annotations

import re
from typing import Any, Dict

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = 1
_REQUIRED_STATE = {
    "schema_version", "run_id", "generation", "next_step", "model_state_dict",
    "optimizer_state_dict", "rng_state", "run_identity", "logger_state", "config",
    "data_order", "storage_contract", "last_metrics", "scheduler_state_dict",
    "scaler_state_dict", "best_eval_final_error", "checkpoint_selection_metric",
    "checkpoint_selection_score",
}
_REQUIRED_RNG = {
    "python", "numpy", "torch_cpu", "torch_cpu_device", "torch_cuda",
    "torch_cuda_devices", "batch_generators", "batch_generator_devices",
    "validation_generator", "validation_generator_device",
}


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_))


def _is_logger_scalar(value: Any) -> bool:
    """Match the JSON scalar values accepted by MetricsLogger.log_scalar."""
    return value is None or isinstance(
        value, (str, bool, int, float, np.integer, np.floating)
    )


def _valid_identity(identity: Any) -> bool:
    if not isinstance(identity, dict):
        return False
    if not isinstance(identity.get("config"), dict):
        return False
    if re.fullmatch(r"[0-9a-f]{64}", str(identity.get("config_hash", ""))) is None:
        return False
    if re.fullmatch(r"[0-9a-f]{64}", str(identity.get("resume_config_hash", ""))) is None:
        return False
    if not isinstance(identity.get("device"), str) or not identity["device"]:
        return False
    try:
        torch.device(identity["device"])
    except (RuntimeError, TypeError, ValueError):
        return False
    if not _is_int(identity.get("cuda_device_count")) or identity["cuda_device_count"] < 0:
        return False
    if not _is_int(identity.get("batch_count")) or identity["batch_count"] < 1:
        return False
    if not isinstance(identity.get("logger_history"), bool):
        return False
    source = identity.get("source")
    return (
        isinstance(source, dict)
        and isinstance(source.get("git_commit"), str)
        and re.fullmatch(r"[0-9a-f]{40,64}", source["git_commit"]) is not None
        and source.get("git_dirty") is False
        and isinstance(source.get("git_status_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", source["git_status_sha256"]) is not None
    )


def _valid_logger_state(logger: Any) -> bool:
    if not isinstance(logger, dict):
        return False
    if not isinstance(logger.get("save_history"), bool):
        return False
    if not isinstance(logger.get("metrics_history"), list):
        return False
    for item in logger["metrics_history"]:
        if (
            not isinstance(item, dict)
            or not _is_int(item.get("step"))
            or item["step"] < 0
            or not isinstance(item.get("name"), str)
            or not _is_logger_scalar(item.get("value"))
        ):
            return False
    if not _is_int(logger.get("step_count")) or logger["step_count"] < 0:
        return False
    if logger["save_history"] and len(logger["metrics_history"]) != logger["step_count"]:
        return False
    if not logger["save_history"] and logger["metrics_history"]:
        return False
    summary = logger.get("summary_state")
    if not isinstance(summary, dict):
        return False
    for name, state in summary.items():
        if (
            not isinstance(name, str)
            or not isinstance(state, dict)
            or not _is_int(state.get("count"))
            or state["count"] < 0
            or any(key not in state for key in ("final", "min", "max"))
            or not _is_number(state.get("sum"))
            or not _is_logger_scalar(state.get("final"))
            or not (_is_number(state.get("min")) or state.get("min") is None)
            or not (_is_number(state.get("max")) or state.get("max") is None)
        ):
            return False
        if state["count"] == 0 and (state["min"] is not None or state["max"] is not None):
            return False
        if state["count"] > 0 and (state["min"] is None or state["max"] is None):
            return False
    return True


def _valid_rng_state(rng: Any, identity: Dict[str, Any]) -> bool:
    if not isinstance(rng, dict) or not _REQUIRED_RNG.issubset(rng):
        return False
    python_state = rng["python"]
    if (
        not isinstance(python_state, tuple)
        or len(python_state) != 3
        or not _is_int(python_state[0])
        or not isinstance(python_state[1], tuple)
        or len(python_state[1]) != 625
        or any(not _is_int(value) for value in python_state[1])
        or (python_state[2] is not None and not _is_number(python_state[2]))
    ):
        return False
    numpy_state = rng["numpy"]
    if not isinstance(numpy_state, tuple) or len(numpy_state) != 5:
        return False
    if (
        not isinstance(numpy_state[0], str)
        or not isinstance(numpy_state[1], np.ndarray)
        or numpy_state[1].dtype != np.uint32
        or numpy_state[1].ndim != 1
        or numpy_state[1].size != 624
        or not _is_int(numpy_state[2])
        or not 0 <= numpy_state[2] <= 624
        or not _is_int(numpy_state[3])
        or numpy_state[3] not in (0, 1)
        or not _is_number(numpy_state[4])
    ):
        return False
    if (
        rng.get("torch_cpu_device") != "cpu"
        or not isinstance(rng["torch_cpu"], torch.Tensor)
        or rng["torch_cpu"].dtype != torch.uint8
        or rng["torch_cpu"].ndim != 1
        or rng["torch_cpu"].numel() < 1
    ):
        return False
    cuda = rng["torch_cuda"]
    cuda_devices = rng["torch_cuda_devices"]
    if (
        not isinstance(cuda, list)
        or not isinstance(cuda_devices, list)
        or len(cuda) != identity["cuda_device_count"]
        or len(cuda_devices) != len(cuda)
        or cuda_devices != [f"cuda:{index}" for index in range(len(cuda))]
    ):
        return False
    if any(
        not isinstance(item, torch.Tensor)
        or item.dtype != torch.uint8
        or item.ndim != 1
        or item.numel() < 1
        for item in cuda
    ):
        return False
    generators = rng["batch_generators"]
    devices = rng.get("batch_generator_devices")
    if not isinstance(generators, list) or not isinstance(devices, list):
        return False
    if len(generators) != len(devices) or len(generators) != identity["batch_count"]:
        return False
    if not all(
        isinstance(item, torch.Tensor)
        and item.dtype == torch.uint8
        and item.ndim == 1
        and item.numel() > 0
        for item in generators
    ):
        return False
    if not all(isinstance(device, str) and device for device in devices):
        return False
    if any(
        device not in {"cpu"} and not re.fullmatch(r"cuda(?::\d+)?", device)
        for device in devices
    ):
        return False
    validation_state = rng["validation_generator"]
    validation_device = rng["validation_generator_device"]
    if validation_state is None:
        if validation_device is not None:
            return False
    else:
        if (
            not isinstance(validation_state, torch.Tensor)
            or validation_state.dtype != torch.uint8
            or validation_state.ndim != 1
            or validation_state.numel() < 1
            or not isinstance(validation_device, str)
            or (
                validation_device != "cpu"
                and re.fullmatch(r"cuda(?::\d+)?", validation_device) is None
            )
        ):
            return False
    return True


def valid_complete_payload(payload: Any) -> bool:
    """Return true only for a complete, exact-continuation payload."""
    if not isinstance(payload, dict) or not _REQUIRED_STATE.issubset(payload):
        return False
    if (
        not _is_int(payload.get("schema_version"))
        or payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION
    ):
        return False
    if not isinstance(payload.get("run_id"), str) or not payload["run_id"]:
        return False
    if not _is_int(payload.get("generation")) or payload["generation"] < 1:
        return False
    if not _is_int(payload.get("next_step")) or payload["next_step"] < 0:
        return False
    if not isinstance(payload.get("model_state_dict"), dict):
        return False
    if not isinstance(payload.get("optimizer_state_dict"), dict):
        return False
    identity = payload.get("run_identity")
    if not _valid_identity(identity) or not _valid_rng_state(payload.get("rng_state"), identity):
        return False
    logger_state = payload.get("logger_state")
    if not _valid_logger_state(logger_state):
        return False
    if logger_state["save_history"] != identity["logger_history"]:
        return False
    if not isinstance(payload.get("config"), dict):
        return False
    order = payload.get("data_order")
    if not isinstance(order, dict):
        return False
    if (
        not _is_int(order.get("num_batches"))
        or order["num_batches"] != identity["batch_count"]
        or not _is_int(order.get("batch_size"))
        or order["batch_size"] < 1
        or not _is_int(order.get("sequence_length"))
        or order["sequence_length"] < 1
        or not _is_int(order.get("seed"))
        or not _is_int(order.get("generator_index"))
        or not 0 <= order["generator_index"] < order["num_batches"]
        or order["generator_index"] != payload["next_step"] % order["num_batches"]
    ):
        return False
    storage = payload.get("storage_contract")
    if not isinstance(payload.get("last_metrics"), dict) or not isinstance(storage, dict):
        return False
    if not _is_int(storage.get("retention")) or storage["retention"] < 2:
        return False
    if not _is_int(storage.get("checkpoint_interval")) or storage["checkpoint_interval"] < 1:
        return False
    score = payload.get("best_eval_final_error")
    if not _is_number(score):
        return False
    selection_score = payload.get("checkpoint_selection_score")
    if selection_score is not None and not _is_number(selection_score):
        return False
    metric = payload.get("checkpoint_selection_metric")
    return metric is None or isinstance(metric, str)
