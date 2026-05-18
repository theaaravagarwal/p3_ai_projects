from __future__ import annotations

import json
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

BUCKET_LABELS = ["Black advantage", "Equal", "White advantage"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CUDA initialization:.*")
        cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="CUDA initialization:.*")
            cuda_available = torch.cuda.is_available()
        if cuda_available:
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def bucket_ids(values_cp: np.ndarray) -> np.ndarray:
    values = np.asarray(values_cp)
    buckets = np.ones(values.shape, dtype=np.int64)
    buckets[values > 150] = 2
    buckets[values < -150] = 0
    return buckets


def normalize_bucket_logit_bias(bias: Any | None) -> np.ndarray | None:
    if bias is None:
        return None
    values = np.asarray(bias, dtype=np.float32)
    if values.shape != (3,):
        raise ValueError(f"bucket_logit_bias must contain exactly 3 values, got shape {values.shape}")
    return values


def apply_bucket_logit_bias(logits: np.ndarray, bias: Any | None = None) -> np.ndarray:
    adjusted = np.asarray(logits, dtype=np.float32)
    logit_bias = normalize_bucket_logit_bias(bias)
    if logit_bias is None:
        return adjusted
    return adjusted + logit_bias.reshape(1, 3)


def bucket_ids_from_logits(logits: np.ndarray, bias: Any | None = None) -> np.ndarray:
    return np.argmax(apply_bucket_logit_bias(logits, bias), axis=1)


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    true_std = float(np.std(y_true))
    pred_std = float(np.std(y_pred))
    if true_std == 0.0 or pred_std == 0.0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def checkpoint_payload(model, config: dict[str, Any], metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    state_model = getattr(model, "_orig_mod", model)
    return {
        "model_state_dict": state_model.state_dict(),
        "config": config,
        "metrics": metrics or {},
    }


def normalize_state_dict_keys(state_dict: dict[str, Any], model_state_dict: dict[str, Any] | None = None) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    if all(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    if model_state_dict is None:
        return state_dict

    remapped: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("head."):
            parts = key.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                layer_idx = int(parts[1])
                suffix = ".".join(parts[2:])
                if layer_idx <= 5:
                    new_key = f"head_body.{layer_idx}.{suffix}"
                elif layer_idx == 6:
                    new_key = f"regression_head.{suffix}"
        remapped[new_key] = value
    return {key: value for key, value in remapped.items() if key in model_state_dict and model_state_dict[key].shape == value.shape}
