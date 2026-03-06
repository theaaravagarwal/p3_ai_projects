#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import torch

from test import CLASSES, IMG_SIZE, build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure stats for one ASL model checkpoint or all .pt files in this directory."
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        help="Path to a .pt/.pth checkpoint. If omitted, all .pt files in the current directory are measured.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used for the dummy inference timing pass.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="Number of timed inference runs per model.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_state_dict_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "_orig_mod."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        normalized[new_key] = value
    return normalized


def load_checkpoint_state_dict(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return normalize_state_dict_keys(checkpoint["state_dict"])
        if all(torch.is_tensor(value) for value in checkpoint.values()):
            return normalize_state_dict_keys(checkpoint)
    raise ValueError(
        f"{model_path} does not look like a supported PyTorch state_dict checkpoint."
    )


def format_int(value: int) -> str:
    return f"{value:,}"


def format_size_mb(value_bytes: int) -> str:
    return f"{value_bytes / (1024 ** 2):.2f} MB"


def checkpoint_tensor_stats(state_dict) -> tuple[int, int]:
    total_params = 0
    total_bytes = 0
    for value in state_dict.values():
        if not torch.is_tensor(value):
            continue
        total_params += value.numel()
        total_bytes += value.numel() * value.element_size()
    return total_params, total_bytes


def measure_inference_ms(model, device: torch.device, runs: int) -> float:
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)

    with torch.inference_mode():
        for _ in range(5):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(runs):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return (elapsed / runs) * 1000.0


def summarize_model(model_path: Path, device: torch.device, runs: int):
    state_dict = load_checkpoint_state_dict(model_path, device)
    checkpoint_params, checkpoint_bytes = checkpoint_tensor_stats(state_dict)
    model = build_model(len(CLASSES)).to(device)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    param_bytes = sum(param.numel() * param.element_size() for param in model.parameters())
    buffer_bytes = sum(buf.numel() * buf.element_size() for buf in model.buffers())

    compatible = True
    compatibility_error = None
    try:
        load_result = model.load_state_dict(state_dict, strict=False)
        model.eval()
    except RuntimeError as exc:
        compatible = False
        compatibility_error = str(exc)
        load_result = None

    print(f"Model: {model_path}")
    print(f"Checkpoint size: {format_size_mb(model_path.stat().st_size)}")
    print(f"Device used: {device}")
    print(f"Classes: {len(CLASSES)}")
    print(f"Checkpoint tensor parameters: {format_int(checkpoint_params)}")
    print(f"Checkpoint tensor memory: {format_size_mb(checkpoint_bytes)}")
    print(f"Reference model parameters: {format_int(total_params)}")
    print(f"Trainable parameters: {format_int(trainable_params)}")
    print(f"Estimated reference model memory: {format_size_mb(param_bytes + buffer_bytes)}")
    print(f"Compatible with current build_model: {'yes' if compatible else 'no'}")

    if compatible:
        inference_ms = measure_inference_ms(model, device, runs)
        print(f"Average inference time: {inference_ms:.2f} ms over {runs} runs")
        print(f"Approx. inferences/sec: {1000.0 / inference_ms:.2f}")
        print(f"Missing keys: {len(load_result.missing_keys)}")
        print(f"Unexpected keys: {len(load_result.unexpected_keys)}")

        if load_result.missing_keys:
            print("First missing key:", load_result.missing_keys[0])
        if load_result.unexpected_keys:
            print("First unexpected key:", load_result.unexpected_keys[0])
    else:
        first_error_line = compatibility_error.splitlines()[0] if compatibility_error else "Unknown compatibility error."
        print(f"Inference timing: skipped")
        print(f"Compatibility error: {first_error_line}")
    print()


def resolve_model_paths(model_path_arg: str | None):
    if model_path_arg:
        model_path = Path(model_path_arg).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        return [model_path]

    cwd = Path.cwd()
    model_paths = sorted(cwd.glob("*.pt")) + sorted(cwd.glob("*.pth"))
    if not model_paths:
        raise FileNotFoundError("No .pt or .pth files found in the current directory.")
    return model_paths


def main():
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")

    device = resolve_device(args.device)
    for model_path in resolve_model_paths(args.model_path):
        summarize_model(model_path, device, args.runs)


if __name__ == "__main__":
    main()
