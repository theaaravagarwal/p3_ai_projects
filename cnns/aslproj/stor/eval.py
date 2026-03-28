#!/usr/bin/env python3
"""Model architecture evaluator for PyTorch models.

Answers the following questions for an input model:
- Number of Convolution Layers
- Filter Size at each layer
- Pooling Details
- Stride and Padding
- Activation Function at each layer
- Fully Connected Neural Network Input Neurons
- FCN Number of hidden layers
- FCN Number of hidden neurons in each layer
- Number of output classes
- Notes
- Total number of parameters in your model

Usage examples:
  python eval.py --model 0/model_weights.pth
  python eval.py --factory webcam_asl_hf:ASLResNet --kwargs '{"num_classes": 5}'
  python eval.py --factory some_module:build_model --kwargs '{"num_classes": 26}' --checkpoint model.pth
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


ACTIVATION_TYPES = (
    nn.ReLU,
    nn.ReLU6,
    nn.LeakyReLU,
    nn.PReLU,
    nn.ELU,
    nn.SELU,
    nn.CELU,
    nn.GELU,
    nn.SiLU,
    nn.Mish,
    nn.Hardswish,
    nn.Hardtanh,
    nn.Sigmoid,
    nn.Tanh,
    nn.Softmax,
    nn.LogSoftmax,
)

POOL_TYPES = (
    nn.MaxPool2d,
    nn.AvgPool2d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveMaxPool2d,
    nn.FractionalMaxPool2d,
    nn.LPPool2d,
)


@dataclass
class LayerEntry:
    name: str
    type_name: str
    details: str


def _tuple2(v: Any) -> tuple[int, int]:
    if isinstance(v, tuple):
        if len(v) == 2:
            return int(v[0]), int(v[1])
        if len(v) == 1:
            return int(v[0]), int(v[0])
    return int(v), int(v)


def _format_k(k: Any) -> str:
    h, w = _tuple2(k)
    return f"{h}x{w}"


def _format_stride_padding(module: nn.Conv2d) -> str:
    s_h, s_w = _tuple2(module.stride)
    p_h, p_w = _tuple2(module.padding)
    return f"stride ({s_h}, {s_w}), padding ({p_h}, {p_w})"


def _load_from_factory(factory_spec: str, kwargs: dict[str, Any]) -> nn.Module:
    if ":" not in factory_spec:
        raise ValueError("--factory must be in format module_path:ClassOrFunction")

    module_path, symbol_name = factory_spec.split(":", 1)
    module = importlib.import_module(module_path)
    symbol = getattr(module, symbol_name)

    model = symbol(**kwargs) if callable(symbol) else symbol
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"Factory '{factory_spec}' did not return nn.Module (got {type(model).__name__})."
        )
    return model


def _unwrap_state_dict(checkpoint_obj: Any) -> dict[str, torch.Tensor] | None:
    if isinstance(checkpoint_obj, dict):
        if all(isinstance(k, str) for k in checkpoint_obj.keys()) and any(
            torch.is_tensor(v) for v in checkpoint_obj.values()
        ):
            return checkpoint_obj
        for key in ("state_dict", "model_state_dict", "model", "net"):
            maybe = checkpoint_obj.get(key)
            if isinstance(maybe, dict) and all(isinstance(k, str) for k in maybe.keys()):
                return maybe
    return None


def _load_checkpoint(model: nn.Module, checkpoint_path: str) -> str:
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        model.load_state_dict(obj.state_dict(), strict=False)
        return "Loaded weights from nn.Module checkpoint with strict=False."

    state_dict = _unwrap_state_dict(obj)
    if state_dict is None:
        return "Checkpoint found, but no compatible state_dict key was detected; skipped loading."

    cleaned_state = {}
    for key, val in state_dict.items():
        if key.startswith("module."):
            cleaned_state[key[len("module.") :]] = val
        else:
            cleaned_state[key] = val

    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    return (
        "Loaded state_dict with strict=False "
        f"(missing={len(missing)}, unexpected={len(unexpected)})."
    )


def _state_dict_param_count(state_dict: dict[str, torch.Tensor]) -> int:
    # Exclude common non-parameter buffers.
    skip_suffixes = ("running_mean", "running_var", "num_batches_tracked")
    return sum(
        int(v.numel())
        for k, v in state_dict.items()
        if torch.is_tensor(v) and not k.endswith(skip_suffixes)
    )


def inspect_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    conv_entries = []
    linear_entries = []
    total_params = _state_dict_param_count(state_dict)

    for key, tensor in state_dict.items():
        if not torch.is_tensor(tensor) or not key.endswith(".weight"):
            continue

        if tensor.ndim == 4:
            layer_name = key[: -len(".weight")]
            out_c, in_c, k_h, k_w = tensor.shape
            conv_entries.append(
                {
                    "layer": layer_name,
                    "filter_size": f"{int(k_h)}x{int(k_w)}",
                    "details": (
                        f"kernel {int(k_h)}x{int(k_w)}, stride/padding unknown from state_dict, "
                        f"in_channels {int(in_c)}, out_channels {int(out_c)}"
                    ),
                }
            )
        elif tensor.ndim == 2:
            layer_name = key[: -len(".weight")]
            out_f, in_f = tensor.shape
            linear_entries.append((layer_name, int(in_f), int(out_f)))

    fcn_input_neurons = linear_entries[0][1] if linear_entries else None
    hidden_layers = max(0, len(linear_entries) - 1)
    hidden_neurons = [out_f for _, _, out_f in linear_entries[:-1]]
    output_classes = linear_entries[-1][2] if linear_entries else None

    return {
        "Number of Convolution Layers": len(conv_entries),
        "Filter Size at each layer": [
            {"layer": e["layer"], "filter_size": e["filter_size"]} for e in conv_entries
        ],
        "Pooling Details": [],
        "Stride and Padding": [
            {"layer": e["layer"], "details": e["details"]} for e in conv_entries
        ],
        "Activation Function at each layer": [],
        "Fully Connected Neural Network Input Neurons": fcn_input_neurons,
        "FCN Number of hidden layers": hidden_layers,
        "FCN Number of hidden neurons in each layer": hidden_neurons,
        "Number of output classes": output_classes,
        "Any other details you want to share": [
            "Model loaded from state_dict only; pooling/activations and exact stride/padding are not fully recoverable."
        ],
        "Total number of parameters in your model": total_params,
        "Trainable parameters": None,
    }


def load_model_from_artifact(model_path: str) -> tuple[nn.Module | None, dict[str, torch.Tensor] | None, list[str]]:
    obj = torch.load(model_path, map_location="cpu", weights_only=False)
    notes: list[str] = []

    if isinstance(obj, nn.Module):
        notes.append("Loaded model artifact as nn.Module.")
        return obj, None, notes

    state_dict = _unwrap_state_dict(obj)
    if state_dict is not None:
        notes.append("Loaded model artifact as state_dict.")
        return None, state_dict, notes

    raise TypeError(
        "Unsupported model artifact. Expected nn.Module or checkpoint containing a state_dict."
    )


def _infer_output_classes(model: nn.Module, input_shape: tuple[int, int, int, int]) -> int | None:
    model.eval()
    try:
        with torch.no_grad():
            x = torch.zeros(input_shape, dtype=torch.float32)
            y = model(x)
            if isinstance(y, (list, tuple)) and y:
                y = y[0]
            if isinstance(y, torch.Tensor) and y.ndim >= 2:
                return int(y.shape[-1])
    except Exception:
        return None
    return None


def inspect_model(model: nn.Module, input_shape: tuple[int, int, int, int]) -> dict[str, Any]:
    convs: list[LayerEntry] = []
    pools: list[LayerEntry] = []
    acts: list[LayerEntry] = []
    linears: list[tuple[str, nn.Linear]] = []

    for name, module in model.named_modules():
        if name == "":
            continue

        if isinstance(module, nn.Conv2d):
            convs.append(
                LayerEntry(
                    name=name,
                    type_name=type(module).__name__,
                    details=(
                        f"kernel {_format_k(module.kernel_size)}, "
                        f"{_format_stride_padding(module)}"
                    ),
                )
            )

        if isinstance(module, POOL_TYPES):
            details = []
            if hasattr(module, "kernel_size"):
                details.append(f"kernel {_format_k(getattr(module, 'kernel_size'))}")
            if hasattr(module, "stride") and getattr(module, "stride") is not None:
                details.append(f"stride {_tuple2(getattr(module, 'stride'))}")
            if hasattr(module, "padding"):
                details.append(f"padding {_tuple2(getattr(module, 'padding'))}")
            if hasattr(module, "output_size"):
                out_size = getattr(module, "output_size")
                if out_size is not None:
                    details.append(f"output_size {_format_k(out_size)}")
            pools.append(
                LayerEntry(
                    name=name,
                    type_name=type(module).__name__,
                    details=", ".join(details) if details else "(no extra attrs)",
                )
            )

        if isinstance(module, ACTIVATION_TYPES):
            acts.append(
                LayerEntry(
                    name=name,
                    type_name=type(module).__name__,
                    details="",
                )
            )

        if isinstance(module, nn.Linear):
            linears.append((name, module))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    fcn_input_neurons = linears[0][1].in_features if linears else None
    output_classes = linears[-1][1].out_features if linears else None
    hidden_layers = max(0, len(linears) - 1)
    hidden_neurons = [lin.out_features for _, lin in linears[:-1]]

    if output_classes is None:
        output_classes = _infer_output_classes(model, input_shape)

    filter_sizes = [
        {
            "layer": layer.name,
            "filter_size": layer.details.split(",")[0].replace("kernel ", ""),
        }
        for layer in convs
    ]

    stride_padding = [{"layer": c.name, "details": c.details} for c in convs]
    pooling_details = [
        {
            "layer": p.name,
            "type": p.type_name,
            "details": p.details,
        }
        for p in pools
    ]
    activation_details = [{"layer": a.name, "activation": a.type_name} for a in acts]

    notes = []
    if pools:
        has_global_gap = any(
            isinstance(m, nn.AdaptiveAvgPool2d) and _tuple2(m.output_size) == (1, 1)
            for m in model.modules()
        )
        if has_global_gap:
            notes.append("Uses AdaptiveAvgPool2d(1x1), i.e., global average pooling.")
    if convs:
        first_conv = convs[0]
        notes.append(f"Stem/first conv: {first_conv.name} -> {first_conv.details}.")

    return {
        "Number of Convolution Layers": len(convs),
        "Filter Size at each layer": filter_sizes,
        "Pooling Details": pooling_details,
        "Stride and Padding": stride_padding,
        "Activation Function at each layer": activation_details,
        "Fully Connected Neural Network Input Neurons": fcn_input_neurons,
        "FCN Number of hidden layers": hidden_layers,
        "FCN Number of hidden neurons in each layer": hidden_neurons,
        "Number of output classes": output_classes,
        "Any other details you want to share": notes,
        "Total number of parameters in your model": total_params,
        "Trainable parameters": trainable_params,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a PyTorch model architecture and report key stats."
    )
    parser.add_argument(
        "--model",
        default="",
        help="Path to a .pt/.pth model artifact (saved nn.Module or state_dict checkpoint).",
    )
    parser.add_argument(
        "--factory",
        default="",
        help="Model factory spec: module_path:ClassOrFunction",
    )
    parser.add_argument(
        "--kwargs",
        default="{}",
        help="JSON kwargs passed into the factory/class constructor.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional .pth/.pt checkpoint to load with strict=False.",
    )
    parser.add_argument(
        "--input-shape",
        default="1,3,224,224",
        help="Input shape for optional output-class inference fallback. Format: N,C,H,W",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional path to save JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model and not args.factory:
        raise ValueError("Provide either --model or --factory.")

    kwargs = json.loads(args.kwargs)
    if not isinstance(kwargs, dict):
        raise ValueError("--kwargs must decode to a JSON object.")

    input_shape = tuple(int(x.strip()) for x in args.input_shape.split(","))
    if len(input_shape) != 4:
        raise ValueError("--input-shape must have 4 integers: N,C,H,W")

    notes = []
    report: dict[str, Any]

    if args.model:
        model_path = str(Path(args.model).expanduser())
        model, loaded_state_dict, artifact_notes = load_model_from_artifact(model_path)
        notes.extend(artifact_notes)
        if model is not None:
            report = inspect_model(model, input_shape)
        else:
            report = inspect_state_dict(loaded_state_dict or {})
    else:
        model = _load_from_factory(args.factory, kwargs)
        if args.checkpoint:
            checkpoint_path = str(Path(args.checkpoint).expanduser())
            notes.append(_load_checkpoint(model, checkpoint_path))
        report = inspect_model(model, input_shape)

    if notes:
        report["Any other details you want to share"].extend(notes)

    print(json.dumps(report, indent=2))

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
