#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
from collections import deque
import json
import os
from typing import Optional
import urllib.request

import cv2
import numpy as np
import torch
import torch.jit
import torch.nn as nn
try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_tasks_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions as MPBaseOptions
except Exception:
    mp = None
    mp_tasks_vision = None
    MPBaseOptions = None


IMAGE_SIZE = 224
CLASS_NAMES = ["A", "B", "C", "D", "E"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEDIAPIPE_MODEL_ENV = "MEDIAPIPE_HAND_LANDMARKER_MODEL"
MEDIAPIPE_DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MEDIAPIPE_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "hand_landmarker.task")
LETTERS_PRESET_RUNS = ["9", "12", "13", "15", "18"]
NUMBERS_PRESET_RUNS = ["24", "25", "27"]


class MediaPipeHandCropper:
    def __init__(
        self,
        model_path: str,
        min_detection_confidence: float = 0.45,
        min_tracking_confidence: float = 0.45,
        padding_ratio: float = 0.30,
    ) -> None:
        if (
            mp is None
            or mp_tasks_vision is None
            or MPBaseOptions is None
            or not hasattr(mp, "Image")
            or not hasattr(mp, "ImageFormat")
        ):
            raise RuntimeError(
                "MediaPipe Tasks APIs are unavailable in this environment. "
                "Install/upgrade mediapipe or run with --disable-mediapipe."
            )
        self.padding_ratio = float(max(0.0, padding_ratio))
        options = mp_tasks_vision.HandLandmarkerOptions(
            base_options=MPBaseOptions(model_asset_path=model_path),
            running_mode=mp_tasks_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self._landmarker = mp_tasks_vision.HandLandmarker.create_from_options(options)

    def _squared_bbox(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        pad_x = int(box_w * self.padding_ratio)
        pad_y = int(box_h * self.padding_ratio)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)

        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        side = max(box_w, box_h)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half = side // 2

        sx1 = max(0, cx - half)
        sy1 = max(0, cy - half)
        sx2 = min(width, sx1 + side)
        sy2 = min(height, sy1 + side)
        sx1 = max(0, sx2 - side)
        sy1 = max(0, sy2 - side)
        return sx1, sy1, sx2, sy2

    def detect_bbox(self, frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        hand_landmarks = getattr(result, "hand_landmarks", None)
        if not hand_landmarks:
            return None

        landmarks = hand_landmarks[0]
        xs = np.array([lm.x for lm in landmarks], dtype=np.float32)
        ys = np.array([lm.y for lm in landmarks], dtype=np.float32)

        x1 = int(np.clip(xs.min() * width, 0, width - 1))
        y1 = int(np.clip(ys.min() * height, 0, height - 1))
        x2 = int(np.clip(xs.max() * width, 0, width - 1))
        y2 = int(np.clip(ys.max() * height, 0, height - 1))
        return self._squared_bbox(x1, y1, x2, y2, width, height)

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass


def resolve_mediapipe_model_path(cli_path: str) -> Optional[str]:
    candidates = []
    if cli_path:
        candidates.append(cli_path)
    env_path = os.getenv(MEDIAPIPE_MODEL_ENV, "").strip()
    if env_path:
        candidates.append(env_path)
    candidates.extend(
        [
            DEFAULT_MEDIAPIPE_MODEL_PATH,
            os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task"),
            os.path.join(PROJECT_ROOT, "hand_landmarker.task"),
        ]
    )

    for candidate in candidates:
        expanded = os.path.abspath(os.path.expanduser(candidate))
        if os.path.isfile(expanded):
            return expanded

    os.makedirs(os.path.dirname(DEFAULT_MEDIAPIPE_MODEL_PATH), exist_ok=True)
    tmp_target = f"{DEFAULT_MEDIAPIPE_MODEL_PATH}.tmp"
    try:
        print(
            f"Downloading MediaPipe model to {DEFAULT_MEDIAPIPE_MODEL_PATH} "
            f"from {MEDIAPIPE_DEFAULT_MODEL_URL}"
        )
        if os.path.exists(tmp_target):
            os.remove(tmp_target)
        urllib.request.urlretrieve(MEDIAPIPE_DEFAULT_MODEL_URL, tmp_target)
        os.replace(tmp_target, DEFAULT_MEDIAPIPE_MODEL_PATH)
        if (
            os.path.isfile(DEFAULT_MEDIAPIPE_MODEL_PATH)
            and os.path.getsize(DEFAULT_MEDIAPIPE_MODEL_PATH) > 0
        ):
            print("MediaPipe hand_landmarker.task downloaded successfully.")
            return DEFAULT_MEDIAPIPE_MODEL_PATH
    except Exception as exc:
        print(f"Failed to auto-download hand_landmarker.task: {exc}")
    finally:
        if os.path.exists(tmp_target):
            try:
                os.remove(tmp_target)
            except OSError:
                pass
    return None


class PredictionSmoother:
    def __init__(self, window_size: int = 5) -> None:
        self.window = deque(maxlen=max(1, int(window_size)))

    def clear(self) -> None:
        self.window.clear()

    def push(self, probs: np.ndarray) -> np.ndarray:
        self.window.append(probs)
        stacked = np.stack(self.window, axis=0)
        return stacked.mean(axis=0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ASLModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 192),
            ConvBlock(192, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.pool(features)
        flattened = torch.flatten(pooled, start_dim=1)
        return self.classifier(flattened)


class LiteInvertedResidual2d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        hidden_ch: int,
        stride: int,
        kernel_size: int,
        first_stage: bool = False,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.use_residual = stride == 1 and in_ch == out_ch
        if first_stage:
            self.block = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(
                        in_ch,
                        in_ch,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=in_ch,
                        bias=False,
                    ),
                    nn.BatchNorm2d(in_ch),
                ),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.block = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_ch),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        hidden_ch,
                        hidden_ch,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=hidden_ch,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_ch),
                ),
                nn.Conv2d(hidden_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if len(self.block) == 4:
            x = self.act(self.block[0](x))
            x = self.act(self.block[1](x))
            x = self.block[2](x)
            x = self.block[3](x)
        else:
            x = self.act(self.block[0](x))
            x = self.block[1](x)
            x = self.block[2](x)
        if self.use_residual:
            x = x + identity
        return self.act(x)


class LiteASLModel(nn.Module):
    def __init__(self, state_dict: dict[str, torch.Tensor], num_classes: int) -> None:
        super().__init__()
        stem_weight = state_dict["stem.0.weight"]
        stem_out = int(stem_weight.shape[0])
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_out),
            nn.SiLU(inplace=True),
        )

        feature_indices = sorted(
            {
                int(key.split(".")[1])
                for key in state_dict.keys()
                if key.startswith("features.") and ".block." in key
            }
        )
        blocks: list[nn.Module] = []
        current_in = stem_out
        for idx in feature_indices:
            prefix = f"features.{idx}.block."
            first_stage_key = f"{prefix}1.weight"
            if first_stage_key in state_dict:
                proj_weight = state_dict[first_stage_key]
                out_ch = int(proj_weight.shape[0])
                dw_weight = state_dict[f"{prefix}0.0.weight"]
                kernel_size = int(dw_weight.shape[-1])
                stride = 1 if out_ch == current_in else 2
                block = LiteInvertedResidual2d(
                    in_ch=current_in,
                    out_ch=out_ch,
                    hidden_ch=current_in,
                    stride=stride,
                    kernel_size=kernel_size,
                    first_stage=True,
                )
            else:
                expand_weight = state_dict[f"{prefix}0.0.weight"]
                hidden_ch = int(expand_weight.shape[0])
                proj_weight = state_dict[f"{prefix}2.weight"]
                out_ch = int(proj_weight.shape[0])
                dw_weight = state_dict[f"{prefix}1.0.weight"]
                kernel_size = int(dw_weight.shape[-1])
                stride = 1 if out_ch == current_in else 2
                block = LiteInvertedResidual2d(
                    in_ch=current_in,
                    out_ch=out_ch,
                    hidden_ch=hidden_ch,
                    stride=stride,
                    kernel_size=kernel_size,
                    first_stage=False,
                )
            blocks.append(block)
            current_in = out_ch

        self.features = nn.Sequential(*blocks)
        head_weight = state_dict["head.0.weight"]
        head_out = int(head_weight.shape[0])
        self.head = nn.Sequential(
            nn.Conv2d(current_in, head_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_out),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(head_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


_torch_load = torch.load


def _torch_load_with_map_call(*args, **kwargs):
    if "map_call" in kwargs and "map_location" not in kwargs:
        kwargs["map_location"] = kwargs.pop("map_call")
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_with_map_call


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    frame = preprocess_roi(frame)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - MEAN) / STD
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    return tensor


def preprocess_roi(image: np.ndarray) -> np.ndarray:
    if image.size == 0:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(
        cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR
    )

    denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)
    blur = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(denoised, 1.4, blur, -0.4, 0)
    return sharpened


def _crop_by_ratio(image: np.ndarray, ratio: float, anchor_x: float, anchor_y: float) -> np.ndarray:
    height, width = image.shape[:2]
    ratio = float(np.clip(ratio, 0.5, 1.0))
    crop_w = max(1, int(round(width * ratio)))
    crop_h = max(1, int(round(height * ratio)))

    max_x = max(0, width - crop_w)
    max_y = max(0, height - crop_h)
    start_x = int(round(np.clip(anchor_x, 0.0, 1.0) * max_x))
    start_y = int(round(np.clip(anchor_y, 0.0, 1.0) * max_y))
    end_x = min(width, start_x + crop_w)
    end_y = min(height, start_y + crop_h)
    return image[start_y:end_y, start_x:end_x]


def build_occlusion_tta_views(image: np.ndarray, ratio: float) -> list[np.ndarray]:
    if image.size == 0:
        return [image]
    height, width = image.shape[:2]
    if min(height, width) < 32:
        return [image]

    anchors = [
        (0.5, 0.5),  # Center
        (0.0, 0.5),  # Left
        (1.0, 0.5),  # Right
        (0.5, 0.0),  # Top
        (0.5, 1.0),  # Bottom
    ]
    views = [image]
    for anchor_x, anchor_y in anchors:
        crop = _crop_by_ratio(image, ratio, anchor_x, anchor_y)
        if crop.size:
            views.append(crop)
    return views


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    preset_group = parser.add_mutually_exclusive_group()
    preset_group.add_argument(
        "--letters",
        action="store_true",
        help="Use preset ensemble for A-E detection: runs 9,12,13,15,18.",
    )
    preset_group.add_argument(
        "--numbers",
        action="store_true",
        help="Use preset ensemble for number detection: runs 24,25,27.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Path to a single model checkpoint (legacy option).",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated checkpoint paths for ensemble averaging.",
    )
    parser.add_argument(
        "--model-weights",
        default="",
        help="Optional comma-separated ensemble weights (same order as --models).",
    )
    parser.add_argument("--manual", help="Path to a single image to run instead of webcam mode.")
    parser.add_argument(
        "--eval-dir",
        default="",
        help=(
            "Run offline evaluation from a labeled directory tree "
            "(eval_dir/<class_name>/*.jpg)."
        ),
    )
    parser.add_argument(
        "--eval-report",
        default="",
        help="Optional JSON output path for evaluation report.",
    )
    parser.add_argument(
        "--eval-preds-csv",
        default="",
        help="Optional CSV output path for per-sample predictions.",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=0,
        help="Optional max number of eval images to process (0 = all).",
    )
    parser.add_argument(
        "--diagnose-stack",
        action="store_true",
        help=(
            "With --eval-dir and multiple models, report per-model metrics and "
            "disagreement/oracle diagnostics."
        ),
    )
    parser.add_argument(
        "--disable-mediapipe",
        action="store_true",
        help="Disable MediaPipe hand detection and only use contour fallback.",
    )
    parser.add_argument(
        "--mp-min-detect-conf",
        type=float,
        default=0.45,
        help="MediaPipe min hand detection confidence in [0, 1].",
    )
    parser.add_argument(
        "--mp-min-track-conf",
        type=float,
        default=0.45,
        help="MediaPipe min tracking confidence in [0, 1].",
    )
    parser.add_argument(
        "--mp-padding",
        type=float,
        default=0.30,
        help="Extra bbox padding ratio for MediaPipe hand crop.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Number of frames for probability smoothing in webcam mode.",
    )
    parser.add_argument(
        "--mediapipe-model-path",
        default="",
        help=(
            "Path to hand_landmarker.task. If unset, uses "
            "$MEDIAPIPE_HAND_LANDMARKER_MODEL, local defaults, then auto-download."
        ),
    )
    parser.add_argument(
        "--occlusion-tta",
        action="store_true",
        help=(
            "Average predictions across center/edge crops to improve robustness when "
            "only part of the hand is visible."
        ),
    )
    parser.add_argument(
        "--occlusion-tta-ratio",
        type=float,
        default=0.82,
        help="Crop ratio in (0.5, 1.0] used by --occlusion-tta.",
    )
    parser.add_argument(
        "--auto-models",
        action="store_true",
        help=(
            "Auto-discover checkpoints from numbered run folders and build a stacked "
            "ensemble from top validation runs."
        ),
    )
    parser.add_argument(
        "--auto-root",
        default=PROJECT_ROOT,
        help="Root folder to scan for numbered run directories when using --auto-models.",
    )
    parser.add_argument(
        "--auto-top-k",
        type=int,
        default=6,
        help="Number of top runs to include when using --auto-models.",
    )
    parser.add_argument(
        "--auto-min-val-acc",
        type=float,
        default=0.90,
        help="Minimum val_accuracy required for auto model selection.",
    )
    parser.add_argument(
        "--auto-min-val-f1",
        type=float,
        default=0.90,
        help="Minimum val_f1 required for auto model selection.",
    )
    parser.add_argument(
        "--auto-weight-by",
        choices=["score", "val_f1", "val_accuracy", "uniform"],
        default="score",
        help=(
            "Weight source for --auto-models. 'score' uses 0.6*val_f1 + 0.4*val_accuracy."
        ),
    )
    return parser.parse_args()


def load_model(model_path: str, device: torch.device) -> nn.Module:
    global CLASS_NAMES
    try:
        loaded = torch.jit.load(model_path, map_location=device)
        loaded.eval()
        return loaded
    except RuntimeError:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(
            state_dict["state_dict"], dict
        ):
            state_dict = state_dict["state_dict"]
        elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
            state_dict = checkpoint["state_dict"]
        classifier_weight = state_dict.get("classifier.weight")
        num_classes = (
            classifier_weight.shape[0]
            if isinstance(classifier_weight, torch.Tensor)
            else len(CLASS_NAMES)
        )
        checkpoint_names = (
            checkpoint.get("class_names")
            or checkpoint.get("classes")
            or checkpoint.get("labels")
            or []
        )
        if isinstance(checkpoint_names, (list, tuple)) and len(checkpoint_names) == num_classes:
            CLASS_NAMES = [str(name) for name in checkpoint_names]
        elif len(CLASS_NAMES) != num_classes:
            CLASS_NAMES = [f"class_{i}" for i in range(num_classes)]

        try:
            model = ASLModel(num_classes=num_classes).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except RuntimeError:
            if not (
                isinstance(state_dict, dict)
                and "stem.0.weight" in state_dict
                and "head.0.weight" in state_dict
                and "classifier.weight" in state_dict
            ):
                raise
            model = LiteASLModel(state_dict=state_dict, num_classes=num_classes).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            return model


def predict_probs(model: nn.Module, image: np.ndarray, device: torch.device) -> np.ndarray:
    input_tensor = preprocess_frame(image).to(device)
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    return probabilities


def _preset_model_paths(run_ids: list[str]) -> list[str]:
    return [os.path.join(PROJECT_ROOT, run_id, "best_model.pt") for run_id in run_ids]


def parse_model_paths(
    single_model: str,
    models_csv: str,
    use_letters_preset: bool,
    use_numbers_preset: bool,
) -> list[str]:
    if use_letters_preset:
        return _preset_model_paths(LETTERS_PRESET_RUNS)
    if use_numbers_preset:
        return _preset_model_paths(NUMBERS_PRESET_RUNS)
    if models_csv.strip():
        model_paths = [path.strip() for path in models_csv.split(",") if path.strip()]
    elif single_model.strip():
        model_paths = [single_model.strip()]
    else:
        raise RuntimeError("Provide --letters, --numbers, --model, or --models.")
    return model_paths


def _read_latest_metrics(metrics_path: str) -> Optional[dict[str, float]]:
    try:
        with open(metrics_path, "r", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return None
        latest = rows[-1]
        val_accuracy = float(latest.get("val_accuracy", "nan"))
        val_f1 = float(latest.get("val_f1", "nan"))
        if not np.isfinite(val_accuracy) or not np.isfinite(val_f1):
            return None
        return {
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
            "score": 0.6 * val_f1 + 0.4 * val_accuracy,
        }
    except Exception:
        return None


def discover_auto_models(
    root_dir: str,
    top_k: int,
    min_val_acc: float,
    min_val_f1: float,
) -> list[dict[str, float | str]]:
    if top_k <= 0:
        raise RuntimeError("--auto-top-k must be >= 1.")

    root_abs = os.path.abspath(os.path.expanduser(root_dir))
    if not os.path.isdir(root_abs):
        raise RuntimeError(f"--auto-root does not exist: {root_abs}")

    candidates: list[dict[str, float | str]] = []
    for entry in os.listdir(root_abs):
        if not entry.isdigit():
            continue
        run_dir = os.path.join(root_abs, entry)
        if not os.path.isdir(run_dir):
            continue
        model_path = os.path.join(run_dir, "best_model.pt")
        metrics_path = os.path.join(run_dir, "metrics.csv")
        if not os.path.isfile(model_path) or not os.path.isfile(metrics_path):
            continue
        metrics = _read_latest_metrics(metrics_path)
        if metrics is None:
            continue
        if metrics["val_accuracy"] < float(min_val_acc):
            continue
        if metrics["val_f1"] < float(min_val_f1):
            continue
        candidates.append(
            {
                "run_id": entry,
                "model_path": model_path,
                "val_accuracy": float(metrics["val_accuracy"]),
                "val_f1": float(metrics["val_f1"]),
                "score": float(metrics["score"]),
            }
        )

    if not candidates:
        raise RuntimeError(
            "No auto-selected models matched thresholds. "
            "Lower --auto-min-val-acc / --auto-min-val-f1 or pass --models directly."
        )

    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    return candidates[:top_k]


def derive_auto_weights(
    selected: list[dict[str, float | str]],
    mode: str,
) -> np.ndarray:
    if mode == "uniform":
        return np.ones((len(selected),), dtype=np.float32) / float(len(selected))

    key = "score"
    if mode == "val_f1":
        key = "val_f1"
    elif mode == "val_accuracy":
        key = "val_accuracy"

    raw = np.array([float(item[key]) for item in selected], dtype=np.float32)
    raw = np.clip(raw, 1e-8, None)
    total = float(raw.sum())
    if total <= 0.0:
        return np.ones((len(selected),), dtype=np.float32) / float(len(selected))
    return raw / total


def parse_model_weights(weights_csv: str, num_models: int) -> np.ndarray:
    if not weights_csv.strip():
        return np.ones((num_models,), dtype=np.float32) / float(num_models)

    weights = np.array(
        [float(item.strip()) for item in weights_csv.split(",") if item.strip()],
        dtype=np.float32,
    )
    if weights.shape[0] != num_models:
        raise RuntimeError(
            f"--model-weights count ({weights.shape[0]}) must match number of models ({num_models})."
        )
    if np.any(weights < 0):
        raise RuntimeError("--model-weights must be non-negative.")
    total = float(weights.sum())
    if total <= 0.0:
        raise RuntimeError("--model-weights sum must be > 0.")
    return weights / total


def load_models(model_paths: list[str], device: torch.device) -> list[nn.Module]:
    global CLASS_NAMES
    models: list[nn.Module] = []
    reference_class_names: Optional[list[str]] = None

    for model_path in model_paths:
        if not os.path.isfile(model_path):
            raise RuntimeError(f"Model file not found: {model_path}")
        model = load_model(model_path, device)
        current_class_names = list(CLASS_NAMES)

        if reference_class_names is None:
            reference_class_names = current_class_names
        else:
            if len(current_class_names) != len(reference_class_names):
                raise RuntimeError(
                    "All ensemble models must have the same number of output classes."
                )
            if current_class_names != reference_class_names:
                print(
                    "Warning: class names differ between checkpoints. "
                    "Using class names from the first model."
                )
            CLASS_NAMES = reference_class_names

        models.append(model)
    return models


def predict_probs_ensemble(
    models: list[nn.Module],
    image: np.ndarray,
    device: torch.device,
    model_weights: np.ndarray,
    occlusion_tta: bool = False,
    occlusion_tta_ratio: float = 0.82,
) -> np.ndarray:
    per_model = predict_probs_per_model(
        models=models,
        image=image,
        device=device,
        occlusion_tta=occlusion_tta,
        occlusion_tta_ratio=occlusion_tta_ratio,
    )
    return combine_weighted_probs(per_model, model_weights)


def predict_probs_per_model(
    models: list[nn.Module],
    image: np.ndarray,
    device: torch.device,
    occlusion_tta: bool = False,
    occlusion_tta_ratio: float = 0.82,
) -> list[np.ndarray]:
    views = (
        build_occlusion_tta_views(image, occlusion_tta_ratio)
        if occlusion_tta
        else [image]
    )
    per_model_sums: list[Optional[np.ndarray]] = [None] * len(models)
    for view in views:
        input_tensor = preprocess_frame(view).to(device)
        for idx, model in enumerate(models):
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
            if per_model_sums[idx] is None:
                per_model_sums[idx] = probabilities
            else:
                per_model_sums[idx] = per_model_sums[idx] + probabilities

    if not per_model_sums:
        raise RuntimeError("No models available for inference.")

    per_model_avg = []
    divisor = float(len(views))
    for model_sum in per_model_sums:
        if model_sum is None:
            raise RuntimeError("No view predictions available for a model.")
        per_model_avg.append(model_sum / divisor)
    return per_model_avg


def combine_weighted_probs(per_model_probs: list[np.ndarray], model_weights: np.ndarray) -> np.ndarray:
    if not per_model_probs:
        raise RuntimeError("No models available for inference.")
    if len(per_model_probs) != int(model_weights.shape[0]):
        raise RuntimeError("Model weights must match number of model predictions.")
    combined: Optional[np.ndarray] = None
    for idx, probabilities in enumerate(per_model_probs):
        weighted = probabilities * float(model_weights[idx])
        if combined is None:
            combined = weighted
        else:
            combined = combined + weighted
    if combined is None:
        raise RuntimeError("Failed to combine ensemble probabilities.")
    return combined


def decode_prediction(probabilities: np.ndarray) -> tuple[str, float]:
    predicted = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted])
    label = CLASS_NAMES[predicted] if predicted < len(CLASS_NAMES) else f"class_{predicted}"
    return label, confidence


def detect_sign_region(image: np.ndarray) -> tuple[int, int, int, int]:
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        height, width = image.shape[:2]
        return 0, 0, width, height

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    pad_x = max(int(w * 0.15), 10)
    pad_y = max(int(h * 0.15), 10)
    height, width = image.shape[:2]

    x1 = max(x - pad_x, 0)
    y1 = max(y - pad_y, 0)
    x2 = min(x + w + pad_x, width)
    y2 = min(y + h + pad_y, height)
    return x1, y1, x2, y2


def detect_roi(
    image: np.ndarray, cropper: Optional[MediaPipeHandCropper]
) -> tuple[int, int, int, int, str]:
    if cropper is not None:
        bbox = cropper.detect_bbox(image)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            return x1, y1, x2, y2, "MediaPipe"

    x1, y1, x2, y2 = detect_sign_region(image)
    return x1, y1, x2, y2, "Fallback"


def run_manual_mode(
    models: list[nn.Module],
    model_weights: np.ndarray,
    image_path: str,
    device: torch.device,
    cropper: Optional[MediaPipeHandCropper],
    occlusion_tta: bool,
    occlusion_tta_ratio: float,
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Unable to read image: {image_path}")

    x1, y1, x2, y2, source = detect_roi(image, cropper)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise RuntimeError("Detected hand crop is empty.")

    probs = predict_probs_ensemble(
        models,
        crop,
        device,
        model_weights,
        occlusion_tta=occlusion_tta,
        occlusion_tta_ratio=occlusion_tta_ratio,
    )
    label, confidence = decode_prediction(probs)
    overlay = f"{label} ({confidence * 100.0:.1f}%) [{source}]"

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text_y = max(y1 - 10, 30)
    cv2.putText(
        image,
        overlay,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("ASL Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _compute_metrics(y_true: list[int], y_pred: list[int], num_classes: int) -> dict[str, object]:
    if not y_true:
        raise RuntimeError("No evaluation samples were processed.")
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        confusion[truth, pred] += 1

    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / total) if total > 0 else 0.0
    class_metrics: list[dict[str, float | int | str]] = []
    f1_values: list[float] = []
    for idx in range(num_classes):
        tp = int(confusion[idx, idx])
        fp = int(confusion[:, idx].sum() - tp)
        fn = int(confusion[idx, :].sum() - tp)
        support = int(confusion[idx, :].sum())
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        if precision + recall > 0.0:
            f1 = float((2.0 * precision * recall) / (precision + recall))
        else:
            f1 = 0.0
        f1_values.append(f1)
        class_metrics.append(
            {
                "class_index": idx,
                "class_name": CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}",
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    macro_f1 = float(np.mean(np.array(f1_values, dtype=np.float64)))
    return {
        "num_samples": len(y_true),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": confusion.tolist(),
        "class_metrics": class_metrics,
    }


def _collect_eval_samples(eval_dir: str, eval_limit: int) -> list[tuple[str, int, str]]:
    root = os.path.abspath(os.path.expanduser(eval_dir))
    if not os.path.isdir(root):
        raise RuntimeError(f"--eval-dir does not exist: {root}")

    class_to_index = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    samples: list[tuple[str, int, str]] = []
    for class_name, class_index in class_to_index.items():
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: missing eval class directory: {class_dir}")
            continue
        for walk_root, _, filenames in os.walk(class_dir):
            for filename in filenames:
                _, ext = os.path.splitext(filename)
                if ext.lower() not in VALID_IMAGE_EXTENSIONS:
                    continue
                image_path = os.path.join(walk_root, filename)
                samples.append((image_path, class_index, class_name))

    samples.sort(key=lambda item: item[0])
    if eval_limit > 0:
        samples = samples[:eval_limit]
    return samples


def run_eval_mode(
    models: list[nn.Module],
    model_weights: np.ndarray,
    model_paths: list[str],
    eval_dir: str,
    device: torch.device,
    cropper: Optional[MediaPipeHandCropper],
    occlusion_tta: bool,
    occlusion_tta_ratio: float,
    eval_report: str,
    eval_preds_csv: str,
    eval_limit: int,
    diagnose_stack: bool,
) -> None:
    samples = _collect_eval_samples(eval_dir, eval_limit)
    if not samples:
        raise RuntimeError("No evaluation images found for the current CLASS_NAMES in --eval-dir.")

    y_true: list[int] = []
    y_pred_ens: list[int] = []
    confidences: list[float] = []
    model_preds_by_sample: list[list[int]] = []
    rows: list[dict[str, object]] = []
    skipped_read = 0
    skipped_empty_crop = 0

    for idx, (image_path, true_index, true_name) in enumerate(samples, start=1):
        image = cv2.imread(image_path)
        if image is None:
            skipped_read += 1
            continue

        x1, y1, x2, y2, source = detect_roi(image, cropper)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            skipped_empty_crop += 1
            continue

        per_model_probs = predict_probs_per_model(
            models=models,
            image=crop,
            device=device,
            occlusion_tta=occlusion_tta,
            occlusion_tta_ratio=occlusion_tta_ratio,
        )
        ensemble_probs = combine_weighted_probs(per_model_probs, model_weights)
        ens_pred = int(np.argmax(ensemble_probs))
        ens_conf = float(ensemble_probs[ens_pred])
        per_model_pred = [int(np.argmax(item)) for item in per_model_probs]

        y_true.append(true_index)
        y_pred_ens.append(ens_pred)
        confidences.append(ens_conf)
        model_preds_by_sample.append(per_model_pred)

        if eval_preds_csv:
            row: dict[str, object] = {
                "path": image_path,
                "true_index": true_index,
                "true_label": true_name,
                "pred_index": ens_pred,
                "pred_label": CLASS_NAMES[ens_pred] if ens_pred < len(CLASS_NAMES) else str(ens_pred),
                "confidence": ens_conf,
                "correct": int(ens_pred == true_index),
                "roi_source": source,
            }
            if diagnose_stack and len(models) > 1:
                for model_idx, model_pred in enumerate(per_model_pred):
                    row[f"model_{model_idx}_path"] = model_paths[model_idx]
                    row[f"model_{model_idx}_pred"] = (
                        CLASS_NAMES[model_pred] if model_pred < len(CLASS_NAMES) else str(model_pred)
                    )
            rows.append(row)

        if idx % 100 == 0:
            print(f"Evaluated {idx}/{len(samples)} images...")

    if not y_true:
        raise RuntimeError(
            "Evaluation finished with zero usable samples. Check image paths/ROI detection."
        )

    ensemble_metrics = _compute_metrics(y_true, y_pred_ens, len(CLASS_NAMES))
    report: dict[str, object] = {
        "mode": "evaluation",
        "num_samples_total": len(samples),
        "num_samples_used": len(y_true),
        "num_samples_skipped_read": skipped_read,
        "num_samples_skipped_empty_crop": skipped_empty_crop,
        "class_names": CLASS_NAMES,
        "models": model_paths,
        "model_weights": model_weights.tolist(),
        "occlusion_tta": bool(occlusion_tta),
        "occlusion_tta_ratio": float(occlusion_tta_ratio),
        "ensemble": ensemble_metrics,
        "mean_confidence": float(np.mean(np.array(confidences, dtype=np.float64))),
    }

    if diagnose_stack and len(models) > 1:
        per_model_metrics: list[dict[str, object]] = []
        for model_idx, model_path in enumerate(model_paths):
            preds = [row[model_idx] for row in model_preds_by_sample]
            per_model_metrics.append(
                {
                    "model_path": model_path,
                    **_compute_metrics(y_true, preds, len(CLASS_NAMES)),
                }
            )

        oracle_correct = 0
        for sample_idx, truth in enumerate(y_true):
            pred_list = model_preds_by_sample[sample_idx]
            if any(pred == truth for pred in pred_list):
                oracle_correct += 1
        oracle_accuracy = float(oracle_correct / len(y_true))

        pairwise_disagreement: list[dict[str, object]] = []
        for i in range(len(model_paths)):
            for j in range(i + 1, len(model_paths)):
                disagree = 0
                for preds in model_preds_by_sample:
                    if preds[i] != preds[j]:
                        disagree += 1
                pairwise_disagreement.append(
                    {
                        "model_i": model_paths[i],
                        "model_j": model_paths[j],
                        "disagreement_rate": float(disagree / len(model_preds_by_sample)),
                    }
                )
        report["diagnostics"] = {
            "per_model": per_model_metrics,
            "oracle_accuracy_any_model_correct": oracle_accuracy,
            "pairwise_disagreement": pairwise_disagreement,
        }

    print(
        "Evaluation summary: "
        f"used={report['num_samples_used']}/{report['num_samples_total']} "
        f"acc={ensemble_metrics['accuracy']:.6f} "
        f"macro_f1={ensemble_metrics['macro_f1']:.6f}"
    )

    if eval_report:
        report_path = os.path.abspath(os.path.expanduser(eval_report))
        report_dir = os.path.dirname(report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Wrote evaluation report: {report_path}")

    if eval_preds_csv:
        csv_path = os.path.abspath(os.path.expanduser(eval_preds_csv))
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        fieldnames: list[str] = []
        if rows:
            fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote per-sample predictions: {csv_path}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    auto_weights: Optional[np.ndarray] = None
    if args.auto_models and (args.letters or args.numbers):
        raise RuntimeError("Use either --auto-models or a preset (--letters/--numbers), not both.")
    if args.auto_models:
        selected = discover_auto_models(
            root_dir=args.auto_root,
            top_k=args.auto_top_k,
            min_val_acc=args.auto_min_val_acc,
            min_val_f1=args.auto_min_val_f1,
        )
        model_paths = [str(item["model_path"]) for item in selected]
        auto_weights = derive_auto_weights(selected, args.auto_weight_by)
        print("Auto-selected models:")
        for item in selected:
            print(
                "  run="
                f"{item['run_id']} val_acc={float(item['val_accuracy']):.6f} "
                f"val_f1={float(item['val_f1']):.6f} score={float(item['score']):.6f}"
            )
    else:
        model_paths = parse_model_paths(
            args.model,
            args.models,
            use_letters_preset=args.letters,
            use_numbers_preset=args.numbers,
        )
    models = load_models(model_paths, device)
    if args.model_weights.strip():
        model_weights = parse_model_weights(args.model_weights, len(models))
    elif auto_weights is not None:
        model_weights = auto_weights
    else:
        model_weights = parse_model_weights(args.model_weights, len(models))
    print(f"Loaded {len(models)} model(s) for inference.")
    print(f"Model weights: {model_weights.tolist()}")
    cropper = None
    if not args.disable_mediapipe:
        model_path = resolve_mediapipe_model_path(args.mediapipe_model_path)
        if not model_path:
            raise RuntimeError(
                "No MediaPipe hand_landmarker.task was found and auto-download failed. "
                "Provide --mediapipe-model-path or set MEDIAPIPE_HAND_LANDMARKER_MODEL."
            )
        cropper = MediaPipeHandCropper(
            model_path=model_path,
            min_detection_confidence=args.mp_min_detect_conf,
            min_tracking_confidence=args.mp_min_track_conf,
            padding_ratio=args.mp_padding,
        )
    smoother = PredictionSmoother(window_size=args.smooth_window)

    try:
        if args.eval_dir:
            with torch.no_grad():
                run_eval_mode(
                    models=models,
                    model_weights=model_weights,
                    model_paths=model_paths,
                    eval_dir=args.eval_dir,
                    device=device,
                    cropper=cropper,
                    occlusion_tta=args.occlusion_tta,
                    occlusion_tta_ratio=args.occlusion_tta_ratio,
                    eval_report=args.eval_report,
                    eval_preds_csv=args.eval_preds_csv,
                    eval_limit=args.eval_limit,
                    diagnose_stack=args.diagnose_stack,
                )
            return

        if args.manual:
            with torch.no_grad():
                run_manual_mode(
                    models,
                    model_weights,
                    args.manual,
                    device,
                    cropper,
                    occlusion_tta=args.occlusion_tta,
                    occlusion_tta_ratio=args.occlusion_tta_ratio,
                )
            return

        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError("Unable to open webcam.")

        try:
            with torch.no_grad():
                while True:
                    ok, frame = camera.read()
                    if not ok:
                        break

                    x1, y1, x2, y2, source = detect_roi(frame, cropper)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        smoother.clear()
                        overlay = "No hand detected"
                    else:
                        probs = predict_probs_ensemble(
                            models,
                            crop,
                            device,
                            model_weights,
                            occlusion_tta=args.occlusion_tta,
                            occlusion_tta_ratio=args.occlusion_tta_ratio,
                        )
                        smoothed = smoother.push(probs)
                        label, confidence = decode_prediction(smoothed)
                        overlay = f"{label} ({confidence * 100.0:.1f}%) [{source}]"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        overlay,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.imshow("ASL Recognition", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
        finally:
            camera.release()
            cv2.destroyAllWindows()
    finally:
        if cropper is not None:
            cropper.close()


if __name__ == "__main__":
    main()
