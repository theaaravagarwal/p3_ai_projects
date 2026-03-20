#!/usr/bin/env python

from __future__ import annotations

import argparse
from collections import deque
import os
from pathlib import Path
from typing import Optional
import urllib.request

import cv2
import numpy as np
import torch
import torch.jit
import torch.nn as nn
import torchvision.models as tv_models
try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_tasks_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions as MPBaseOptions
except Exception:
    mp = None
    mp_tasks_vision = None
    MPBaseOptions = None


IMAGE_SIZE = 128
CLASS_NAMES = ["A", "B", "C", "D", "E"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MEDIAPIPE_MODEL_ENV = "MEDIAPIPE_HAND_LANDMARKER_MODEL"
MEDIAPIPE_DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MEDIAPIPE_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "hand_landmarker.task")
DEFAULT_BULK_ROOT = os.path.expanduser("~/.cache/kagglehub/datasets")
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


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

    def detect(
        self, frame: np.ndarray
    ) -> tuple[Optional[tuple[int, int, int, int]], Optional[list[tuple[int, int]]]]:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        hand_landmarks = getattr(result, "hand_landmarks", None)
        if not hand_landmarks:
            return None, None

        landmarks = hand_landmarks[0]
        xs = np.array([lm.x for lm in landmarks], dtype=np.float32)
        ys = np.array([lm.y for lm in landmarks], dtype=np.float32)

        x1 = int(np.clip(xs.min() * width, 0, width - 1))
        y1 = int(np.clip(ys.min() * height, 0, height - 1))
        x2 = int(np.clip(xs.max() * width, 0, width - 1))
        y2 = int(np.clip(ys.max() * height, 0, height - 1))
        bbox = self._squared_bbox(x1, y1, x2, y2, width, height)
        points: list[tuple[int, int]] = []
        for landmark in landmarks:
            px = int(np.clip(landmark.x * width, 0, width - 1))
            py = int(np.clip(landmark.y * height, 0, height - 1))
            points.append((px, py))
        return bbox, points

    def detect_bbox(self, frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        bbox, _ = self.detect(frame)
        return bbox

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


class ASLResNet50Model(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = tv_models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ASLEfficientNetB0Model(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = tv_models.efficientnet_b0(weights=None)
        if isinstance(self.backbone.classifier, nn.Sequential) and len(self.backbone.classifier) >= 2:
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            raise RuntimeError("Unexpected EfficientNet classifier structure.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


_torch_load = torch.load


def _torch_load_with_map_call(*args, **kwargs):
    if "map_call" in kwargs and "map_location" not in kwargs:
        kwargs["map_location"] = kwargs.pop("map_call")
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_with_map_call


def preprocess_frame(frame: np.ndarray, use_roi_enhancement: bool = True) -> torch.Tensor:
    if use_roi_enhancement:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the model checkpoint.")
    parser.add_argument("--manual", help="Path to a single image to run instead of webcam mode.")
    parser.add_argument(
        "--bulk",
        action="store_true",
        help=(
            "Run inference over every image in a directory. "
            "Use --bulk-dir to pick a dataset folder."
        ),
    )
    parser.add_argument(
        "--bulk-root",
        default=DEFAULT_BULK_ROOT,
        help=(
            "Root directory to search when choosing a bulk folder interactively. "
            f"Default: {DEFAULT_BULK_ROOT}"
        ),
    )
    parser.add_argument(
        "--bulk-dir",
        default="",
        help=(
            "Directory of images to process in bulk mode. "
            "If omitted, you will be prompted to choose from --bulk-root."
        ),
    )
    parser.add_argument(
        "--bulk-show",
        default="",
        help=(
            "Comma-separated image names or 1-based indices to display in bulk mode. "
            "Use 'all' to show every processed image."
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
        "--no-mp-landmarks",
        action="store_true",
        help="Disable drawing MediaPipe hand landmarks in the preview.",
    )
    parser.add_argument(
        "--show-isolated-hand",
        action="store_true",
        help="Show an additional preview window of the isolated hand crop used for inference.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Run heavy webcam processing every N frames. 1 means process every frame.",
    )
    parser.add_argument(
        "--webcam-width",
        type=int,
        default=640,
        help="Requested webcam frame width for live mode. Set to 0 to keep the camera default.",
    )
    parser.add_argument(
        "--webcam-height",
        type=int,
        default=480,
        help="Requested webcam frame height for live mode. Set to 0 to keep the camera default.",
    )
    parser.add_argument(
        "--webcam-enhance-crop",
        action="store_true",
        help="Apply the slower CLAHE/bilateral preprocessing path in webcam mode.",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS


def find_bulk_image_files(directory: str) -> list[str]:
    root = Path(os.path.expanduser(directory))
    if not root.is_dir():
        raise RuntimeError(f"Bulk directory does not exist or is not a directory: {directory}")
    files = [path for path in sorted(root.rglob("*")) if is_image_file(path)]
    if not files:
        raise RuntimeError(f"No supported image files found in: {directory}")
    return [str(path) for path in files]


def discover_bulk_directories(root: str) -> list[str]:
    search_root = Path(os.path.expanduser(root))
    if not search_root.exists():
        return []

    candidates: list[Path] = []
    for current_root, dirnames, filenames in os.walk(search_root):
        current_path = Path(current_root)
        if any(Path(name).suffix.lower() in VALID_IMAGE_EXTENSIONS for name in filenames):
            candidates.append(current_path)
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]

    unique_candidates = sorted({str(path) for path in candidates})
    return unique_candidates


def choose_bulk_directory(root: str) -> str:
    candidates = discover_bulk_directories(root)
    if not candidates:
        raise RuntimeError(
            f"No image directories found under bulk root: {os.path.expanduser(root)}"
        )
    if len(candidates) == 1:
        print(f"Using bulk directory: {candidates[0]}")
        return candidates[0]

    print("Choose a bulk directory:")
    for idx, candidate in enumerate(candidates, start=1):
        print(f"  {idx:>2}. {candidate}")

    while True:
        choice = input(f"Enter a number [1-{len(candidates)}]: ").strip()
        try:
            index = int(choice)
        except ValueError:
            print("Enter a numeric selection.")
            continue
        if 1 <= index <= len(candidates):
            selected = candidates[index - 1]
            print(f"Using bulk directory: {selected}")
            return selected
        print("Selection out of range.")


def parse_bulk_show_selection(selection: str, image_files: list[str]) -> set[str]:
    if not selection.strip():
        return set()
    tokens = [token.strip() for token in selection.split(",") if token.strip()]
    if not tokens:
        return set()
    if any(token.lower() == "all" for token in tokens):
        return {os.path.abspath(path) for path in image_files}

    selected: set[str] = set()
    basename_to_paths: dict[str, list[str]] = {}
    for path in image_files:
        basename_to_paths.setdefault(os.path.basename(path).lower(), []).append(path)

    for token in tokens:
        if token.isdigit():
            index = int(token)
            if 1 <= index <= len(image_files):
                selected.add(os.path.abspath(image_files[index - 1]))
            continue
        token_lower = token.lower()
        if token_lower in basename_to_paths:
            selected.update(os.path.abspath(path) for path in basename_to_paths[token_lower])
            continue
        for path in image_files:
            if token_lower in os.path.basename(path).lower():
                selected.add(os.path.abspath(path))
    return selected


def annotate_image(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    label: str,
    confidence: float,
    source: str,
    landmarks: Optional[list[tuple[int, int]]],
) -> np.ndarray:
    frame = image.copy()
    x1, y1, x2, y2 = bbox
    box_color = draw_detection_overlay(
        frame,
        x1,
        y1,
        x2,
        y2,
        source,
        landmarks,
        draw_landmarks=True,
    )
    overlay = f"{label} ({confidence * 100.0:.1f}%) [{source}]"
    text_y = max(y1 - 10, 30)
    cv2.putText(
        frame,
        overlay,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        box_color,
        2,
        cv2.LINE_AA,
    )
    return frame


def run_bulk_mode(
    model: nn.Module,
    directory: str,
    device: torch.device,
    cropper: Optional[MediaPipeHandCropper],
    show_isolated_hand: bool,
    bulk_show_selection: str,
) -> None:
    image_files = find_bulk_image_files(directory)
    selected_to_show = parse_bulk_show_selection(bulk_show_selection, image_files)

    print(f"Found {len(image_files)} image(s) in {directory}")
    if bulk_show_selection.strip() or len(image_files) <= 20:
        print("Processing order:")
        for index, image_path in enumerate(image_files, start=1):
            print(f"  {index:>3}. {os.path.relpath(image_path, directory)}")
    print("Processing in bulk mode...")

    with torch.no_grad():
        for index, image_path in enumerate(image_files, start=1):
            image = cv2.imread(image_path)
            if image is None:
                print(f"[{index}/{len(image_files)}] {os.path.basename(image_path)} -> unreadable")
                continue

            x1, y1, x2, y2, source, landmarks = detect_roi(image, cropper)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                print(f"[{index}/{len(image_files)}] {os.path.basename(image_path)} -> no hand detected")
                continue

            if source == "MediaPipe":
                crop = isolate_hand_from_crop(crop, (x1, y1, x2, y2), landmarks)

            probs = predict_probs(model, crop, device)
            label, confidence = decode_prediction(probs)
            print(
                f"[{index}/{len(image_files)}] {os.path.basename(image_path)} -> "
                f"{label} ({confidence * 100.0:.1f}%) [{source}]"
            )

            abs_path = os.path.abspath(image_path)
            if selected_to_show and abs_path not in selected_to_show:
                continue

            annotated = annotate_image(image, (x1, y1, x2, y2), label, confidence, source, landmarks)
            cv2.imshow("ASL Recognition", annotated)
            if show_isolated_hand:
                cv2.imshow("Isolated Hand", crop)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
    cv2.destroyAllWindows()


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
        if isinstance(state_dict, dict) and any(key.startswith("module.") for key in state_dict):
            state_dict = {
                (key[7:] if key.startswith("module.") else key): value
                for key, value in state_dict.items()
            }
        classifier_weight = state_dict.get("classifier.weight")
        if not isinstance(classifier_weight, torch.Tensor):
            classifier_weight = state_dict.get("backbone.fc.weight")
        if not isinstance(classifier_weight, torch.Tensor):
            classifier_weight = state_dict.get("backbone.classifier.1.weight")
        if not isinstance(classifier_weight, torch.Tensor):
            classifier_weight = state_dict.get("backbone.classifier.weight")
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

        if (
            isinstance(state_dict, dict)
            and "backbone.conv1.weight" in state_dict
            and "backbone.fc.weight" in state_dict
        ):
            model = ASLResNet50Model(num_classes=num_classes).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            return model

        if (
            isinstance(state_dict, dict)
            and "backbone.features.0.0.weight" in state_dict
            and "backbone.classifier.1.weight" in state_dict
        ):
            model = ASLEfficientNetB0Model(num_classes=num_classes).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            return model

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


def predict_probs(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    use_roi_enhancement: bool = True,
) -> np.ndarray:
    input_tensor = preprocess_frame(image, use_roi_enhancement=use_roi_enhancement).to(device)
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    return probabilities


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
) -> tuple[int, int, int, int, str, Optional[list[tuple[int, int]]]]:
    if cropper is not None:
        bbox, landmarks = cropper.detect(image)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            return x1, y1, x2, y2, "MediaPipe", landmarks

    x1, y1, x2, y2 = detect_sign_region(image)
    return x1, y1, x2, y2, "Fallback", None


def isolate_hand_from_crop(
    crop: np.ndarray,
    bbox: tuple[int, int, int, int],
    landmarks: Optional[list[tuple[int, int]]],
) -> np.ndarray:
    if crop.size == 0 or not landmarks:
        return crop

    x1, y1, x2, y2 = bbox
    crop_h, crop_w = crop.shape[:2]
    points: list[tuple[int, int]] = []
    for lx, ly in landmarks:
        px = int(np.clip(lx - x1, 0, crop_w - 1))
        py = int(np.clip(ly - y1, 0, crop_h - 1))
        points.append((px, py))
    if len(points) < 3:
        return crop

    pts = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    kernel = np.ones((7, 7), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    masked_area = int(np.count_nonzero(mask))
    min_area = int(0.03 * crop_h * crop_w)
    if masked_area < min_area:
        return crop

    isolated = cv2.bitwise_and(crop, crop, mask=mask)
    return isolated


def draw_detection_overlay(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    source: str,
    landmarks: Optional[list[tuple[int, int]]],
    draw_landmarks: bool,
) -> tuple[int, int, int]:
    color = (0, 0, 255) if source == "MediaPipe" else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if source == "MediaPipe" and draw_landmarks and landmarks:
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                cv2.line(frame, landmarks[start_idx], landmarks[end_idx], (0, 128, 255), 2)
        for point in landmarks:
            cv2.circle(frame, point, 3, (0, 0, 255), -1)
    return color


def run_manual_mode(
    model: nn.Module,
    image_path: str,
    device: torch.device,
    cropper: Optional[MediaPipeHandCropper],
    show_isolated_hand: bool,
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Unable to read image: {image_path}")

    x1, y1, x2, y2, source, landmarks = detect_roi(image, cropper)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise RuntimeError("Detected hand crop is empty.")
    if source == "MediaPipe":
        crop = isolate_hand_from_crop(crop, (x1, y1, x2, y2), landmarks)
    model_crop_preview = crop.copy()

    probs = predict_probs(model, crop, device)
    label, confidence = decode_prediction(probs)
    overlay = f"{label} ({confidence * 100.0:.1f}%) [{source}]"

    box_color = draw_detection_overlay(
        image,
        x1,
        y1,
        x2,
        y2,
        source,
        landmarks,
        draw_landmarks=True,
    )
    text_y = max(y1 - 10, 30)
    cv2.putText(
        image,
        overlay,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        box_color,
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("ASL Recognition", image)
    if show_isolated_hand:
        cv2.imshow("Isolated Hand", model_crop_preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_webcam_mode(
    model: nn.Module,
    device: torch.device,
    cropper: Optional[MediaPipeHandCropper],
    show_isolated_hand: bool,
    no_mp_landmarks: bool,
    frame_skip: int,
    webcam_width: int,
    webcam_height: int,
    use_roi_enhancement: bool,
    smooth_window: int,
) -> None:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Unable to open webcam.")

    if webcam_width > 0:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(webcam_width))
    if webcam_height > 0:
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(webcam_height))
    try:
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    frame_skip = max(1, int(frame_skip))
    smoother = PredictionSmoother(window_size=smooth_window)
    last_bbox: Optional[tuple[int, int, int, int]] = None
    last_source = "Fallback"
    last_landmarks: Optional[list[tuple[int, int]]] = None
    last_overlay = "Searching for hand..."
    last_box_color = (0, 255, 0)
    last_isolated_preview: Optional[np.ndarray] = None
    frame_index = 0

    try:
        with torch.inference_mode():
            while True:
                ok, frame = camera.read()
                if not ok:
                    break

                should_refresh = frame_index % frame_skip == 0 or last_bbox is None
                if should_refresh:
                    x1, y1, x2, y2, source, landmarks = detect_roi(frame, cropper)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        smoother.clear()
                        last_bbox = None
                        last_source = source
                        last_landmarks = landmarks
                        last_overlay = "No hand detected"
                        last_isolated_preview = None
                    else:
                        if source == "MediaPipe":
                            crop = isolate_hand_from_crop(crop, (x1, y1, x2, y2), landmarks)
                        last_isolated_preview = crop.copy()
                        probs = predict_probs(
                            model,
                            crop,
                            device,
                            use_roi_enhancement=use_roi_enhancement,
                        )
                        smoothed = smoother.push(probs)
                        label, confidence = decode_prediction(smoothed)
                        last_bbox = (x1, y1, x2, y2)
                        last_source = source
                        last_landmarks = landmarks
                        last_overlay = f"{label} ({confidence * 100.0:.1f}%) [{source}]"
                        last_box_color = (0, 0, 255) if source == "MediaPipe" else (0, 255, 0)

                if last_bbox is not None:
                    x1, y1, x2, y2 = last_bbox
                    draw_detection_overlay(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        last_source,
                        last_landmarks,
                        draw_landmarks=not no_mp_landmarks,
                    )
                    cv2.putText(
                        frame,
                        last_overlay,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        last_box_color,
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        frame,
                        last_overlay,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow("ASL Recognition", frame)
                if show_isolated_hand and last_isolated_preview is not None:
                    cv2.imshow("Isolated Hand", last_isolated_preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                frame_index += 1
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
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
        if args.manual:
            with torch.no_grad():
                run_manual_mode(
                    model,
                    args.manual,
                    device,
                    cropper,
                    show_isolated_hand=args.show_isolated_hand,
                )
            return

        if args.bulk:
            bulk_dir = args.bulk_dir.strip() or choose_bulk_directory(args.bulk_root)
            with torch.no_grad():
                run_bulk_mode(
                    model,
                    bulk_dir,
                    device,
                    cropper,
                    show_isolated_hand=args.show_isolated_hand,
                    bulk_show_selection=args.bulk_show,
                )
            return

        run_webcam_mode(
            model=model,
            device=device,
            cropper=cropper,
            show_isolated_hand=args.show_isolated_hand,
            no_mp_landmarks=args.no_mp_landmarks,
            frame_skip=args.frame_skip,
            webcam_width=args.webcam_width,
            webcam_height=args.webcam_height,
            use_roi_enhancement=args.webcam_enhance_crop,
            smooth_window=args.smooth_window,
        )
    finally:
        if cropper is not None:
            cropper.close()


if __name__ == "__main__":
    main()
