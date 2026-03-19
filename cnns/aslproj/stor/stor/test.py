#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import urllib.request

import cv2
import numpy as np
import torch
import torch.jit
import torch.nn as nn
from PIL import Image, ImageOps

try:
    from mediapipe.tasks.python import vision as mp_tasks_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions as MPBaseOptions
    from mediapipe.tasks.python.vision.core.image import (
        Image as MPImage,
        ImageFormat as MPImageFormat,
    )
except Exception:
    mp_tasks_vision = None
    MPBaseOptions = None
    MPImage = None
    MPImageFormat = None


CLASS_NAMES = ["A", "B", "C", "D", "E"]
MEDIAPIPE_DEFAULT_MIN_DET_CONF = 0.35
MEDIAPIPE_MODEL_ENV = "MEDIAPIPE_HAND_LANDMARKER_MODEL"
MEDIAPIPE_DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MEDIAPIPE_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "hand_landmarker.task"
)


class SEBlock1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class SEBlock2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(16, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class InvertedResidual1d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expansion: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        use_se: bool = True,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        hidden = in_ch * expansion
        padding = kernel_size // 2
        self.use_residual = stride == 1 and in_ch == out_ch
        self.drop_prob = drop_prob

        layers = [
            nn.Conv1d(in_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv1d(
                hidden,
                hidden,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm1d(hidden),
            nn.SiLU(inplace=True),
        ]
        if use_se:
            layers.append(SEBlock1d(hidden, reduction=8))
        layers.extend(
            [
                nn.Conv1d(hidden, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            if self.training and self.drop_prob > 0.0:
                keep_prob = 1.0 - self.drop_prob
                shape = (out.shape[0], 1, 1)
                mask = (
                    keep_prob + torch.rand(shape, dtype=out.dtype, device=out.device)
                ).floor_()
                out = out.div(keep_prob) * mask
            out = out + x
        return out


class InvertedResidual2d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expansion: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        use_se: bool = True,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        hidden = in_ch * expansion
        padding = kernel_size // 2
        self.use_residual = stride == 1 and in_ch == out_ch
        self.drop_prob = drop_prob

        layers = [
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        ]
        if use_se:
            layers.append(SEBlock2d(hidden, reduction=8))
        layers.extend(
            [
                nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            if self.training and self.drop_prob > 0.0:
                keep_prob = 1.0 - self.drop_prob
                shape = (out.shape[0], 1, 1, 1)
                mask = (
                    keep_prob + torch.rand(shape, dtype=out.dtype, device=out.device)
                ).floor_()
                out = out.div(keep_prob) * mask
            out = out + x
        return out


class LandmarkCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(3, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(48),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            InvertedResidual1d(48, 64, expansion=2, kernel_size=3, stride=1, use_se=True),
            InvertedResidual1d(64, 96, expansion=2, kernel_size=5, stride=2, use_se=True),
            InvertedResidual1d(96, 96, expansion=2, kernel_size=3, stride=1, use_se=True),
            InvertedResidual1d(96, 128, expansion=3, kernel_size=5, stride=2, use_se=True),
            InvertedResidual1d(128, 128, expansion=2, kernel_size=3, stride=1, use_se=True),
            InvertedResidual1d(128, 160, expansion=2, kernel_size=3, stride=1, use_se=True),
        )
        self.head = nn.Sequential(
            nn.Conv1d(160, 192, kernel_size=1, bias=False),
            nn.BatchNorm1d(192),
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2 and x.shape[1] == 63:
            x = x.view(x.shape[0], 21, 3)
        elif x.ndim != 3 or x.shape[1:] != (21, 3):
            raise ValueError(f"Unexpected landmark tensor shape: {tuple(x.shape)}")
        x = x.transpose(1, 2).contiguous()
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return self.classifier(x)


class SlimLandmarkCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
        )
        self.features = nn.Sequential(
            InvertedResidual2d(24, 32, expansion=2, kernel_size=3, stride=1, use_se=True),
            InvertedResidual2d(32, 48, expansion=3, kernel_size=3, stride=2, use_se=True),
            InvertedResidual2d(48, 64, expansion=3, kernel_size=3, stride=1, use_se=True),
            InvertedResidual2d(64, 64, expansion=3, kernel_size=3, stride=2, use_se=True),
            InvertedResidual2d(64, 96, expansion=3, kernel_size=3, stride=1, use_se=True),
            InvertedResidual2d(96, 96, expansion=3, kernel_size=3, stride=2, use_se=True),
            InvertedResidual2d(96, 128, expansion=3, kernel_size=3, stride=1, use_se=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2 and x.shape[1] == 63:
            x = x.view(x.shape[0], 21, 3)
        elif x.ndim != 3 or x.shape[1:] != (21, 3):
            raise ValueError(f"Unexpected landmark tensor shape: {tuple(x.shape)}")
        x = x.transpose(1, 2).contiguous().unsqueeze(-1)
        x = self.stem(x)
        x = self.features(x)
        return self.head(x)


class HandLandmarkFeatureExtractor:
    def __init__(
        self,
        model_path: str,
        min_detection_confidence: float = MEDIAPIPE_DEFAULT_MIN_DET_CONF,
    ):
        if (
            mp_tasks_vision is None
            or MPBaseOptions is None
            or MPImage is None
            or MPImageFormat is None
        ):
            raise RuntimeError(
                "mediapipe tasks APIs are unavailable. Install mediapipe with tasks support."
            )
        self.model_path = model_path
        self.min_detection_confidence = float(min_detection_confidence)
        self._landmarker = None

    def _get_landmarker(self):
        if self._landmarker is None:
            base_options = MPBaseOptions(model_asset_path=self.model_path)
            options = mp_tasks_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_tasks_vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=self.min_detection_confidence,
            )
            self._landmarker = mp_tasks_vision.HandLandmarker.create_from_options(
                options
            )
        return self._landmarker

    def extract(self, frame: np.ndarray) -> torch.Tensor | None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        width, height = image.size
        if width != height:
            side = max(width, height)
            pad_left = (side - width) // 2
            pad_top = (side - height) // 2
            pad_right = side - width - pad_left
            pad_bottom = side - height - pad_top
            image = ImageOps.expand(
                image,
                border=(pad_left, pad_top, pad_right, pad_bottom),
                fill=(0, 0, 0),
            )

        rgb = np.asarray(image, dtype=np.uint8)
        mp_image = MPImage(image_format=MPImageFormat.SRGB, data=rgb)
        results = self._get_landmarker().detect(mp_image)
        hands = getattr(results, "hand_landmarks", None)
        if not hands:
            return None

        points = np.array([[lm.x, lm.y, lm.z] for lm in hands[0]], dtype=np.float32)
        if points.shape != (21, 3):
            return None

        points = points - points[0]
        scale = float(np.linalg.norm(points[9]))
        if scale < 1e-6:
            return None
        points = points / scale
        return torch.from_numpy(points.reshape(63)).unsqueeze(0)

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None


_torch_load = torch.load


def _torch_load_with_map_call(*args, **kwargs):
    if "map_call" in kwargs and "map_location" not in kwargs:
        kwargs["map_location"] = kwargs.pop("map_call")
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_with_map_call


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the model checkpoint.")
    parser.add_argument("--manual", help="Path to a single image to run instead of webcam mode.")
    parser.add_argument(
        "--slim",
        action="store_true",
        help="Use the slim landmark architecture for *_slim_*.pt checkpoints.",
    )
    parser.add_argument(
        "--mediapipe-model-path",
        default="",
        help=(
            "Path to MediaPipe HandLandmarker .task model. "
            f"If unset, uses ${MEDIAPIPE_MODEL_ENV} or common local paths."
        ),
    )
    parser.add_argument(
        "--mediapipe-min-detect-conf",
        type=float,
        default=MEDIAPIPE_DEFAULT_MIN_DET_CONF,
        help="MediaPipe minimum hand detection confidence in [0, 1].",
    )
    return parser.parse_args()


def resolve_mediapipe_model_path(cli_path: str) -> str | None:
    candidates = []
    if cli_path:
        candidates.append(cli_path)
    env_path = os.getenv(MEDIAPIPE_MODEL_ENV, "").strip()
    if env_path:
        candidates.append(env_path)
    candidates.extend(
        [
            DEFAULT_MEDIAPIPE_MODEL_PATH,
            os.path.join(PROJECT_ROOT, "hand_landmarker.task"),
            os.path.join(PROJECT_ROOT, "assets", "hand_landmarker.task"),
            os.path.join(
                PROJECT_ROOT,
                ".venv",
                "lib",
                f"python{os.sys.version_info.major}.{os.sys.version_info.minor}",
                "site-packages",
                "mediapipe",
                "modules",
                "hand_landmarker",
                "hand_landmarker.task",
            ),
        ]
    )

    for candidate in candidates:
        if not candidate:
            continue
        expanded = os.path.abspath(os.path.expanduser(candidate))
        if os.path.isfile(expanded):
            return expanded

    download_target = DEFAULT_MEDIAPIPE_MODEL_PATH
    os.makedirs(os.path.dirname(download_target), exist_ok=True)
    tmp_target = f"{download_target}.tmp"
    try:
        print(f"Downloading HandLandmarker model to {download_target} ...")
        if os.path.exists(tmp_target):
            os.remove(tmp_target)
        urllib.request.urlretrieve(MEDIAPIPE_DEFAULT_MODEL_URL, tmp_target)
        os.replace(tmp_target, download_target)
        if os.path.isfile(download_target) and os.path.getsize(download_target) > 0:
            print("Downloaded MediaPipe HandLandmarker model.")
            return download_target
    except Exception as exc:
        print(f"Auto-download failed: {exc}")
    finally:
        if os.path.exists(tmp_target):
            try:
                os.remove(tmp_target)
            except OSError:
                pass
    return None


def load_model(model_path: str, device: torch.device, slim: bool = False) -> nn.Module:
    try:
        loaded = torch.jit.load(model_path, map_location=device)
        loaded.eval()
        return loaded
    except RuntimeError:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model_cls = SlimLandmarkCNN if slim else LandmarkCNN
        model = model_cls(num_classes=len(CLASS_NAMES)).to(device)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            if not slim and any(key.startswith("features.") for key in state_dict):
                raise RuntimeError(
                    "This checkpoint uses the slim landmark architecture. "
                    "Re-run with --slim."
                ) from exc
            raise
        model.eval()
        return model


def predict_landmarks(
    model: nn.Module, landmarks: torch.Tensor, device: torch.device
) -> tuple[str, float]:
    logits = model(landmarks.to(device))
    probabilities = torch.softmax(logits, dim=1)
    confidence, predicted_index = torch.max(probabilities, dim=1)
    return CLASS_NAMES[predicted_index.item()], confidence.item()


def run_manual_mode(
    model: nn.Module,
    image_path: str,
    extractor: HandLandmarkFeatureExtractor,
    device: torch.device,
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Unable to read image: {image_path}")

    landmarks = extractor.extract(image)
    if landmarks is None:
        raise RuntimeError("No hand detected in the provided image.")

    label, confidence = predict_landmarks(model, landmarks, device)
    overlay = f"{label} ({confidence * 100.0:.1f}%)"

    cv2.putText(
        image,
        overlay,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("ASL Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mediapipe_model_path = resolve_mediapipe_model_path(args.mediapipe_model_path)
    if not mediapipe_model_path:
        raise RuntimeError(
            "No MediaPipe HandLandmarker model was found. "
            "Auto-download also failed. Provide --mediapipe-model-path or set "
            "MEDIAPIPE_HAND_LANDMARKER_MODEL."
        )

    model = load_model(args.model, device, slim=args.slim)
    extractor = HandLandmarkFeatureExtractor(
        model_path=mediapipe_model_path,
        min_detection_confidence=args.mediapipe_min_detect_conf,
    )

    try:
        if args.manual:
            with torch.no_grad():
                run_manual_mode(model, args.manual, extractor, device)
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

                    landmarks = extractor.extract(frame)
                    if landmarks is None:
                        overlay = "No hand detected"
                    else:
                        label, confidence = predict_landmarks(model, landmarks, device)
                        overlay = f"{label} ({confidence * 100.0:.1f}%)"

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
        extractor.close()


if __name__ == "__main__":
    main()
