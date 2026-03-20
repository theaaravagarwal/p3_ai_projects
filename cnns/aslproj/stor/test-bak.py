#!/usr/bin/env python

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch
import torch.jit
import torch.nn as nn
import torchvision.models as tv_models


IMAGE_SIZE = 128
CLASS_NAMES = ["A", "B", "C", "D", "E"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


class ASLResNet50Model(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = tv_models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)


class ASLEfficientNetB0Model(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = tv_models.efficientnet_b0(weights=None)
        if isinstance(self.backbone.classifier, nn.Sequential) and len(self.backbone.classifier) >= 2:
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            raise RuntimeError("Unexpected EfficientNet classifier structure.")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)


_torch_load = torch.load


def _torch_load_with_map_call(*args, **kwargs):
    if "map_call" in kwargs and "map_location" not in kwargs:
        kwargs["map_location"] = kwargs.pop("map_call")
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_with_map_call


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - MEAN) / STD
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    return tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the model checkpoint.")
    parser.add_argument("--manual", help="Path to a single image to run instead of webcam mode.")
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

        model = ASLModel(num_classes=num_classes).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        return model


def predict_image(model: nn.Module, image: np.ndarray, device: torch.device) -> tuple[str, float]:
    input_tensor = preprocess_frame(image).to(device)
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)
    confidence, predicted_index = torch.max(probabilities, dim=1)
    predicted = predicted_index.item()
    label = CLASS_NAMES[predicted] if predicted < len(CLASS_NAMES) else f"class_{predicted}"
    return label, confidence.item()


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


def run_manual_mode(model: nn.Module, image_path: str, device: torch.device) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Unable to read image: {image_path}")

    x1, y1, x2, y2 = detect_sign_region(image)
    crop = image[y1:y2, x1:x2]
    label, confidence = predict_image(model, crop, device)
    overlay = f"{label} ({confidence * 100.0:.1f}%)"

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


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)

    if args.manual:
        with torch.no_grad():
            run_manual_mode(model, args.manual, device)
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

                label, confidence = predict_image(model, frame, device)
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


if __name__ == "__main__":
    main()
