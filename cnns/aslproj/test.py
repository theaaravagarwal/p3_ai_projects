#!/usr/bin/env python

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn


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


_torch_load = torch.load


def _torch_load_with_map_call(*args, **kwargs):
    if "map_call" in kwargs and "map_location" not in kwargs:
        kwargs["map_location"] = kwargs.pop("map_call")
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_with_map_call


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - MEAN) / STD
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    return tensor


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script because the model is configured to run on the GPU.")

    device = torch.device("cuda")

    model = ASLModel(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load("model.pth", map_call="cuda"))
    model.eval()

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Unable to open webcam.")

    try:
        with torch.no_grad():
            while True:
                ok, frame = camera.read()
                if not ok:
                    break

                input_tensor = preprocess_frame(frame).to(device)
                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted_index = torch.max(probabilities, dim=1)

                label = CLASS_NAMES[predicted_index.item()]
                confidence_percent = confidence.item() * 100.0
                overlay = f"{label} ({confidence_percent:.1f}%)"

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
