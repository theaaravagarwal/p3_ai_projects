#!/usr/bin/env python3
"""Live ASL alphabet classification from webcam using Hugging Face model.

Usage:
  python webcam_asl_hf.py
  python webcam_asl_hf.py --camera 1 --min-conf 0.45 --window 7
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModelForImageClassification

try:
    import mediapipe as mp

    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False


@dataclass
class Prediction:
    label: str
    confidence: float


@dataclass
class PreprocessConfig:
    image_size: int = 224
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class ASLResNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = tv_models.resnet50(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live ASL letter prediction from webcam")
    parser.add_argument(
        "--model-id",
        default="Abuzaid01/asl-sign-language-classifier",
        help="Hugging Face model id",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--min-conf", type=float, default=0.40, help="Confidence threshold")
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Number of recent predictions used for smoothing",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Run inference every N frames (1 = every frame)",
    )
    parser.add_argument(
        "--no-hand-detector",
        action="store_true",
        help="Disable MediaPipe hand detection and classify full frame",
    )
    parser.add_argument(
        "--hand-model",
        default="",
        help="Path to MediaPipe hand_landmarker.task (auto-detected if omitted)",
    )
    return parser.parse_args()


def resolve_default_hand_model_path() -> Optional[str]:
    candidates = [
        Path("model/hand_landmarker.task"),
        Path("models/hand_landmarker.task"),
        Path("stor/models/hand_landmarker.task"),
        Path(__file__).resolve().parent / "model" / "hand_landmarker.task",
        Path(__file__).resolve().parent / "models" / "hand_landmarker.task",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def setup_hand_detector(disabled: bool, hand_model_path: str):
    if disabled or not HAS_MEDIAPIPE:
        return None, "disabled"

    # Preferred: MediaPipe Tasks API (newer versions).
    if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
        model_path = hand_model_path or resolve_default_hand_model_path()
        if not model_path:
            print("[WARN] Could not find hand_landmarker.task. Hand detector disabled.")
            return None, "disabled"

        try:
            base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
            print(f"[INFO] Hand detector: MediaPipe Tasks ({model_path})")
            return detector, "tasks"
        except Exception as exc:
            print(f"[WARN] Failed to initialize MediaPipe Tasks hand detector: {exc}")

    # Fallback: legacy Solutions API if present.
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("[INFO] Hand detector: MediaPipe Solutions")
        return detector, "solutions"

    print("[WARN] No supported MediaPipe hand API found. Hand detector disabled.")
    return None, "disabled"


def hand_bbox_from_landmarks(landmarks, frame_shape, pad_ratio: float = 0.20) -> Optional[Tuple[int, int, int, int]]:
    if not landmarks:
        return None

    h, w = frame_shape[:2]
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    x_min = max(0, int(min(xs) * w))
    y_min = max(0, int(min(ys) * h))
    x_max = min(w - 1, int(max(xs) * w))
    y_max = min(h - 1, int(max(ys) * h))

    bw = x_max - x_min
    bh = y_max - y_min
    if bw <= 0 or bh <= 0:
        return None

    pad_w = int(bw * pad_ratio)
    pad_h = int(bh * pad_ratio)

    x1 = max(0, x_min - pad_w)
    y1 = max(0, y_min - pad_h)
    x2 = min(w - 1, x_max + pad_w)
    y2 = min(h - 1, y_max + pad_h)
    return x1, y1, x2, y2


def detect_hand_bbox(frame_bgr: np.ndarray, detector: Any, detector_mode: str) -> Optional[Tuple[int, int, int, int]]:
    if detector is None or detector_mode == "disabled":
        return None

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if detector_mode == "tasks":
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = detector.detect(mp_image)
        if not result or not result.hand_landmarks:
            return None
        return hand_bbox_from_landmarks(result.hand_landmarks[0], frame_bgr.shape)

    if detector_mode == "solutions":
        result = detector.process(frame_rgb)
        if not result or not result.multi_hand_landmarks:
            return None
        return hand_bbox_from_landmarks(result.multi_hand_landmarks[0].landmark, frame_bgr.shape)

    return None


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_and_preprocess(
    model_id: str, device: torch.device
) -> Tuple[Any, Optional[Any], list[str], PreprocessConfig]:
    config_path = hf_hub_download(repo_id=model_id, filename="config.json")
    preproc_path = hf_hub_download(repo_id=model_id, filename="preprocessor_config.json")
    weights_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")

    cfg = _read_json(config_path)
    preproc_cfg = _read_json(preproc_path)
    labels = cfg.get("class_names", [])
    num_classes = int(cfg.get("num_classes", len(labels) if labels else 29))
    prep = PreprocessConfig(
        image_size=int(preproc_cfg.get("size", 224)),
        image_mean=tuple(preproc_cfg.get("image_mean", [0.485, 0.456, 0.406])),
        image_std=tuple(preproc_cfg.get("image_std", [0.229, 0.224, 0.225])),
    )

    # This repo ships a custom ASLResNet checkpoint, not HF ResNet heads.
    if cfg.get("architectures", [""])[0] == "ASLResNet":
        model = ASLResNet(num_classes=num_classes)
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        cleaned = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            cleaned[nk] = v
        model.load_state_dict(cleaned, strict=True)
        model = model.to(device).eval()
        print("[INFO] Loaded model via custom ASLResNet checkpoint")
        return model, None, labels, prep

    # Fallback for regular HF image-classification repos.
    model = AutoModelForImageClassification.from_pretrained(model_id).to(device).eval()
    processor: Optional[Any] = None
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        print("[INFO] Loaded processor via AutoImageProcessor")
    except Exception as exc:
        print(f"[WARN] AutoImageProcessor failed: {exc}")
        try:
            processor = AutoFeatureExtractor.from_pretrained(model_id)
            print("[INFO] Loaded processor via AutoFeatureExtractor")
        except Exception as exc2:
            print(f"[WARN] AutoFeatureExtractor failed: {exc2}")
            print("[WARN] Falling back to built-in preprocessing from preprocessor_config.json.")
    if not labels and hasattr(model, "config") and hasattr(model.config, "id2label"):
        id2label = model.config.id2label
        labels = [id2label[i] for i in sorted(id2label.keys(), key=int)]
    return model, processor, labels, prep


def predict_image(
    image_bgr: np.ndarray,
    processor: Optional[Any],
    model: Any,
    device: torch.device,
    labels: list[str],
    preprocess: PreprocessConfig,
) -> Prediction:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    if processor is not None:
        inputs = processor(images=pil_image, return_tensors="pt")
    else:
        resized = pil_image.resize((preprocess.image_size, preprocess.image_size), Image.BILINEAR)
        arr = np.asarray(resized).astype(np.float32) / 255.0
        mean = np.array(preprocess.image_mean, dtype=np.float32)
        std = np.array(preprocess.image_std, dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        inputs = {"pixel_values": torch.from_numpy(arr).unsqueeze(0)}

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        if "pixel_values" in inputs:
            try:
                out = model(**inputs)
            except TypeError:
                out = model(inputs["pixel_values"])
        else:
            out = model(**inputs)
        logits = out.logits if hasattr(out, "logits") else out
        probs = torch.softmax(logits, dim=-1)
        conf, idx = torch.max(probs, dim=-1)

    pred_idx = idx.item()
    pred_label = labels[pred_idx] if labels and pred_idx < len(labels) else str(pred_idx)
    pred_conf = conf.item()
    return Prediction(label=pred_label, confidence=pred_conf)


def smoothed_prediction(history: Deque[Prediction]) -> Optional[Prediction]:
    if not history:
        return None

    labels = [p.label for p in history]
    majority_label, _ = Counter(labels).most_common(1)[0]
    confs = [p.confidence for p in history if p.label == majority_label]
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return Prediction(label=majority_label, confidence=avg_conf)


def draw_overlay(
    frame: np.ndarray,
    current_pred: Optional[Prediction],
    stable_pred: Optional[Prediction],
    bbox: Optional[Tuple[int, int, int, int]],
    min_conf: float,
    hand_detector_enabled: bool,
):
    h, _ = frame.shape[:2]

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)

    detector_status = "ON" if hand_detector_enabled else "OFF"
    cv2.putText(
        frame,
        f"Hand detector: {detector_status}",
        (10, h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Keys: q=quit",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    if current_pred is not None:
        cv2.putText(
            frame,
            f"Raw: {current_pred.label} ({current_pred.confidence:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (80, 220, 80),
            2,
            cv2.LINE_AA,
        )

    if stable_pred is not None:
        color = (0, 255, 0) if stable_pred.confidence >= min_conf else (0, 165, 255)
        text = stable_pred.label if stable_pred.confidence >= min_conf else "UNSURE"
        cv2.putText(
            frame,
            f"Stable: {text} ({stable_pred.confidence:.2f})",
            (10, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            color,
            2,
            cv2.LINE_AA,
        )


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading model: {args.model_id}")
    print(f"[INFO] Device: {device}")

    model, processor, labels, preprocess = load_model_and_preprocess(args.model_id, device)

    hand_detector, hand_detector_mode = setup_hand_detector(args.no_hand_detector, args.hand_model)
    if not HAS_MEDIAPIPE and not args.no_hand_detector:
        print("[WARN] mediapipe not installed. Falling back to full-frame classification.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera}")

    history: Deque[Prediction] = deque(maxlen=max(args.window, 1))

    frame_idx = 0
    current_pred: Optional[Prediction] = None
    stable_pred: Optional[Prediction] = None

    print("[INFO] Webcam started. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame_idx += 1

        bbox = None
        roi = frame

        if hand_detector is not None:
            bbox = detect_hand_bbox(frame, hand_detector, hand_detector_mode)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                roi = frame[y1:y2, x1:x2]

        if roi.size > 0 and frame_idx % max(args.frame_skip, 1) == 0:
            current_pred = predict_image(roi, processor, model, device, labels, preprocess)
            history.append(current_pred)
            stable_pred = smoothed_prediction(history)

        draw_overlay(
            frame,
            current_pred=current_pred,
            stable_pred=stable_pred,
            bbox=bbox,
            min_conf=args.min_conf,
            hand_detector_enabled=(hand_detector is not None and hand_detector_mode != "disabled"),
        )

        cv2.imshow("ASL Webcam Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if hand_detector is not None and hasattr(hand_detector, "close"):
        hand_detector.close()


if __name__ == "__main__":
    main()
