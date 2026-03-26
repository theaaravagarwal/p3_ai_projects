#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib>=3.10.0",
#   "mediapipe>=0.10.21",
#   "numpy>=2.2.0",
#   "pillow>=11.1.0",
#   "psutil>=5.9.8",
#   "scikit-learn>=1.6.0",
#   "torch>=2.7.0",
#   "torchvision>=0.22.0",
#   "tqdm>=4.67.1",
# ]
# ///

from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import csv
import gc
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
from collections import deque
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, get_worker_info
from torchvision import models
from torchvision.transforms import v2
from tqdm.auto import tqdm

try:
    import psutil
except ModuleNotFoundError:
    psutil = None

try:
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError:
    accuracy_score = None
    f1_score = None
    train_test_split = None

CLASS_NAMES = ("A", "B", "C", "D", "E")
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASS_NAMES)}
TARGET_LABELS = set(CLASS_NAMES)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MEDIAPIPE_MODEL_PATH = Path("model") / "hand_landmarker.task"
DEFAULT_TARGET_VRAM_GB = 11.0
DEFAULT_TARGET_RAM_GB = 30.0
CPU_UTILIZATION_TARGET = 0.90
ESTIMATED_MEDIAPIPE_WORKER_GB = 1.25
ESTIMATED_RAW_WORKER_GB = 0.40
MIN_RAM_HEADROOM_GB = 1.5
MIN_VRAM_HEADROOM_GB = 1.0
AGGRESSIVE_PREFETCH_FACTOR = 10
AGGRESSIVE_GPU_PREFETCH_BATCHES = 6


@dataclass(frozen=True)
class CropperConfig:
    model_path: Path
    image_size: int
    padding: float
    min_detection_confidence: float
    raw_mode: bool


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label_name: str
    label_idx: int
    source_name: str


_PROCESS_CROPPER: "LandmarkCropper | None" = None
_PROCESS_CROPPER_CONFIG: CropperConfig | None = None
_PROCESS_CROPPER_ATEXIT_REGISTERED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an A-E ASL classifier from a local folder dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root directory containing one subdirectory per class: A, B, C, D, E.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for training artifacts.",
    )
    parser.add_argument(
        "--mediapipe-model-path",
        type=Path,
        default=DEFAULT_MEDIAPIPE_MODEL_PATH,
        help="Local Hand Landmarker task bundle.",
    )
    parser.add_argument("--image-size", type=int, default=128, help="Final crop size.")
    parser.add_argument(
        "--padding",
        type=float,
        default=0.10,
        help="Relative padding applied to the landmark bounding box.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Mini-batch size. Use 0 for aggressive auto-sizing (targets >=128 on CUDA when memory allows).",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="AdamW learning rate. Default auto-selects 1.5e-4 for pretrained and 3e-4 otherwise.",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="AdamW weight decay."
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Validation ratio for the stratified train/val split.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count. Use 0 to auto-size against CPU and RAM limits.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=AGGRESSIVE_PREFETCH_FACTOR,
        help="CPU-side DataLoader prefetch factor (aggressive by default).",
    )
    parser.add_argument(
        "--gpu-prefetch-batches",
        type=int,
        default=AGGRESSIVE_GPU_PREFETCH_BATCHES,
        help="Number of batches asynchronously staged on CUDA (0 disables CUDA prefetch loader).",
    )
    parser.add_argument(
        "--heavy",
        action="store_true",
        help="Enable maximum-throughput mode: pushes worker/prefetch/batch allocation aggressively.",
    )
    parser.add_argument(
        "--target-vram-gb",
        type=float,
        default=DEFAULT_TARGET_VRAM_GB,
        help="VRAM budget used by dynamic batch sizing.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--min-hand-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe hand detection confidence.",
    )
    parser.add_argument(
        "--raw-train",
        action="store_true",
        help="Bypass MediaPipe and train on resized full-frame images.",
    )
    parser.add_argument(
        "--horizontal-flip-prob",
        type=float,
        default=0.0,
        help="Optional horizontal flip probability. Default is 0.0 because hand orientation can change sign appearance.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on the discovered local dataset for debugging.",
    )
    parser.add_argument(
        "--preprocess-cache-dir",
        type=Path,
        default=Path("data") / "preprocessed_cache",
        help="Directory for preprocessed hand-crop cache files.",
    )
    parser.add_argument(
        "--refresh-preprocess-cache",
        action="store_true",
        help="Rebuild the preprocessed cache instead of reusing existing cached crops.",
    )
    parser.add_argument(
        "--disable-preprocess-cache",
        action="store_true",
        help="Disable offline preprocessing cache and crop on-the-fly in DataLoader workers.",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=0,
        help="Worker processes for offline preprocessing cache. Use 0 for auto-sizing.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights for EfficientNet-B0.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print dataset discovery details."
    )
    args = parser.parse_args()
    if not 0.0 <= args.horizontal_flip_prob <= 1.0:
        raise ValueError("--horizontal-flip-prob must be between 0 and 1.")
    if args.batch_size < 0:
        raise ValueError("--batch-size must be >= 0.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be >= 1.")
    if args.gpu_prefetch_batches < 0:
        raise ValueError("--gpu-prefetch-batches must be >= 0.")
    if args.target_vram_gb <= 0:
        raise ValueError("--target-vram-gb must be > 0.")
    if args.preprocess_workers < 0:
        raise ValueError("--preprocess-workers must be >= 0.")
    if args.lr is None:
        args.lr = 1.5e-4 if not args.no_pretrained else 3e-4
    return args


def debug_log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[debug] {message}")


def ensure_sklearn_available() -> None:
    if accuracy_score is None or f1_score is None or train_test_split is None:
        raise RuntimeError(
            "scikit-learn is required for splitting and metric computation. "
            "Run the script with `uv run python .\\main.py ...` so inline dependencies are installed, "
            "or install scikit-learn into your current environment."
        )


def ensure_psutil_available() -> None:
    if psutil is None:
        raise RuntimeError(
            "psutil is required for resource-aware worker sizing and memory telemetry. "
            "Run the script with `uv run python .\\main.py ...` so inline dependencies are installed, "
            "or install psutil into your current environment."
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    return torch.device("cpu")


def print_device_summary(device: torch.device) -> None:
    print(f"Python executable: {sys.executable}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"CUDA device: {props.name}")
        print(f"CUDA total memory GiB: {props.total_memory / (1024**3):.2f}")
    else:
        print("CUDA device: unavailable (running on CPU)")


def bytes_to_gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def round_down_power_of_two(value: int) -> int:
    power = 1
    while power * 2 <= value:
        power *= 2
    return power


def gib_to_bytes(value_gib: float) -> int:
    return int(value_gib * (1024**3))


def current_ram_bytes() -> int:
    ensure_psutil_available()
    return int(psutil.virtual_memory().used)


def available_ram_bytes() -> int:
    ensure_psutil_available()
    return int(psutil.virtual_memory().available)


def current_vram_bytes(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return int(torch.cuda.memory_reserved(device))


def available_vram_bytes(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    free_bytes, _ = torch.cuda.mem_get_info(device=device)
    return int(free_bytes)


def current_cpu_percent() -> float:
    ensure_psutil_available()
    return float(psutil.cpu_percent(interval=None))


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def normalize_label_token(token: str) -> str | None:
    cleaned = token.strip().upper()
    return cleaned if cleaned in TARGET_LABELS else None


def infer_label_from_filename(path: Path) -> str | None:
    stem = path.stem.upper()
    for candidate in (stem, stem.split("_", 1)[0], stem.split("-", 1)[0]):
        label = normalize_label_token(candidate)
        if label is not None:
            return label
    return None


def discover_class_directories(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    if root.name in TARGET_LABELS:
        return [root]
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name in TARGET_LABELS),
        key=lambda path: path.as_posix().lower(),
    )


def discover_samples_under_root(
    root: Path, source_name: str, debug: bool = False
) -> list[ImageSample]:
    samples: list[ImageSample] = []
    seen_paths: set[Path] = set()

    class_directories = discover_class_directories(root)
    if class_directories:
        debug_log(debug, f"Found {len(class_directories)} class directories under {root}")
        for class_dir in class_directories:
            label_name = class_dir.name
            for image_path in sorted(
                (path for path in class_dir.rglob("*") if is_image_file(path)),
                key=lambda path: path.as_posix().lower(),
            ):
                resolved = image_path.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                samples.append(
                    ImageSample(
                        path=image_path,
                        label_name=label_name,
                        label_idx=CLASS_TO_IDX[label_name],
                        source_name=source_name,
                    )
                )
        if samples:
            return samples

    debug_log(
        debug,
        f"No usable A-E class directories found under {root}; falling back to filename parsing",
    )
    for image_path in sorted(
        (path for path in root.rglob("*") if is_image_file(path)),
        key=lambda path: path.as_posix().lower(),
    ):
        label_name = infer_label_from_filename(image_path)
        if label_name is None:
            continue
        resolved = image_path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        samples.append(
            ImageSample(
                path=image_path,
                label_name=label_name,
                label_idx=CLASS_TO_IDX[label_name],
                source_name=source_name,
            )
        )
    return samples


def build_master_pool(dataset_root: Path, debug: bool = False) -> list[ImageSample]:
    resolved_root = dataset_root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root does not exist or is not a directory: {resolved_root}"
        )
    discovered = discover_samples_under_root(
        resolved_root, source_name=resolved_root.as_posix(), debug=debug
    )
    if not discovered:
        raise ValueError(
            f"No A-E image samples were discovered under dataset root {resolved_root}"
        )
    print(f"Using dataset root at {resolved_root}")
    print(f"  -> discovered {len(discovered)} images")
    return discovered


def build_explicit_splits(
    dataset_root: Path, debug: bool = False
) -> tuple[list[ImageSample], list[ImageSample]] | None:
    resolved_root = dataset_root.expanduser().resolve()
    train_root = resolved_root / "train"
    test_root = resolved_root / "test"
    if not train_root.is_dir() or not test_root.is_dir():
        return None

    train_samples = discover_samples_under_root(train_root, source_name="train", debug=debug)
    val_samples = discover_samples_under_root(test_root, source_name="test", debug=debug)
    if not train_samples:
        raise ValueError(f"No A-E image samples were discovered under {train_root}")
    if not val_samples:
        raise ValueError(f"No A-E image samples were discovered under {test_root}")

    print(f"Using explicit train/test layout at {resolved_root}")
    print(f"  -> train images: {len(train_samples)}")
    print(f"  -> test images: {len(val_samples)}")
    return train_samples, val_samples


def maybe_cap_samples(
    samples: list[ImageSample], max_samples: int, seed: int
) -> list[ImageSample]:
    if max_samples <= 0 or len(samples) <= max_samples:
        return samples
    rng = random.Random(seed)
    sampled = samples.copy()
    rng.shuffle(sampled)
    return sampled[:max_samples]


def take_sample_cap(
    samples: list[ImageSample], cap: int, seed: int
) -> list[ImageSample]:
    if cap < 0:
        raise ValueError("Sample cap must be >= 0.")
    if cap == 0:
        return []
    if len(samples) <= cap:
        return samples
    rng = random.Random(seed)
    sampled = samples.copy()
    rng.shuffle(sampled)
    return sampled[:cap]


def cap_split_pair(
    train_samples: list[ImageSample],
    val_samples: list[ImageSample],
    max_samples: int,
    seed: int,
) -> tuple[list[ImageSample], list[ImageSample]]:
    total = len(train_samples) + len(val_samples)
    if max_samples <= 0 or total <= max_samples:
        return train_samples, val_samples
    if total == 0:
        return train_samples, val_samples
    if len(train_samples) > 0 and len(val_samples) > 0 and max_samples < 2:
        raise ValueError(
            "--max-samples is too small to keep both train and validation splits non-empty."
        )

    train_share = len(train_samples) / total
    train_cap = min(len(train_samples), max(1, int(round(max_samples * train_share))))
    val_cap = min(len(val_samples), max_samples - train_cap)
    if val_cap < 1 and len(val_samples) > 0:
        val_cap = 1
        train_cap = min(len(train_samples), max_samples - val_cap)

    if train_cap + val_cap < max_samples:
        remaining = max_samples - (train_cap + val_cap)
        train_extra = min(remaining, len(train_samples) - train_cap)
        train_cap += train_extra
        remaining -= train_extra
        if remaining > 0:
            val_cap += min(remaining, len(val_samples) - val_cap)

    return (
        take_sample_cap(train_samples, train_cap, seed),
        take_sample_cap(val_samples, val_cap, seed),
    )


def split_master_pool(
    samples: list[ImageSample], val_ratio: float, seed: int
) -> tuple[list[ImageSample], list[ImageSample]]:
    class_counts = Counter(sample.label_name for sample in samples)
    missing_classes = [label for label in CLASS_NAMES if class_counts.get(label, 0) == 0]
    undersized_classes = [
        label for label in CLASS_NAMES if class_counts.get(label, 0) < 2
    ]
    if missing_classes:
        raise ValueError(
            "Missing one or more class folders or images: "
            + ", ".join(missing_classes)
            + ". Expected one folder per class under the data root."
        )
    if undersized_classes:
        raise ValueError(
            "Each class needs at least 2 images for a stratified split. "
            "Too-small classes: "
            + ", ".join(undersized_classes)
        )
    labels = [sample.label_idx for sample in samples]
    train_samples, val_samples = train_test_split(
        samples,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    return list(train_samples), list(val_samples)


def print_dataset_summary(name: str, samples: Iterable[ImageSample]) -> None:
    class_counts = Counter(sample.label_name for sample in samples)
    source_counts = Counter(sample.source_name for sample in samples)
    class_summary = ", ".join(
        f"{label}:{class_counts.get(label, 0)}" for label in CLASS_NAMES
    )
    source_summary = ", ".join(
        f"{source}:{count}" for source, count in sorted(source_counts.items())
    )
    print(f"{name} class distribution -> {class_summary}")
    print(f"{name} source distribution -> {source_summary}")


def create_numbered_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        int(path.name)
        for path in base_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    run_id = max(existing, default=-1) + 1
    run_dir = base_dir / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def sample_cache_key(sample: ImageSample, cropper_config: CropperConfig) -> str:
    resolved = sample.path.resolve()
    stat = resolved.stat()
    payload = "|".join(
        (
            resolved.as_posix(),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            str(cropper_config.image_size),
            f"{cropper_config.padding:.6f}",
            f"{cropper_config.min_detection_confidence:.6f}",
            "1" if cropper_config.raw_mode else "0",
        )
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_preprocess_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            loaded = pickle.load(handle)
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def cache_preprocessed_samples(
    samples: list[ImageSample],
    cropper_config: CropperConfig,
    cache_dir: Path,
    refresh: bool,
    preprocess_workers: int,
) -> list[ImageSample]:
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.pkl"
    manifest: dict[str, dict[str, str]] = (
        {} if refresh else load_preprocess_manifest(manifest_path)
    )
    cached_samples: list[ImageSample] = []
    created = 0
    reused = 0
    keyed_samples: list[tuple[ImageSample, str]] = []
    missing_samples: list[tuple[ImageSample, str]] = []
    for sample in samples:
        key = sample_cache_key(sample, cropper_config)
        keyed_samples.append((sample, key))
        record = manifest.get(key, {})
        relative_path = record.get("relative_path")
        if isinstance(relative_path, str) and (cache_dir / relative_path).is_file():
            reused += 1
            continue
        missing_samples.append((sample, key))

    worker_count = (
        max(1, min(max(1, (os.cpu_count() or 1) - 1), 16))
        if preprocess_workers <= 0
        else max(1, min(preprocess_workers, max(1, os.cpu_count() or 1)))
    )

    if missing_samples:
        print(
            f"Preprocess cache build: {len(missing_samples)} missing / {len(samples)} total (workers={worker_count})"
        )
    if worker_count == 1:
        for sample, key in tqdm(
            missing_samples, desc="Preprocessing cache", leave=False
        ):
            relative_path = preprocess_and_save_sample(
                sample=sample,
                key=key,
                cropper_config=cropper_config,
                cache_dir=cache_dir,
            )
            manifest[key] = {"relative_path": relative_path}
            created += 1
    elif missing_samples:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count
        ) as executor:
            future_map = {
                executor.submit(
                    preprocess_and_save_sample,
                    sample=sample,
                    key=key,
                    cropper_config=cropper_config,
                    cache_dir=cache_dir,
                ): key
                for sample, key in missing_samples
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_map),
                total=len(future_map),
                desc="Preprocessing cache",
                leave=False,
            ):
                key = future_map[future]
                relative_path = future.result()
                manifest[key] = {"relative_path": relative_path}
                created += 1

    for sample, key in keyed_samples:
        record = manifest.get(key, {})
        relative_path = record.get("relative_path")
        if not isinstance(relative_path, str):
            raise RuntimeError(
                f"Cache manifest entry missing for sample: {sample.path.as_posix()}"
            )
        cached_path = cache_dir / relative_path
        if not cached_path.is_file():
            raise RuntimeError(
                f"Cached preprocessed file missing for sample: {cached_path.as_posix()}"
            )
        cached_samples.append(
            ImageSample(
                path=cached_path,
                label_name=sample.label_name,
                label_idx=sample.label_idx,
                source_name=sample.source_name,
            )
        )

    close_process_cropper()
    with manifest_path.open("wb") as handle:
        pickle.dump(manifest, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"Preprocess cache -> total:{len(samples)} reused:{reused} created:{created} dir:{cache_dir}"
    )
    return cached_samples


def preprocess_and_save_sample(
    sample: ImageSample,
    key: str,
    cropper_config: CropperConfig,
    cache_dir: Path,
) -> str:
    record = {
        "relative_path": f"{sample.label_name}/{key}.jpg",
    }
    relative_path = record["relative_path"]
    cached_path = cache_dir / relative_path
    if cached_path.is_file():
        return relative_path
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(sample.path) as image:
        cropper = get_process_cropper(cropper_config)
        processed = cropper.preprocess(image)
    processed.save(cached_path, format="JPEG", quality=95)
    return relative_path


class LandmarkCropper:
    def __init__(self, config: CropperConfig) -> None:
        self.config = config
        self.landmarker: vision.HandLandmarker | None = None

    def close(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def _get_landmarker(self) -> vision.HandLandmarker:
        if self.config.raw_mode:
            raise RuntimeError("Landmarker access is invalid in raw training mode.")
        if not self.config.model_path.is_file():
            raise FileNotFoundError(
                f"Missing MediaPipe task bundle at {self.config.model_path}. "
                "Place hand_landmarker.task there or override --mediapipe-model-path."
            )
        if self.landmarker is None:
            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(self.config.model_path.resolve())
                ),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=self.config.min_detection_confidence,
                min_hand_presence_confidence=self.config.min_detection_confidence,
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
        return self.landmarker

    def preprocess(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if self.config.raw_mode:
            return ImageOps.fit(
                image,
                (self.config.image_size, self.config.image_size),
                method=Image.Resampling.BILINEAR,
            )

        rgb_array = np.asarray(image, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_array)
        result = self._get_landmarker().detect(mp_image)

        if result.hand_landmarks:
            crop = self._crop_to_landmarks(image, result.hand_landmarks[0])
        else:
            crop = ImageOps.fit(
                image,
                (self.config.image_size, self.config.image_size),
                method=Image.Resampling.BILINEAR,
            )
            return crop

        return crop.resize(
            (self.config.image_size, self.config.image_size), Image.Resampling.BILINEAR
        )

    def _crop_to_landmarks(
        self, image: Image.Image, landmarks: list[object]
    ) -> Image.Image:
        width, height = image.size
        xs = [float(landmark.x) * width for landmark in landmarks]
        ys = [float(landmark.y) * height for landmark in landmarks]

        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        hand_width = max(max_x - min_x, 1.0)
        hand_height = max(max_y - min_y, 1.0)
        pad_x = hand_width * self.config.padding
        pad_y = hand_height * self.config.padding

        left = max(0, int(math.floor(min_x - pad_x)))
        top = max(0, int(math.floor(min_y - pad_y)))
        right = min(width, int(math.ceil(max_x + pad_x)))
        bottom = min(height, int(math.ceil(max_y + pad_y)))

        if right <= left or bottom <= top:
            return ImageOps.fit(
                image,
                (self.config.image_size, self.config.image_size),
                method=Image.Resampling.BILINEAR,
            )
        return image.crop((left, top, right, bottom))


def close_process_cropper() -> None:
    global _PROCESS_CROPPER, _PROCESS_CROPPER_CONFIG
    if _PROCESS_CROPPER is not None:
        _PROCESS_CROPPER.close()
        _PROCESS_CROPPER = None
        _PROCESS_CROPPER_CONFIG = None


def get_process_cropper(config: CropperConfig) -> LandmarkCropper:
    global _PROCESS_CROPPER, _PROCESS_CROPPER_CONFIG, _PROCESS_CROPPER_ATEXIT_REGISTERED

    if not _PROCESS_CROPPER_ATEXIT_REGISTERED:
        atexit.register(close_process_cropper)
        _PROCESS_CROPPER_ATEXIT_REGISTERED = True

    if _PROCESS_CROPPER is None or _PROCESS_CROPPER_CONFIG != config:
        close_process_cropper()
        _PROCESS_CROPPER = LandmarkCropper(config)
        _PROCESS_CROPPER_CONFIG = config
        if not config.raw_mode:
            _PROCESS_CROPPER._get_landmarker()

    return _PROCESS_CROPPER


def dataloader_worker_init(worker_id: int) -> None:
    worker_info = get_worker_info()
    if worker_info is None:
        return
    dataset = worker_info.dataset
    if isinstance(dataset, ASLDataset) and dataset.preprocess_on_load:
        get_process_cropper(dataset.cropper_config)


class ASLDataset(Dataset):
    def __init__(
        self,
        samples: list[ImageSample],
        cropper_config: CropperConfig,
        preprocess_on_load: bool = True,
    ) -> None:
        self.samples = samples
        self.cropper_config = cropper_config
        self.preprocess_on_load = preprocess_on_load

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            if self.preprocess_on_load:
                cropper = get_process_cropper(self.cropper_config)
                processed = cropper.preprocess(image)
            else:
                processed = image.convert("RGB")
                target_size = (
                    self.cropper_config.image_size,
                    self.cropper_config.image_size,
                )
                if processed.size != target_size:
                    processed = ImageOps.fit(
                        processed,
                        target_size,
                        method=Image.Resampling.BILINEAR,
                    )
        image_array = np.array(processed, dtype=np.uint8, copy=True)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        return image_tensor, sample.label_idx


class ASLModel(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, num_classes)
        self.backbone = backbone

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


class BatchTransformPipeline:
    def __init__(self, training: bool, horizontal_flip_prob: float) -> None:
        transforms: list[object] = [v2.ToDtype(torch.float32, scale=True)]
        if training:
            transforms.extend(
                [
                    v2.RandomRotation(20),
                    v2.ColorJitter(brightness=0.2, contrast=0.2),
                ]
            )
            if horizontal_flip_prob > 0.0:
                transforms.append(v2.RandomHorizontalFlip(p=horizontal_flip_prob))
        transforms.append(
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        )
        self.transform = v2.Compose(transforms)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.transform(images)


def estimate_batch_memory_bytes(
    batch_size: int, image_size: int, parameter_count: int
) -> int:
    parameter_bytes = parameter_count * 4
    static_training_bytes = parameter_bytes * 4
    per_sample_bytes = max(8 * 1024 * 1024, image_size * image_size * 3 * 2 * 96)
    return static_training_bytes + batch_size * per_sample_bytes


def estimate_cpu_batch_bytes(batch_size: int, image_size: int) -> int:
    per_sample_bytes = image_size * image_size * 3
    # Account for decoded image tensor, prefetch queue copies, and a small bookkeeping margin.
    return max(32 * 1024 * 1024, batch_size * per_sample_bytes * 6)


def resolve_num_workers(
    requested_num_workers: int,
    batch_size: int,
    image_size: int,
    prefetch_factor: int,
    raw_mode: bool,
    heavy: bool = False,
    ram_limit_gb: float = DEFAULT_TARGET_RAM_GB,
) -> tuple[int, int, int]:
    logical_cores = max(1, os.cpu_count() or 1)
    worker_cap = (
        logical_cores
        if requested_num_workers <= 0
        else min(requested_num_workers, logical_cores)
    )

    ram_headroom_gb = 0.25 if heavy else MIN_RAM_HEADROOM_GB
    dynamic_ram_limit_bytes = max(
        gib_to_bytes(ram_headroom_gb),
        available_ram_bytes() - gib_to_bytes(ram_headroom_gb),
    )
    configured_ram_limit_bytes = gib_to_bytes(ram_limit_gb)
    ram_limit_bytes = min(dynamic_ram_limit_bytes, configured_ram_limit_bytes)
    base_process_bytes = gib_to_bytes(2.0)
    per_worker_overhead_bytes = gib_to_bytes(
        ESTIMATED_RAW_WORKER_GB if raw_mode else ESTIMATED_MEDIAPIPE_WORKER_GB
    )
    prefetched_batch_bytes = (
        estimate_cpu_batch_bytes(batch_size, image_size) * prefetch_factor
    )

    resolved_num_workers = worker_cap
    while resolved_num_workers > 0:
        estimated_ram_bytes = base_process_bytes + resolved_num_workers * (
            per_worker_overhead_bytes + prefetched_batch_bytes
        )
        if estimated_ram_bytes <= ram_limit_bytes:
            return resolved_num_workers, estimated_ram_bytes, ram_limit_bytes
        resolved_num_workers -= 1

    estimated_ram_bytes = base_process_bytes + estimate_cpu_batch_bytes(
        batch_size, image_size
    )
    return 0, estimated_ram_bytes, ram_limit_bytes


def resolve_batch_size(
    requested_batch_size: int,
    image_size: int,
    parameter_count: int,
    available_vram_bytes: int,
    total_vram_bytes: int,
    target_vram_gb: float,
    heavy: bool = False,
) -> tuple[int, int, int]:
    if total_vram_bytes <= 0:
        # CPU fallback: keep batches conservative to avoid OOM on memory-constrained hosts.
        max_safe_batch_size = 64 if heavy else 32
        resolved_batch_size = (
            max(1, min(requested_batch_size, max_safe_batch_size))
            if requested_batch_size > 0
            else max_safe_batch_size
        )
        expected_bytes = estimate_batch_memory_bytes(
            resolved_batch_size, image_size, parameter_count
        )
        return resolved_batch_size, expected_bytes, 0

    vram_headroom_gb = 0.25 if heavy else MIN_VRAM_HEADROOM_GB
    total_vram_ratio = 0.97 if heavy else 0.93
    available_budget_bytes = max(
        0, available_vram_bytes - gib_to_bytes(vram_headroom_gb)
    )
    target_vram_bytes = min(
        int(total_vram_bytes * total_vram_ratio),
        int(target_vram_gb * (1024**3)),
        available_budget_bytes,
    )
    parameter_bytes = parameter_count * 4
    static_training_bytes = parameter_bytes * 4
    per_sample_bytes = max(8 * 1024 * 1024, image_size * image_size * 3 * 2 * 96)
    available_batch_bytes = max(
        target_vram_bytes - static_training_bytes, 64 * 1024 * 1024
    )
    raw_batch_size = max(1, available_batch_bytes // per_sample_bytes)
    max_safe_batch_size = min(
        4096 if heavy else 2048, round_down_power_of_two(int(raw_batch_size))
    )
    if max_safe_batch_size < 1:
        max_safe_batch_size = 1
    if requested_batch_size > 0:
        resolved_batch_size = min(requested_batch_size, max_safe_batch_size)
    else:
        resolved_batch_size = (
            max_safe_batch_size
            if raw_batch_size < 128
            else max(128, max_safe_batch_size)
        )
    expected_bytes = estimate_batch_memory_bytes(
        resolved_batch_size, image_size, parameter_count
    )
    return resolved_batch_size, expected_bytes, target_vram_bytes


def print_resource_report(
    device: torch.device,
    heavy: bool,
    num_workers: int,
    prefetch_factor: int,
    gpu_prefetch_batches: int,
    batch_size: int,
    estimated_ram_bytes: int,
    ram_limit_bytes: int,
    total_vram_bytes: int,
    available_vram_bytes: int,
    target_vram_bytes: int,
    expected_batch_bytes: int,
) -> None:
    print("Resource Report")
    print(f"  Heavy mode: {heavy}")
    print(f"  CPU logical cores: {os.cpu_count() or 1}")
    print(f"  CPU workers active: {num_workers}")
    print(f"  Prefetch factor: {prefetch_factor if num_workers > 0 else 'disabled'}")
    print(
        f"  CUDA prefetch batches: {gpu_prefetch_batches if device.type == 'cuda' else 'disabled'}"
    )
    print(f"  Pin memory: True")
    print(f"  Persistent workers: {num_workers > 0}")
    print(f"  Calculated batch size: {batch_size}")
    print(
        f"  Estimated RAM footprint: {bytes_to_gib(estimated_ram_bytes):.2f} GiB / {bytes_to_gib(ram_limit_bytes):.2f} GiB"
    )
    if device.type == "cuda":
        print(f"  GPU VRAM total: {bytes_to_gib(total_vram_bytes):.2f} GiB")
        print(
            f"  GPU VRAM currently available: {bytes_to_gib(available_vram_bytes):.2f} GiB"
        )
        print(f"  Target VRAM budget: {bytes_to_gib(target_vram_bytes):.2f} GiB")
        print(
            f"  Expected VRAM footprint: {bytes_to_gib(expected_batch_bytes):.2f} GiB"
        )
    else:
        print("  GPU VRAM total: n/a (CPU mode)")
        print("  Target VRAM budget: n/a (CPU mode)")
        print(
            f"  Expected training memory estimate: {bytes_to_gib(expected_batch_bytes):.2f} GiB"
        )
    print(f"  AMP enabled: {device.type == 'cuda'}")


class CUDAPrefetchLoader:
    def __init__(
        self, loader: DataLoader, device: torch.device, prefetch_batches: int
    ) -> None:
        self.loader = loader
        self.device = device
        self.prefetch_batches = max(0, prefetch_batches)

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        if self.device.type != "cuda" or self.prefetch_batches == 0:
            yield from self.loader
            return

        data_iter = iter(self.loader)
        stream = torch.cuda.Stream(device=self.device)
        queue: deque[tuple[torch.Tensor, torch.Tensor]] = deque()

        def _enqueue() -> bool:
            try:
                cpu_inputs, cpu_targets = next(data_iter)
            except StopIteration:
                return False
            with torch.cuda.stream(stream):
                gpu_inputs = cpu_inputs.to(self.device, non_blocking=True)
                gpu_targets = cpu_targets.to(self.device, non_blocking=True)
            queue.append((gpu_inputs, gpu_targets))
            return True

        for _ in range(self.prefetch_batches):
            if not _enqueue():
                break

        while queue:
            torch.cuda.current_stream(self.device).wait_stream(stream)
            inputs, targets = queue.popleft()
            _enqueue()
            yield inputs, targets


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
    drop_last: bool,
    device: torch.device,
) -> DataLoader:
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "worker_init_fn": dataloader_worker_init,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    if device.type == "cuda":
        loader_kwargs["pin_memory_device"] = "cuda"
    return DataLoader(**loader_kwargs)


def run_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    batch_transform: BatchTransformPipeline,
    description: str,
    debug: bool,
    ram_limit_bytes: int,
    vram_limit_bytes: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    num_classes = len(CLASS_NAMES)
    total_examples = 0
    total_loss = torch.zeros(1, device=device, dtype=torch.float32)
    total_correct = torch.zeros(1, device=device, dtype=torch.int64)
    confusion = torch.zeros(
        (num_classes, num_classes), device=device, dtype=torch.int64
    )
    epoch_start = time.perf_counter()
    current_cpu_percent()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    progress = tqdm(loader, desc=description, leave=False)
    total_steps = 0
    for step_index, (inputs, targets) in enumerate(progress, start=1):
        total_steps = step_index
        if inputs.device != device:
            inputs = inputs.to(device, non_blocking=True)
        if targets.device != device:
            targets = targets.to(device, non_blocking=True)
        inputs = batch_transform(inputs)
        inputs = inputs.contiguous(memory_format=torch.channels_last)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with (
            torch.set_grad_enabled(training),
            torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.float16),
        ):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        predictions = logits.argmax(dim=1)
        batch_size = targets.size(0)
        total_examples += int(batch_size)
        total_loss += loss.detach().to(torch.float32) * batch_size
        total_correct += (predictions == targets).sum()
        flat_pairs = targets * num_classes + predictions
        confusion += torch.bincount(
            flat_pairs, minlength=num_classes * num_classes
        ).reshape(num_classes, num_classes)

        if step_index % 50 == 0:
            ram_used_bytes = current_ram_bytes()
            vram_used_bytes = current_vram_bytes(device)
            if ram_used_bytes > ram_limit_bytes or vram_used_bytes > vram_limit_bytes:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if debug and step_index % 20 == 0:
            elapsed = max(time.perf_counter() - epoch_start, 1e-6)
            cpu_util = current_cpu_percent()
            ram_used_gb = bytes_to_gib(current_ram_bytes())
            vram_used_gb = bytes_to_gib(current_vram_bytes(device))
            speed = step_index / elapsed
            tqdm.write(
                f"[Resource Log] CPU: {cpu_util:.0f}% | "
                f"RAM: {ram_used_gb:.1f}GB/{bytes_to_gib(ram_limit_bytes):.0f}GB | "
                f"VRAM: {vram_used_gb:.1f}GB/{bytes_to_gib(vram_limit_bytes):.0f}GB | "
                f"Speed: {speed:.2f} it/s"
            )

    if total_examples == 0:
        raise RuntimeError("No usable training samples were processed.")

    epoch_seconds = max(time.perf_counter() - epoch_start, 1e-6)
    cpu_util = current_cpu_percent()
    ram_used_gb = bytes_to_gib(current_ram_bytes())
    vram_used_gb = bytes_to_gib(current_vram_bytes(device))
    peak_vram_gb = (
        bytes_to_gib(int(torch.cuda.max_memory_allocated(device)))
        if device.type == "cuda"
        else 0.0
    )

    confusion_cpu = confusion.detach().cpu().numpy()
    true_positives = np.diag(confusion_cpu).astype(np.float64)
    false_positives = confusion_cpu.sum(axis=0).astype(np.float64) - true_positives
    false_negatives = confusion_cpu.sum(axis=1).astype(np.float64) - true_positives
    f1_denominator = (2.0 * true_positives) + false_positives + false_negatives
    class_f1 = np.divide(
        2.0 * true_positives,
        f1_denominator,
        out=np.zeros_like(true_positives, dtype=np.float64),
        where=f1_denominator > 0,
    )

    return {
        "loss": float(total_loss.item()) / total_examples,
        "accuracy": float(total_correct.item()) / total_examples,
        "f1": float(class_f1.mean()),
        "examples": float(total_examples),
        "steps": float(total_steps),
        "epoch_seconds": float(epoch_seconds),
        "examples_per_sec": float(total_examples / epoch_seconds),
        "steps_per_sec": float(total_steps / epoch_seconds),
        "cpu_util": float(cpu_util),
        "ram_used_gb": float(ram_used_gb),
        "vram_used_gb": float(vram_used_gb),
        "vram_peak_gb": float(peak_vram_gb),
    }


def write_metrics_csv(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(
        epochs, [row["train_loss"] for row in history], label="Train Loss", linewidth=2
    )
    axes[0].plot(
        epochs, [row["val_loss"] for row in history], label="Val Loss", linewidth=2
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        epochs,
        [row["train_accuracy"] for row in history],
        label="Train Acc",
        linewidth=2,
    )
    axes[1].plot(
        epochs, [row["val_accuracy"] for row in history], label="Val Acc", linewidth=2
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(
        epochs, [row["train_f1"] for row in history], label="Train F1", linewidth=2
    )
    axes[2].plot(
        epochs, [row["val_f1"] for row in history], label="Val F1", linewidth=2
    )
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_checkpoint(
    path: Path, model: nn.Module, args: argparse.Namespace, metrics: dict[str, float]
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def save_text(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_sklearn_available()
    ensure_psutil_available()
    seed_everything(args.seed)
    device = select_device()
    if device.type == "cuda":
        logical_cores = max(1, os.cpu_count() or 1)
        target_workers = max(2, int(logical_cores * CPU_UTILIZATION_TARGET))
        if args.num_workers <= 0:
            args.num_workers = target_workers
        args.prefetch_factor = max(args.prefetch_factor, AGGRESSIVE_PREFETCH_FACTOR)
        args.gpu_prefetch_batches = max(
            args.gpu_prefetch_batches, AGGRESSIVE_GPU_PREFETCH_BATCHES
        )
    if args.heavy:
        args.num_workers = max(1, os.cpu_count() or 1)
        args.prefetch_factor = max(args.prefetch_factor, 10)
        args.gpu_prefetch_batches = max(args.gpu_prefetch_batches, 6)
        if device.type == "cuda":
            args.target_vram_gb = max(
                args.target_vram_gb,
                bytes_to_gib(torch.cuda.get_device_properties(0).total_memory),
            )
    print_device_summary(device)
    print(f"Classes: {', '.join(CLASS_NAMES)}")
    print(
        f"Pipeline mode: {'raw full-frame' if args.raw_train else 'MediaPipe hand-localized crop'}"
    )

    cropper_config = CropperConfig(
        model_path=args.mediapipe_model_path,
        image_size=args.image_size,
        padding=args.padding,
        min_detection_confidence=args.min_hand_detection_confidence,
        raw_mode=args.raw_train,
    )
    preprocess_on_load = True
    explicit_splits = build_explicit_splits(args.data_dir, debug=args.debug)
    if explicit_splits is not None:
        train_samples, val_samples = explicit_splits
        train_samples, val_samples = cap_split_pair(
            train_samples=train_samples,
            val_samples=val_samples,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        print(f"Train samples: {len(train_samples)}")
        print(f"Val samples: {len(val_samples)}")
        print_dataset_summary("Train", train_samples)
        print_dataset_summary("Val", val_samples)
        if not args.disable_preprocess_cache:
            train_samples = cache_preprocessed_samples(
                samples=train_samples,
                cropper_config=cropper_config,
                cache_dir=args.preprocess_cache_dir / "train",
                refresh=args.refresh_preprocess_cache,
                preprocess_workers=args.preprocess_workers,
            )
            val_samples = cache_preprocessed_samples(
                samples=val_samples,
                cropper_config=cropper_config,
                cache_dir=args.preprocess_cache_dir / "test",
                refresh=args.refresh_preprocess_cache,
                preprocess_workers=args.preprocess_workers,
            )
            preprocess_on_load = False
    else:
        master_pool = build_master_pool(args.data_dir, debug=args.debug)
        master_pool = maybe_cap_samples(master_pool, args.max_samples, args.seed)
        print(f"Master pool size: {len(master_pool)}")
        print_dataset_summary("Master pool", master_pool)
        if not args.disable_preprocess_cache:
            master_pool = cache_preprocessed_samples(
                samples=master_pool,
                cropper_config=cropper_config,
                cache_dir=args.preprocess_cache_dir,
                refresh=args.refresh_preprocess_cache,
                preprocess_workers=args.preprocess_workers,
            )
            preprocess_on_load = False

        train_samples, val_samples = split_master_pool(
            master_pool, val_ratio=args.val_ratio, seed=args.seed
        )
        print(f"Train samples: {len(train_samples)}")
        print(f"Val samples: {len(val_samples)}")
        print_dataset_summary("Train", train_samples)
        print_dataset_summary("Val", val_samples)

    run_dir = create_numbered_run_dir(args.output_dir)
    print(f"Writing artifacts to {run_dir}")
    save_text(run_dir / "label_map.json", json.dumps(CLASS_TO_IDX, indent=2))

    train_dataset = ASLDataset(
        samples=train_samples,
        cropper_config=cropper_config,
        preprocess_on_load=preprocess_on_load,
    )
    val_dataset = ASLDataset(
        samples=val_samples,
        cropper_config=cropper_config,
        preprocess_on_load=preprocess_on_load,
    )

    model = ASLModel(
        num_classes=len(CLASS_NAMES),
        pretrained=not args.no_pretrained,
    ).to(device=device, memory_format=torch.channels_last)
    parameter_count = count_trainable_parameters(model)
    print(f"Trainable parameters: {parameter_count}")

    total_vram_bytes = (
        torch.cuda.get_device_properties(0).total_memory if device.type == "cuda" else 0
    )
    free_vram_bytes = available_vram_bytes(device)
    resolved_batch_size, expected_batch_bytes, target_vram_bytes = resolve_batch_size(
        requested_batch_size=args.batch_size,
        image_size=args.image_size,
        parameter_count=parameter_count,
        available_vram_bytes=free_vram_bytes,
        total_vram_bytes=total_vram_bytes,
        target_vram_gb=args.target_vram_gb,
        heavy=args.heavy,
    )
    args.batch_size = resolved_batch_size
    resolved_num_workers, estimated_ram_bytes, ram_limit_bytes = resolve_num_workers(
        requested_num_workers=args.num_workers,
        batch_size=args.batch_size,
        image_size=args.image_size,
        prefetch_factor=args.prefetch_factor,
        raw_mode=args.raw_train,
        heavy=args.heavy,
        ram_limit_gb=DEFAULT_TARGET_RAM_GB,
    )
    args.num_workers = resolved_num_workers
    save_text(run_dir / "config.json", json.dumps(vars(args), indent=2, default=str))

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        drop_last=True,
        device=device,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        drop_last=False,
        device=device,
    )
    train_epoch_loader: Iterable[tuple[torch.Tensor, torch.Tensor]] = (
        CUDAPrefetchLoader(
            train_loader,
            device=device,
            prefetch_batches=args.gpu_prefetch_batches,
        )
    )
    val_epoch_loader: Iterable[tuple[torch.Tensor, torch.Tensor]] = CUDAPrefetchLoader(
        val_loader,
        device=device,
        prefetch_batches=args.gpu_prefetch_batches,
    )

    train_batch_transform = BatchTransformPipeline(
        training=True, horizontal_flip_prob=args.horizontal_flip_prob
    )
    eval_batch_transform = BatchTransformPipeline(
        training=False, horizontal_flip_prob=0.0
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=(device.type == "cuda"),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history: list[dict[str, float]] = []
    best_val_f1 = -1.0
    best_checkpoint_path = run_dir / "best_model.pt"

    try:
        print_resource_report(
            device=device,
            heavy=args.heavy,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            gpu_prefetch_batches=args.gpu_prefetch_batches,
            batch_size=args.batch_size,
            estimated_ram_bytes=estimated_ram_bytes,
            ram_limit_bytes=ram_limit_bytes,
            total_vram_bytes=total_vram_bytes,
            available_vram_bytes=free_vram_bytes,
            target_vram_bytes=target_vram_bytes,
            expected_batch_bytes=expected_batch_bytes,
        )
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model=model,
                loader=train_epoch_loader,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                batch_transform=train_batch_transform,
                description=f"Train {epoch}/{args.epochs}",
                debug=args.debug,
                ram_limit_bytes=ram_limit_bytes,
                vram_limit_bytes=target_vram_bytes,
            )
            val_metrics = run_epoch(
                model=model,
                loader=val_epoch_loader,
                device=device,
                criterion=criterion,
                optimizer=None,
                scaler=scaler,
                batch_transform=eval_batch_transform,
                description=f"Val {epoch}/{args.epochs}",
                debug=args.debug,
                ram_limit_bytes=ram_limit_bytes,
                vram_limit_bytes=target_vram_bytes,
            )
            scheduler.step()

            row = {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "lr": optimizer.param_groups[0]["lr"],
                "train_epoch_seconds": train_metrics["epoch_seconds"],
                "val_epoch_seconds": val_metrics["epoch_seconds"],
                "train_examples_per_sec": train_metrics["examples_per_sec"],
                "val_examples_per_sec": val_metrics["examples_per_sec"],
                "train_steps_per_sec": train_metrics["steps_per_sec"],
                "val_steps_per_sec": val_metrics["steps_per_sec"],
                "train_cpu_util": train_metrics["cpu_util"],
                "val_cpu_util": val_metrics["cpu_util"],
                "train_ram_used_gb": train_metrics["ram_used_gb"],
                "val_ram_used_gb": val_metrics["ram_used_gb"],
                "train_vram_used_gb": train_metrics["vram_used_gb"],
                "val_vram_used_gb": val_metrics["vram_used_gb"],
                "train_vram_peak_gb": train_metrics["vram_peak_gb"],
                "val_vram_peak_gb": val_metrics["vram_peak_gb"],
            }
            history.append(row)
            write_metrics_csv(run_dir / "metrics.csv", history)
            plot_history(history, run_dir / "training_curves.png")

            train_loss_gap = val_metrics["loss"] - train_metrics["loss"]
            acc_gap = train_metrics["accuracy"] - val_metrics["accuracy"]
            f1_gap = train_metrics["f1"] - val_metrics["f1"]
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}/{args.epochs} | lr {current_lr:.6g} | "
                f"batch {args.batch_size} | workers {args.num_workers} | "
                f"prefetch {args.prefetch_factor if args.num_workers > 0 else 0} | "
                f"gpu_prefetch {args.gpu_prefetch_batches if device.type == 'cuda' else 0} | "
                f"amp {device.type == 'cuda'}"
            )
            print(
                f"  train -> loss {train_metrics['loss']:.4f} | acc {train_metrics['accuracy']:.4f} | f1 {train_metrics['f1']:.4f} | "
                f"time {train_metrics['epoch_seconds']:.1f}s | {train_metrics['examples_per_sec']:.1f} ex/s | {train_metrics['steps_per_sec']:.2f} it/s"
            )
            print(
                f"  val   -> loss {val_metrics['loss']:.4f} | acc {val_metrics['accuracy']:.4f} | f1 {val_metrics['f1']:.4f} | "
                f"time {val_metrics['epoch_seconds']:.1f}s | {val_metrics['examples_per_sec']:.1f} ex/s | {val_metrics['steps_per_sec']:.2f} it/s"
            )
            print(
                f"  gap   -> val-train loss {train_loss_gap:+.4f} | train-val acc {acc_gap:+.4f} | train-val f1 {f1_gap:+.4f}"
            )
            print(
                f"  sys   -> CPU {train_metrics['cpu_util']:.0f}%/{val_metrics['cpu_util']:.0f}% | "
                f"RAM {train_metrics['ram_used_gb']:.1f}/{val_metrics['ram_used_gb']:.1f} GB | "
                f"VRAM cur {train_metrics['vram_used_gb']:.1f}/{val_metrics['vram_used_gb']:.1f} GB | "
                f"VRAM peak {train_metrics['vram_peak_gb']:.1f}/{val_metrics['vram_peak_gb']:.1f} GB"
            )

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                save_checkpoint(
                    path=best_checkpoint_path,
                    model=model,
                    args=args,
                    metrics={
                        "epoch": epoch,
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"],
                        "val_f1": val_metrics["f1"],
                        "parameter_count": parameter_count,
                    },
                )
    finally:
        close_process_cropper()

    print(f"Best validation macro F1: {best_val_f1:.4f}")
    print(f"Best checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
