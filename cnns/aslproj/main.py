#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "kagglehub>=0.3.13",
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
import csv
import gc
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import kagglehub
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, get_worker_info
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


DATASET_HANDLES = (
    "grassknoted/asl-alphabet",
    "debashishsau/aslamerican-sign-language-aplhabet-dataset",
    "danrasband/asl-alphabet-test",
)
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
    parser = argparse.ArgumentParser(description="Train an A-E ASL classifier from a triple-source master pool.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Directory for training artifacts.")
    parser.add_argument("--mediapipe-model-path", type=Path, default=DEFAULT_MEDIAPIPE_MODEL_PATH, help="Local Hand Landmarker task bundle.")
    parser.add_argument("--image-size", type=int, default=128, help="Final crop size.")
    parser.add_argument("--padding", type=float, default=0.10, help="Relative padding applied to the landmark bounding box.")
    parser.add_argument("--batch-size", type=int, default=0, help="Mini-batch size. Use 0 to auto-size against the 11 GiB VRAM cap.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Validation ratio for the global master pool split.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count. Use 0 to auto-size against CPU and RAM limits.",
    )
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Number of batches prefetched by each worker.")
    parser.add_argument("--target-vram-gb", type=float, default=DEFAULT_TARGET_VRAM_GB, help="VRAM budget used by dynamic batch sizing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--min-hand-detection-confidence", type=float, default=0.5, help="MediaPipe hand detection confidence.")
    parser.add_argument("--raw-train", action="store_true", help="Bypass MediaPipe and train on resized full-frame images.")
    parser.add_argument(
        "--horizontal-flip-prob",
        type=float,
        default=0.0,
        help="Optional horizontal flip probability. Default is 0.0 because hand orientation can change sign appearance.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap on the merged master pool for debugging.")
    parser.add_argument("--debug", action="store_true", help="Print dataset discovery details.")
    args = parser.parse_args()
    if not 0.0 <= args.horizontal_flip_prob <= 1.0:
        raise ValueError("--horizontal-flip-prob must be between 0 and 1.")
    if args.batch_size < 0:
        raise ValueError("--batch-size must be >= 0.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be >= 1.")
    if args.target_vram_gb <= 0:
        raise ValueError("--target-vram-gb must be > 0.")
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
        print(f"CUDA total memory GiB: {props.total_memory / (1024 ** 3):.2f}")
    else:
        print("CUDA device: unavailable (running on CPU)")


def bytes_to_gib(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def round_down_power_of_two(value: int) -> int:
    power = 1
    while power * 2 <= value:
        power *= 2
    return power


def gib_to_bytes(value_gib: float) -> int:
    return int(value_gib * (1024 ** 3))


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
    directories: list[Path] = []
    if root.is_dir() and root.name in TARGET_LABELS:
        directories.append(root)
    directories.extend(path for path in root.rglob("*") if path.is_dir() and path.name in TARGET_LABELS)
    return sorted(set(directories), key=lambda path: path.as_posix().lower())


def discover_samples_under_root(root: Path, source_name: str, debug: bool = False) -> list[ImageSample]:
    samples: list[ImageSample] = []
    seen_paths: set[Path] = set()

    class_directories = discover_class_directories(root)
    if class_directories:
        debug_log(debug, f"Found {len(class_directories)} A-E class directories under {root}")
        for class_dir in class_directories:
            label_name = class_dir.name
            for image_path in sorted((path for path in class_dir.rglob("*") if is_image_file(path)), key=lambda path: path.as_posix().lower()):
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

    debug_log(debug, f"No usable A-E class directories found under {root}; falling back to filename parsing")
    for image_path in sorted((path for path in root.rglob("*") if is_image_file(path)), key=lambda path: path.as_posix().lower()):
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


def download_dataset(handle: str) -> Path:
    resolved = Path(kagglehub.dataset_download(handle))
    print(f"Using dataset {handle} at {resolved}")
    return resolved


def build_master_pool(handles: Iterable[str], debug: bool = False) -> list[ImageSample]:
    master_pool: list[ImageSample] = []
    for handle in handles:
        dataset_root = download_dataset(handle)
        discovered = discover_samples_under_root(dataset_root, source_name=handle, debug=debug)
        if not discovered:
            raise ValueError(f"No A-E image samples were discovered under dataset {handle} at {dataset_root}")
        master_pool.extend(discovered)
        print(f"  {handle} -> {len(discovered)} A-E images")
    return master_pool


def maybe_cap_samples(samples: list[ImageSample], max_samples: int, seed: int) -> list[ImageSample]:
    if max_samples <= 0 or len(samples) <= max_samples:
        return samples
    rng = random.Random(seed)
    sampled = samples.copy()
    rng.shuffle(sampled)
    return sampled[:max_samples]


def split_master_pool(samples: list[ImageSample], val_ratio: float, seed: int) -> tuple[list[ImageSample], list[ImageSample]]:
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
    class_summary = ", ".join(f"{label}:{class_counts.get(label, 0)}" for label in CLASS_NAMES)
    source_summary = ", ".join(f"{source}:{count}" for source, count in sorted(source_counts.items()))
    print(f"{name} class distribution -> {class_summary}")
    print(f"{name} source distribution -> {source_summary}")


def create_numbered_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = [int(path.name) for path in base_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    run_id = max(existing, default=-1) + 1
    run_dir = base_dir / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


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
                base_options=python.BaseOptions(model_asset_path=str(self.config.model_path.resolve())),
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
            return ImageOps.fit(image, (self.config.image_size, self.config.image_size), method=Image.Resampling.BILINEAR)

        rgb_array = np.asarray(image, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_array)
        result = self._get_landmarker().detect(mp_image)

        if result.hand_landmarks:
            crop = self._crop_to_landmarks(image, result.hand_landmarks[0])
        else:
            crop = ImageOps.fit(image, (self.config.image_size, self.config.image_size), method=Image.Resampling.BILINEAR)
            return crop

        return crop.resize((self.config.image_size, self.config.image_size), Image.Resampling.BILINEAR)

    def _crop_to_landmarks(self, image: Image.Image, landmarks: list[object]) -> Image.Image:
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
            return ImageOps.fit(image, (self.config.image_size, self.config.image_size), method=Image.Resampling.BILINEAR)
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
    if isinstance(dataset, ASLDataset):
        get_process_cropper(dataset.cropper_config)


class ASLDataset(Dataset):
    def __init__(
        self,
        samples: list[ImageSample],
        cropper_config: CropperConfig,
    ) -> None:
        self.samples = samples
        self.cropper_config = cropper_config

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        cropper = get_process_cropper(self.cropper_config)
        with Image.open(sample.path) as image:
            processed = cropper.preprocess(image)
        image_array = np.array(processed, dtype=np.uint8, copy=True)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        return image_tensor, sample.label_idx


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


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


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
        transforms.append(v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        self.transform = v2.Compose(transforms)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.transform(images)


def estimate_batch_memory_bytes(batch_size: int, image_size: int, parameter_count: int) -> int:
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
    ram_limit_gb: float = DEFAULT_TARGET_RAM_GB,
) -> tuple[int, int, int]:
    logical_cores = max(1, os.cpu_count() or 1)
    cpu_util = float(psutil.cpu_percent(interval=0.2))
    available_cpu_ratio = max(0.10, 1.0 - (cpu_util / 100.0))
    cpu_target = max(1, int(logical_cores * available_cpu_ratio * CPU_UTILIZATION_TARGET))
    worker_cap = cpu_target if requested_num_workers <= 0 else min(requested_num_workers, cpu_target)

    dynamic_ram_limit_bytes = max(
        gib_to_bytes(MIN_RAM_HEADROOM_GB),
        available_ram_bytes() - gib_to_bytes(MIN_RAM_HEADROOM_GB),
    )
    configured_ram_limit_bytes = gib_to_bytes(ram_limit_gb)
    ram_limit_bytes = min(dynamic_ram_limit_bytes, configured_ram_limit_bytes)
    base_process_bytes = gib_to_bytes(2.0)
    per_worker_overhead_bytes = gib_to_bytes(ESTIMATED_RAW_WORKER_GB if raw_mode else ESTIMATED_MEDIAPIPE_WORKER_GB)
    prefetched_batch_bytes = estimate_cpu_batch_bytes(batch_size, image_size) * prefetch_factor

    resolved_num_workers = worker_cap
    while resolved_num_workers > 0:
        estimated_ram_bytes = base_process_bytes + resolved_num_workers * (per_worker_overhead_bytes + prefetched_batch_bytes)
        if estimated_ram_bytes <= ram_limit_bytes:
            return resolved_num_workers, estimated_ram_bytes, ram_limit_bytes
        resolved_num_workers -= 1

    estimated_ram_bytes = base_process_bytes + estimate_cpu_batch_bytes(batch_size, image_size)
    return 0, estimated_ram_bytes, ram_limit_bytes


def resolve_batch_size(
    requested_batch_size: int,
    image_size: int,
    parameter_count: int,
    available_vram_bytes: int,
    total_vram_bytes: int,
    target_vram_gb: float,
) -> tuple[int, int, int]:
    if total_vram_bytes <= 0:
        # CPU fallback: keep batches conservative to avoid OOM on memory-constrained hosts.
        max_safe_batch_size = 32
        resolved_batch_size = max(1, min(requested_batch_size, max_safe_batch_size)) if requested_batch_size > 0 else max_safe_batch_size
        expected_bytes = estimate_batch_memory_bytes(resolved_batch_size, image_size, parameter_count)
        return resolved_batch_size, expected_bytes, 0

    available_budget_bytes = max(0, available_vram_bytes - gib_to_bytes(MIN_VRAM_HEADROOM_GB))
    target_vram_bytes = min(int(total_vram_bytes * 0.85), int(target_vram_gb * (1024 ** 3)), available_budget_bytes)
    parameter_bytes = parameter_count * 4
    static_training_bytes = parameter_bytes * 4
    per_sample_bytes = max(8 * 1024 * 1024, image_size * image_size * 3 * 2 * 96)
    available_batch_bytes = max(target_vram_bytes - static_training_bytes, 512 * 1024 * 1024)
    raw_batch_size = max(32, available_batch_bytes // per_sample_bytes)
    max_safe_batch_size = min(1024, round_down_power_of_two(int(raw_batch_size)))
    if requested_batch_size > 0:
        resolved_batch_size = min(requested_batch_size, max_safe_batch_size)
    else:
        resolved_batch_size = max_safe_batch_size
    expected_bytes = estimate_batch_memory_bytes(resolved_batch_size, image_size, parameter_count)
    return resolved_batch_size, expected_bytes, target_vram_bytes


def print_resource_report(
    device: torch.device,
    num_workers: int,
    prefetch_factor: int,
    batch_size: int,
    estimated_ram_bytes: int,
    ram_limit_bytes: int,
    total_vram_bytes: int,
    available_vram_bytes: int,
    target_vram_bytes: int,
    expected_batch_bytes: int,
) -> None:
    print("Resource Report")
    print(f"  CPU logical cores: {os.cpu_count() or 1}")
    print(f"  CPU workers active: {num_workers}")
    print(f"  Prefetch factor: {prefetch_factor if num_workers > 0 else 'disabled'}")
    print(f"  Pin memory: True")
    print(f"  Persistent workers: {num_workers > 0}")
    print(f"  Calculated batch size: {batch_size}")
    print(f"  Estimated RAM footprint: {bytes_to_gib(estimated_ram_bytes):.2f} GiB / {bytes_to_gib(ram_limit_bytes):.2f} GiB")
    if device.type == "cuda":
        print(f"  GPU VRAM total: {bytes_to_gib(total_vram_bytes):.2f} GiB")
        print(f"  GPU VRAM currently available: {bytes_to_gib(available_vram_bytes):.2f} GiB")
        print(f"  Target VRAM budget: {bytes_to_gib(target_vram_bytes):.2f} GiB")
        print(f"  Expected VRAM footprint: {bytes_to_gib(expected_batch_bytes):.2f} GiB")
    else:
        print("  GPU VRAM total: n/a (CPU mode)")
        print("  Target VRAM budget: n/a (CPU mode)")
        print(f"  Expected training memory estimate: {bytes_to_gib(expected_batch_bytes):.2f} GiB")
    print(f"  AMP enabled: {device.type == 'cuda'}")


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
    drop_last: bool,
) -> DataLoader:
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
        "worker_init_fn": dataloader_worker_init,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
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

    total_examples = 0
    total_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []
    epoch_start = time.perf_counter()
    current_cpu_percent()

    progress = tqdm(loader, desc=description, leave=False)
    for step_index, (inputs, targets) in enumerate(progress, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        inputs = batch_transform(inputs)
        inputs = inputs.contiguous(memory_format=torch.channels_last)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        predictions = logits.argmax(dim=1)
        batch_size = targets.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

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

    return {
        "loss": total_loss / total_examples,
        "accuracy": accuracy_score(all_targets, all_predictions),
        "f1": f1_score(all_targets, all_predictions, average="macro"),
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

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="Train Acc", linewidth=2)
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="Val Acc", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(epochs, [row["train_f1"] for row in history], label="Train F1", linewidth=2)
    axes[2].plot(epochs, [row["val_f1"] for row in history], label="Val F1", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_checkpoint(path: Path, model: nn.Module, args: argparse.Namespace, metrics: dict[str, float]) -> None:
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
    print_device_summary(device)
    print(f"Classes: {', '.join(CLASS_NAMES)}")
    print(f"Pipeline mode: {'raw full-frame' if args.raw_train else 'MediaPipe hand-localized crop'}")

    master_pool = build_master_pool(DATASET_HANDLES, debug=args.debug)
    master_pool = maybe_cap_samples(master_pool, args.max_samples, args.seed)
    print(f"Master pool size: {len(master_pool)}")
    print_dataset_summary("Master pool", master_pool)

    train_samples, val_samples = split_master_pool(master_pool, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print_dataset_summary("Train", train_samples)
    print_dataset_summary("Val", val_samples)

    run_dir = create_numbered_run_dir(args.output_dir)
    print(f"Writing artifacts to {run_dir}")
    save_text(run_dir / "label_map.json", json.dumps(CLASS_TO_IDX, indent=2))

    cropper_config = CropperConfig(
        model_path=args.mediapipe_model_path,
        image_size=args.image_size,
        padding=args.padding,
        min_detection_confidence=args.min_hand_detection_confidence,
        raw_mode=args.raw_train,
    )
    train_dataset = ASLDataset(
        samples=train_samples,
        cropper_config=cropper_config,
    )
    val_dataset = ASLDataset(
        samples=val_samples,
        cropper_config=cropper_config,
    )

    model = ASLModel(num_classes=len(CLASS_NAMES)).to(device=device, memory_format=torch.channels_last)
    parameter_count = count_trainable_parameters(model)
    print(f"Trainable parameters: {parameter_count}")
    if parameter_count >= 1_000_000:
        raise ValueError(f"Model exceeds the <1M parameter limit: {parameter_count}")

    total_vram_bytes = torch.cuda.get_device_properties(0).total_memory if device.type == "cuda" else 0
    free_vram_bytes = available_vram_bytes(device)
    resolved_batch_size, expected_batch_bytes, target_vram_bytes = resolve_batch_size(
        requested_batch_size=args.batch_size,
        image_size=args.image_size,
        parameter_count=parameter_count,
        available_vram_bytes=free_vram_bytes,
        total_vram_bytes=total_vram_bytes,
        target_vram_gb=args.target_vram_gb,
    )
    args.batch_size = resolved_batch_size
    resolved_num_workers, estimated_ram_bytes, ram_limit_bytes = resolve_num_workers(
        requested_num_workers=args.num_workers,
        batch_size=args.batch_size,
        image_size=args.image_size,
        prefetch_factor=args.prefetch_factor,
        raw_mode=args.raw_train,
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
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        drop_last=False,
    )

    train_batch_transform = BatchTransformPipeline(training=True, horizontal_flip_prob=args.horizontal_flip_prob)
    eval_batch_transform = BatchTransformPipeline(training=False, horizontal_flip_prob=0.0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history: list[dict[str, float]] = []
    best_val_f1 = -1.0
    best_checkpoint_path = run_dir / "best_model.pt"

    try:
        print_resource_report(
            device=device,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
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
                loader=train_loader,
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
                loader=val_loader,
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
            }
            history.append(row)
            write_metrics_csv(run_dir / "metrics.csv", history)
            plot_history(history, run_dir / "training_curves.png")

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train loss {train_metrics['loss']:.4f} | train acc {train_metrics['accuracy']:.4f} | train f1 {train_metrics['f1']:.4f} | "
                f"val loss {val_metrics['loss']:.4f} | val acc {val_metrics['accuracy']:.4f} | val f1 {val_metrics['f1']:.4f}"
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
