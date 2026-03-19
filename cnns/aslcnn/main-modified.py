import os
import pickle
import random
import argparse
import gc
import csv
import re
import io
import subprocess
from dataclasses import dataclass
from contextlib import nullcontext

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from PIL import Image

try:
    from torchcodec.decoders import VideoDecoder as TorchcodecVideoDecoder
except Exception:
    TorchcodecVideoDecoder = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Configuration
IMG_SIZE = 128
DEFAULT_BATCH_SIZE = 32
EPOCHS = 20
CLASSES = [
    "A",
    "B",
    "C",
    "D",
    "E",
]
SEED = 67
DATASET_OPTIONS = {
    "grassknoted": {
        "kaggle_id": "grassknoted/asl-alphabet",
        "output_tag": "grassknoted",
        "train_dir_candidates": [
            "",
            "asl_alphabet_train",
            os.path.join("asl_alphabet_train", "asl_alphabet_train"),
            "train",
            "Train",
        ],
    },
    "synthetic": {
        "kaggle_id": "lexset/synthetic-asl-alphabet",
        "output_tag": "synthetic",
        "train_dir_candidates": [
            "",
            "Train_Alphabet",
            "train",
            "Train",
        ],
    },
    "asl-citizen": {
        "kaggle_id": "abd0kamel/asl-citizen",
        "output_tag": "asl_citizen",
        "train_dir_candidates": [
            "",
            "asl-citizen",
            "asl_citizen",
            "ASL_Citizen",
            "train",
            "Train",
        ],
    },
}
CHANNEL_MEAN = (0.485, 0.456, 0.406)
CHANNEL_STD = (0.229, 0.224, 0.225)


def normalize_class_name(name: str) -> str:
    return name.strip().upper()


REQUIRED_CLASSES_NORMALIZED = {normalize_class_name(name) for name in CLASSES}


@dataclass
class RuntimeConfig:
    device: torch.device
    use_amp: bool
    cpu_threads: int


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    kaggle_id: str
    output_tag: str
    train_dir_candidates: list[str]


def configure_runtime(device_pref: str, amp_mode: str) -> RuntimeConfig:
    cpu_count = os.cpu_count() or 4
    cpu_threads = max(2, min(8, cpu_count // 2))

    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        # set_num_interop_threads can fail if parallel work already started.
        pass

    requested_cuda = device_pref == "cuda"
    auto_cuda = device_pref == "auto"
    if requested_cuda or auto_cuda:
        try:
            if not torch.cuda.is_available():
                if requested_cuda:
                    raise RuntimeError(
                        "CUDA requested but torch.cuda.is_available() is False."
                    )
                raise RuntimeError("CUDA not available.")
            if torch.cuda.device_count() < 1:
                raise RuntimeError("CUDA reports zero visible devices.")
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            use_amp = amp_mode == "on"
            if use_amp:
                print(f"GPU detected: AMP enabled on {device_name}.")
            else:
                print(f"GPU detected on {device_name}. AMP disabled for stability.")
            return RuntimeConfig(
                device=torch.device("cuda"),
                use_amp=use_amp,
                cpu_threads=cpu_threads,
            )
        except Exception as exc:
            if requested_cuda:
                raise RuntimeError(
                    f"Failed to initialize CUDA with --device=cuda: {exc}"
                ) from exc
            print(f"CUDA init failed, falling back to CPU: {exc}")

    print("No GPU detected. Training will run on CPU.")
    return RuntimeConfig(
        device=torch.device("cpu"), use_amp=False, cpu_threads=cpu_threads
    )


def create_grad_scaler(use_amp: bool):
    if not use_amp:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=True)


def autocast_ctx(device: torch.device, use_amp: bool):
    if not use_amp:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, dtype=torch.float16)
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an ASL classifier on the first five letters."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_OPTIONS),
        default="grassknoted",
        help="Which dataset source to train on.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device selection: auto picks CUDA when safe, else CPU.",
    )
    parser.add_argument(
        "--amp",
        choices=("on", "off"),
        default="off",
        help="Enable/disable AMP. Default off for CUDA-event stability.",
    )
    return parser.parse_args()


def get_dataset_config(dataset_key: str) -> DatasetConfig:
    config = DATASET_OPTIONS[dataset_key]
    return DatasetConfig(
        key=dataset_key,
        kaggle_id=config["kaggle_id"],
        output_tag=config["output_tag"],
        train_dir_candidates=config["train_dir_candidates"],
    )


def resolve_batch_size(device: torch.device, train_size: int) -> int:
    if train_size < 1:
        raise ValueError("Training split is empty. Cannot resolve batch size.")

    env_batch = os.getenv("BATCH_SIZE")
    if env_batch is not None:
        try:
            batch_size = int(env_batch)
            if batch_size < 1:
                raise ValueError
        except ValueError as exc:
            raise ValueError("BATCH_SIZE must be a positive integer.") from exc
        return min(batch_size, train_size)

    if device.type == "cuda":
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024**3)
        except Exception as exc:
            print(f"CUDA memory probe failed ({exc}); using conservative batch size.")
            free_gb = 0.0

        if free_gb >= 18:
            batch_size = 256
        elif free_gb >= 10:
            batch_size = 128
        elif free_gb >= 6:
            batch_size = 64
        else:
            batch_size = 16
    else:
        cpu_count = os.cpu_count() or 4
        batch_size = DEFAULT_BATCH_SIZE if cpu_count >= 8 else 32

    min_batch = 8 if train_size >= 8 else 1
    return max(min_batch, min(batch_size, train_size))


def resolve_num_workers(cpu_threads: int) -> int:
    env_workers = os.getenv("NUM_WORKERS")
    if env_workers is not None:
        try:
            workers = int(env_workers)
            if workers < 0:
                raise ValueError
        except ValueError as exc:
            raise ValueError("NUM_WORKERS must be a non-negative integer.") from exc
        return workers

    if torch.cuda.is_available():
        return 16
    return max(0, min(4, cpu_threads))


def resolve_pickle_reader_workers(cpu_threads: int, device: torch.device) -> int:
    env_workers = os.getenv("PICKLE_READER_WORKERS")
    if env_workers is not None:
        try:
            workers = int(env_workers)
            if workers < 0:
                raise ValueError
        except ValueError as exc:
            raise ValueError(
                "PICKLE_READER_WORKERS must be a non-negative integer."
            ) from exc
        return workers

    if device.type == "cuda":
        return 16
    return max(2, min(8, cpu_threads))


def resolve_pin_memory(device: torch.device) -> bool:
    env_pin_memory = os.getenv("PIN_MEMORY")
    if env_pin_memory is not None:
        return env_pin_memory.strip().lower() in {"1", "true", "yes", "on"}
    return device.type == "cuda"


def move_batch_to_device(images, targets, device: torch.device):
    non_blocking = (
        device.type == "cuda"
        and isinstance(images, torch.Tensor)
        and images.is_pinned()
        and isinstance(targets, torch.Tensor)
        and targets.is_pinned()
    )
    return (
        images.to(device, non_blocking=non_blocking),
        targets.to(device, non_blocking=non_blocking),
    )


def is_cuda_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in message and "cuda" in message


def resolve_micro_batch_size() -> int:
    env_value = os.getenv("MICRO_BATCH_SIZE")
    if env_value is None:
        return 0
    try:
        value = int(env_value)
    except ValueError as exc:
        raise ValueError("MICRO_BATCH_SIZE must be an integer >= 0.") from exc
    if value < 0:
        raise ValueError("MICRO_BATCH_SIZE must be an integer >= 0.")
    return value


def resolve_dataset_fraction() -> float:
    env_value = os.getenv("DATASET_FRACTION")
    if env_value is None:
        return 1.0
    try:
        value = float(env_value)
    except ValueError as exc:
        raise ValueError("DATASET_FRACTION must be a float in (0, 1].") from exc
    if value <= 0.0 or value > 1.0:
        raise ValueError("DATASET_FRACTION must be a float in (0, 1].")
    return value


def resolve_dataset_cap(env_name: str) -> int:
    env_value = os.getenv(env_name)
    if env_value is None:
        return 0
    try:
        value = int(env_value)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be an integer >= 0.") from exc
    if value < 0:
        raise ValueError(f"{env_name} must be an integer >= 0.")
    return value


def apply_dataset_chunking(
    train_ds: Dataset,
    val_ds: Dataset,
    seed: int,
    fraction: float,
    max_train_samples: int,
    max_val_samples: int,
):
    def subset_dataset(ds: Dataset, cap: int) -> Dataset:
        if cap <= 0 or cap >= len(ds):
            return ds
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(ds), generator=g)[:cap].tolist()
        return Subset(ds, indices)

    if fraction < 1.0:
        train_cap = max(1, int(len(train_ds) * fraction))
        val_cap = max(1, int(len(val_ds) * fraction))
        train_ds = subset_dataset(train_ds, train_cap)
        val_ds = subset_dataset(val_ds, val_cap)

    train_ds = subset_dataset(train_ds, max_train_samples)
    val_ds = subset_dataset(val_ds, max_val_samples)
    return train_ds, val_ds


def use_tqdm_progress(device: torch.device) -> bool:
    env_progress = os.getenv("USE_TQDM")
    if env_progress is not None:
        return env_progress.strip().lower() in {"1", "true", "yes", "on"}
    # tqdm generators can produce noisy finalizer traces on unstable CUDA stacks.
    return device.type != "cuda"


def resolve_use_epoch_pickle_cache(device: torch.device) -> bool:
    env_value = os.getenv("USE_EPOCH_PICKLE_CACHE")
    if env_value is not None:
        return env_value.strip().lower() in {"1", "true", "yes", "on"}
    # Default off on CUDA to avoid generator/finalizer teardown instability.
    return device.type != "cuda"


def asl_collate_fn(batch):
    images = []
    targets = []
    for image, target in batch:
        if not torch.is_tensor(image):
            raise TypeError(f"Expected tensor image, got {type(image)}")
        images.append(image.contiguous())
        targets.append(int(target))

    heights = [img.shape[-2] for img in images]
    widths = [img.shape[-1] for img in images]
    max_h = max(heights)
    max_w = max(widths)

    if len(set(heights)) == 1 and len(set(widths)) == 1:
        batch_images = torch.stack(images, dim=0)
    else:
        # Preserve native resolution by padding only to batch-local max size.
        padded = []
        for img in images:
            pad_h = max_h - img.shape[-2]
            pad_w = max_w - img.shape[-1]
            padded.append(F.pad(img, (0, pad_w, 0, pad_h), value=0.0))
        batch_images = torch.stack(padded, dim=0)

    batch_targets = torch.tensor(targets, dtype=torch.long)
    return batch_images, batch_targets


class FilteredImageDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if torch.is_tensor(image):
            image = image.contiguous()
        return image, target


class VideoFrameDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None, random_frame: bool = True):
        self.samples = samples
        self.transform = transform
        self.random_frame = random_frame
        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.samples)

    def _decoded_to_pil(self, decoded):
        tensor = None
        if torch.is_tensor(decoded):
            tensor = decoded
        elif isinstance(decoded, dict):
            for key in ("data", "frame", "image", "video"):
                value = decoded.get(key)
                if torch.is_tensor(value):
                    tensor = value
                    break
        elif isinstance(decoded, (list, tuple)) and decoded:
            first = decoded[0]
            if torch.is_tensor(first):
                tensor = first
            elif hasattr(first, "data") and torch.is_tensor(first.data):
                tensor = first.data
        elif hasattr(decoded, "data") and torch.is_tensor(decoded.data):
            tensor = decoded.data

        if tensor is None:
            raise RuntimeError(f"Unsupported torchcodec frame type: {type(decoded)}")

        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise RuntimeError(f"Expected 3D frame tensor, got shape={tuple(tensor.shape)}")
        if tensor.shape[0] not in (1, 3, 4) and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)

        return self.to_pil(tensor)

    def _load_frame_torchcodec(self, video_path: str):
        if TorchcodecVideoDecoder is None:
            raise RuntimeError("torchcodec is not installed")

        decoder = TorchcodecVideoDecoder(video_path)
        total_frames = None
        try:
            total_frames = int(len(decoder))
        except Exception:
            total_frames = None

        frame_index = 0
        if total_frames is not None and total_frames > 0:
            frame_index = random.randrange(total_frames) if self.random_frame else total_frames // 2

        if total_frames is not None and total_frames > 0:
            try:
                return self._decoded_to_pil(decoder[frame_index])
            except Exception:
                pass

        for method_name in ("get_frame_at_index", "get_frame_at", "get_frame"):
            method = getattr(decoder, method_name, None)
            if method is None:
                continue
            try:
                return self._decoded_to_pil(method(frame_index))
            except Exception:
                continue

        iterator = iter(decoder)
        return self._decoded_to_pil(next(iterator))

    def _load_frame_ffmpeg(self, video_path: str):
        duration = None
        try:
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=8,
            )
            if probe.returncode == 0:
                duration = float(probe.stdout.strip())
        except Exception:
            duration = None

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        if self.random_frame and duration is not None and duration > 0.2:
            t = random.uniform(0.0, max(0.0, duration - 0.1))
            cmd.extend(["-ss", f"{t:.3f}"])
        cmd.extend(
            [
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ]
        )

        proc = subprocess.run(cmd, capture_output=True, check=False, timeout=20)
        if proc.returncode != 0 or not proc.stdout:
            raise RuntimeError(
                f"ffmpeg decode failed for {video_path}: {proc.stderr.decode(errors='ignore')[:300]}"
            )
        return Image.open(io.BytesIO(proc.stdout)).convert("RGB")

    def __getitem__(self, idx):
        max_attempts = min(8, len(self.samples))
        last_exc = None
        image = None
        target = None

        for attempt in range(max_attempts):
            sample_idx = (idx + attempt) % len(self.samples)
            video_path, target = self.samples[sample_idx]
            try:
                image = self._load_frame_torchcodec(video_path)
                break
            except Exception as exc:
                last_exc = exc
                try:
                    image = self._load_frame_ffmpeg(video_path)
                    break
                except Exception as ffmpeg_exc:
                    last_exc = ffmpeg_exc
                    continue

        if image is None:
            raise RuntimeError(f"Failed to decode video after retries: {last_exc}")

        if self.transform is not None:
            image = self.transform(image)
        if torch.is_tensor(image):
            image = image.contiguous()
        return image, target


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=12,
                translate=(0.06, 0.06),
                scale=(0.92, 1.08),
                shear=8,
            ),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.25),
            transforms.ColorJitter(
                brightness=0.12, contrast=0.18, saturation=0.08, hue=0.02
            ),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=0.18,
                scale=(0.01, 0.08),
                ratio=(0.5, 1.8),
                value="random",
            ),
            transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD),
        ]
    )
    return train_transform, val_transform


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(16, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.shape
        scale = self.pool(x).view(batch_size, channels)
        scale = self.fc(scale).view(batch_size, channels, 1, 1)
        return x * scale


class CurvePatternMixer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # Multi-scale depthwise branches emphasize curved contours and local strokes.
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.branch_dilated = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        bd = self.branch_dilated(x)
        return self.fuse(torch.cat([b3, b5, bd], dim=1))


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ResidualASLBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        drop_path_prob: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.pattern_mixer = CurvePatternMixer(out_channels)
        self.drop_path = DropPath(drop_path_prob)
        self.act = nn.GELU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se(x)
        x = self.pattern_mixer(x)
        x = self.drop_path(x)
        x = x + residual
        return self.act(x)


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)


def build_datasets(root_dir: str):
    from torchvision.datasets import ImageFolder

    folder_dataset = ImageFolder(root_dir)

    normalized_to_old_idx = {}
    for class_name, old_idx in folder_dataset.class_to_idx.items():
        normalized_to_old_idx.setdefault(normalize_class_name(class_name), old_idx)

    missing = [
        name
        for name in CLASSES
        if normalize_class_name(name) not in normalized_to_old_idx
    ]
    if missing:
        raise ValueError(f"Missing class folders: {missing}")

    class_to_new_idx = {
        normalized_to_old_idx[normalize_class_name(name)]: idx
        for idx, name in enumerate(CLASSES)
    }

    filtered_samples = []
    for path, target in folder_dataset.samples:
        if target in class_to_new_idx:
            filtered_samples.append((path, class_to_new_idx[target]))

    if not filtered_samples:
        raise ValueError("No samples found for requested classes.")

    train_transform, val_transform = build_transforms()

    train_dataset = FilteredImageDataset(filtered_samples, transform=train_transform)
    val_dataset = FilteredImageDataset(filtered_samples, transform=val_transform)

    total = len(filtered_samples)
    val_size = int(total * 0.2)
    train_size = total - val_size

    g = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(total, generator=g).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    return train_subset, val_subset


def normalize_csv_col(col_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", col_name.strip().lower())


def resolve_csv_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    normalized_candidates = {normalize_csv_col(name) for name in candidates}
    for name in fieldnames:
        if normalize_csv_col(name) in normalized_candidates:
            return name
    return None


def parse_label_to_target(raw_label) -> int | None:
    if raw_label is None:
        return None
    label_text = str(raw_label).strip()
    if not label_text:
        return None

    normalized = normalize_class_name(label_text)
    if normalized in REQUIRED_CLASSES_NORMALIZED:
        return CLASSES.index(normalized)

    try:
        numeric = int(float(label_text))
    except ValueError:
        numeric = None

    if numeric is not None:
        if 0 <= numeric < len(CLASSES):
            return numeric
        if 1 <= numeric <= len(CLASSES):
            return numeric - 1

    for token in re.split(r"[^A-Z0-9]+", normalized):
        if token in REQUIRED_CLASSES_NORMALIZED:
            return CLASSES.index(token)
    return None


def build_video_lookup(videos_dir: str) -> dict[str, str]:
    lookup = {}
    for root, _, files in os.walk(videos_dir):
        for file_name in files:
            full_path = os.path.join(root, file_name)
            rel_path = os.path.relpath(full_path, videos_dir)
            lookup[file_name.lower()] = full_path
            lookup[rel_path.replace("\\", "/").lower()] = full_path
    return lookup


def resolve_video_path(
    raw_path: str, dataset_path: str, videos_dir: str, lookup: dict[str, str]
) -> str | None:
    if not raw_path:
        return None
    raw_path = str(raw_path).strip()
    if not raw_path:
        return None

    direct_candidates = [
        raw_path,
        os.path.join(dataset_path, raw_path),
        os.path.join(videos_dir, raw_path),
        os.path.join(videos_dir, os.path.basename(raw_path)),
    ]
    for candidate in direct_candidates:
        if os.path.isfile(candidate):
            return candidate

    normalized = raw_path.replace("\\", "/").lower()
    if normalized in lookup:
        return lookup[normalized]

    base_name = os.path.basename(raw_path).lower()
    return lookup.get(base_name)


def load_asl_citizen_samples(
    csv_path: str, dataset_path: str, videos_dir: str, lookup: dict[str, str]
):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")

        path_col = resolve_csv_column(
            reader.fieldnames,
            [
                "video",
                "video_path",
                "video_file",
                "file",
                "file_path",
                "filepath",
                "filename",
                "name",
                "path",
            ],
        )
        label_col = resolve_csv_column(
            reader.fieldnames,
            [
                "label",
                "class",
                "gesture",
                "sign",
                "target",
                "y",
                "glos",
                "gloss",
                "asl-lex code",
                "asl_lex_code",
                "asllexcode",
            ],
        )
        if path_col is None or label_col is None:
            raise ValueError(
                f"Could not detect video/label columns in {csv_path}. "
                f"Found columns: {reader.fieldnames}"
            )

        samples = []
        for row in reader:
            target = parse_label_to_target(row.get(label_col))
            if target is None:
                continue
            video_path = resolve_video_path(
                row.get(path_col, ""), dataset_path, videos_dir, lookup
            )
            if video_path is None:
                continue
            samples.append((video_path, target))

    if not samples:
        raise ValueError(
            f"No usable A-E samples found in {csv_path}. "
            f"Check label values and video paths."
        )
    return samples


def build_asl_citizen_datasets(dataset_path: str):
    dataset_root = dataset_path
    asl_citizen_root = os.path.join(dataset_path, "ASL_Citizen")
    if os.path.isdir(asl_citizen_root):
        dataset_root = asl_citizen_root

    split_dir = os.path.join(dataset_root, "splits")
    videos_dir = os.path.join(dataset_root, "videos")
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Missing splits directory: {split_dir}")
    if not os.path.isdir(videos_dir):
        raise FileNotFoundError(f"Missing videos directory: {videos_dir}")

    lookup = build_video_lookup(videos_dir)
    train_csv = os.path.join(split_dir, "train.csv")
    val_csv = os.path.join(split_dir, "val.csv")

    train_samples = load_asl_citizen_samples(
        train_csv, dataset_path, videos_dir, lookup
    )
    val_samples = load_asl_citizen_samples(val_csv, dataset_path, videos_dir, lookup)

    train_transform, val_transform = build_transforms()
    train_dataset = VideoFrameDataset(
        train_samples, transform=train_transform, random_frame=True
    )
    val_dataset = VideoFrameDataset(
        val_samples, transform=val_transform, random_frame=False
    )
    return train_dataset, val_dataset


def resolve_train_dir(dataset_path: str, dir_candidates: list[str]) -> str:
    def has_required_class_dirs(path: str) -> bool:
        class_dirs = {
            normalize_class_name(name)
            for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))
        }
        return REQUIRED_CLASSES_NORMALIZED.issubset(class_dirs)

    candidates = [
        dataset_path
        if relative_path == ""
        else os.path.join(dataset_path, relative_path)
        for relative_path in dir_candidates
    ]

    checked = []
    for candidate in candidates:
        if not os.path.isdir(candidate):
            checked.append(candidate)
            continue

        raw_class_dirs = [
            name
            for name in os.listdir(candidate)
            if os.path.isdir(os.path.join(candidate, name))
        ]
        if has_required_class_dirs(candidate):
            return candidate
        checked.append(candidate)

        nested_dirs = [os.path.join(candidate, name) for name in raw_class_dirs]
        for nested_dir in nested_dirs:
            if has_required_class_dirs(nested_dir):
                return nested_dir
            checked.append(nested_dir)

    raise FileNotFoundError(
        "Could not find a training directory containing class folders "
        f"{CLASSES}. Checked: " + ", ".join(checked)
    )


def build_model(num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 48, kernel_size=5, stride=1, padding=2, bias=False),
        nn.BatchNorm2d(48),
        nn.GELU(),
        nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.GELU(),
        ResidualASLBlock(64, 96, stride=2),
        ResidualASLBlock(96, 96),
        nn.Dropout2d(0.08),
        ResidualASLBlock(96, 160, stride=2),
        ResidualASLBlock(160, 160),
        nn.Dropout2d(0.12),
        ResidualASLBlock(160, 224, stride=2),
        ResidualASLBlock(224, 224),
        nn.Dropout2d(0.16),
        ResidualASLBlock(224, 320, stride=2),
        ResidualASLBlock(320, 320),
        nn.Dropout2d(0.2),
        ResidualASLBlock(320, 384, stride=1),
        ResidualASLBlock(384, 384),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.LayerNorm(384),
        nn.Linear(384, 256),
        nn.GELU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    use_amp,
    epoch: int | None = None,
    total_epochs: int | None = None,
    micro_batch_size: int = 0,
    allow_auto_microbatch_reduce: bool = True,
    num_classes: int = len(CLASSES),
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    iterator = loader
    if tqdm is not None and use_tqdm_progress(device):
        if epoch is not None and total_epochs is not None:
            desc = f"Epoch {epoch:02d}/{total_epochs:02d} train"
        else:
            desc = "train"
        iterator = tqdm(loader, total=len(loader), desc=desc, unit="batch", leave=False)

    for images, targets in iterator:
        images, targets = move_batch_to_device(images, targets, device)
        valid_mask = (targets >= 0) & (targets < num_classes)
        if not torch.all(valid_mask):
            bad_count = int((~valid_mask).sum().item())
            print(
                f"Skipping {bad_count} samples with invalid labels outside [0, {num_classes - 1}]"
            )
            if not torch.any(valid_mask):
                continue
            images = images[valid_mask]
            targets = targets[valid_mask]
        batch_total = targets.size(0)
        current_micro = (
            min(batch_total, micro_batch_size) if micro_batch_size > 0 else batch_total
        )
        batch_loss_sum = 0.0
        batch_correct = 0

        while True:
            optimizer.zero_grad(set_to_none=True)
            try:
                for start in range(0, batch_total, current_micro):
                    end = min(start + current_micro, batch_total)
                    mb_images = images[start:end]
                    mb_targets = targets[start:end]
                    weight = mb_targets.size(0) / batch_total

                    if use_amp:
                        with autocast_ctx(device, use_amp=True):
                            outputs = model(mb_images)
                            loss = criterion(outputs, mb_targets)
                        scaler.scale(loss * weight).backward()
                    else:
                        outputs = model(mb_images)
                        loss = criterion(outputs, mb_targets)
                        (loss * weight).backward()

                    batch_loss_sum += loss.detach().item() * mb_targets.size(0)
                    _, predicted = outputs.detach().max(1)
                    batch_correct += predicted.eq(mb_targets).sum().item()

                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                break
            except Exception as exc:
                if (
                    device.type == "cuda"
                    and is_cuda_oom_error(exc)
                    and allow_auto_microbatch_reduce
                    and current_micro > 1
                ):
                    current_micro = max(1, current_micro // 2)
                    batch_loss_sum = 0.0
                    batch_correct = 0
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    print(f"CUDA OOM fallback: reducing micro-batch to {current_micro}")
                    continue
                raise

        running_loss += batch_loss_sum
        correct += batch_correct
        total += batch_total
        if tqdm is not None and use_tqdm_progress(device):
            iterator.set_postfix(
                loss=f"{(running_loss / total):.4f}",
                acc=f"{(correct / total):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
    if tqdm is not None and use_tqdm_progress(device) and hasattr(iterator, "close"):
        iterator.close()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(
    model,
    loader,
    criterion,
    device,
    epoch: int | None = None,
    total_epochs: int | None = None,
    micro_batch_size: int = 0,
    allow_auto_microbatch_reduce: bool = True,
    num_classes: int = len(CLASSES),
):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    iterator = loader
    if tqdm is not None and use_tqdm_progress(device):
        if epoch is not None and total_epochs is not None:
            desc = f"Epoch {epoch:02d}/{total_epochs:02d} val"
        else:
            desc = "val"
        iterator = tqdm(loader, total=len(loader), desc=desc, unit="batch", leave=False)

    with torch.no_grad():
        for images, targets in iterator:
            images, targets = move_batch_to_device(images, targets, device)
            valid_mask = (targets >= 0) & (targets < num_classes)
            if not torch.all(valid_mask):
                bad_count = int((~valid_mask).sum().item())
                print(
                    f"Skipping {bad_count} eval samples with invalid labels outside [0, {num_classes - 1}]"
                )
                if not torch.any(valid_mask):
                    continue
                images = images[valid_mask]
                targets = targets[valid_mask]
            batch_total = targets.size(0)
            current_micro = (
                min(batch_total, micro_batch_size)
                if micro_batch_size > 0
                else batch_total
            )
            batch_loss_sum = 0.0
            batch_correct = 0

            while True:
                try:
                    for start in range(0, batch_total, current_micro):
                        end = min(start + current_micro, batch_total)
                        mb_images = images[start:end]
                        mb_targets = targets[start:end]
                        outputs = model(mb_images)
                        loss = criterion(outputs, mb_targets)
                        batch_loss_sum += loss.item() * mb_targets.size(0)
                        _, predicted = outputs.max(1)
                        batch_correct += predicted.eq(mb_targets).sum().item()
                    break
                except Exception as exc:
                    if (
                        device.type == "cuda"
                        and is_cuda_oom_error(exc)
                        and allow_auto_microbatch_reduce
                        and current_micro > 1
                    ):
                        current_micro = max(1, current_micro // 2)
                        batch_loss_sum = 0.0
                        batch_correct = 0
                        torch.cuda.empty_cache()
                        print(
                            f"CUDA OOM fallback (eval): reducing micro-batch to {current_micro}"
                        )
                        continue
                    raise

            running_loss += batch_loss_sum
            correct += batch_correct
            total += batch_total
            if tqdm is not None and use_tqdm_progress(device):
                iterator.set_postfix(
                    loss=f"{(running_loss / total):.4f}",
                    acc=f"{(correct / total):.4f}",
                )
    if tqdm is not None and use_tqdm_progress(device) and hasattr(iterator, "close"):
        iterator.close()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def shutdown_dataloader_workers(loader: DataLoader | None) -> None:
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        try:
            iterator._shutdown_workers()
        except Exception:
            pass


class PickleBatchDataset(Dataset):
    def __init__(self, pickle_path: str, offsets: list[int]):
        self.pickle_path = pickle_path
        self.offsets = offsets

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        with open(self.pickle_path, "rb") as handle:
            handle.seek(self.offsets[idx])
            return pickle.load(handle)


def unwrap_single_item(batch):
    return batch[0]


class EpochPickleDataLoader:
    def __init__(
        self,
        dataloader: DataLoader,
        split_name: str,
        cache_dir: str,
        reader_num_workers: int,
        pin_memory: bool,
        reader_shuffle: bool,
    ):
        self.dataloader = dataloader
        self.split_name = split_name
        safe_split_name = split_name.lower().replace(" ", "_")
        self.cache_dir = cache_dir
        self.pickle_path = os.path.join(self.cache_dir, f"{safe_split_name}_epoch.pkl")
        self.meta_path = os.path.join(
            self.cache_dir, f"{safe_split_name}_epoch.meta.pkl"
        )
        self.reader_num_workers = max(16, reader_num_workers)
        self.pin_memory = pin_memory
        self.reader_shuffle = reader_shuffle
        self.batch_count = 0
        self.offsets: list[int] = []
        self.reader_loader: DataLoader | None = None

    def _build_reader_loader(self) -> None:
        dataset = PickleBatchDataset(self.pickle_path, self.offsets)
        self.reader_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=self.reader_shuffle,
            num_workers=self.reader_num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False,
            collate_fn=unwrap_single_item,
        )

    def _load_existing_metadata(self) -> bool:
        if not os.path.exists(self.meta_path):
            return False

        try:
            with open(self.meta_path, "rb") as handle:
                metadata = pickle.load(handle)
        except Exception:
            return False

        offsets = metadata.get("offsets")
        batch_count = metadata.get("batch_count")
        if not isinstance(offsets, list) or not isinstance(batch_count, int):
            return False
        if batch_count != len(offsets):
            return False

        self.offsets = offsets
        self.batch_count = batch_count
        return self.batch_count > 0

    def _generate_pickle_cache(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        temp_pickle_path = f"{self.pickle_path}.tmp"
        temp_meta_path = f"{self.meta_path}.tmp"
        iterator = self.dataloader
        if tqdm is not None and os.getenv("USE_TQDM", "0") == "1":
            iterator = tqdm(
                self.dataloader,
                total=len(self.dataloader),
                desc=f"{self.split_name} pickle build",
                unit="batch",
            )

        batch_count = 0
        offsets: list[int] = []
        with open(temp_pickle_path, "wb") as handle:
            for images, targets in iterator:
                offsets.append(handle.tell())
                pickle.dump((images, targets), handle, protocol=pickle.HIGHEST_PROTOCOL)
                batch_count += 1

        os.replace(temp_pickle_path, self.pickle_path)
        with open(temp_meta_path, "wb") as handle:
            pickle.dump(
                {"offsets": offsets, "batch_count": batch_count},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(temp_meta_path, self.meta_path)

        self.batch_count = batch_count
        self.offsets = offsets
        print(
            f"Created {self.split_name} pickle cache with "
            f"{self.batch_count} batches: {self.pickle_path} "
            f"(reader_workers={self.reader_num_workers})"
        )

    def prepare_pickle_cache(self) -> None:
        has_pickle = os.path.exists(self.pickle_path)
        has_valid_meta = self._load_existing_metadata()

        if has_pickle and has_valid_meta:
            print(
                f"Reusing existing {self.split_name} pickle cache: {self.pickle_path} "
                f"({self.batch_count} batches)"
            )
        else:
            self._generate_pickle_cache()

        self._build_reader_loader()

    def __iter__(self):
        if (
            self.batch_count < 1
            or not os.path.exists(self.pickle_path)
            or self.reader_loader is None
        ):
            raise RuntimeError(
                f"{self.split_name} pickle cache not ready. "
                "Call prepare_pickle_cache() first."
            )
        return iter(self.reader_loader)

    def __len__(self):
        return self.batch_count

    def close(self) -> None:
        shutdown_dataloader_workers(self.reader_loader)


def main():
    args = parse_args()
    dataset_config = get_dataset_config(args.dataset)

    print("--- Checking Dataset Status ---")
    dataset_path = kagglehub.dataset_download(dataset_config.kaggle_id)
    if dataset_config.key == "asl-citizen":
        train_dir = os.path.join(dataset_path, "splits")
    else:
        train_dir = resolve_train_dir(dataset_path, dataset_config.train_dir_candidates)

    seed_everything(SEED)
    runtime = configure_runtime(args.device, args.amp)
    if args.amp == "on" and runtime.device.type != "cuda":
        print("AMP requested but CUDA is not active; AMP disabled.")

    best_model_path = f"best_asl_classifier_top5_{dataset_config.output_tag}.pt"
    final_model_path = f"asl_classifier_top5_{dataset_config.output_tag}.pt"
    curve_path = f"training_accuracy_{dataset_config.output_tag}.png"

    print(f"Using dataset: {dataset_config.kaggle_id}")
    print(f"Training directory: {train_dir}")
    print(f"Best checkpoint path: {best_model_path}")
    print(f"Final checkpoint path: {final_model_path}")

    if dataset_config.key == "asl-citizen":
        train_ds, val_ds = build_asl_citizen_datasets(dataset_path)
    else:
        train_ds, val_ds = build_datasets(train_dir)

    num_workers = resolve_num_workers(runtime.cpu_threads)
    pickle_reader_workers = resolve_pickle_reader_workers(
        runtime.cpu_threads, runtime.device
    )
    pin_memory = resolve_pin_memory(runtime.device)
    reader_pin_memory = False
    micro_batch_size = resolve_micro_batch_size()
    allow_auto_microbatch_reduce = micro_batch_size == 0
    dataset_fraction = resolve_dataset_fraction()
    max_train_samples = resolve_dataset_cap("MAX_TRAIN_SAMPLES")
    max_val_samples = resolve_dataset_cap("MAX_VAL_SAMPLES")

    train_ds, val_ds = apply_dataset_chunking(
        train_ds,
        val_ds,
        seed=SEED,
        fraction=dataset_fraction,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )
    batch_size = resolve_batch_size(runtime.device, len(train_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        collate_fn=asl_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        collate_fn=asl_collate_fn,
    )

    use_epoch_pickle_cache = resolve_use_epoch_pickle_cache(runtime.device)
    train_epoch_buffer = None
    val_epoch_buffer = None
    train_iterable = train_loader
    val_iterable = val_loader
    if use_epoch_pickle_cache:
        cache_dir = os.path.join(
            os.getcwd(), ".epoch_pickle_cache", dataset_config.output_tag
        )
        train_epoch_buffer = EpochPickleDataLoader(
            train_loader,
            split_name="train",
            cache_dir=cache_dir,
            reader_num_workers=pickle_reader_workers,
            pin_memory=reader_pin_memory,
            reader_shuffle=True,
        )
        val_epoch_buffer = EpochPickleDataLoader(
            val_loader,
            split_name="validation",
            cache_dir=cache_dir,
            reader_num_workers=pickle_reader_workers,
            pin_memory=reader_pin_memory,
            reader_shuffle=False,
        )

    model = None
    model_old = None
    optimizer = None
    scheduler = None
    scaler = None

    try:
        model = build_model(len(CLASSES)).to(runtime.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=7.5e-4, weight_decay=3e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            threshold=1e-3,
            cooldown=1,
            min_lr=1e-6,
        )

        scaler = create_grad_scaler(runtime.use_amp)

        best_val_acc = 0.0
        epochs_without_improve = 0

        history = {"train_accuracy": [], "val_accuracy": []}

        print("\n--- Starting Training ---")
        print(
            f"Using batch_size={batch_size}, num_workers={num_workers}, "
            f"pickle_reader_workers={pickle_reader_workers}, "
            f"pin_memory={pin_memory}, reader_pin_memory={reader_pin_memory}, "
            f"micro_batch_size={'auto' if micro_batch_size == 0 else micro_batch_size}, "
            f"auto_microbatch_reduce={allow_auto_microbatch_reduce}"
        )
        print(
            f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)} "
            f"(fraction={dataset_fraction}, "
            f"max_train_samples={max_train_samples or 'off'}, "
            f"max_val_samples={max_val_samples or 'off'})"
        )
        if use_epoch_pickle_cache:
            print("Preparing pickle caches (create once, then reuse on next runs)...")
            train_epoch_buffer.prepare_pickle_cache()
            val_epoch_buffer.prepare_pickle_cache()
            train_iterable = train_epoch_buffer
            val_iterable = val_epoch_buffer
        else:
            print("Epoch pickle cache disabled for CUDA/runtime stability.")

        model_old = model
        use_compile = os.getenv("ENABLE_TORCH_COMPILE", "0") == "1"
        if use_compile:
            print("torch.compile enabled via ENABLE_TORCH_COMPILE=1")
            model = torch.compile(model)
        else:
            print("torch.compile disabled by default for CUDA stability.")

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_epoch(
                model,
                train_iterable,
                criterion,
                optimizer,
                runtime.device,
                scaler,
                runtime.use_amp,
                epoch=epoch,
                total_epochs=EPOCHS,
                micro_batch_size=micro_batch_size,
                allow_auto_microbatch_reduce=allow_auto_microbatch_reduce,
                num_classes=len(CLASSES),
            )
            val_loss, val_acc = evaluate(
                model,
                val_iterable,
                criterion,
                runtime.device,
                epoch=epoch,
                total_epochs=EPOCHS,
                micro_batch_size=micro_batch_size,
                allow_auto_microbatch_reduce=allow_auto_microbatch_reduce,
                num_classes=len(CLASSES),
            )

            previous_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            history["train_accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)

            print(
                f"Epoch {epoch:02d}: "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"lr={current_lr:.6f}"
            )
            if current_lr < previous_lr:
                print(
                    f"Learning rate reduced from {previous_lr:.6f} to {current_lr:.6f}"
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improve = 0
                torch.save(model_old.state_dict(), best_model_path)
                print(f"Saved best model to {best_model_path}")
            else:
                epochs_without_improve += 1

            if epochs_without_improve >= 8:
                print("Early stopping triggered.")
                break

        torch.save(model_old.state_dict(), final_model_path)
        print(f"\nModel saved as '{final_model_path}'")

        plt.figure(figsize=(8, 5))
        plt.plot(history["train_accuracy"], label="train_accuracy")
        plt.plot(history["val_accuracy"], label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(curve_path, dpi=150)
        print(f"Training curve saved as '{curve_path}'")
    finally:
        if runtime.device.type == "cuda":
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

        if model_old is not None and runtime.device.type == "cuda":
            try:
                model_old.to("cpu")
            except Exception:
                pass
        if train_epoch_buffer is not None:
            train_epoch_buffer.close()
        if val_epoch_buffer is not None:
            val_epoch_buffer.close()
        shutdown_dataloader_workers(train_loader)
        shutdown_dataloader_workers(val_loader)

        del scaler, scheduler, optimizer, model_old, model
        gc.collect()
        if runtime.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    main()
