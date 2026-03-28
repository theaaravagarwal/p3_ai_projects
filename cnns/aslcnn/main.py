import os
import random
import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import default_loader

# Configuration
IMG_SIZE = 128
DEFAULT_BATCH_SIZE = 16
EPOCHS = 75
CLASSES = [
    "A",
    "B",
    "C",
    "D",
    "E",
]
SEED = 42
CLASS_SET = set(CLASSES)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CHANNEL_MEAN = (0.485, 0.456, 0.406)
CHANNEL_STD = (0.229, 0.224, 0.225)


@dataclass
class RuntimeConfig:
    device: torch.device
    use_amp: bool
    cpu_threads: int


def configure_runtime() -> RuntimeConfig:
    cpu_count = os.cpu_count() or 4
    cpu_threads = max(2, min(8, cpu_count // 2))

    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    if use_amp:
        print(
            f"GPU detected: mixed precision enabled on {torch.cuda.get_device_name(0)}."
        )
    else:
        print("No GPU detected. Training will run on CPU.")

    return RuntimeConfig(device=device, use_amp=use_amp, cpu_threads=cpu_threads)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an ASL classifier on the first five letters."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Local dataset directory containing class folders A, B, C, D, E.",
    )
    return parser.parse_args()


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
        free_bytes, _ = torch.cuda.mem_get_info(device=device)
        free_gb = free_bytes / (1024**3)

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
        return image, target


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def normalize_label_token(token: str) -> str | None:
    cleaned = token.strip().upper()
    return cleaned if cleaned in CLASS_SET else None


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
    if root.name in CLASS_SET:
        return [root]
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name in CLASS_SET),
        key=lambda path: path.as_posix().lower(),
    )


def discover_samples_under_root(root: Path) -> list[tuple[str, int]]:
    samples: list[tuple[str, int]] = []
    seen_paths: set[Path] = set()
    class_to_idx = {name: idx for idx, name in enumerate(CLASSES)}

    class_directories = discover_class_directories(root)
    if class_directories:
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
                samples.append((image_path.as_posix(), class_to_idx[label_name]))
        if samples:
            return samples

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
        samples.append((image_path.as_posix(), class_to_idx[label_name]))
    return samples


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


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_map, max_map], dim=1)
        return x * self.sigmoid(self.conv(attn))


class ResidualASLBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.spatial = SpatialAttention()
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
        x = self.spatial(x)
        x = x + residual
        return self.act(x)


def stratified_split(samples: list[tuple[str, int]]) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    grouped: dict[int, list[tuple[str, int]]] = {}
    for sample in samples:
        grouped.setdefault(sample[1], []).append(sample)

    missing_classes = [name for idx, name in enumerate(CLASSES) if idx not in grouped]
    if missing_classes:
        raise ValueError(f"Missing class folders or samples: {missing_classes}")

    train_samples: list[tuple[str, int]] = []
    val_samples: list[tuple[str, int]] = []
    rng = random.Random(SEED)
    for label_idx in range(len(CLASSES)):
        class_samples = grouped[label_idx]
        if len(class_samples) < 2:
            raise ValueError(
                f"Class {CLASSES[label_idx]} requires at least 2 images for 80/20 split."
            )
        class_copy = class_samples.copy()
        rng.shuffle(class_copy)
        val_count = max(1, int(round(len(class_copy) * 0.2)))
        if val_count >= len(class_copy):
            val_count = len(class_copy) - 1
        val_samples.extend(class_copy[:val_count])
        train_samples.extend(class_copy[val_count:])
    return train_samples, val_samples


def build_datasets(dataset_root: Path):
    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root is not a directory: {dataset_root}")

    explicit_train = dataset_root / "train"
    explicit_test = dataset_root / "test"
    if explicit_train.is_dir() and explicit_test.is_dir():
        train_samples = discover_samples_under_root(explicit_train)
        val_samples = discover_samples_under_root(explicit_test)
        if not train_samples or not val_samples:
            raise ValueError("Explicit train/test layout found, but one split is empty.")
    else:
        discovered = discover_samples_under_root(dataset_root)
        if not discovered:
            raise ValueError(
                f"No A-E samples discovered under local dataset root: {dataset_root}"
            )
        train_samples, val_samples = stratified_split(discovered)

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE + 12, IMG_SIZE + 12)),
            transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
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
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD),
        ]
    )

    train_dataset = FilteredImageDataset(train_samples, transform=train_transform)
    val_dataset = FilteredImageDataset(val_samples, transform=val_transform)
    return train_dataset, val_dataset


def build_model(num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.GELU(),
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.GELU(),
        ResidualASLBlock(32, 64, stride=2),
        ResidualASLBlock(64, 64),
        nn.Dropout2d(0.08),
        ResidualASLBlock(64, 128, stride=2),
        ResidualASLBlock(128, 128),
        nn.Dropout2d(0.12),
        ResidualASLBlock(128, 192, stride=2),
        ResidualASLBlock(192, 192),
        nn.Dropout2d(0.16),
        ResidualASLBlock(192, 256, stride=1),
        ResidualASLBlock(256, 256),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.LayerNorm(256),
        nn.Linear(256, 192),
        nn.GELU(),
        nn.Dropout(0.35),
        nn.Linear(192, num_classes),
    )


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    args = parse_args()
    output_tag = args.data_dir.name or "local"

    seed_everything(SEED)
    runtime = configure_runtime()

    best_model_path = f"best_asl_classifier_top5_{output_tag}.pt"
    final_model_path = f"asl_classifier_top5_{output_tag}.pt"
    curve_path = f"training_accuracy_{output_tag}.png"

    print(f"Using local dataset root: {args.data_dir.expanduser().resolve()}")
    print(f"Best checkpoint path: {best_model_path}")
    print(f"Final checkpoint path: {final_model_path}")

    train_ds, val_ds = build_datasets(args.data_dir)
    batch_size = resolve_batch_size(runtime.device, len(train_ds))

    # num_workers = min(4, runtime.cpu_threads)
    num_workers = 32
    pin_memory = runtime.device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

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

    scaler = torch.cuda.amp.GradScaler(enabled=runtime.use_amp)

    best_val_acc = 0.0
    epochs_without_improve = 0

    history = {"train_accuracy": [], "val_accuracy": []}

    print("\n--- Starting Training ---")
    print(f"Using batch_size={batch_size}, num_workers={num_workers}")
    model_old = model
    model = torch.compile(model)
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            runtime.device,
            scaler,
            runtime.use_amp,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, runtime.device)

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
            print(f"Learning rate reduced from {previous_lr:.6f} to {current_lr:.6f}")

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


if __name__ == "__main__":
    main()
