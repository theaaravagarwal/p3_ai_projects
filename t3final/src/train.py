from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .board_encoding import EXTRA_FEATURES_DIM
from .dataset import ChessEvaluationDataset, add_dataset_args, inverse_target_to_cp, load_splits
from .model import create_model
from .utils import bucket_ids, bucket_ids_from_logits, checkpoint_payload, ensure_dir, get_device, normalize_state_dict_keys, pearson_corr, save_json, set_seed


NUM_BUCKETS = 3


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a CNN to predict chess evaluations from FEN.")
    add_dataset_args(parser)
    parser.add_argument("--preset", choices=["smoke", "quick", "strong", "max"], default=None)
    parser.add_argument("--model-name", choices=["resnet18", "resnet34", "efficientnet_b0", "cnn", "legacy_cnn"], default="resnet18")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--model-width", type=int, default=192)
    parser.add_argument("--model-depth", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--head-hidden", type=int, default=512)
    parser.add_argument("--huber-beta", type=float, default=0.25)
    parser.add_argument(
        "--advantage-weight",
        type=float,
        default=0.0,
        help="Extra SmoothL1 weight for decisive positions, scaled by abs(target_cp) / eval_clip.",
    )
    parser.add_argument(
        "--bucket-loss-weight",
        type=float,
        default=0.0,
        help="Weight for direct Black/Equal/White classification loss. Enables a 3-class bucket head when > 0.",
    )
    parser.add_argument(
        "--bucket-margin-cp",
        type=float,
        default=0.0,
        help="Ignore bucket classification labels within this many cp of +/-150 thresholds; regression still uses them.",
    )
    parser.add_argument(
        "--bucket-class-balance",
        action="store_true",
        help="Use inverse-frequency class weights for bucket loss within each batch.",
    )
    parser.add_argument("--no-bucket-class-balance", dest="bucket_class_balance", action="store_false")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--checkpoint-dir", default="models")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--warm-start",
        default=None,
        help="Load checkpoint weights but start optimizer, scheduler, logs, and epoch count from scratch.",
    )
    parser.add_argument("--fresh", action="store_true", help="Start a fresh training run and ignore --resume.")
    parser.add_argument("--use-extra-features", dest="use_extra_features", action="store_true", default=True)
    parser.add_argument("--no-extra-features", dest="use_extra_features", action="store_false")
    parser.add_argument("--no-preencode", action="store_true", help="Disable up-front parallel FEN encoding.")
    parser.add_argument("--no-mirror-augment", action="store_true", help="Disable random color/board mirror augmentation.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile on CUDA for faster repeated epochs.")
    parser.add_argument("--no-channels-last", action="store_true", help="Disable channels-last CUDA convolution layout.")
    return parser


PRESETS = {
    "smoke": {
        "max_samples": 10_000,
        "epochs": 2,
        "batch_size": 256,
        "model_name": "resnet18",
    },
    "quick": {
        "max_samples": 250_000,
        "epochs": 6,
        "batch_size": 512,
        "model_name": "resnet18",
    },
    "strong": {
        "max_samples": 1_000_000,
        "epochs": 12,
        "batch_size": 512,
        "model_name": "resnet34",
        "use_extra_features": True,
        "eval_clip": 3000.0,
        "target_transform": "tanh",
        "target_scale": 900.0,
        "drop_mate_labels": True,
        "advantage_weight": 0.5,
        "bucket_loss_weight": 0.6,
        "bucket_margin_cp": 75.0,
        "bucket_class_balance": True,
    },
    "max": {
        "max_samples": 0,
        "epochs": 40,
        "batch_size": 4096,
        "model_name": "resnet34",
        "use_extra_features": True,
        "eval_clip": 3000.0,
        "target_transform": "tanh",
        "target_scale": 900.0,
        "patience": 10,
        "drop_mate_labels": True,
        "advantage_weight": 0.5,
        "bucket_loss_weight": 0.6,
        "bucket_margin_cp": 75.0,
        "bucket_class_balance": True,
    },
}


def split_model_output(output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(output, tuple):
        return output[0], output[1]
    return output, None


def target_cp_tensor(target: torch.Tensor, transform: str, eval_clip: float, target_scale: float) -> torch.Tensor:
    if transform == "linear":
        return (target * eval_clip).clamp(-eval_clip, eval_clip)
    if transform == "tanh":
        return (torch.atanh(target.clamp(-0.999, 0.999)) * target_scale).clamp(-eval_clip, eval_clip)
    raise ValueError(f"Unsupported target transform: {transform}")


def bucket_targets_from_norm(
    target: torch.Tensor,
    transform: str,
    eval_clip: float,
    target_scale: float,
    bucket_margin_cp: float = 0.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    cp = target_cp_tensor(target, transform, eval_clip, target_scale)
    margin = max(0.0, float(bucket_margin_cp))
    if margin > 0:
        buckets = torch.full_like(target, ignore_index, dtype=torch.long)
        equal_limit = max(0.0, 150.0 - margin)
        buckets = torch.where(cp < (-150.0 - margin), torch.zeros_like(buckets), buckets)
        buckets = torch.where(cp.abs() <= equal_limit, torch.ones_like(buckets), buckets)
        buckets = torch.where(cp > (150.0 + margin), torch.full_like(buckets, 2), buckets)
        return buckets

    buckets = torch.ones_like(target, dtype=torch.long)
    buckets = torch.where(cp > 150.0, torch.full_like(buckets, 2), buckets)
    buckets = torch.where(cp < -150.0, torch.zeros_like(buckets), buckets)
    return buckets


def bucket_targets_from_cp_np(values_cp: np.ndarray, bucket_margin_cp: float = 0.0, ignore_index: int = -100) -> np.ndarray:
    values = np.asarray(values_cp)
    margin = max(0.0, float(bucket_margin_cp))
    if margin <= 0:
        return bucket_ids(values)

    buckets = np.full(values.shape, ignore_index, dtype=np.int64)
    equal_limit = max(0.0, 150.0 - margin)
    buckets[values < (-150.0 - margin)] = 0
    buckets[np.abs(values) <= equal_limit] = 1
    buckets[values > (150.0 + margin)] = 2
    return buckets


class WeightedSmoothL1Loss(nn.Module):
    def __init__(
        self,
        beta: float,
        advantage_weight: float = 0.0,
        eval_clip: float = 1500.0,
        target_transform: str = "tanh",
        target_scale: float = 600.0,
        bucket_loss_weight: float = 0.0,
        bucket_margin_cp: float = 0.0,
        bucket_class_balance: bool = False,
    ):
        super().__init__()
        self.beta = beta
        self.advantage_weight = max(0.0, float(advantage_weight))
        self.eval_clip = float(eval_clip)
        self.target_transform = target_transform
        self.target_scale = float(target_scale)
        self.bucket_loss_weight = max(0.0, float(bucket_loss_weight))
        self.bucket_margin_cp = max(0.0, float(bucket_margin_cp))
        self.bucket_class_balance = bool(bucket_class_balance)

    def forward(self, output: torch.Tensor | tuple[torch.Tensor, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        pred, bucket_logits = split_model_output(output)
        loss = nn.functional.smooth_l1_loss(pred, target, beta=self.beta, reduction="none")
        if self.advantage_weight <= 0:
            regression_loss = loss.mean()
        else:
            with torch.no_grad():
                target_cp = target_cp_tensor(target, self.target_transform, self.eval_clip, self.target_scale)
                weights = 1.0 + self.advantage_weight * (target_cp.abs() / self.eval_clip).clamp(0.0, 1.0)
            regression_loss = (loss * weights).mean()

        if self.bucket_loss_weight <= 0 or bucket_logits is None:
            return regression_loss
        bucket_target = bucket_targets_from_norm(
            target,
            self.target_transform,
            self.eval_clip,
            self.target_scale,
            self.bucket_margin_cp,
        )
        valid = bucket_target != -100
        if not bool(valid.any()):
            return regression_loss
        class_weights = None
        if self.bucket_class_balance:
            counts = torch.bincount(bucket_target[valid], minlength=NUM_BUCKETS).float()
            class_weights = valid.sum().float() / (NUM_BUCKETS * counts.clamp_min(1.0))
            class_weights = class_weights.to(bucket_logits.device)
        bucket_loss = nn.functional.cross_entropy(bucket_logits, bucket_target, weight=class_weights, ignore_index=-100)
        return regression_loss + self.bucket_loss_weight * bucket_loss


def apply_preset_overrides(args: argparse.Namespace, argv: list[str]) -> argparse.Namespace:
    if not args.preset:
        return args
    explicit = {arg.split("=")[0] for arg in argv if arg.startswith("--")}
    for key, value in PRESETS[args.preset].items():
        flag = "--" + key.replace("_", "-")
        if flag not in explicit:
            setattr(args, key, value)
    return args


def compute_metrics(
    target_norm: np.ndarray,
    pred_norm: np.ndarray,
    eval_clip: float,
    target_transform: str = "linear",
    target_scale: float = 600.0,
    bucket_logits: np.ndarray | None = None,
    bucket_margin_cp: float = 0.0,
    bucket_logit_bias: list[float] | np.ndarray | None = None,
) -> dict[str, float]:
    y_true = inverse_target_to_cp(target_norm, target_transform, eval_clip, target_scale)
    y_pred = inverse_target_to_cp(pred_norm, target_transform, eval_clip, target_scale)
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    corr = pearson_corr(y_true, y_pred)
    bucket_accuracy = float(np.mean(bucket_ids(y_true) == bucket_ids(y_pred)))
    metrics = {
        "mae_cp": mae,
        "mse": mse,
        "rmse_cp": rmse,
        "pearson": corr,
        "bucket_accuracy": bucket_accuracy,
    }
    if bucket_logits is not None:
        direct = bucket_ids_from_logits(bucket_logits, bucket_logit_bias)
        true_buckets = bucket_ids(y_true)
        metrics["bucket_accuracy_direct"] = float(np.mean(true_buckets == direct))
        clean_buckets = bucket_targets_from_cp_np(y_true, bucket_margin_cp)
        clean_mask = clean_buckets != -100
        if np.any(clean_mask):
            metrics["bucket_accuracy_clean"] = float(np.mean(true_buckets[clean_mask] == bucket_ids(y_pred)[clean_mask]))
            metrics["bucket_accuracy_direct_clean"] = float(np.mean(clean_buckets[clean_mask] == direct[clean_mask]))
            metrics["bucket_clean_coverage"] = float(np.mean(clean_mask))
    return metrics


def evaluate_loader(
    model,
    loader,
    criterion,
    device,
    eval_clip: float,
    target_transform: str = "linear",
    target_scale: float = 600.0,
    channels_last: bool = False,
    bucket_margin_cp: float = 0.0,
) -> dict[str, float]:
    model.eval()
    losses = []
    targets = []
    preds = []
    bucket_logits = []
    with torch.no_grad():
        for board, extras, target in loader:
            board = board.to(device, non_blocking=True).float()
            if channels_last:
                board = board.contiguous(memory_format=torch.channels_last)
            extras = extras.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(board, extras)
            pred, logits = split_model_output(output)
            loss = criterion(output, target)
            losses.append(float(loss.item()))
            targets.append(target.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
            if logits is not None:
                bucket_logits.append(logits.detach().cpu().numpy())
    if not targets:
        return {"loss": 0.0, "mae_cp": 0.0, "mse": 0.0, "rmse_cp": 0.0, "pearson": 0.0, "bucket_accuracy": 0.0}
    target_np = np.concatenate(targets)
    pred_np = np.concatenate(preds)
    logits_np = np.concatenate(bucket_logits) if bucket_logits else None
    metrics = compute_metrics(target_np, pred_np, eval_clip, target_transform, target_scale, logits_np, bucket_margin_cp)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    use_amp: bool,
    channels_last: bool,
    grad_clip: float,
) -> float:
    model.train()
    losses = []
    progress = tqdm(loader, desc="Training", leave=False)
    for board, extras, target in progress:
        board = board.to(device, non_blocking=True).float()
        if channels_last:
            board = board.contiguous(memory_format=torch.channels_last)
        extras = extras.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = model(board, extras)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.item()))
        progress.set_postfix(loss=f"{losses[-1]:.4f}")
    return float(np.mean(losses)) if losses else 0.0


def save_checkpoint(
    path: Path,
    model,
    config: dict,
    metrics: dict,
    epoch: int,
    optimizer=None,
    scheduler=None,
    scaler=None,
    best_mae: float | None = None,
    stale_epochs: int = 0,
    rows: list[dict[str, float]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(model, config, metrics)
    payload.update(
        {
            "epoch": epoch,
            "best_mae": best_mae,
            "stale_epochs": stale_epochs,
            "training_rows": rows or [],
        }
    )
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, path)


def save_training_plot(rows: list[dict[str, float]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [r["epoch"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, [r["train_loss"] for r in rows], label="Train loss")
    ax1.plot(epochs, [r["val_loss"] for r in rows], label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [r["val_mae_cp"] for r in rows], color="tab:red", label="Val MAE cp")
    ax2.set_ylabel("MAE cp")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_training_graphics(rows: list[dict[str, float]], output_dir: Path) -> None:
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [r["train_loss"] for r in rows], label="Train loss")
    ax.plot(epochs, [r["val_loss"] for r in rows], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_loss.png", dpi=160)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, [r["val_mae_cp"] for r in rows], label="MAE cp")
    ax1.plot(epochs, [r["val_rmse_cp"] for r in rows], label="RMSE cp")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Centipawns")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [r["val_pearson"] for r in rows], color="tab:green", label="Correlation")
    ax2.set_ylabel("Pearson correlation")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "validation_metrics.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [r["val_bucket_accuracy"] for r in rows], label="Bucket accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "bucket_accuracy.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = apply_preset_overrides(build_parser().parse_args(), sys.argv[1:])
    if args.fresh:
        args.resume = None
        args.warm_start = None
    explicit = {arg.split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")}
    resume_config = {}
    config_source = args.resume or args.warm_start
    if config_source and Path(config_source).exists():
        resume_config = torch.load(config_source, map_location="cpu").get("config", {})
        resume_overrides = {
            "--model-name": "model_name",
            "--eval-clip": "eval_clip",
            "--target-transform": "target_transform",
            "--target-scale": "target_scale",
            "--model-width": "model_width",
            "--model-depth": "model_depth",
            "--dropout": "dropout",
            "--head-hidden": "head_hidden",
            "--bucket-loss-weight": "bucket_loss_weight",
        }
        for flag, key in resume_overrides.items():
            if flag not in explicit and key in resume_config:
                setattr(args, key.replace("-", "_"), resume_config[key])
        if "--use-extra-features" not in explicit and "--no-extra-features" not in explicit:
            args.use_extra_features = int(resume_config.get("extra_features_dim", EXTRA_FEATURES_DIM)) > 0
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    checkpoint_dir = ensure_dir(args.checkpoint_dir)
    save_json(vars(args), output_dir / "training_config.json")

    splits = load_splits(
        data_dir=args.data_dir,
        fen_column=args.fen_column,
        eval_column=args.eval_column,
        eval_clip=args.eval_clip,
        target_transform=args.target_transform,
        target_scale=args.target_scale,
        max_samples=args.max_samples,
        seed=args.seed,
        output_dir=output_dir,
        preprocess_workers=args.preprocess_workers,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        drop_mate_labels=args.drop_mate_labels,
    )
    if len(splits["train"]) == 0 or len(splits["val"]) == 0:
        raise SystemExit("Not enough valid rows to create train/validation splits.")

    device = get_device(args.device)
    channels_last = device.type == "cuda" and not args.no_channels_last
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    extra_features_dim = EXTRA_FEATURES_DIM if args.use_extra_features else 0
    train_dataset = ChessEvaluationDataset(
        splits["train"],
        preencode=not args.no_preencode,
        encode_workers=args.preprocess_workers,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        mirror_augment=not args.no_mirror_augment,
        extra_features_dim=extra_features_dim,
    )
    val_dataset = ChessEvaluationDataset(
        splits["val"],
        preencode=not args.no_preencode,
        encode_workers=args.preprocess_workers,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        extra_features_dim=extra_features_dim,
    )
    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, device)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, device)

    model = create_model(
        model_name=args.model_name,
        extra_features_dim=extra_features_dim,
        width=args.model_width,
        depth=args.model_depth,
        dropout=args.dropout,
        head_hidden=args.head_hidden,
        num_buckets=NUM_BUCKETS if args.bucket_loss_weight > 0 else 0,
    ).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.resume or args.warm_start:
        checkpoint = torch.load(args.resume or args.warm_start, map_location=device)
        model.load_state_dict(normalize_state_dict_keys(checkpoint["model_state_dict"], model.state_dict()), strict=False)
    if args.compile and device.type == "cuda":
        model = torch.compile(model)

    criterion = WeightedSmoothL1Loss(
        beta=args.huber_beta,
        advantage_weight=args.advantage_weight,
        eval_clip=args.eval_clip,
        target_transform=args.target_transform,
        target_scale=args.target_scale,
        bucket_loss_weight=args.bucket_loss_weight,
        bucket_margin_cp=args.bucket_margin_cp,
        bucket_class_balance=args.bucket_class_balance,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = {
        "model_name": args.model_name,
        "extra_features_dim": extra_features_dim,
        "use_extra_features": args.use_extra_features,
        "model_width": args.model_width,
        "model_depth": args.model_depth,
        "dropout": args.dropout,
        "head_hidden": args.head_hidden,
        "huber_beta": args.huber_beta,
        "advantage_weight": args.advantage_weight,
        "bucket_loss_weight": args.bucket_loss_weight,
        "bucket_margin_cp": args.bucket_margin_cp,
        "bucket_class_balance": args.bucket_class_balance,
        "num_buckets": NUM_BUCKETS if args.bucket_loss_weight > 0 else 0,
        "target_transform": args.target_transform,
        "target_scale": args.target_scale,
        "grad_clip": args.grad_clip,
        "mirror_augment": not args.no_mirror_augment,
        "channels_last": channels_last,
        "eval_clip": args.eval_clip,
        "drop_mate_labels": args.drop_mate_labels,
        "warm_start": args.warm_start,
        "input": f"12x8x8 board planes plus {extra_features_dim} scalar features",
    }

    best_mae = float("inf")
    best_selection = -float("inf")
    stale_epochs = 0
    rows: list[dict[str, float]] = []
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        optimizer_state_loaded = False
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                optimizer_state_loaded = True
            except ValueError:
                print("Checkpoint optimizer state is incompatible with the current model heads; starting optimizer fresh.")
        if optimizer_state_loaded and "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception:
                print("Checkpoint scheduler state is incompatible with the current scheduler; starting scheduler fresh.")
        if optimizer_state_loaded and "scaler_state_dict" in checkpoint:
            try:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except Exception:
                print("Checkpoint scaler state is incompatible with the current run; starting scaler fresh.")
        best_mae = float(checkpoint.get("best_mae") or checkpoint.get("metrics", {}).get("mae_cp") or best_mae)
        if args.bucket_loss_weight > 0:
            best_selection = float(checkpoint.get("metrics", {}).get("bucket_accuracy_direct", -float("inf")))
        else:
            best_selection = -best_mae
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        rows = list(checkpoint.get("training_rows", []))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if start_epoch > 1:
            print(f"Resuming from epoch {start_epoch} with best validation MAE {best_mae:.1f}cp")
    elif args.warm_start:
        print(f"Warm-started model weights from {args.warm_start}; starting optimizer and scheduler fresh.")
    log_path = output_dir / "training_log.csv"
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
            channels_last,
            args.grad_clip,
        )
        val_metrics = evaluate_loader(
            model,
            val_loader,
            criterion,
            device,
            args.eval_clip,
            args.target_transform,
            args.target_scale,
            channels_last,
            args.bucket_margin_cp,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mae_cp": val_metrics["mae_cp"],
            "val_rmse_cp": val_metrics["rmse_cp"],
            "val_pearson": val_metrics["pearson"],
            "val_bucket_accuracy": val_metrics["bucket_accuracy"],
            "val_bucket_accuracy_direct": val_metrics.get("bucket_accuracy_direct", 0.0),
            "val_bucket_accuracy_clean": val_metrics.get("bucket_accuracy_clean", 0.0),
            "val_bucket_accuracy_direct_clean": val_metrics.get("bucket_accuracy_direct_clean", 0.0),
            "val_bucket_clean_coverage": val_metrics.get("bucket_clean_coverage", 0.0),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        rows.append(row)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_mae={val_metrics['mae_cp']:.1f}cp "
            f"val_rmse={val_metrics['rmse_cp']:.1f}cp corr={val_metrics['pearson']:.3f} "
            f"bucket_acc={val_metrics['bucket_accuracy']:.3f}"
            + (f" bucket_acc_direct={val_metrics['bucket_accuracy_direct']:.3f}" if "bucket_accuracy_direct" in val_metrics else "")
            + (f" bucket_acc_direct_clean={val_metrics['bucket_accuracy_direct_clean']:.3f}" if "bucket_accuracy_direct_clean" in val_metrics else "")
        )
        best_mae = min(best_mae, val_metrics["mae_cp"])
        selection_score = (
            val_metrics.get("bucket_accuracy_direct", val_metrics["bucket_accuracy"])
            if args.bucket_loss_weight > 0
            else -val_metrics["mae_cp"]
        )
        if selection_score > best_selection:
            best_selection = selection_score
            stale_epochs = 0
            save_checkpoint(
                checkpoint_dir / "best_model.pt",
                model,
                config,
                val_metrics,
                epoch,
                optimizer,
                scheduler,
                scaler,
                best_mae,
                stale_epochs,
                rows,
            )
        else:
            stale_epochs += 1
        save_checkpoint(
            checkpoint_dir / "latest_model.pt",
            model,
            config,
            val_metrics,
            epoch,
            optimizer,
            scheduler,
            scaler,
            best_mae,
            stale_epochs,
            rows,
        )
        if stale_epochs >= args.patience:
            print(f"Early stopping after {args.patience} epochs without validation MAE improvement.")
            break

    final_metrics = rows[-1] if rows else {}
    save_checkpoint(
        checkpoint_dir / "final_model.pt",
        model,
        config,
        final_metrics,
        rows[-1]["epoch"] if rows else 0,
        optimizer,
        scheduler,
        scaler,
        best_mae,
        stale_epochs,
        rows,
    )
    save_training_plot(rows, output_dir / "training_plot.png")
    save_training_graphics(rows, output_dir)
    print(f"Saved best checkpoint to {checkpoint_dir / 'best_model.pt'}")
    print(f"Saved final checkpoint to {checkpoint_dir / 'final_model.pt'}")


if __name__ == "__main__":
    main()
