from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from .board_encoding import EXTRA_FEATURES_DIM, pretty_label_from_eval
from .dataset import ChessEvaluationDataset, add_dataset_args, inverse_target_to_cp, load_splits
from .model import create_model
from .train import compute_metrics, split_model_output
from .utils import BUCKET_LABELS, apply_bucket_logit_bias, bucket_ids, bucket_ids_from_logits, ensure_dir, get_device, normalize_state_dict_keys, save_json
from .visualization import save_confusion_matrix


def make_loader(dataset, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained chess evaluation CNN.")
    add_dataset_args(parser)
    parser.add_argument("--model", default="models/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--no-preencode", action="store_true", help="Disable up-front parallel FEN encoding.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="Dataset split to evaluate.")
    return parser


def load_model(model_path: str | Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})
    model_name = str(config.get("model_name", "cnn"))
    if model_name == "cnn" and "model_width" not in config:
        model_name = "legacy_cnn"
    model = create_model(
        model_name=model_name,
        extra_features_dim=int(config.get("extra_features_dim", EXTRA_FEATURES_DIM)),
        width=int(config.get("model_width", 192)),
        depth=int(config.get("model_depth", 8)),
        dropout=float(config.get("dropout", 0.15)),
        head_hidden=int(config.get("head_hidden", 512)),
        num_buckets=int(config.get("num_buckets", 0)),
    ).to(device)
    model.load_state_dict(normalize_state_dict_keys(checkpoint["model_state_dict"], model.state_dict()))
    model.eval()
    return model, config


def predict_dataframe(
    model,
    config: dict,
    data_dir: str = "data/chess_evaluations",
    fen_column: str | None = None,
    eval_column: str | None = None,
    eval_clip: float | None = None,
    target_transform: str | None = None,
    target_scale: float | None = None,
    max_samples: int | None = 500_000,
    seed: int = 42,
    output_dir: str | Path = "runs",
    preprocess_workers: int | None = None,
    cache_dir: str | Path = "data/cache",
    use_cache: bool = True,
    batch_size: int = 2048,
    num_workers: int | None = None,
    device: torch.device | None = None,
    preencode: bool = True,
    split: str = "test",
) -> tuple[pd.DataFrame, dict[str, float]]:
    device = device or torch.device("cpu")
    eval_clip = float(config.get("eval_clip", eval_clip or 1500.0))
    target_transform = str(config.get("target_transform", target_transform or "tanh"))
    target_scale = float(config.get("target_scale", target_scale or 600.0))
    extra_features_dim = int(config.get("extra_features_dim", EXTRA_FEATURES_DIM))
    splits = load_splits(
        data_dir=data_dir,
        fen_column=fen_column,
        eval_column=eval_column,
        eval_clip=eval_clip,
        target_transform=target_transform,
        target_scale=target_scale,
        max_samples=max_samples,
        seed=seed,
        output_dir=output_dir,
        preprocess_workers=preprocess_workers,
        cache_dir=cache_dir,
        use_cache=use_cache,
        drop_mate_labels=bool(config.get("drop_mate_labels", False)),
    )
    eval_df = splits[split]
    if len(eval_df) == 0:
        raise ValueError(f"No {split} rows are available after preprocessing.")
    eval_dataset = ChessEvaluationDataset(
        eval_df,
        preencode=preencode,
        encode_workers=preprocess_workers,
        cache_dir=cache_dir,
        use_cache=use_cache,
        extra_features_dim=extra_features_dim,
    )
    loader = make_loader(eval_dataset, batch_size, num_workers if num_workers is not None else (os.cpu_count() or 4), device)
    targets = []
    preds = []
    bucket_logits = []
    model.eval()
    with torch.no_grad():
        for board, extras, target in tqdm(loader, desc="Evaluating"):
            board = board.to(device, non_blocking=True).float()
            extras = extras.to(device, non_blocking=True)
            output = model(board, extras)
            pred, logits = split_model_output(output)
            pred = pred.detach().cpu().numpy()
            preds.append(pred)
            if logits is not None:
                bucket_logits.append(logits.detach().cpu().numpy())
            targets.append(target.numpy())
    target_norm = np.concatenate(targets)
    pred_norm = np.concatenate(preds)
    y_true = inverse_target_to_cp(target_norm, target_transform, eval_clip, target_scale)
    y_pred = inverse_target_to_cp(pred_norm, target_transform, eval_clip, target_scale)
    logits_np = np.concatenate(bucket_logits) if bucket_logits else None
    bucket_logit_bias = config.get("bucket_logit_bias")
    metrics = compute_metrics(
        target_norm,
        pred_norm,
        eval_clip,
        target_transform,
        target_scale,
        logits_np,
        float(config.get("bucket_margin_cp", 0.0)),
        bucket_logit_bias,
    )
    direct_labels = None if logits_np is None else bucket_ids_from_logits(logits_np, bucket_logit_bias)
    predictions = pd.DataFrame(
        {
            "fen": eval_df["fen"],
            "split": split,
            "actual_cp": y_true,
            "predicted_cp": y_pred,
            "absolute_error": np.abs(y_pred - y_true),
            "actual_label": [pretty_label_from_eval(v) for v in y_true],
            "predicted_label": [pretty_label_from_eval(v) for v in y_pred],
        }
    )
    if direct_labels is not None:
        adjusted_logits = apply_bucket_logit_bias(logits_np, bucket_logit_bias)
        probs = torch.softmax(torch.from_numpy(adjusted_logits), dim=-1).numpy()
        label_names = np.array(BUCKET_LABELS, dtype=object)
        predictions["predicted_label_direct"] = label_names[direct_labels]
        for idx, name in enumerate(["black", "equal", "white"]):
            predictions[f"bucket_logit_{name}"] = logits_np[:, idx]
            predictions[f"bucket_prob_{name}"] = probs[:, idx]
    return predictions, metrics


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    device = get_device(args.device)
    if not Path(args.model).exists():
        raise SystemExit(f"Model not found: {args.model}. Train first with: uv run python -m src.train")

    model, config = load_model(args.model, device)
    predictions, metrics = predict_dataframe(
        model,
        config,
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
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        preencode=not args.no_preencode,
        split=args.split,
    )
    metrics["model"] = str(args.model)
    metrics["model_name"] = str(config.get("model_name", "unknown"))
    metrics["target_transform"] = str(config.get("target_transform", args.target_transform))
    metrics["num_samples"] = int(len(predictions))
    metrics["num_test_samples"] = int(len(predictions))
    metrics["split"] = args.split
    save_json(metrics, output_dir / "evaluation_metrics.json")

    labels = BUCKET_LABELS
    predicted_bucket_ids = (
        np.argmax(np.stack([predictions["predicted_label_direct"].to_numpy() == label for label in labels], axis=1), axis=1)
        if "predicted_label_direct" in predictions
        else bucket_ids(predictions["predicted_cp"].to_numpy())
    )
    cm = confusion_matrix(bucket_ids(predictions["actual_cp"].to_numpy()), predicted_bucket_ids, labels=[0, 1, 2])
    save_confusion_matrix(cm, labels, str(output_dir / "confusion_matrix.png"))
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics, predictions, and confusion matrix under {output_dir}")


if __name__ == "__main__":
    main()
