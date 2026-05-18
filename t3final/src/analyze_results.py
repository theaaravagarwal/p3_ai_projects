from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix

from .board_encoding import pretty_label_from_eval
from .dataset import add_dataset_args
from .evaluate import load_model, predict_dataframe
from .utils import bucket_ids, ensure_dir, get_device, save_json
from .visualization import render_board_image, save_confusion_matrix


BUCKET_LABELS = ["Black advantage", "Equal", "White advantage"]
EVAL_RANGES = [(-1500, -700), (-700, -300), (-300, -150), (-150, 150), (150, 300), (300, 700), (700, 1500)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate evaluation graphics and report artifacts.")
    add_dataset_args(parser)
    parser.add_argument("--model", default="models/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--output-dir", default="outputs/analysis")
    parser.add_argument("--scatter-samples", type=int, default=20_000)
    parser.add_argument("--no-preencode", action="store_true")
    return parser


def write_summary(metrics: dict, output_dir: Path) -> None:
    save_json(metrics, output_dir / "metrics_summary.json")
    lines = [
        "Chess Evaluation CNN Analysis",
        "",
        f"Model: {metrics.get('model')}",
        f"Model name: {metrics.get('model_name')}",
        f"Target transform: {metrics.get('target_transform')}",
        f"Test samples: {metrics.get('num_test_samples')}",
        f"MAE: {metrics['mae_cp']:.2f} cp",
        f"MSE: {metrics['mse']:.2f}",
        f"RMSE: {metrics['rmse_cp']:.2f} cp",
        f"Pearson correlation: {metrics['pearson']:.4f}",
        f"Bucket accuracy: {metrics['bucket_accuracy']:.4f}",
    ]
    (output_dir / "metrics_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def scatter_plot(df: pd.DataFrame, output_dir: Path, max_points: int) -> None:
    sample = df.sample(n=min(len(df), max_points), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(sample["actual_cp"], sample["predicted_cp"], s=5, alpha=0.25)
    lim = max(abs(sample["actual_cp"]).max(), abs(sample["predicted_cp"]).max(), 100)
    ax.plot([-lim, lim], [-lim, lim], color="red", linewidth=1.5, label="y = x")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("True Stockfish eval (cp)")
    ax.set_ylabel("Predicted eval (cp)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "predicted_vs_actual.png", dpi=180)
    plt.close(fig)


def error_histogram(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["absolute_error"], bins=80, color="steelblue", alpha=0.85)
    ax.set_xlabel("Absolute error (cp)")
    ax.set_ylabel("Positions")
    fig.tight_layout()
    fig.savefig(output_dir / "error_histogram.png", dpi=180)
    plt.close(fig)


def residual_plot(df: pd.DataFrame, output_dir: Path, max_points: int) -> None:
    sample = df.sample(n=min(len(df), max_points), random_state=43).copy()
    sample["error"] = sample["predicted_cp"] - sample["actual_cp"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sample["actual_cp"], sample["error"], s=5, alpha=0.25)
    ax.axhline(0, color="red", linewidth=1.5)
    ax.set_xlabel("True eval (cp)")
    ax.set_ylabel("Prediction error (cp)")
    fig.tight_layout()
    fig.savefig(output_dir / "residual_plot.png", dpi=180)
    plt.close(fig)


def bucket_accuracy_bar(df: pd.DataFrame, output_dir: Path) -> None:
    actual = bucket_ids(df["actual_cp"].to_numpy())
    pred = bucket_ids(df["predicted_cp"].to_numpy())
    accuracies = []
    for bucket in [0, 1, 2]:
        mask = actual == bucket
        accuracies.append(float(np.mean(pred[mask] == bucket)) if np.any(mask) else 0.0)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(BUCKET_LABELS, accuracies, color=["#444444", "#7aa6c2", "#e9e9e9"], edgecolor="black")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_dir / "bucket_accuracy_bar.png", dpi=180)
    plt.close(fig)


def error_by_eval_range(df: pd.DataFrame, output_dir: Path) -> None:
    labels = []
    maes = []
    for low, high in EVAL_RANGES:
        mask = (df["actual_cp"] >= low) & (df["actual_cp"] < high)
        labels.append(f"[{low},{high}]")
        maes.append(float(df.loc[mask, "absolute_error"].mean()) if mask.any() else 0.0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, maes, color="#8fb3d9")
    ax.set_ylabel("MAE (cp)")
    ax.set_xlabel("True eval range (cp)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "error_by_eval_range.png", dpi=180)
    plt.close(fig)


def save_example_boards(df: pd.DataFrame, output_dir: Path) -> None:
    board_dir = ensure_dir(output_dir / "sample_prediction_boards")
    examples = [
        ("best", df.nsmallest(10, "absolute_error")),
        ("worst", df.nlargest(10, "absolute_error")),
    ]
    for prefix, rows in examples:
        for idx, (_, row) in enumerate(rows.iterrows(), start=1):
            board = render_board_image(row["fen"]).resize((420, 420))
            canvas = Image.new("RGB", (420, 520), "white")
            canvas.paste(board, (0, 0))
            draw = ImageDraw.Draw(canvas)
            text = (
                f"True: {row['actual_cp']:.1f} cp ({row['actual_label']})\n"
                f"Pred: {row['predicted_cp']:.1f} cp ({row['predicted_label']})\n"
                f"Abs error: {row['absolute_error']:.1f} cp"
            )
            draw.multiline_text((12, 432), text, fill=(20, 20, 20), spacing=5)
            canvas.save(board_dir / f"{prefix}_{idx:02d}.png")


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    device = get_device(args.device)
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
    )
    metrics.update(
        {
            "model": args.model,
            "model_name": config.get("model_name", "unknown"),
            "target_transform": config.get("target_transform", args.target_transform),
            "num_test_samples": int(len(predictions)),
        }
    )
    write_summary(metrics, output_dir)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    predictions.nlargest(100, "absolute_error").to_csv(output_dir / "worst_predictions.csv", index=False)
    predictions.nsmallest(100, "absolute_error").to_csv(output_dir / "best_predictions.csv", index=False)

    scatter_plot(predictions, output_dir, args.scatter_samples)
    error_histogram(predictions, output_dir)
    residual_plot(predictions, output_dir, args.scatter_samples)
    cm = confusion_matrix(bucket_ids(predictions["actual_cp"].to_numpy()), bucket_ids(predictions["predicted_cp"].to_numpy()), labels=[0, 1, 2])
    save_confusion_matrix(cm, BUCKET_LABELS, str(output_dir / "confusion_matrix.png"))
    bucket_accuracy_bar(predictions, output_dir)
    error_by_eval_range(predictions, output_dir)
    save_example_boards(predictions, output_dir)
    print(json.dumps(metrics, indent=2))
    print(f"Saved analysis report to {output_dir}")


if __name__ == "__main__":
    main()

