from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .dataset import add_dataset_args
from .evaluate import load_model, predict_dataframe
from .utils import ensure_dir, get_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare multiple chess evaluation checkpoints.")
    add_dataset_args(parser)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--no-preencode", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    device = get_device(args.device)
    rows = []
    for model_path in args.models:
        model, config = load_model(model_path, device)
        _, metrics = predict_dataframe(
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
        rows.append(
            {
                "model": model_path,
                "model_name": config.get("model_name", "unknown"),
                "target_transform": config.get("target_transform", args.target_transform),
                "mae_cp": metrics["mae_cp"],
                "rmse_cp": metrics["rmse_cp"],
                "pearson": metrics["pearson"],
                "bucket_accuracy": metrics["bucket_accuracy"],
            }
        )
    results = pd.DataFrame(rows).sort_values("mae_cp")
    results.to_csv(output_dir / "model_comparison.csv", index=False)
    print(results.to_string(index=False))

    labels = [Path(m).name for m in results["model"]]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(labels, results["mae_cp"], color="#7aa6c2", label="MAE cp")
    ax1.set_ylabel("MAE (cp)")
    ax1.tick_params(axis="x", rotation=20)
    ax2 = ax1.twinx()
    ax2.plot(labels, results["bucket_accuracy"], color="tab:green", marker="o", label="Bucket accuracy")
    ax2.set_ylabel("Bucket accuracy")
    ax2.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Saved comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()

