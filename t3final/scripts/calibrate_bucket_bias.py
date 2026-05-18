from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def bucket_ids(values_cp: np.ndarray) -> np.ndarray:
    values = np.asarray(values_cp)
    buckets = np.ones(values.shape, dtype=np.int64)
    buckets[values > 150.0] = 2
    buckets[values < -150.0] = 0
    return buckets


def clean_mask(values_cp: np.ndarray, margin_cp: float) -> np.ndarray:
    margin = max(0.0, float(margin_cp))
    values = np.asarray(values_cp)
    return (values < (-150.0 - margin)) | (np.abs(values) <= (150.0 - margin)) | (values > (150.0 + margin))


def score_bias(logits: np.ndarray, true_buckets: np.ndarray, bias: np.ndarray, mask: np.ndarray | None = None) -> float:
    pred = np.argmax(logits + bias.reshape(1, 3), axis=1)
    if mask is not None:
        pred = pred[mask]
        true_buckets = true_buckets[mask]
    if len(true_buckets) == 0:
        return 0.0
    return float(np.mean(pred == true_buckets))


def search_bias(
    logits: np.ndarray,
    true_buckets: np.ndarray,
    coarse_min: float,
    coarse_max: float,
    coarse_step: float,
    fine_radius: float,
    fine_step: float,
) -> tuple[np.ndarray, dict[str, float]]:
    values = np.arange(coarse_min, coarse_max + coarse_step * 0.5, coarse_step, dtype=np.float32)
    best_bias = np.zeros(3, dtype=np.float32)
    best_score = -1.0
    best_norm = float("inf")
    # Black is fixed at 0 because adding the same constant to all logits changes nothing.
    for equal_bias in values:
        for white_bias in values:
            bias = np.array([0.0, equal_bias, white_bias], dtype=np.float32)
            score = score_bias(logits, true_buckets, bias)
            norm = float(np.linalg.norm(bias))
            if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and norm < best_norm):
                best_score = score
                best_bias = bias
                best_norm = norm

    fine_equal = np.arange(best_bias[1] - fine_radius, best_bias[1] + fine_radius + fine_step * 0.5, fine_step, dtype=np.float32)
    fine_white = np.arange(best_bias[2] - fine_radius, best_bias[2] + fine_radius + fine_step * 0.5, fine_step, dtype=np.float32)
    for equal_bias in fine_equal:
        for white_bias in fine_white:
            bias = np.array([0.0, equal_bias, white_bias], dtype=np.float32)
            score = score_bias(logits, true_buckets, bias)
            norm = float(np.linalg.norm(bias))
            if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and norm < best_norm):
                best_score = score
                best_bias = bias
                best_norm = norm

    metrics = {
        "bucket_accuracy_direct": score_bias(logits, true_buckets, best_bias),
    }
    best_bias[np.abs(best_bias) < fine_step * 0.5] = 0.0
    return best_bias, metrics


def compute_metrics(logits: np.ndarray, true_cp: np.ndarray, bias: np.ndarray, margin_cp: float) -> dict[str, float]:
    true_buckets = bucket_ids(true_cp)
    mask = clean_mask(true_cp, margin_cp)
    metrics = {
        "bucket_accuracy_direct": score_bias(logits, true_buckets, bias),
        "bucket_accuracy_direct_clean": score_bias(logits, true_buckets, bias, mask),
        "bucket_clean_coverage": float(np.mean(mask)),
    }
    return metrics


def update_checkpoint_config(checkpoint_in: Path, checkpoint_out: Path, bias: np.ndarray, metrics: dict[str, float]) -> None:
    checkpoint = torch.load(checkpoint_in, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    config["bucket_logit_bias"] = [float(x) for x in bias]
    config["bucket_calibration_metrics"] = metrics
    checkpoint["config"] = config
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grid-search a small class-logit bias for bucket-head checkpoints.")
    parser.add_argument("--predictions", required=True, help="predictions.csv produced by src.evaluate after this patch.")
    parser.add_argument("--output", default="bucket_calibration.json")
    parser.add_argument("--margin-cp", type=float, default=75.0)
    parser.add_argument("--coarse-min", type=float, default=-1.5)
    parser.add_argument("--coarse-max", type=float, default=1.5)
    parser.add_argument("--coarse-step", type=float, default=0.05)
    parser.add_argument("--fine-radius", type=float, default=0.06)
    parser.add_argument("--fine-step", type=float, default=0.01)
    parser.add_argument("--checkpoint-in", help="Optional checkpoint to copy and annotate with bucket_logit_bias.")
    parser.add_argument("--checkpoint-out", help="Output checkpoint path when --checkpoint-in is provided.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.predictions)
    required = ["actual_cp", "bucket_logit_black", "bucket_logit_equal", "bucket_logit_white"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(f"{args.predictions} is missing {missing}; rerun src.evaluate so it writes bucket logits.")

    logits = df[["bucket_logit_black", "bucket_logit_equal", "bucket_logit_white"]].to_numpy(dtype=np.float32)
    true_cp = df["actual_cp"].to_numpy(dtype=np.float32)
    true_buckets = bucket_ids(true_cp)
    before = compute_metrics(logits, true_cp, np.zeros(3, dtype=np.float32), args.margin_cp)

    best_bias, _ = search_bias(
        logits,
        true_buckets,
        args.coarse_min,
        args.coarse_max,
        args.coarse_step,
        args.fine_radius,
        args.fine_step,
    )
    after = compute_metrics(logits, true_cp, best_bias, args.margin_cp)
    payload = {
        "bucket_logit_bias": [float(x) for x in best_bias],
        "before": before,
        "after": after,
        "predictions": str(args.predictions),
        "margin_cp": float(args.margin_cp),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.checkpoint_in:
        if not args.checkpoint_out:
            raise SystemExit("--checkpoint-out is required with --checkpoint-in")
        update_checkpoint_config(Path(args.checkpoint_in), Path(args.checkpoint_out), best_bias, after)
        payload["checkpoint_out"] = args.checkpoint_out
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
