from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import add_dataset_args, load_splits
from src.utils import bucket_ids, ensure_dir, pearson_corr


PIECE_VALUES_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def material_eval_cp(fen: str) -> float:
    board = chess.Board(fen)
    score = 0
    for piece in board.piece_map().values():
        value = PIECE_VALUES_CP[piece.piece_type]
        score += value if piece.color == chess.WHITE else -value
    return float(score)


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    diff = y_pred - y_true
    return {
        "mae_cp": float(np.mean(np.abs(diff))),
        "rmse_cp": float(np.sqrt(np.mean(diff**2))),
        "pearson": pearson_corr(y_true, y_pred),
        "bucket_accuracy": float(np.mean(bucket_ids(y_true) == bucket_ids(y_pred))),
    }


def format_markdown(rows: list[dict[str, float | str]]) -> str:
    lines = [
        "| Model | MAE cp | RMSE cp | Pearson | Bucket Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | "
            f"{float(row['mae_cp']):.2f} | "
            f"{float(row['rmse_cp']):.2f} | "
            f"{float(row['pearson']):.3f} | "
            f"{float(row['bucket_accuracy']) * 100:.2f}% |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare the CNN against a material-only baseline on the same split.")
    add_dataset_args(parser)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--predictions",
        help="Optional predictions.csv from src.evaluate. When provided, FENs and actual_cp are read from this file instead of the raw dataset.",
    )
    parser.add_argument("--cnn-predictions", help="Optional predictions.csv from src.evaluate for the same split.")
    parser.add_argument("--output-dir", default="outputs/baseline_comparison")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    source_predictions = args.predictions or args.cnn_predictions
    if args.predictions:
        frame = pd.read_csv(args.predictions)
        required = {"fen", "actual_cp"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise SystemExit(f"{args.predictions} is missing required columns: {missing}")
    else:
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
        frame = splits[args.split]
    if "target_cp" not in frame and "actual_cp" in frame:
        y_true = frame["actual_cp"].to_numpy(dtype=np.float32)
    else:
        y_true = frame["target_cp"].to_numpy(dtype=np.float32)
    material_pred = np.array([material_eval_cp(fen) for fen in frame["fen"]], dtype=np.float32)

    rows: list[dict[str, float | str]] = [{"model": "Material baseline", **metrics_for(y_true, material_pred)}]
    per_row = pd.DataFrame(
        {
            "fen": frame["fen"],
            "actual_cp": y_true,
            "material_cp": material_pred,
            "material_abs_error": np.abs(material_pred - y_true),
        }
    )

    if source_predictions:
        cnn_df = pd.read_csv(source_predictions)
        if len(cnn_df) != len(frame):
            raise SystemExit(
                f"CNN predictions row count ({len(cnn_df)}) does not match {args.split} split row count ({len(frame)})."
            )
        cnn_pred = cnn_df["predicted_cp"].to_numpy(dtype=np.float32)
        rows.append({"model": "CNN", **metrics_for(y_true, cnn_pred)})
        if "predicted_label_direct" in cnn_df:
            direct_map = {"Black advantage": 0, "Equal": 1, "White advantage": 2}
            direct_pred = cnn_df["predicted_label_direct"].map(direct_map).to_numpy(dtype=np.int64)
            rows[-1]["bucket_accuracy_direct"] = float(np.mean(bucket_ids(y_true) == direct_pred))
        per_row["cnn_cp"] = cnn_pred
        per_row["cnn_abs_error"] = np.abs(cnn_pred - y_true)

    summary = {
        "split": args.split,
        "num_samples": int(len(frame)),
        "rows": rows,
    }
    (output_dir / "baseline_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "baseline_metrics.md").write_text(format_markdown(rows), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_dir / "baseline_metrics.csv", index=False)
    per_row.to_csv(output_dir / "baseline_predictions.csv", index=False)
    print(format_markdown(rows))
    if any("bucket_accuracy_direct" in row for row in rows):
        direct = next(row["bucket_accuracy_direct"] for row in rows if row["model"] == "CNN")
        print(f"CNN direct bucket accuracy: {float(direct) * 100:.2f}%")
    print(f"Saved baseline comparison under {output_dir}")


if __name__ == "__main__":
    main()
