#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Trial:
    name: str
    bucket_loss_weight: float
    bucket_margin_cp: float
    lr: float | None = None


DEFAULT_TRIALS = [
    Trial("w06_m75", 0.6, 75.0),
    Trial("w08_m75", 0.8, 75.0),
    Trial("w10_m75", 1.0, 75.0),
    Trial("w06_m100", 0.6, 100.0),
    Trial("w08_m100", 0.8, 100.0),
    Trial("w10_m100", 1.0, 100.0),
    Trial("w08_m125", 0.8, 125.0),
    Trial("w10_m125", 1.0, 125.0),
]


def run(cmd: list[str], dry_run: bool) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def load_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an overnight sweep for bucket-head chess evaluation training.")
    parser.add_argument("--warm-start", default="models/best_model.pt")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--eval-max-samples", type=int, default=1_000_000)
    parser.add_argument("--root", default="sweeps/bucket")
    parser.add_argument("--preset", default="max")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-lr-low",
        action="store_true",
        help="Also test lower-lr variants for the strongest margin configs.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    trials = list(DEFAULT_TRIALS)
    if args.include_lr_low:
        trials.extend(
            [
                Trial("w08_m100_lr3e4", 0.8, 100.0, 3e-4),
                Trial("w10_m100_lr3e4", 1.0, 100.0, 3e-4),
                Trial("w10_m125_lr3e4", 1.0, 125.0, 3e-4),
            ]
        )

    results = []
    for trial in trials:
        out_dir = root / "runs" / trial.name
        ckpt_dir = root / "models" / trial.name
        eval_dir = root / "eval" / trial.name
        best_model = ckpt_dir / "best_model.pt"
        metrics_path = eval_dir / "evaluation_metrics.json"

        if not best_model.exists():
            train_cmd = [
                sys.executable,
                "-m",
                "src.train",
                "--preset",
                args.preset,
                "--warm-start",
                args.warm_start,
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--bucket-loss-weight",
                str(trial.bucket_loss_weight),
                "--bucket-margin-cp",
                str(trial.bucket_margin_cp),
                "--output-dir",
                str(out_dir),
                "--checkpoint-dir",
                str(ckpt_dir),
            ]
            if trial.lr is not None:
                train_cmd.extend(["--lr", str(trial.lr)])
            run(train_cmd, args.dry_run)
        else:
            print(f"\nSkipping training for {trial.name}; {best_model} already exists.", flush=True)

        if best_model.exists() and not metrics_path.exists():
            eval_cmd = [
                sys.executable,
                "-m",
                "src.evaluate",
                "--model",
                str(best_model),
                "--max-samples",
                str(args.eval_max_samples),
                "--output-dir",
                str(eval_dir),
            ]
            run(eval_cmd, args.dry_run)

        metrics = load_metrics(metrics_path)
        if metrics:
            row = {
                "trial": trial.name,
                "bucket_loss_weight": trial.bucket_loss_weight,
                "bucket_margin_cp": trial.bucket_margin_cp,
                "lr": trial.lr,
                "model": str(best_model),
                **metrics,
            }
            results.append(row)

    if not results:
        print("\nNo completed trial metrics yet.")
        return

    leaderboard = sorted(
        results,
        key=lambda row: (
            float(row.get("bucket_accuracy_direct", row.get("bucket_accuracy", 0.0))),
            float(row.get("bucket_accuracy_direct_clean", 0.0)),
            -float(row.get("mae_cp", 1e9)),
        ),
        reverse=True,
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")

    print("\nLeaderboard:")
    for row in leaderboard:
        print(
            f"{row['trial']:16s} "
            f"direct={float(row.get('bucket_accuracy_direct', 0.0)):.4f} "
            f"clean={float(row.get('bucket_accuracy_direct_clean', 0.0)):.4f} "
            f"coverage={float(row.get('bucket_clean_coverage', 0.0)):.3f} "
            f"mae={float(row.get('mae_cp', 0.0)):.1f} "
            f"model={row['model']}"
        )


if __name__ == "__main__":
    main()
