#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def print_row(title: str, row) -> None:
    print(f"\n{title}")
    for key, value in row.items():
        print(f"  {key}: {value}")


def summarize_log(path: Path) -> None:
    df = pd.read_csv(path)
    if df.empty:
        print(f"\n### {path}\nempty")
        return

    print(f"\n### {path}")
    print(f"epochs: {len(df)}")
    print_row("last", df.tail(1).iloc[0].to_dict())

    if "val_bucket_accuracy_direct" in df:
        print_row("best full direct", df.loc[df["val_bucket_accuracy_direct"].idxmax()].to_dict())
    if "val_bucket_accuracy_direct_clean" in df:
        print_row("best clean direct", df.loc[df["val_bucket_accuracy_direct_clean"].idxmax()].to_dict())
    if "val_mae_cp" in df:
        print_row("best MAE", df.loc[df["val_mae_cp"].idxmin()].to_dict())


def summarize_leaderboard(path: Path) -> None:
    if not path.exists():
        return
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"\n### {path}\nCould not parse JSON: {exc}")
        return
    if not rows:
        return

    print(f"\n### {path}")
    for row in rows[:10]:
        print(
            f"{row.get('trial', '<unknown>'):16s} "
            f"direct={float(row.get('bucket_accuracy_direct', row.get('bucket_accuracy', 0.0))):.4f} "
            f"clean={float(row.get('bucket_accuracy_direct_clean', 0.0)):.4f} "
            f"coverage={float(row.get('bucket_clean_coverage', 0.0)):.3f} "
            f"mae={float(row.get('mae_cp', 0.0)):.1f} "
            f"model={row.get('model', '')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize all training logs and sweep leaderboards.")
    parser.add_argument("--root", default=".", help="Directory to search.")
    parser.add_argument("--log", action="append", default=[], help="Specific training_log.csv path to summarize.")
    args = parser.parse_args()

    root = Path(args.root)
    paths = [Path(p) for p in args.log] if args.log else sorted(root.glob("**/training_log.csv"))
    if not paths:
        print(f"No training_log.csv files found under {root}")
    for path in paths:
        summarize_log(path)

    summarize_leaderboard(root / "sweeps" / "bucket" / "leaderboard.json")


if __name__ == "__main__":
    main()
