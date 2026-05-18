#!/usr/bin/env python
from __future__ import annotations

import contextlib
import io
import os
import zipfile
import argparse
from pathlib import Path


TARGET_DIR = Path("data/chess_evaluations")
DATASETS = {
    "primary": {
        "slug": "ronakbadhe/chess-evaluations",
        "path": TARGET_DIR / "ronakbadhe_chess_evaluations",
        "url": "https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations",
        "note": "Current depth-22 Stockfish evaluation dataset.",
        "default": True,
    },
    "extra": {
        "slug": "dev102/chess-fens-evaluations-dataset",
        "path": TARGET_DIR / "dev102_chess_fens_evaluations_dataset",
        "url": "https://www.kaggle.com/datasets/dev102/chess-fens-evaluations-dataset",
        "note": "Extra public-domain FEN/Evaluation CSVs, roughly 8M rows; Stockfish depth 0.",
        "default": True,
    },
    "lichess": {
        "slug": "lichess/chess-evaluations",
        "path": TARGET_DIR / "lichess_chess_evaluations",
        "url": "https://www.kaggle.com/datasets/lichess/chess-evaluations",
        "note": "Huge Lichess evaluation corpus, hundreds of millions of Stockfish evaluations. Download explicitly.",
        "default": False,
    },
}


def print_manual_instructions() -> None:
    print("\nManual setup:")
    print("1. Open one of:")
    for dataset in DATASETS.values():
        print(f"   - {dataset['url']}")
    print("2. Download the dataset ZIP or CSV files.")
    print(f"3. Place/extract all CSV files under: {TARGET_DIR}")
    print("4. Kaggle API users can authenticate with one of:")
    print("   - kaggle auth login")
    print("   - ~/.kaggle/access_token from a Kaggle API token")
    print("   - ~/.kaggle/kaggle.json")
    print("   - KAGGLE_API_TOKEN or KAGGLE_USERNAME/KAGGLE_KEY environment variables")


def discover_csvs() -> list[Path]:
    return sorted(TARGET_DIR.rglob("*.csv")) if TARGET_DIR.exists() else []


def has_kaggle_credentials() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return (config_dir / "access_token").exists() or (config_dir / "kaggle.json").exists()


def selected_datasets(selection: str) -> list[dict[str, object]]:
    if selection == "all":
        return [dataset for dataset in DATASETS.values() if dataset.get("default", True)]
    if selection == "mega":
        return list(DATASETS.values())
    return [DATASETS[selection]]


def download_dataset(api, dataset: dict[str, object]) -> None:
    slug = str(dataset["slug"])
    target = Path(dataset["path"])
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kaggle dataset '{slug}' to {target}...")
    api.dataset_download_files(slug, path=str(target), unzip=True, quiet=False)
    for zip_path in target.glob("*.zip"):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
        zip_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download supported Kaggle chess evaluation datasets.")
    parser.add_argument(
        "--dataset",
        choices=["primary", "extra", "lichess", "all", "mega"],
        default="all",
        help="all=primary+extra, lichess=huge Lichess corpus, mega=all supported datasets including Lichess.",
    )
    args = parser.parse_args()

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    attempted_download = False
    if has_kaggle_credentials():
        attempted_download = True
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                api.authenticate()
            for dataset in selected_datasets(args.dataset):
                download_dataset(api, dataset)
        except (Exception, SystemExit) as exc:
            print("Could not download with the Kaggle API.")
            reason = "" if isinstance(exc, SystemExit) else str(exc).strip()
            if reason:
                print(f"Reason: {reason}")
            else:
                print("Reason: Kaggle authentication failed.")
    else:
        print("Could not download with the Kaggle API.")
        print("Reason: Kaggle authentication is not configured.")

    csvs = discover_csvs()
    if csvs:
        print("\nDiscovered CSV files:")
        for path in csvs:
            print(f"- {path}")
    else:
        print(f"\nNo CSV files found under {TARGET_DIR}.")
        print_manual_instructions()
        if attempted_download:
            print("\nThe Kaggle request ran, but no CSV files were discovered after download/extract.")


if __name__ == "__main__":
    main()
