##only used once to get data from dataset before opening the notebook
from __future__ import annotations

from pathlib import Path
import zipfile


def main() -> None:
    ##zip_path = Path.home() / "Downloads" / "steam-reviews.zip"
    zip_path = "./steam-reviews.zip"
    if not zip_path.exists():
        raise SystemExit(f"Zip not found: {zip_path}")

    dest_dir = Path(__file__).resolve().parent / "data"
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)

    extracted = dest_dir / "dataset.csv"
    if extracted.exists():
        print(f"Extracted to {extracted}")
    else:
        print(f"Extracted files to {dest_dir}")


if __name__ == "__main__":
    main()
