#!/usr/bin/env python3

from __future__ import annotations

import os
from typing import Iterable

try:
    import kagglehub
except Exception as exc:  # pragma: no cover - import-time environment issue
    raise SystemExit(
        "kagglehub is required for this script. Install it first, then rerun."
    ) from exc


# Edit these lists directly.
LETTER_DATASETS = [
    "grassknoted/asl-alphabet",
    "debashishsau/aslamerican-sign-language-aplhabet-dataset",
    "danrasband/asl-alphabet-test",
    "kapillondhe/american-sign-language",
]

NUMBER_DATASETS = [
    "jeyasrissenthil/hand-signs-asl-hand-sign-data",
    "jakubboczar/asl-alphabet-numbers-dataset",
]

DATASET_HANDLES = LETTER_DATASETS + NUMBER_DATASETS
DEFAULT_CACHE_ROOT = os.path.expanduser("~/.cache/kagglehub")


def ensure_cache_root(cache_root: str) -> str:
    expanded = os.path.abspath(os.path.expanduser(cache_root))
    os.makedirs(expanded, exist_ok=True)
    os.environ.setdefault("KAGGLEHUB_CACHE_DIR", expanded)
    return expanded


def download_handles(handles: Iterable[str]) -> None:
    for handle in handles:
        print(f"Downloading: {handle}")
        dataset_path = kagglehub.dataset_download(handle)
        print(f"Cached at: {dataset_path}")


def main() -> None:
    cache_root = ensure_cache_root(DEFAULT_CACHE_ROOT)
    print(f"Using cache root: {cache_root}")
    print(f"Downloading {len(DATASET_HANDLES)} dataset(s)...")
    download_handles(DATASET_HANDLES)


if __name__ == "__main__":
    main()
