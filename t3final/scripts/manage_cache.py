#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BITBOARD_MASKS = np.left_shift(np.uint64(1), np.arange(64, dtype=np.uint64))


def human_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def cache_files(cache_dir: Path) -> list[Path]:
    if not cache_dir.exists():
        return []
    patterns = ["*.npy", "*.npz", "*.pkl", "*.json"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(cache_dir.glob(pattern))
    return sorted(files)


def summarize(cache_dir: Path) -> None:
    files = cache_files(cache_dir)
    total = sum(path.stat().st_size for path in files)
    by_suffix: dict[str, int] = {}
    for path in files:
        by_suffix[path.suffix] = by_suffix.get(path.suffix, 0) + path.stat().st_size
    print(f"Cache directory: {cache_dir}")
    print(f"Files: {len(files)}")
    print(f"Total: {human_size(total)}")
    for suffix, size in sorted(by_suffix.items()):
        print(f"  {suffix or '<none>'}: {human_size(size)}")


def compress_legacy(cache_dir: Path, delete_legacy: bool) -> None:
    board_files = sorted(cache_dir.glob("encoded_*_boards.npy"))
    converted = 0
    saved_bytes = 0
    for boards_path in board_files:
        key = boards_path.name.removeprefix("encoded_").removesuffix("_boards.npy")
        extras_path = cache_dir / f"encoded_{key}_extras.npy"
        meta_path = cache_dir / f"encoded_{key}.json"
        npz_path = cache_dir / f"encoded_{key}.npz"
        if not extras_path.exists():
            continue
        old_size = boards_path.stat().st_size + extras_path.stat().st_size
        if npz_path.exists():
            new_size = npz_path.stat().st_size
        else:
            print(f"Compressing {key}...")
            boards = np.load(boards_path)
            extras = np.load(extras_path)
            np.savez_compressed(npz_path, boards=boards, extras=extras)
            new_size = npz_path.stat().st_size
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    meta = {}
            else:
                meta = {}
            meta.update({"format": "npz", "compressed_path": npz_path.name})
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        converted += 1
        saved_bytes += max(0, old_size - new_size)
        if delete_legacy:
            boards_path.unlink(missing_ok=True)
            extras_path.unlink(missing_ok=True)
    print(f"Compressed/checked {converted} encoded cache groups.")
    print(f"Potential space saved: {human_size(saved_bytes)}")
    if delete_legacy:
        print("Deleted legacy .npy board/extras files after compression.")


def pack_board_batch(boards: np.ndarray) -> np.ndarray:
    planes = np.asarray(boards, dtype=np.uint8).reshape(len(boards), 12, 64)
    return np.sum(planes.astype(np.uint64) * BITBOARD_MASKS, axis=2, dtype=np.uint64)


def pack_legacy(cache_dir: Path, delete_legacy: bool, dry_run: bool, chunk_size: int) -> None:
    board_files = sorted(cache_dir.glob("encoded_*_boards.npy"))
    converted = 0
    saved_bytes = 0
    for boards_path in board_files:
        key = boards_path.name.removeprefix("encoded_").removesuffix("_boards.npy")
        extras_path = cache_dir / f"encoded_{key}_extras.npy"
        meta_path = cache_dir / f"encoded_{key}.json"
        packed_path = cache_dir / f"encoded_{key}_boards_packed.tmp.npy"
        if not extras_path.exists() or not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        if meta.get("format") == "packed_bitboard_memmap":
            continue

        boards = np.load(boards_path, mmap_mode="r")
        if boards.ndim != 4 or boards.shape[1:] != (12, 8, 8):
            continue

        old_size = boards_path.stat().st_size
        new_size = boards.shape[0] * 12 * np.dtype(np.uint64).itemsize
        converted += 1
        saved_bytes += max(0, old_size - new_size)
        print(f"{'Would pack' if dry_run else 'Packing'} {boards_path.name}: {human_size(old_size)} -> about {human_size(new_size)}")
        if dry_run:
            continue

        packed = np.lib.format.open_memmap(packed_path, mode="w+", dtype=np.uint64, shape=(boards.shape[0], 12))
        for start in range(0, boards.shape[0], chunk_size):
            end = min(start + chunk_size, boards.shape[0])
            packed[start:end] = pack_board_batch(boards[start:end])
        packed.flush()
        del packed
        del boards

        if delete_legacy:
            boards_path.unlink()
            packed_path.replace(boards_path)
        else:
            packed_path.replace(cache_dir / f"encoded_{key}_boards_packed.npy")

        meta.update(
            {
                "format": "packed_bitboard_memmap" if delete_legacy else "packed_bitboard_memmap_sidecar",
                "boards_shape": [int(np.load(boards_path if delete_legacy else cache_dir / f"encoded_{key}_boards_packed.npy", mmap_mode="r").shape[0]), 12],
                "packed_from": "12x8x8 uint8 board planes",
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Packed/checked {converted} legacy encoded board caches.")
    print(f"Potential space saved: {human_size(saved_bytes)}")
    if delete_legacy and not dry_run:
        print("Replaced legacy full-plane board caches with packed bitboard caches.")


def prune(cache_dir: Path, keep_latest: int, dry_run: bool) -> None:
    files = cache_files(cache_dir)
    groups: dict[str, list[Path]] = {}
    for path in files:
        if path.name.startswith("prepared_"):
            key = path.stem
        elif path.name.startswith("encoded_"):
            key = path.stem.replace("_boards", "").replace("_extras", "")
        else:
            key = path.stem
        groups.setdefault(key, []).append(path)
    ordered = sorted(groups.items(), key=lambda item: max(p.stat().st_mtime for p in item[1]), reverse=True)
    to_delete = [p for _, paths in ordered[keep_latest:] for p in paths]
    size = sum(p.stat().st_size for p in to_delete if p.exists())
    action = "Would delete" if dry_run else "Deleting"
    print(f"{action} {len(to_delete)} files, {human_size(size)}")
    for path in to_delete:
        print(f"- {path}")
        if not dry_run:
            path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize, compress, and prune dataset cache files.")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--summary", action="store_true", help="Print cache size summary.")
    parser.add_argument("--compress", action="store_true", help="Compress legacy encoded .npy caches to .npz.")
    parser.add_argument("--delete-legacy", action="store_true", help="Delete .npy files after compression.")
    parser.add_argument("--pack-legacy", action="store_true", help="Convert legacy full-plane board .npy caches to packed bitboard .npy caches.")
    parser.add_argument("--pack-chunk-size", type=int, default=100_000, help="Rows per chunk while packing legacy board caches.")
    parser.add_argument("--prune-keep-latest", type=int, default=None, help="Keep N newest cache groups and remove the rest.")
    parser.add_argument("--dry-run", action="store_true", help="Show prune targets without deleting.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if args.summary or not (args.compress or args.pack_legacy or args.prune_keep_latest is not None):
        summarize(cache_dir)
    if args.compress:
        compress_legacy(cache_dir, args.delete_legacy)
    if args.pack_legacy:
        pack_legacy(cache_dir, args.delete_legacy, args.dry_run, args.pack_chunk_size)
    if args.prune_keep_latest is not None:
        prune(cache_dir, args.prune_keep_latest, args.dry_run)
    if args.compress or args.pack_legacy or args.prune_keep_latest is not None:
        summarize(cache_dir)


if __name__ == "__main__":
    main()
