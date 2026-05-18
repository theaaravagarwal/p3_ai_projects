from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .board_encoding import EXTRA_FEATURES_DIM, encode_fen, mirror_encoded_position, validate_fen


FEN_COLUMN_CANDIDATES = ("FEN", "fen", "position")
EVAL_COLUMN_CANDIDATES = ("Evaluation", "evaluation", "eval", "cp")
MATE_VALUE_CP = 10000.0
CACHE_VERSION = 3
MEMMAP_ENCODE_THRESHOLD = 2_000_000
BITBOARD_MASKS = np.left_shift(np.uint64(1), np.arange(64, dtype=np.uint64))


@dataclass(frozen=True)
class SplitData:
    fens: list[str]
    targets: np.ndarray
    raw_cp: np.ndarray


@dataclass(frozen=True)
class ParsedEvaluation:
    cp: float
    is_mate: bool


def discover_csv_files(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    if not root.exists():
        return []
    return sorted(root.rglob("*.csv"))


def _json_cache_key(payload: dict[str, object]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def _csv_fingerprint(csv_files: list[Path]) -> list[dict[str, object]]:
    fingerprint = []
    for path in csv_files:
        stat = path.stat()
        fingerprint.append(
            {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return fingerprint


def _fen_digest(fens: list[str]) -> str:
    digest = hashlib.sha256()
    for fen in fens:
        digest.update(fen.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _validate_fen_batch(fens: list[str]) -> list[bool]:
    return [validate_fen(fen) for fen in fens]


def transform_cp_to_target(cp: np.ndarray | pd.Series, transform: str, eval_clip: float, target_scale: float) -> np.ndarray:
    clipped = np.asarray(cp, dtype=np.float32).clip(-eval_clip, eval_clip)
    if transform == "linear":
        return clipped / float(eval_clip)
    if transform == "tanh":
        return np.tanh(clipped / float(target_scale)).astype(np.float32)
    raise ValueError(f"Unsupported target transform: {transform}")


def inverse_target_to_cp(target: np.ndarray, transform: str, eval_clip: float, target_scale: float) -> np.ndarray:
    values = np.asarray(target, dtype=np.float32)
    if transform == "linear":
        return (values * float(eval_clip)).clip(-eval_clip, eval_clip)
    if transform == "tanh":
        clipped = np.clip(values, -0.999, 0.999)
        return (np.arctanh(clipped) * float(target_scale)).clip(-eval_clip, eval_clip)
    raise ValueError(f"Unsupported target transform: {transform}")


def _encode_fen_batch(fens: list[str]) -> tuple[np.ndarray, np.ndarray]:
    boards = np.empty((len(fens), 12, 8, 8), dtype=np.uint8)
    extras = np.empty((len(fens), EXTRA_FEATURES_DIM), dtype=np.float32)
    for i, fen in enumerate(fens):
        board, extra = encode_fen(fen, include_extras=True)
        boards[i] = board.astype(np.uint8, copy=False)
        extras[i] = extra
    return boards, extras


def pack_board_planes(board: np.ndarray) -> np.ndarray:
    planes = np.asarray(board, dtype=np.uint8).reshape(12, 64)
    return np.sum(planes.astype(np.uint64) * BITBOARD_MASKS, axis=1, dtype=np.uint64)


def pack_board_batch(boards: np.ndarray) -> np.ndarray:
    planes = np.asarray(boards, dtype=np.uint8).reshape(len(boards), 12, 64)
    return np.sum(planes.astype(np.uint64) * BITBOARD_MASKS, axis=2, dtype=np.uint64)


def unpack_board_planes(packed: np.ndarray) -> np.ndarray:
    values = np.asarray(packed, dtype=np.uint64).reshape(12, 1)
    return ((values & BITBOARD_MASKS) != 0).astype(np.uint8).reshape(12, 8, 8)


def detect_column(columns: Iterable[str], candidates: Iterable[str], override: Optional[str] = None) -> str:
    cols = list(columns)
    if override:
        if override not in cols:
            raise ValueError(f"Column override '{override}' was not found. Available columns: {cols}")
        return override
    lower_to_original = {c.lower(): c for c in cols}
    for candidate in candidates:
        if candidate in cols:
            return candidate
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    raise ValueError(f"Could not auto-detect column. Tried {list(candidates)}. Available columns: {cols}")


def parse_evaluation(value: object) -> float:
    return parse_evaluation_detail(value).cp


def parse_evaluation_detail(value: object) -> ParsedEvaluation:
    if value is None:
        return ParsedEvaluation(math.nan, False)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return ParsedEvaluation(float(value), False)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ParsedEvaluation(math.nan, False)
    text = text.replace(" ", "")
    mate_match = re.fullmatch(r"([+-])?#([+-]?\d+)", text) or re.fullmatch(r"#([+-]?\d+)", text)
    if mate_match:
        if len(mate_match.groups()) == 2:
            explicit_sign, moves = mate_match.groups()
            moves_i = int(moves)
            sign = -1.0 if explicit_sign == "-" or moves_i < 0 else 1.0
        else:
            moves_i = int(mate_match.group(1))
            sign = -1.0 if moves_i < 0 else 1.0
        return ParsedEvaluation(sign * MATE_VALUE_CP, True)
    try:
        return ParsedEvaluation(float(text.replace("+", "")), False)
    except ValueError:
        return ParsedEvaluation(math.nan, False)


def load_raw_dataframe(data_dir: str | Path, fen_column: str | None = None, eval_column: str | None = None) -> pd.DataFrame:
    csv_files = discover_csv_files(data_dir)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found under {data_dir}. Run scripts/download_dataset.py or place Kaggle CSV files there."
        )

    frames = []
    detected_fen = None
    detected_eval = None
    for csv_path in csv_files:
        sample = pd.read_csv(csv_path, nrows=5)
        local_fen = detect_column(sample.columns, FEN_COLUMN_CANDIDATES, fen_column)
        local_eval = detect_column(sample.columns, EVAL_COLUMN_CANDIDATES, eval_column)
        detected_fen = detected_fen or local_fen
        detected_eval = detected_eval or local_eval
        frame = pd.read_csv(csv_path, usecols=[local_fen, local_eval])
        frame = frame.rename(columns={local_fen: "fen", local_eval: "evaluation"})
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df.attrs["csv_files"] = [str(p) for p in csv_files]
    df.attrs["fen_column"] = detected_fen
    df.attrs["eval_column"] = detected_eval
    return df


def prepare_dataframe(
    data_dir: str | Path,
    fen_column: str | None = None,
    eval_column: str | None = None,
    eval_clip: float = 1000.0,
    target_transform: str = "linear",
    target_scale: float = 600.0,
    max_samples: int | None = 500_000,
    seed: int = 42,
    preprocess_workers: int | None = None,
    cache_dir: str | Path = "data/cache",
    use_cache: bool = True,
    drop_mate_labels: bool = False,
) -> pd.DataFrame:
    csv_files = discover_csv_files(data_dir)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found under {data_dir}. Run scripts/download_dataset.py or place Kaggle CSV files there."
        )
    workers = preprocess_workers if preprocess_workers is not None else (os.cpu_count() or 1)
    workers = max(1, workers)
    cache_payload = {
        "version": CACHE_VERSION,
        "csv_files": _csv_fingerprint(csv_files),
        "fen_column": fen_column,
        "eval_column": eval_column,
        "eval_clip": eval_clip,
        "target_transform": target_transform,
        "target_scale": target_scale,
        "max_samples": max_samples,
        "seed": seed,
        "drop_mate_labels": drop_mate_labels,
    }
    cache_key = _json_cache_key(cache_payload)
    cache_path = Path(cache_dir) / f"prepared_{cache_key}.pkl"
    if use_cache and cache_path.exists():
        print(f"Loading cached validated rows from {cache_path}")
        df = pd.read_pickle(cache_path)
        df.attrs["loaded_from_cache"] = True
        df.attrs["cache_key"] = cache_key
        df.attrs["preprocess_workers"] = workers
        df.attrs["eval_clip"] = eval_clip
        df.attrs["target_transform"] = target_transform
        df.attrs["target_scale"] = target_scale
        df.attrs["max_samples"] = max_samples
        df.attrs["drop_mate_labels"] = drop_mate_labels
        df.attrs["mate_label_rows"] = int(df["is_mate_label"].sum()) if "is_mate_label" in df else None
        return df

    df = load_raw_dataframe(data_dir, fen_column, eval_column)
    before = len(df)
    df = df.dropna(subset=["fen", "evaluation"]).copy()
    df["fen"] = df["fen"].astype(str).str.strip()
    df = df[df["fen"] != ""]
    parsed = df["evaluation"].map(parse_evaluation_detail)
    df["raw_cp"] = parsed.map(lambda item: item.cp)
    df["is_mate_label"] = parsed.map(lambda item: item.is_mate)
    df = df.dropna(subset=["raw_cp"])
    mate_label_rows = int(df["is_mate_label"].sum())
    if drop_mate_labels:
        df = df[~df["is_mate_label"]].copy()
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if max_samples and max_samples > 0:
        candidate_count = min(len(df), max(max_samples + 10_000, int(max_samples * 1.25)))
        df = df.head(candidate_count).copy()

    fens = df["fen"].tolist()
    if workers == 1 or len(fens) < 10_000:
        valid_mask = [validate_fen(fen) for fen in tqdm(fens, desc="Validating FEN")]
    else:
        chunk_size = max(1_000, len(fens) // (workers * 8))
        chunks = [fens[i : i + chunk_size] for i in range(0, len(fens), chunk_size)]
        valid_mask = []
        print(f"Validating {len(fens):,} FENs with {workers} CPU workers...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for result in tqdm(executor.map(_validate_fen_batch, chunks), total=len(chunks), desc="Validating FEN"):
                valid_mask.extend(result)
    df = df[valid_mask]
    df["target_cp"] = df["raw_cp"].clip(-eval_clip, eval_clip)
    df["target"] = transform_cp_to_target(df["target_cp"], target_transform, eval_clip, target_scale)
    df = df.reset_index(drop=True)
    if max_samples and max_samples > 0:
        df = df.head(max_samples).copy()
    df.attrs.update(
        {
            "rows_before_cleaning": before,
            "rows_after_cleaning": len(df),
            "eval_clip": eval_clip,
            "target_transform": target_transform,
            "target_scale": target_scale,
            "max_samples": max_samples,
            "preprocess_workers": workers,
            "drop_mate_labels": drop_mate_labels,
            "mate_label_rows": mate_label_rows,
            "loaded_from_cache": False,
            "cache_key": cache_key,
        }
    )
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)
        print(f"Cached validated rows to {cache_path}")
    return df


def split_dataframe(df: pd.DataFrame, train_frac: float = 0.8, val_frac: float = 0.1) -> dict[str, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return {
        "train": df.iloc[:train_end].reset_index(drop=True),
        "val": df.iloc[train_end:val_end].reset_index(drop=True),
        "test": df.iloc[val_end:].reset_index(drop=True),
    }


def save_data_summary(df: pd.DataFrame, splits: dict[str, pd.DataFrame], output_dir: str | Path = "runs") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary = {
        "csv_files": df.attrs.get("csv_files", []),
        "fen_column": df.attrs.get("fen_column"),
        "eval_column": df.attrs.get("eval_column"),
        "rows_before_cleaning": df.attrs.get("rows_before_cleaning"),
        "rows_after_cleaning": df.attrs.get("rows_after_cleaning", len(df)),
        "eval_clip": df.attrs.get("eval_clip"),
        "target_transform": df.attrs.get("target_transform"),
        "target_scale": df.attrs.get("target_scale"),
        "max_samples": df.attrs.get("max_samples"),
        "preprocess_workers": df.attrs.get("preprocess_workers"),
        "drop_mate_labels": df.attrs.get("drop_mate_labels"),
        "mate_label_rows": df.attrs.get("mate_label_rows"),
        "loaded_from_cache": df.attrs.get("loaded_from_cache", False),
        "cache_key": df.attrs.get("cache_key"),
        "splits": {name: len(split) for name, split in splits.items()},
    }
    (Path(output_dir) / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def load_splits(
    data_dir: str | Path = "data/chess_evaluations",
    fen_column: str | None = None,
    eval_column: str | None = None,
    eval_clip: float = 1000.0,
    target_transform: str = "linear",
    target_scale: float = 600.0,
    max_samples: int | None = 500_000,
    seed: int = 42,
    output_dir: str | Path = "runs",
    preprocess_workers: int | None = None,
    cache_dir: str | Path = "data/cache",
    use_cache: bool = True,
    drop_mate_labels: bool = False,
) -> dict[str, pd.DataFrame]:
    df = prepare_dataframe(
        data_dir,
        fen_column,
        eval_column,
        eval_clip,
        target_transform,
        target_scale,
        max_samples,
        seed,
        preprocess_workers,
        cache_dir,
        use_cache,
        drop_mate_labels,
    )
    splits = split_dataframe(df)
    for name, split in splits.items():
        split.attrs.update(df.attrs)
        split.attrs["split_name"] = name
    save_data_summary(df, splits, output_dir)
    return splits


class ChessEvaluationDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        preencode: bool = True,
        encode_workers: int | None = None,
        cache_dir: str | Path = "data/cache",
        use_cache: bool = True,
        mirror_augment: bool = False,
        extra_features_dim: int = EXTRA_FEATURES_DIM,
    ):
        self.length = len(frame)
        self.fens: list[str] | None = frame["fen"].tolist()
        self.targets = frame["target"].astype(np.float32).to_numpy()
        self.raw_cp = frame["target_cp"].astype(np.float32).to_numpy()
        self.mirror_augment = mirror_augment
        self.extra_features_dim = extra_features_dim
        self.boards: np.ndarray | None = None
        self.extras: np.ndarray | None = None
        self.boards_cache_path: str | None = None
        self.extras_cache_path: str | None = None
        if preencode:
            split_name = str(frame.attrs.get("split_name", "dataset"))
            parent_key = str(frame.attrs.get("cache_key", "uncached"))
            self.boards, self.extras = load_or_encode_fens(
                self.fens or [],
                workers=encode_workers,
                cache_dir=cache_dir,
                cache_key=f"{parent_key}_{split_name}_{_fen_digest(self.fens)}",
                use_cache=use_cache,
            )
            if isinstance(self.boards, np.memmap):
                self.boards_cache_path = str(Path(self.boards.filename))
            if isinstance(self.extras, np.memmap):
                self.extras_cache_path = str(Path(self.extras.filename))

    def __len__(self) -> int:
        return self.length

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        if self.boards_cache_path and self.extras_cache_path:
            state["boards"] = None
            state["extras"] = None
            state["fens"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        if self.boards is None and self.boards_cache_path:
            self.boards = np.load(self.boards_cache_path, mmap_mode="r")
        if self.extras is None and self.extras_cache_path:
            self.extras = np.load(self.extras_cache_path, mmap_mode="r")

    def __getitem__(self, idx: int):
        if self.boards is None or self.extras is None:
            if self.fens is None:
                raise RuntimeError("FEN rows are unavailable and encoded cache was not loaded.")
            board, extras = encode_fen(self.fens[idx], include_extras=True)
        else:
            board = self.boards[idx]
            extras = self.extras[idx]
        if np.asarray(board).shape == (12,) and np.asarray(board).dtype == np.uint64:
            board = unpack_board_planes(board)
        extras = adjust_extra_features(extras, self.extra_features_dim)
        if not board.flags.writeable:
            board = np.array(board, copy=True)
        if not extras.flags.writeable:
            extras = np.array(extras, copy=True)
        target = float(self.targets[idx])
        if self.mirror_augment and torch.rand(()) < 0.5:
            board, extras, mirrored_target = mirror_encoded_position(board, extras, target)
            target = float(mirrored_target)
        return (
            torch.from_numpy(board),
            torch.from_numpy(extras),
            torch.tensor(target, dtype=torch.float32),
        )


def encode_fens_parallel(fens: list[str], workers: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    workers = workers if workers is not None else (os.cpu_count() or 1)
    workers = max(1, workers)
    if not fens:
        return np.empty((0, 12, 8, 8), dtype=np.uint8), np.empty((0, EXTRA_FEATURES_DIM), dtype=np.float32)
    if workers == 1 or len(fens) < 10_000:
        boards = np.empty((len(fens), 12, 8, 8), dtype=np.uint8)
        extras = np.empty((len(fens), EXTRA_FEATURES_DIM), dtype=np.float32)
        for i, fen in enumerate(tqdm(fens, desc="Encoding FEN")):
            board, extra = encode_fen(fen, include_extras=True)
            boards[i] = board.astype(np.uint8, copy=False)
            extras[i] = extra
        return boards, extras

    chunk_size = max(1_000, len(fens) // (workers * 8))
    chunks = [fens[i : i + chunk_size] for i in range(0, len(fens), chunk_size)]
    board_parts = []
    extra_parts = []
    print(f"Pre-encoding {len(fens):,} FENs with {workers} CPU workers...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for boards, extras in tqdm(executor.map(_encode_fen_batch, chunks), total=len(chunks), desc="Encoding FEN"):
            board_parts.append(boards)
            extra_parts.append(extras)
    return np.concatenate(board_parts, axis=0), np.concatenate(extra_parts, axis=0)


def encode_fens_to_memmap(
    fens: list[str],
    boards_path: Path,
    extras_path: Path,
    workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    workers = workers if workers is not None else (os.cpu_count() or 1)
    workers = max(1, workers)
    boards = np.lib.format.open_memmap(boards_path, mode="w+", dtype=np.uint64, shape=(len(fens), 12))
    extras = np.lib.format.open_memmap(extras_path, mode="w+", dtype=np.float32, shape=(len(fens), EXTRA_FEATURES_DIM))

    if not fens:
        return boards, extras

    if workers == 1 or len(fens) < 10_000:
        for i, fen in enumerate(tqdm(fens, desc="Encoding FEN")):
            board, extra = encode_fen(fen, include_extras=True)
            boards[i] = pack_board_planes(board)
            extras[i] = extra
    else:
        chunk_size = max(1_000, len(fens) // (workers * 8))
        chunks = [fens[i : i + chunk_size] for i in range(0, len(fens), chunk_size)]
        print(f"Pre-encoding {len(fens):,} FENs to packed memmap cache with {workers} CPU workers...")
        offset = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for board_chunk, extra_chunk in tqdm(
                executor.map(_encode_fen_batch, chunks),
                total=len(chunks),
                desc="Encoding FEN",
            ):
                end = offset + len(board_chunk)
                boards[offset:end] = pack_board_batch(board_chunk)
                extras[offset:end] = extra_chunk
                offset = end

    boards.flush()
    extras.flush()
    return np.load(boards_path, mmap_mode="r"), np.load(extras_path, mmap_mode="r")


def adjust_extra_features(extras: np.ndarray, dim: int) -> np.ndarray:
    if dim <= 0:
        return np.zeros((0,), dtype=np.float32)
    values = np.asarray(extras, dtype=np.float32)
    if len(values) == dim:
        return values
    if len(values) > dim:
        return values[:dim]
    padded = np.zeros((dim,), dtype=np.float32)
    padded[: len(values)] = values
    return padded


def load_or_encode_fens(
    fens: list[str],
    workers: int | None = None,
    cache_dir: str | Path = "data/cache",
    cache_key: str | None = None,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if not use_cache:
        return encode_fens_parallel(fens, workers)

    key = cache_key or _fen_digest(fens)
    root = Path(cache_dir)
    compressed_path = root / f"encoded_{key}.npz"
    boards_path = root / f"encoded_{key}_boards.npy"
    extras_path = root / f"encoded_{key}_extras.npy"
    meta_path = root / f"encoded_{key}.json"
    if compressed_path.exists():
        if len(fens) >= MEMMAP_ENCODE_THRESHOLD:
            print(f"Replacing legacy compressed cache with packed memmap cache: {compressed_path}")
            compressed_path.unlink()
        else:
            print(f"Loading compressed encoded FENs from {compressed_path}")
            with np.load(compressed_path) as data:
                return data["boards"], data["extras"]
    if boards_path.exists() and extras_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        if len(fens) >= MEMMAP_ENCODE_THRESHOLD and meta.get("format") != "packed_bitboard_memmap":
            print(f"Replacing legacy encoded cache with packed memmap cache: {boards_path}")
            boards_path.unlink(missing_ok=True)
            extras_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
        else:
            print(f"Loading cached encoded FENs from {root} ({key})")
            return np.load(boards_path, mmap_mode="r"), np.load(extras_path, mmap_mode="r")
    if compressed_path.exists():
        print(f"Loading compressed encoded FENs from {compressed_path}")
        with np.load(compressed_path) as data:
            return data["boards"], data["extras"]

    root.mkdir(parents=True, exist_ok=True)
    if len(fens) >= MEMMAP_ENCODE_THRESHOLD:
        boards, extras = encode_fens_to_memmap(fens, boards_path, extras_path, workers)
        meta = {
            "version": CACHE_VERSION,
            "format": "packed_bitboard_memmap",
            "rows": len(fens),
            "boards_shape": list(boards.shape),
            "extras_shape": list(extras.shape),
            "fen_digest": _fen_digest(fens),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Cached packed memory-mapped encoded FENs to {boards_path} and {extras_path}")
        return boards, extras

    boards, extras = encode_fens_parallel(fens, workers)
    np.savez_compressed(compressed_path, boards=boards, extras=extras)
    meta = {
        "version": CACHE_VERSION,
        "format": "npz",
        "rows": len(fens),
        "boards_shape": list(boards.shape),
        "extras_shape": list(extras.shape),
        "fen_digest": _fen_digest(fens),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Cached compressed encoded FENs to {compressed_path}")
    return boards, extras


def add_dataset_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data-dir", default="data/chess_evaluations")
    parser.add_argument("--fen-column", default=None)
    parser.add_argument("--eval-column", default=None)
    parser.add_argument("--eval-clip", type=float, default=1500.0)
    parser.add_argument("--target-transform", choices=["linear", "tanh"], default="tanh")
    parser.add_argument("--target-scale", type=float, default=600.0)
    parser.add_argument("--max-samples", type=int, default=500_000, help="Maximum rows to use; pass 0 to use all valid rows.")
    parser.add_argument(
        "--drop-mate-labels",
        action="store_true",
        help="Drop forced-mate labels before training; useful when mate signs are noisy or not white-relative.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--no-cache", action="store_true", help="Disable cached validation and FEN encoding.")
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=os.cpu_count() or 1,
        help="CPU processes for FEN validation. Use 1 for serial validation.",
    )
    return parser
