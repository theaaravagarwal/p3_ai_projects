#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import stockfish_result
from src.predict import predict_fen


POSITIONS = {
    "start": chess.STARTING_FEN,
    "e4": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "bare_kings": "8/8/8/4k3/8/8/8/4K3 w - - 0 1",
    "white_up_queen": "rnb1kbnr/pppp1ppp/8/4p3/4Q3/8/PPPP1PPP/RNB1KBNR b KQkq - 0 3",
    "black_up_queen": "rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPP1PPP/RNB1KBNR w KQkq - 0 3",
    "italian": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 3",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CNN predictions to local Stockfish on sanity positions.")
    parser.add_argument("--model", default="models/best_model.pt")
    parser.add_argument("--stockfish-depth", type=int, default=12)
    parser.add_argument("--stockfish-time", type=float, default=0.5)
    parser.add_argument("--max-abs-equal-cp", type=float, default=175.0)
    parser.add_argument("--min-queen-odds-cp", type=float, default=350.0)
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"Model not found: {args.model}")

    rows = []
    failures = []
    for name, fen in POSITIONS.items():
        cnn = predict_fen(fen, model_path=args.model)
        stockfish = stockfish_result(fen, depth=args.stockfish_depth, time_limit=args.stockfish_time)
        cnn_cp = float(cnn["predicted_cp"])
        sf_cp = None if stockfish["cp"] is None else float(stockfish["cp"])
        rows.append({"name": name, "fen": fen, "cnn_cp": cnn_cp, "stockfish_cp": sf_cp, "stockfish_note": stockfish["note"]})

        if name in {"start", "e4", "bare_kings", "italian"} and abs(cnn_cp) > args.max_abs_equal_cp:
            failures.append(f"{name}: CNN should stay near equal, got {cnn_cp:.1f}cp")
        if name == "white_up_queen" and cnn_cp < args.min_queen_odds_cp:
            failures.append(f"{name}: CNN should show a clear white edge, got {cnn_cp:.1f}cp")
        if name == "black_up_queen" and cnn_cp > -args.min_queen_odds_cp:
            failures.append(f"{name}: CNN should show a clear black edge, got {cnn_cp:.1f}cp")

    print(json.dumps(rows, indent=2))
    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("\nPASS CNN sanity positions")


if __name__ == "__main__":
    main()
