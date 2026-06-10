#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

import chess
import chess.engine
from flask import Flask, jsonify, render_template_string, request

from src.board_encoding import pretty_label_from_eval, validate_fen
from src.predict import ChessEvaluationPredictor


MODEL_PATH = Path("models/best_model.pt")
MODEL_INFO = {
    "name": "CNN",
    "mae_cp": 125.22,
    "bucket_accuracy": 89.01,
    "path": str(MODEL_PATH),
}
START_FEN = chess.STARTING_FEN
PROJECT_STOCKFISH_BINARY = Path("models/stockfish/official/stockfish-macos-m1-apple-silicon")
STOCKFISH_CANDIDATES = [
    os.environ.get("STOCKFISH_BINARY", "/usr/games/stockfish"),
    str(PROJECT_STOCKFISH_BINARY),
    "/usr/games/stockfish",
    "/opt/homebrew/bin/stockfish",
    "/usr/local/bin/stockfish",
    "/usr/bin/stockfish",
]
CUSTOM_ENGINE_CANDIDATES = [
    os.environ.get("CUSTOM_CHESS_ENGINE_BINARY", ""),
    "~/Documents/coding/apps/chess/cpp/chess_engine_fast",
]
STOCKFISH_NET_NAMES = ("nn-7bf13f9655c8.nnue", "nn-47fc8b7fff06.nnue")
CUSTOM_STOCKFISH_EVALFILE_CANDIDATES = (
    "models/stockfish/*.nnue",
    "~/Documents/coding/apps/chess/eval_results/*/a_net_3.nnue",
    "~/Documents/coding/apps/chess/eval_results/*/nn-7bf13f9655c8.nnue",
    "~/Documents/coding/apps/chess/models/a_net_3.nnue",
)
STOCKFISH_NET_DIRS = (
    "models/stockfish",
    "/opt/homebrew/share/stockfish",
    "/opt/homebrew/opt/stockfish/share/stockfish",
    "/opt/homebrew/Cellar/stockfish/18/share/stockfish",
    "/usr/local/share/stockfish",
    "/usr/local/opt/stockfish/share/stockfish",
)

PIECE_VALUES_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

EXAMPLE_FENS = {
    "Opera Game": {
        "pgn": "1. e4 e5 2. Nf3 d6 3. d4 Bg4 4. dxe5 Bxf3 5. Qxf3 dxe5 6. Bc4 Nf6 7. Qb3 Qe7 8. Nc3 c6 9. Bg5 b5 10. Nxb5 cxb5 11. Bxb5+ Nbd7 12. O-O-O Rd8 13. Rxd7 Rxd7 14. Rd1 Qe6 15. Bxd7+ Nxd7",
        "continuation": ["Qb8+", "Nxb8", "Rd8#"],
        "description": "Morphy vs Duke Karl/Count Isouard, Paris 1858. White to move: Morphy's pieces are fully active and the final queen sacrifice is on the board.",
    },
    "Game of the Century": {
        "pgn": "1. Nf3 Nf6 2. c4 g6 3. Nc3 Bg7 4. d4 O-O 5. Bf4 d5 6. Qb3 dxc4 7. Qxc4 c6 8. e4 Nbd7 9. Rd1 Nb6 10. Qc5 Bg4 11. Bg5 Na4 12. Qa3 Nxc3 13. bxc3 Nxe4 14. Bxe7 Qb6 15. Bc4 Nxc3 16. Bc5 Rfe8+ 17. Kf1",
        "continuation": ["Be6", "Bxb6", "Bxc4+", "Kg1", "Ne2+", "Kf1", "Nxd4+", "Kg1", "Ne2+", "Kf1", "Nc3+", "Kg1", "axb6", "Qb4", "Ra4", "Qxb6", "Nxd1", "h3", "Rxa2", "Kh2", "Nxf2", "Re1", "Rxe1", "Qd8+", "Bf8", "Nxe1", "Bd5", "Nf3", "Ne4", "Qb8", "b5", "h4", "h5", "Ne5", "Kg7", "Kg1", "Bc5+", "Kf1", "Ng3+", "Ke1", "Bb4+", "Kd1", "Bb3+", "Kc1", "Ne2+", "Kb1", "Nc3+", "Kc1", "Rc2#"],
        "description": "Byrne vs Fischer, New York 1956. Black to move: Fischer is 13 and is about to play the queen sacrifice that made this game famous.",
    },
    "Kasparov's Immortal": {
        "pgn": "1. e4 d6 2. d4 Nf6 3. Nc3 g6 4. Be3 Bg7 5. Qd2 c6 6. f3 b5 7. Nge2 Nbd7 8. Bh6 Bxh6 9. Qxh6 Bb7 10. a3 e5 11. O-O-O Qe7 12. Kb1 a6 13. Nc1 O-O-O 14. Nb3 exd4 15. Rxd4 c5 16. Rd1 Nb6 17. g3 Kb8 18. Na5 Ba8 19. Bh3 d5 20. Qf4+ Ka7 21. Rhe1 d4 22. Nd5 Nbxd5 23. exd5 Qd6",
        "continuation": ["Rxd4", "cxd4", "Re7+", "Kb6", "Qxd4+", "Kxa5", "b4+", "Ka4", "Qc3", "Qxd5", "Ra7", "Bb7", "Rxb7", "Qc4", "Qxf6", "Kxa3", "Qxa6+", "Kxb4", "c3+", "Kxc3", "Qa1+", "Kd2", "Qb2+", "Kd1", "Bf1", "Rd2", "Rd7", "Rxd7", "Bxc4", "bxc4", "Qxh8", "Rd3", "Qa8", "c3", "Qa4+", "Ke1", "f4", "f5", "Kc1", "Rd2", "Qa7"],
        "description": "Kasparov vs Topalov, Wijk aan Zee 1999. White to move: this is the moment before 24.Rxd4 starts the legendary king hunt.",
    },
    "Immortal Game": {
        "pgn": "1. e4 e5 2. f4 exf4 3. Bc4 Qh4+ 4. Kf1 b5 5. Bxb5 Nf6 6. Nf3 Qh6 7. d3 Nh5 8. Nh4 Qg5 9. Nf5 c6 10. g4 Nf6 11. Rg1 cxb5 12. h4 Qg6 13. h5 Qg5 14. Qf3 Ng8 15. Bxf4 Qf6 16. Nc3 Bc5 17. Nd5 Qxb2 18. Bd6 Bxg1 19. e5 Qxa1+ 20. Ke2 Na6 21. Nxg7+ Kd8",
        "continuation": ["Qf6+", "Nxf6", "Be7#"],
        "description": "Anderssen vs Kieseritzky, London 1851. White to move: Anderssen is down huge material but has a forced finish with the queen sacrifice.",
    },
    "Evergreen Game": {
        "pgn": "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. b4 Bxb4 5. c3 Ba5 6. d4 exd4 7. O-O d3 8. Qb3 Qf6 9. e5 Qg6 10. Re1 Nge7 11. Ba3 b5 12. Qxb5 Rb8 13. Qa4 Bb6 14. Nbd2 Bb7 15. Ne4 Qf5 16. Bxd3 Qh5 17. Nf6+ gxf6 18. exf6 Rg8",
        "continuation": ["Rad1", "Qxf3", "Rxe7+", "Nxe7", "Qxd7+", "Kxd7", "Bf5+", "Ke8", "Bd7+", "Kf8", "Bxe7#"],
        "description": "Anderssen vs Dufresne, Berlin 1852. White to move: the pieces are aimed at the king and 19.Rad1 begins the classic Evergreen combination.",
    },
}
COMMON_OPENING_LINES = [
    ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"],
    ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"],
    ["e4", "e6", "d4", "d5", "Nc3", "Nf6", "e5", "Nfd7", "f4", "c5"],
    ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6", "Nf3", "O-O"],
    ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "e3", "O-O"],
    ["d4", "Nf6", "c4", "e6", "Nf3", "d5", "g3", "Be7", "Bg2", "O-O"],
    ["c4", "e5", "Nc3", "Nf6", "g3", "d5", "cxd5", "Nxd5", "Bg2", "Nb6"],
    ["Nf3", "d5", "g3", "Nf6", "Bg2", "g6", "O-O", "Bg7", "d3", "O-O"],
    ["e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Bf5", "Ng3", "Bg6"],
    ["e4", "d5", "exd5", "Qxd5", "Nc3", "Qa5", "d4", "Nf6", "Nf3", "c6"],
    ["d4", "f5", "g3", "Nf6", "Bg2", "e6", "Nf3", "Be7", "O-O", "O-O"],
    ["e4", "g6", "d4", "Bg7", "Nc3", "d6", "Be3", "a6", "Qd2", "b5"],
]
STOCKFISH_SANITY_FEN = "rnb1kbnr/pppp1ppp/8/4p3/4Q3/8/PPPP1PPP/RNB1KBNR b KQkq - 0 3"
STOCKFISH_BLACK_SANITY_FEN = "rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPP1PPP/RNB1KBNR w KQkq - 0 3"
STOCKFISH_KINGS_SANITY_FEN = "8/8/8/4k3/8/8/8/4K3 w - - 0 1"


@lru_cache(maxsize=1)
def get_predictor() -> ChessEvaluationPredictor:
    return ChessEvaluationPredictor(MODEL_PATH)


def find_stockfish() -> str | None:
    for path in STOCKFISH_CANDIDATES:
        if Path(path).exists():
            return str(Path(path).resolve())
    found = shutil.which("stockfish")
    if found:
        return found
    return None


def find_custom_engine() -> str | None:
    for path in CUSTOM_ENGINE_CANDIDATES:
        if not path:
            continue
        candidate = Path(path).expanduser()
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    return None


def is_project_stockfish(stockfish_path: str | None) -> bool:
    if not stockfish_path:
        return False
    try:
        return Path(stockfish_path).resolve() == PROJECT_STOCKFISH_BINARY.resolve()
    except OSError:
        return False


def find_stockfish_nets() -> dict[str, str]:
    found = {}
    for directory in STOCKFISH_NET_DIRS:
        root = Path(directory)
        if not root.exists():
            continue
        for name in STOCKFISH_NET_NAMES:
            candidate = root / name
            if candidate.exists():
                found[name] = str(candidate.resolve())
    return found


def find_custom_stockfish_eval_file() -> str | None:
    for pattern in CUSTOM_STOCKFISH_EVALFILE_CANDIDATES:
        matches = sorted(Path().glob(str(Path(pattern).expanduser())) if not pattern.startswith("~") else Path.home().glob(pattern[2:]))
        for candidate in reversed(matches):
            if candidate.is_file():
                return str(candidate.resolve())
    return None


def stockfish_net_diagnostics() -> list[str]:
    if is_project_stockfish(find_stockfish()):
        return ["Using official Stockfish 18 binary with embedded NNUE networks."]

    messages = []
    discovered = find_custom_stockfish_eval_file()
    if discovered:
        messages.append(f"Using custom NNUE candidate: {discovered}")
    elif not find_stockfish_nets():
        requested = Path("~/Documents/coding/apps/chess/models/a_net_3.nnue").expanduser()
        if requested.exists() and requested.is_dir():
            messages.append(f"Requested NNUE path is a directory, not a file: {requested}")
        elif not requested.exists():
            messages.append(f"Requested NNUE file was not found: {requested}")
        messages.append("No usable .nnue files were found in the configured Stockfish search paths.")
    return messages


def stockfish_engine_options() -> dict[str, Any]:
    options: dict[str, Any] = {"Threads": 1, "Skill Level": 20}
    if is_project_stockfish(find_stockfish()):
        return options

    nets = find_stockfish_nets()
    # This Homebrew Stockfish build rejects the 85MiB main net even when its
    # hash is correct, then continues in a broken relaxed-load mode. The small
    # net is compatible and gives sane search scores, so use it explicitly.
    if STOCKFISH_NET_NAMES[1] in nets:
        options["EvalFileSmall"] = nets[STOCKFISH_NET_NAMES[1]]
    if "EvalFileSmall" in options:
        return options

    custom_eval = find_custom_stockfish_eval_file()
    if custom_eval:
        options["EvalFile"] = custom_eval
    return options


@lru_cache(maxsize=4)
def stockfish_setup_error(stockfish_path: str) -> str | None:
    try:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            try:
                engine.configure(stockfish_engine_options())
            except chess.engine.EngineError:
                pass

            scores: dict[str, float] = {}
            sanity_positions = {
                "starting position": START_FEN,
                "bare kings": STOCKFISH_KINGS_SANITY_FEN,
                "white queen-odds": STOCKFISH_SANITY_FEN,
                "black queen-odds": STOCKFISH_BLACK_SANITY_FEN,
            }
            for name, fen in sanity_positions.items():
                info = engine.analyse(chess.Board(fen), chess.engine.Limit(depth=6, time=0.2), multipv=1)
                if isinstance(info, list):
                    info = info[0]
                if int(info.get("nodes", 0) or 0) <= 0:
                    return (
                        "Local Stockfish appears misconfigured: it returned zero searched nodes. "
                        + " ".join(stockfish_net_diagnostics())
                    )
                cp = info["score"].white().score(mate_score=10000)
                if cp is None:
                    return f"Local Stockfish did not return a centipawn sanity score for {name}."
                scores[name] = float(cp)

        if abs(scores["starting position"]) > 150 or abs(scores["bare kings"]) > 100:
            return (
                "Local Stockfish appears misconfigured: it evaluates an equal sanity position as winning. "
                + " ".join(stockfish_net_diagnostics())
            )
        if scores["white queen-odds"] < 300 or scores["black queen-odds"] > -300:
            return (
                "Local Stockfish appears misconfigured: it fails a queen-odds sanity position. "
                "The NNUE files are probably missing or incompatible. "
                + " ".join(stockfish_net_diagnostics())
            )
        return None
    except Exception as exc:
        return f"Stockfish setup check failed: {exc}"


def material_evaluation_cp(board: chess.Board) -> int:
    score = 0
    for piece in board.piece_map().values():
        value = PIECE_VALUES_CP[piece.piece_type]
        score += value if piece.color == chess.WHITE else -value
    return score


def result_payload(name: str, cp: float | None, label: str, note: str, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "cp": None if cp is None else round(float(cp), 1),
        "label": label,
        "note": note,
        **extra,
    }


def mate_display(mate: int | None) -> str | None:
    if mate is None:
        return None
    return f"M{abs(int(mate))}"


def material_result(board: chess.Board) -> dict[str, Any]:
    cp = float(material_evaluation_cp(board))
    return result_payload("Material", cp, pretty_label_from_eval(cp), "Piece values only; no tactics or position.")


def cnn_result(fen: str) -> dict[str, Any]:
    if not MODEL_PATH.exists():
        return result_payload("CNN", None, "Unavailable", "models/best_model.pt is missing. Train or pull artifacts first.")
    try:
        result = get_predictor().predict(fen)
    except Exception as exc:
        return result_payload("CNN", None, "Unavailable", f"CNN failed: {exc}")
    cp = float(result["predicted_cp"])
    return result_payload("CNN", cp, str(result["label"]), "Neural network static evaluation; no move search.")


def stockfish_result(fen: str, depth: int = 14, time_limit: float = 0.25) -> dict[str, Any]:
    stockfish_path = find_stockfish()
    if not stockfish_path:
        return result_payload("Stockfish", None, "Unavailable", "Stockfish binary was not found locally.")
    setup_error = stockfish_setup_error(stockfish_path)
    if setup_error:
        return result_payload("Stockfish", None, "Unavailable", setup_error)

    board = chess.Board(fen)
    try:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            try:
                engine.configure(stockfish_engine_options())
            except chess.engine.EngineError:
                pass
            info = engine.analyse(board, chess.engine.Limit(depth=depth, time=time_limit), multipv=1)
            if isinstance(info, list):
                info = info[0]

        score = info["score"].white()
        cp = score.score(mate_score=10000)
        mate = score.mate()
        nodes = int(info.get("nodes", 0) or 0)
        reached_depth = int(info.get("depth", 0) or 0)
        if nodes <= 0:
            return result_payload(
                "Stockfish",
                None,
                "Unavailable",
                "Stockfish returned zero searched nodes. Its NNUE files are probably missing; reinstall Stockfish or add the nn-*.nnue files.",
            )
        if reached_depth <= 1 and depth > 2:
            return result_payload(
                "Stockfish",
                None,
                "Unavailable",
                f"Stockfish only reached depth {reached_depth}; local engine setup looks broken.",
            )
        if cp is None:
            return result_payload("Stockfish", None, "Unavailable", "Stockfish did not return a centipawn score.")

        pv_board = board.copy(stack=False)
        pv_moves = []
        best_move = None
        for move in info.get("pv", [])[:5]:
            san = pv_board.san(move)
            if best_move is None:
                best_move = {"san": san, "uci": move.uci()}
            pv_moves.append(san)
            pv_board.push(move)
        note = f"Local Stockfish depth {reached_depth}, nodes {nodes:,}" + (f"; PV: {' '.join(pv_moves)}" if pv_moves else "")
        return result_payload(
            "Stockfish",
            float(cp),
            pretty_label_from_eval(float(cp)),
            note,
            mate_in=None if mate is None else abs(int(mate)),
            display_value=mate_display(mate),
            best_move_san=best_move["san"] if best_move else None,
            best_move_uci=best_move["uci"] if best_move else None,
            pv=pv_moves,
        )
    except Exception as exc:
        return result_payload("Stockfish", None, "Unavailable", f"Stockfish failed: {exc}")


def normalize_custom_engine_cp(cp: int) -> tuple[float, bool]:
    if abs(cp) >= 20000:
        return (10000.0 if cp > 0 else -10000.0), True
    return float(cp), False


def custom_engine_result(fen: str, depth: int = 10, time_limit: float = 0.25) -> dict[str, Any]:
    engine_path = find_custom_engine()
    if not engine_path:
        return result_payload(
            "Custom Engine",
            None,
            "Unavailable",
            "Custom C++ engine was not found at ~/Documents/coding/apps/chess/cpp/chess_engine_fast.",
        )

    board = chess.Board(fen)
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
            info = engine.analyse(board, chess.engine.Limit(depth=depth, time=time_limit), multipv=1)
            if isinstance(info, list):
                info = info[0]

        score = info["score"].white()
        cp = score.score(mate_score=10000)
        mate = score.mate()
        if cp is None:
            return result_payload("Custom Engine", None, "Unavailable", "Custom engine did not return a centipawn score.")
        display_cp, mate_like = normalize_custom_engine_cp(int(cp))

        nodes = int(info.get("nodes", 0) or 0)
        reached_depth = int(info.get("depth", 0) or 0)
        pv_board = board.copy(stack=False)
        pv_moves = []
        best_move = None
        for move in info.get("pv", [])[:5]:
            san = pv_board.san(move)
            if best_move is None:
                best_move = {"san": san, "uci": move.uci()}
            pv_moves.append(san)
            pv_board.push(move)

        note = f"Custom C++ engine depth {reached_depth}, nodes {nodes:,}"
        if mate_like:
            note += f"; raw mate-like score {cp} normalized for display"
        if pv_moves:
            note += f"; PV: {' '.join(pv_moves)}"
        return result_payload(
            "Custom Engine",
            display_cp,
            pretty_label_from_eval(display_cp),
            note,
            raw_cp=float(cp),
            mate_like=mate_like or mate is not None,
            mate_in=None if mate is None else abs(int(mate)),
            display_value=mate_display(mate),
            best_move_san=best_move["san"] if best_move else None,
            best_move_uci=best_move["uci"] if best_move else None,
            pv=pv_moves,
        )
    except Exception as exc:
        return result_payload("Custom Engine", None, "Unavailable", f"Custom engine failed: {exc}")


def evaluate_position(fen: str, stockfish_depth: int = 14, stockfish_time: float = 0.25) -> dict[str, Any]:
    fen = fen.strip()
    if not validate_fen(fen):
        raise ValueError("Invalid FEN.")

    board = chess.Board(fen)
    return {
        "fen": board.fen(),
        "turn": "White" if board.turn == chess.WHITE else "Black",
        "is_check": board.is_check(),
        "is_game_over": board.is_game_over(),
        "model_info": MODEL_INFO,
        "evaluations": [
            material_result(board),
            cnn_result(board.fen()),
            custom_engine_result(board.fen(), min(stockfish_depth, 12), stockfish_time),
            stockfish_result(board.fen(), stockfish_depth, stockfish_time),
        ],
    }


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Chess Evaluation CNN</title>
  <link rel="icon" href="data:," />
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ["Inter", "SF Pro", "system-ui", "sans-serif"]
          }
        }
      }
    };
  </script>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet" />
  <style>
    :root {
      color-scheme: light dark;
      --bg: #e8edf3;
      --surface: #f8fafc;
      --surface-strong: #ffffff;
      --surface-soft: #edf2f7;
      --ink: #111827;
      --muted: #5d6878;
      --line: #b8c4d2;
      --line-strong: #7f8ea1;
      --accent: #1d7f92;
      --accent-dark: #125a68;
      --good: #167247;
      --warn: #a45116;
      --board-light: #eeeed2;
      --board-dark: #769656;
      --board-line: #4b6f3e;
      --piece-white-bg: #f7f4e7;
      --piece-white-ink: #172033;
      --piece-black-bg: #111827;
      --piece-black-ink: #f8fafc;
      --shadow: 0 18px 34px rgba(17, 24, 39, 0.14);
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #10151c;
        --surface: #151b24;
        --surface-strong: #1c2430;
        --surface-soft: #111821;
        --ink: #e7edf5;
        --muted: #9aa7b8;
        --line: #334052;
        --line-strong: #56677d;
        --accent: #29a2b8;
        --accent-dark: #6ecfe0;
        --good: #48b57b;
        --warn: #d18a42;
        --board-light: #eeeed2;
        --board-dark: #769656;
        --board-line: #4b6f3e;
        --piece-white-bg: #f4f0dc;
        --piece-white-ink: #111827;
        --piece-black-bg: #0b111a;
        --piece-black-ink: #f8fafc;
        --shadow: 0 22px 46px rgba(0, 0, 0, 0.35);
      }
    }
    * {
      box-sizing: border-box;
    }
    html,
    body {
      margin: 0;
      min-height: 100%;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, sans-serif;
    }
    .shell {
      width: min(1480px, 100%);
      min-height: 100vh;
      margin: 0 auto;
      padding: clamp(1rem, 2vw, 1.7rem);
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      gap: 1rem;
    }
    .headline {
      margin: 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
    }
    .headline-copy {
      display: grid;
      gap: 0.3rem;
    }
    .headline h1 {
      margin: 0;
      font-weight: 800;
      letter-spacing: 0;
      font-size: clamp(1.75rem, 3vw, 3rem);
      line-height: 1.08;
    }
    .headline p {
      display: none;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 0.25rem;
      padding: 1rem;
      background: color-mix(in srgb, var(--surface) 94%, transparent);
      box-shadow: var(--shadow);
    }
    .section-title {
      margin: 0 0 0.75rem;
      font-size: 0.8rem;
      font-weight: 800;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .workspace {
      display: grid;
      gap: 1rem;
      grid-template-columns: minmax(460px, 0.95fr) minmax(480px, 1.05fr);
      align-items: start;
      min-height: 0;
    }
    @media (max-width: 1100px) {
      .workspace {
        grid-template-columns: 1fr;
      }
    }
    @media (max-width: 640px) {
      .shell {
        padding: 0.9rem;
      }
    }
    .board-panel,
    .compare-panel {
      min-height: 100%;
    }
    .compare-panel {
      display: grid;
      grid-template-rows: auto auto auto;
      align-content: start;
    }
    .board-wrap {
      display: grid;
      place-items: center;
      min-height: 0;
    }
    .chess-board {
      position: relative;
      width: min(72vh, 100%);
      max-width: 680px;
      min-width: min(100%, 320px);
      aspect-ratio: 1 / 1;
    }
    @media (max-width: 640px) {
      .board-wrap {
        min-height: auto;
      }
      .chess-board {
        width: 100%;
        min-width: 0;
      }
    }
    .chess-grid {
      width: 100%;
      height: 100%;
      display: grid;
      grid-template-columns: repeat(8, minmax(0, 1fr));
      grid-template-rows: repeat(8, minmax(0, 1fr));
      overflow: hidden;
      border: 2px solid var(--board-line);
      border-radius: 0.15rem;
      box-shadow: 0 14px 30px rgba(17, 24, 39, 0.24);
    }
    .square {
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
      user-select: none;
      cursor: pointer;
      font-size: clamp(1rem, 3.4vw, 2.25rem);
      line-height: 1;
      letter-spacing: 0;
      font-weight: 800;
    }
    .square.selected {
      outline: 3px solid var(--accent);
      outline-offset: -3px;
    }
    .square.target::after {
      content: "";
      width: 28%;
      aspect-ratio: 1;
      border-radius: 0.15rem;
      background: color-mix(in srgb, var(--accent) 58%, transparent);
      position: absolute;
    }
    .square.target:has(.piece)::after {
      width: 72%;
      background: transparent;
      border: 3px solid color-mix(in srgb, var(--accent) 62%, transparent);
    }
    .light-square {
      background: var(--board-light);
      color: var(--ink);
    }
    .dark-square {
      background: var(--board-dark);
      color: var(--ink);
    }
    .coord {
      position: absolute;
      z-index: 1;
      color: rgba(32, 48, 28, 0.72);
      font-size: clamp(0.48rem, 1.1vw, 0.68rem);
      font-weight: 800;
      line-height: 1;
      pointer-events: none;
    }
    .rank-coord {
      top: 0.22rem;
      left: 0.24rem;
    }
    .file-coord {
      right: 0.24rem;
      bottom: 0.2rem;
      text-transform: uppercase;
    }
    .piece {
      position: relative;
      z-index: 2;
      width: 100%;
      height: 100%;
      display: grid;
      place-items: center;
      border: none;
      border-radius: 0;
      box-shadow: none;
      pointer-events: none;
    }
    .piece img {
      width: 92%;
      height: 92%;
      object-fit: contain;
      display: block;
      filter: drop-shadow(0 2px 1px rgba(0, 0, 0, 0.22));
    }
    .piece.white {
      background: transparent;
    }
    .piece.black {
      background: transparent;
    }
    .board-meta {
      margin-top: 0.8rem;
      display: flex;
      justify-content: space-between;
      gap: 0.75rem;
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-weight: 700;
    }
    .dot-row {
      margin-top: 0.8rem;
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
    }
    .dot-row button {
      appearance: none;
      border: 1px solid var(--line);
      background: var(--surface);
      color: var(--ink);
      min-width: 2.2rem;
      height: 2.1rem;
      border-radius: 0.2rem;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 0.74rem;
      font-weight: 800;
      cursor: pointer;
    }
    .dot-row button.active,
    .dot-row button:hover {
      color: #ffffff;
      background: var(--accent);
      border-color: var(--accent);
    }
    .position-note {
      display: none;
    }
    .position-note h3 {
      margin: 0 0 0.25rem;
      color: var(--ink);
      font-size: 0.86rem;
      font-weight: 800;
    }
    .position-note p {
      margin: 0;
      color: var(--muted);
      font-size: 0.86rem;
      line-height: 1.45;
    }
    .demo-controls {
      margin-top: 0.7rem;
      display: grid;
      grid-template-columns: auto 1fr auto;
      align-items: center;
      gap: 0.6rem;
    }
    .demo-controls button {
      border: 1px solid var(--line);
      border-radius: 0.18rem;
      background: var(--surface);
      color: var(--ink);
      height: 2.2rem;
      padding: 0 0.8rem;
      font-size: 0.76rem;
      font-weight: 800;
      cursor: pointer;
    }
    .demo-controls button:not(:disabled):hover {
      border-color: var(--accent);
      color: var(--accent-dark);
    }
    .demo-controls button:disabled {
      opacity: 0.45;
      cursor: default;
    }
    .demo-next {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 0.2rem;
      background: var(--surface-strong);
      padding: 0.55rem 0.7rem;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.85);
    }
    .demo-next-card {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: center;
      gap: 0.65rem;
      min-height: 2.35rem;
    }
    .demo-next-kicker {
      color: var(--muted);
      font-size: 0.64rem;
      font-weight: 800;
      letter-spacing: 0.08em;
      line-height: 1.1;
      text-transform: uppercase;
    }
    .demo-next-main {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--ink);
      font-size: 0.98rem;
      font-weight: 800;
      line-height: 1.2;
    }
    .demo-next-sub {
      color: var(--muted);
      font-size: 0.7rem;
      font-weight: 700;
      line-height: 1.2;
      white-space: nowrap;
    }
    .demo-next-count {
      min-width: 2.3rem;
      border-radius: 0.15rem;
      background: color-mix(in srgb, var(--accent) 14%, transparent);
      color: var(--accent-dark);
      padding: 0.32rem 0.5rem;
      text-align: center;
      font-size: 0.68rem;
      font-weight: 800;
    }
    .move-arrow-layer {
      position: absolute;
      inset: 0;
      pointer-events: none;
      z-index: 4;
    }
    .move-arrow-layer line {
      stroke: color-mix(in srgb, var(--accent) 76%, transparent);
      stroke-width: 2.1;
      stroke-linecap: round;
      filter: drop-shadow(0 1px 1px rgba(21, 34, 56, 0.18));
      stroke-dasharray: 150;
      stroke-dashoffset: 150;
      animation: arrowDraw 180ms ease-out forwards;
    }
    .move-arrow-layer polygon {
      fill: color-mix(in srgb, var(--accent) 76%, transparent);
      filter: drop-shadow(0 1px 1px rgba(21, 34, 56, 0.18));
      opacity: 0;
      transform-box: fill-box;
      transform-origin: center;
      animation: arrowHeadIn 130ms ease-out 120ms forwards;
    }
    @keyframes arrowDraw {
      to {
        stroke-dashoffset: 0;
      }
    }
    @keyframes arrowHeadIn {
      from {
        opacity: 0;
        transform: scale(0.78);
      }
      to {
        opacity: 1;
        transform: scale(1);
      }
    }
    .comparison-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 0.75rem;
      margin-bottom: 0.75rem;
    }
    @media (max-width: 720px) {
      .comparison-grid {
        grid-template-columns: 1fr;
      }
    }
    .model-card {
      border: 1px solid var(--line);
      border-radius: 0.22rem;
      padding: 0.8rem;
      min-height: 8.25rem;
      background: var(--surface);
    }
    .model-title {
      font-size: 0.72rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 800;
    }
    .model-value {
      margin-top: 0.45rem;
      font-size: clamp(1.75rem, 3vw, 2.4rem);
      line-height: 1;
      font-weight: 800;
      letter-spacing: 0;
      white-space: nowrap;
      color: var(--ink);
    }
    .sub-tag {
      margin-top: 0.2rem;
      color: var(--muted);
      font-size: 0.75rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      font-weight: 700;
    }
    .sparkline {
      margin-top: 0.55rem;
      height: 36px;
      border-top: 1px solid var(--line);
      padding-top: 0.5rem;
    }
    .sparkline svg {
      width: 100%;
      height: 100%;
      overflow: visible;
    }
    .sparkline path {
      fill: none;
      stroke-width: 2;
      stroke-linecap: round;
      stroke-linejoin: round;
    }
    .master-block {
      display: grid;
      gap: 0.65rem;
      align-content: start;
    }
    .master-caption {
      font-size: 0.72rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 800;
    }
    .master-bar {
      position: relative;
      height: 1.2rem;
      border: 1px solid var(--line);
      border-radius: 0.2rem;
      overflow: hidden;
      background: var(--surface-soft);
    }
    .master-bar::before {
      content: "";
      position: absolute;
      top: 0;
      bottom: 0;
      left: 50%;
      width: 2px;
      background: var(--line);
      z-index: 2;
    }
    .master-fill {
      position: absolute;
      top: 0;
      bottom: 0;
      border-radius: 0.12rem;
      transform-origin: center;
      z-index: 1;
    }
    .master-value {
      font-size: clamp(1.8rem, 3vw, 2.35rem);
      line-height: 1;
      font-weight: 800;
      letter-spacing: 0;
    }
    .adv-ring {
      display: none;
    }
    .adv-ring span {
      width: 2.8rem;
      aspect-ratio: 1;
      border-radius: 0.18rem;
      background: currentColor;
      opacity: 0.9;
      box-shadow: 0 0 0 0.8rem rgba(31, 122, 140, 0.12);
    }
    .adv-ring.white {
      color: var(--good);
      background: color-mix(in srgb, var(--good) 12%, var(--surface));
    }
    .adv-ring.black {
      color: var(--ink);
      background: color-mix(in srgb, var(--ink) 10%, var(--surface));
    }
    .adv-ring.equal {
      color: var(--accent-dark);
      background: color-mix(in srgb, var(--accent) 10%, var(--surface));
    }
    .ring-text {
      display: none;
    }
    .best-move {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: center;
      gap: 0.8rem;
      border: 1px solid var(--line);
      border-radius: 0.22rem;
      background: var(--surface-strong);
      padding: 0.7rem 0.8rem;
      min-height: 3.75rem;
    }
    .best-move-label {
      color: var(--muted);
      font-size: 0.72rem;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .best-move-line {
      margin-top: 0.3rem;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--muted);
      font-size: 0.76rem;
      font-weight: 700;
    }
    .best-move-value {
      color: var(--ink);
      font-size: 1.45rem;
      font-weight: 800;
      white-space: nowrap;
    }
    .best-move-value.empty {
      color: var(--muted);
      font-size: 1.1rem;
    }
    .fen-shell {
      margin-top: 1rem;
    }
    .fen-row {
      width: 100%;
      display: flex;
      align-items: center;
      gap: 0.65rem;
    }
    @media (max-width: 640px) {
      .fen-row {
        align-items: stretch;
        flex-direction: column;
      }
    }
    #fenInput {
      width: 100%;
      min-width: 0;
      background: var(--surface);
      color: var(--ink);
      border: 1px solid var(--line);
      border-radius: 0.2rem;
      padding: 0.75rem 0.85rem;
      outline: none;
      font-size: 0.9rem;
      font-family: inherit;
    }
    #fenInput:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(31, 122, 140, 0.14);
    }
    #evalBtn {
      border: 1px solid var(--accent);
      border-radius: 0.2rem;
      background: var(--accent);
      color: #ffffff;
      padding: 0.75rem 1rem;
      font-size: 0.78rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      cursor: pointer;
      white-space: nowrap;
    }
    #evalBtn:hover {
      background: var(--accent-dark);
      border-color: var(--accent-dark);
    }
    #evalBtn:disabled {
      opacity: 0.65;
      cursor: wait;
    }
    #status {
      display: none;
    }
    #status.error {
      color: #b42318;
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="headline">
      <div class="headline-copy">
        <h1>Chess Evaluation CNN</h1>
      </div>
    </header>

    <section class="workspace">
      <section class="panel board-panel">
        <h2 class="section-title">Position</h2>
        <div class="board-wrap">
          <div id="board" class="chess-board" aria-label="Chess board"></div>
        </div>
        <div class="board-meta">
          <span id="turnBadge">Turn · White</span>
          <span id="stateBadge">Live</span>
        </div>
        <div id="exampleDots" class="dot-row"></div>
        <div class="position-note">
          <h3 id="exampleTitle">Make a move</h3>
          <p id="exampleDescription">Click a piece, then click a legal destination. You can also paste a FEN below.</p>
        </div>
        <div class="demo-controls">
          <button id="prevLineBtn" type="button">Back</button>
          <div id="demoMoveLabel" class="demo-next">Pick a game</div>
          <button id="nextLineBtn" type="button">Next</button>
        </div>
        <div id="status">Type or pick a position below</div>
      </section>

      <section class="panel compare-panel">
        <h2 class="section-title">Evaluations</h2>
        <div class="comparison-grid">
          <article class="model-card">
            <div class="model-title">Baseline</div>
            <div id="matValue" class="model-value">—</div>
            <div class="sub-tag">Material</div>
            <div id="matSpark" class="sparkline" aria-hidden="true"></div>
          </article>
          <article class="model-card">
            <div class="model-title">Model</div>
            <div id="cnnValue" class="model-value">—</div>
            <div class="sub-tag">CNN</div>
            <div id="cnnSpark" class="sparkline" aria-hidden="true"></div>
          </article>
          <article class="model-card">
            <div class="model-title">Custom</div>
            <div id="customValue" class="model-value">—</div>
            <div class="sub-tag">C++ Engine</div>
            <div id="customSpark" class="sparkline" aria-hidden="true"></div>
          </article>
          <article class="model-card">
            <div class="model-title">Engine</div>
            <div id="sfValue" class="model-value">—</div>
            <div class="sub-tag">Stockfish</div>
            <div id="sfSpark" class="sparkline" aria-hidden="true"></div>
          </article>
        </div>
        <div class="master-block">
          <div class="master-caption">Master Eval Bar</div>
          <div class="master-bar">
            <div id="masterFill" class="master-fill"></div>
          </div>
          <div id="masterValue" class="master-value">—</div>
          <div class="best-move">
            <div>
              <div class="best-move-label">Stockfish best move</div>
              <div id="bestMoveLine" class="best-move-line">Waiting for engine line</div>
            </div>
            <div id="bestMoveValue" class="best-move-value empty">—</div>
          </div>
          <div id="advRing" class="adv-ring equal"><span></span></div>
          <div id="ringLabel" class="ring-text">Equal</div>
        </div>
      </section>
    </section>

    <div class="fen-shell">
      <div class="fen-row">
        <input id="fenInput" type="text" spellcheck="false" autocomplete="off" aria-label="FEN position" />
        <button id="evalBtn" type="button">Evaluate</button>
      </div>
    </div>
  </div>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
  <script>
    const START_FEN = {{ start_fen|tojson }};
    const EXAMPLES = {{ examples|tojson }};
    const COMMON_OPENING_LINES = {{ common_opening_lines|tojson }};
    const MAX_SPARK = 8;
    const boardEl = document.getElementById("board");
    const fenInput = document.getElementById("fenInput");
    const turnBadge = document.getElementById("turnBadge");
    const stateBadge = document.getElementById("stateBadge");
    const statusEl = document.getElementById("status");
    const dotRow = document.getElementById("exampleDots");
    const exampleTitle = document.getElementById("exampleTitle");
    const exampleDescription = document.getElementById("exampleDescription");
    const prevLineBtn = document.getElementById("prevLineBtn");
    const nextLineBtn = document.getElementById("nextLineBtn");
    const demoMoveLabel = document.getElementById("demoMoveLabel");
    const masterFill = document.getElementById("masterFill");
    const masterValueEl = document.getElementById("masterValue");
    const bestMoveValue = document.getElementById("bestMoveValue");
    const bestMoveLine = document.getElementById("bestMoveLine");
    const advRing = document.getElementById("advRing");
    const ringLabel = document.getElementById("ringLabel");
    const evalBtn = document.getElementById("evalBtn");

    const pieceImages = {
      K: "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wk.png",
      Q: "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wq.png",
      R: "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wr.png",
      B: "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wb.png",
      N: "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wn.png",
      P: "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wp.png",
      k: "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bk.png",
      q: "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bq.png",
      r: "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/br.png",
      b: "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bb.png",
      n: "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bn.png",
      p: "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bp.png",
    };

    let requestSeq = 0;
    let currentFen = START_FEN;
    let game = null;
    let selectedSquare = null;
    let demoStartFen = START_FEN;
    let demoLine = [];
    let demoIndex = 0;
    const histories = {
      mat: [],
      cnn: [],
      custom: [],
      sf: [],
    };

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function toPawns(value) {
      return value / 100;
    }

    function cpToLabel(cp) {
      if (!Number.isFinite(cp)) return "—";
      const v = toPawns(cp);
      const prefix = v > 0 ? "+" : "";
      return `${prefix}${v.toFixed(1)}`;
    }

    function setStatus(message, isError = false) {
      statusEl.textContent = message;
      statusEl.classList.toggle("error", isError);
    }

    function syncGameFromFen(fen) {
      if (!window.Chess) return false;
      try {
        game = null;
        game = new Chess(fen);
        return true;
      } catch (error) {
        game = null;
        return false;
      }
    }

    function squareName(row, file) {
      return `${"abcdefgh"[file]}${8 - row}`;
    }

    function legalTargets(square) {
      if (!game || !square) return [];
      return game.moves({ square, verbose: true }).map((move) => move.to);
    }

    function previewDemoMove() {
      if (!game || demoIndex >= demoLine.length) return null;
      const replay = new Chess(game.fen());
      return replay.move(demoLine[demoIndex], { sloppy: true });
    }

    function clearDemoLine() {
      demoLine = [];
      demoIndex = 0;
      demoStartFen = currentFen;
      updateDemoControls();
    }

    function updateDemoControls() {
      const move = previewDemoMove();
      prevLineBtn.disabled = demoIndex <= 0;
      nextLineBtn.disabled = !move;
      const card = document.createElement("div");
      card.className = "demo-next-card";
      const copy = document.createElement("div");
      const kicker = document.createElement("div");
      kicker.className = "demo-next-kicker";
      const main = document.createElement("div");
      main.className = "demo-next-main";
      const sub = document.createElement("div");
      sub.className = "demo-next-sub";
      const count = document.createElement("div");
      count.className = "demo-next-count";

      if (!demoLine.length) {
        kicker.textContent = "Preview";
        main.textContent = "Pick a game";
        sub.textContent = "No line loaded";
        count.textContent = "0/0";
      } else if (move) {
        const side = move.color === "w" ? "White" : "Black";
        kicker.textContent = "Next move";
        main.textContent = `${side}: ${move.san}`;
        sub.textContent = `${move.from} to ${move.to}`;
        count.textContent = `${demoIndex + 1}/${demoLine.length}`;
      } else {
        kicker.textContent = "Line";
        main.textContent = "Complete";
        sub.textContent = `${demoLine.length} moves reviewed`;
        count.textContent = `${demoLine.length}/${demoLine.length}`;
      }

      copy.append(kicker, main, sub);
      card.append(copy, count);
      demoMoveLabel.replaceChildren(card);
    }

    function squarePoint(square) {
      const file = "abcdefgh".indexOf(square[0]);
      const rank = Number(square[1]);
      return {
        x: (file + 0.5) * 12.5,
        y: (8 - rank + 0.5) * 12.5,
      };
    }

    function moveArrowSvg(move) {
      if (!move) return "";
      const from = squarePoint(move.from);
      const to = squarePoint(move.to);
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      const length = Math.hypot(dx, dy);
      if (!length) return "";
      const ux = dx / length;
      const uy = dy / length;
      const px = -uy;
      const py = ux;
      const headLength = 3.4;
      const headHalfWidth = 1.55;
      const shaftEnd = {
        x: to.x - ux * headLength,
        y: to.y - uy * headLength,
      };
      const left = {
        x: shaftEnd.x + px * headHalfWidth,
        y: shaftEnd.y + py * headHalfWidth,
      };
      const right = {
        x: shaftEnd.x - px * headHalfWidth,
        y: shaftEnd.y - py * headHalfWidth,
      };
      const point = ({ x, y }) => `${x.toFixed(2)},${y.toFixed(2)}`;
      return `
        <svg class="move-arrow-layer" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          <line x1="${from.x.toFixed(2)}" y1="${from.y.toFixed(2)}" x2="${shaftEnd.x.toFixed(2)}" y2="${shaftEnd.y.toFixed(2)}" />
          <polygon points="${point(to)} ${point(left)} ${point(right)}" />
        </svg>
      `;
    }

    function fenFromExample(item) {
      if (item.fen) return item.fen;
      if (!window.Chess || !item.pgn) return START_FEN;
      const replay = new Chess();
      const loaded = replay.load_pgn(`${item.pgn} *`, { sloppy: true });
      return loaded ? replay.fen() : START_FEN;
    }

    function randomPracticalFen() {
      if (!window.Chess) return START_FEN;
      const replay = new Chess();
      const line = COMMON_OPENING_LINES[Math.floor(Math.random() * COMMON_OPENING_LINES.length)] || [];
      const prefixLength = Math.max(4, Math.floor(Math.random() * (line.length + 1)));
      for (const san of line.slice(0, prefixLength)) {
        if (!replay.move(san, { sloppy: true })) break;
      }
      const extraPlies = 6 + Math.floor(Math.random() * 18);
      for (let i = 0; i < extraPlies && !replay.game_over(); i++) {
        const moves = replay.moves({ verbose: true });
        if (!moves.length) break;
        const checks = moves.filter((move) => move.san.includes("+") || move.san.includes("#"));
        const captures = moves.filter((move) => move.captured);
        const quiet = moves.filter((move) => !move.captured && !move.san.includes("+") && !move.san.includes("#"));
        const pool = checks.length && Math.random() < 0.16
          ? checks
          : captures.length && Math.random() < 0.36
            ? captures
            : quiet.length
              ? quiet
              : moves;
        replay.move(pool[Math.floor(Math.random() * pool.length)]);
      }
      return replay.fen();
    }

    function animateValue(element, targetCp) {
      const startCp = Number(element.dataset.cp || 0);
      const endCp = Number.isFinite(targetCp) ? targetCp : startCp;
      const startTs = performance.now();
      const duration = 300;

      if (startCp === endCp) {
        element.textContent = cpToLabel(endCp);
        return;
      }

      function tick(now) {
        const elapsed = Math.min(1, (now - startTs) / duration);
        const eased = 1 - Math.pow(1 - elapsed, 2.7);
        const current = startCp + (endCp - startCp) * eased;
        element.textContent = cpToLabel(current);
        if (elapsed < 1) {
          requestAnimationFrame(tick);
        }
      }
      requestAnimationFrame(tick);
      element.dataset.cp = endCp;
    }

    function boardFromFen(fen) {
      const boardPart = String(fen || "").trim().split(" ")[0];
      const rows = boardPart.split("/");
      const grid = document.createElement("div");
      grid.className = "chess-grid";
      const targets = legalTargets(selectedSquare);

      function makeSquare(rowIndex, fileIndex, pieceCode = null) {
        const square = document.createElement("div");
        const name = squareName(rowIndex, fileIndex);
        square.dataset.square = name;
        square.className = `square ${((rowIndex + fileIndex) % 2 === 0) ? "light-square" : "dark-square"}${selectedSquare === name ? " selected" : ""}${targets.includes(name) ? " target" : ""}`;

        if (fileIndex === 0) {
          const rank = document.createElement("span");
          rank.className = "coord rank-coord";
          rank.textContent = String(8 - rowIndex);
          square.appendChild(rank);
        }
        if (rowIndex === 7) {
          const fileLabel = document.createElement("span");
          fileLabel.className = "coord file-coord";
          fileLabel.textContent = "abcdefgh"[fileIndex];
          square.appendChild(fileLabel);
        }

        const imageSrc = pieceCode ? pieceImages[pieceCode] : null;
        if (imageSrc) {
          const p = document.createElement("span");
          p.className = `piece ${pieceCode === pieceCode.toUpperCase() ? "white" : "black"}`;
          const img = document.createElement("img");
          img.src = imageSrc;
          img.alt = "";
          img.draggable = false;
          p.appendChild(img);
          square.appendChild(p);
        }

        return square;
      }

      for (let r = 0; r < 8; r++) {
        const row = rows[r] || "";
        let file = 0;
        for (const ch of row) {
          if (file >= 8) break;
          if (/\d/.test(ch)) {
            const count = Number(ch);
            for (let i = 0; i < count; i++) {
              grid.appendChild(makeSquare(r, file));
              file++;
            }
            continue;
          }
          grid.appendChild(makeSquare(r, file, ch));
          file++;
        }
        while (file < 8) {
          grid.appendChild(makeSquare(r, file));
          file++;
        }
      }

      while (grid.childElementCount < 64) {
        const rowIndex = Math.floor(grid.childElementCount / 8);
        const column = grid.childElementCount % 8;
        grid.appendChild(makeSquare(rowIndex, column));
      }
      boardEl.replaceChildren(grid);
      boardEl.insertAdjacentHTML("beforeend", moveArrowSvg(previewDemoMove()));
      updateDemoControls();
    }

    function buildSpark(values, color) {
      const width = 260;
      const height = 52;
      const safe = values.length ? values : [0, 0];
      const minRaw = Math.min(...safe);
      const maxRaw = Math.max(...safe);
      const range = Math.max(Math.abs(minRaw), Math.abs(maxRaw), 200);
      const norm = safe.map(v => clamp(v, -range, range));
      const xStep = width / Math.max(1, safe.length - 1);
      const pts = norm.map((value, index) => {
        const x = index * xStep;
        const y = (0.5 - value / (2 * range)) * (height - 8) + 4;
        return `${index === 0 ? "M" : "L"}${x.toFixed(2)} ${y.toFixed(2)}`;
      }).join(" ");
      const id = `line-${color.replace("#", "")}`;
      return `
        <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
          <defs>
            <linearGradient id="${id}" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stop-color="${color}" stop-opacity="0.45" />
              <stop offset="100%" stop-color="${color}" stop-opacity="0.1" />
            </linearGradient>
          </defs>
          <path d="${pts}" stroke="url(#${id})" />
        </svg>
      `;
    }

    function updateMaster(cp, displayValue = null) {
      const clamped = clamp(Number.isFinite(cp) ? cp : 0, -1000, 1000);
      const ratio = (clamped + 1000) / 2000;
      const spread = Math.abs(ratio - 0.5) * 2;
      const widthPercent = clamp(Math.max(4, 50 * spread), 4, 50);

      if (clamped >= 0) {
        masterFill.style.left = "50%";
      } else {
        masterFill.style.left = `${50 - widthPercent}%`;
      }
      masterFill.style.width = `${widthPercent}%`;

      if (clamped > 60) {
        masterFill.style.background = "var(--accent)";
        masterFill.style.boxShadow = "none";
      } else if (clamped < -60) {
        masterFill.style.background = "var(--ink)";
        masterFill.style.boxShadow = "none";
      } else {
        masterFill.style.background = "var(--line-strong)";
        masterFill.style.boxShadow = "none";
      }

      masterValueEl.textContent = displayValue || cpToLabel(clamped);
      animateRing(clamped);
    }

    function formatUciSquares(uci) {
      if (!uci || uci.length < 4) return "";
      return `${uci.slice(0, 2)} to ${uci.slice(2, 4)}`;
    }

    function updateBestMove(stockfish, turn) {
      const move = stockfish?.best_move_san || stockfish?.best_move_uci;
      bestMoveValue.textContent = move || "—";
      bestMoveValue.classList.toggle("empty", !move);
      const squares = formatUciSquares(stockfish?.best_move_uci);
      const pv = stockfish?.pv?.length ? `PV: ${stockfish.pv.join(" ")}` : "";
      bestMoveLine.textContent = move
        ? `${turn || "Side to move"}${squares ? ` · ${squares}` : ""}${pv ? ` · ${pv}` : ""}`
        : "Engine move unavailable";
      bestMoveValue.title = pv || "";
      bestMoveLine.title = pv || stockfish?.note || "";
    }

    function animateRing(cp) {
      advRing.classList.remove("white", "equal", "black");
      if (cp > 60) {
        advRing.classList.add("white");
        ringLabel.textContent = "White Advantage";
      } else if (cp < -60) {
        advRing.classList.add("black");
        ringLabel.textContent = "Black Advantage";
      } else {
        advRing.classList.add("equal");
        ringLabel.textContent = "Equal";
      }
    }

    function updateModel(side, payload) {
      const valueEl = document.getElementById(`${side}Value`);
      const sparkEl = document.getElementById(`${side}Spark`);

      if (!payload || payload.cp === null || payload.cp === undefined) {
        valueEl.textContent = "—";
        valueEl.dataset.cp = 0;
        sparkEl.innerHTML = buildSpark([0, 0], "#98a2b3");
        return;
      }

      const cp = Number(payload.cp);
      if (Number.isFinite(cp)) {
        histories[side].push(cp);
        if (histories[side].length > MAX_SPARK) histories[side].shift();
        if (payload.display_value) {
          valueEl.textContent = payload.display_value;
          valueEl.dataset.cp = cp;
        } else {
          animateValue(valueEl, cp);
        }
        const colors = { mat: "#667085", cnn: "#1f7a8c", custom: "#a35f1f", sf: "#18212f" };
        sparkEl.innerHTML = buildSpark(histories[side], colors[side] || "#1f7a8c");
      }
    }

    function renderExamples() {
      const names = Object.entries(EXAMPLES);
      names.forEach(([label, item], index) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = String(index + 1);
        btn.title = label;
        btn.addEventListener("click", () => {
          dotRow.querySelectorAll("button").forEach((el) => el.classList.remove("active"));
          btn.classList.add("active");
          exampleTitle.textContent = label;
          exampleDescription.textContent = item.description;
          fenInput.value = fenFromExample(item);
          demoStartFen = fenInput.value;
          demoLine = item.continuation || [];
          demoIndex = 0;
          triggerEvaluate(false);
        });
        dotRow.appendChild(btn);
      });
      const randomBtn = document.createElement("button");
      randomBtn.type = "button";
      randomBtn.textContent = String(names.length + 1);
      randomBtn.title = "Random practical position";
      randomBtn.addEventListener("click", () => {
        dotRow.querySelectorAll("button").forEach((el) => el.classList.remove("active"));
        randomBtn.classList.add("active");
        exampleTitle.textContent = "Random practical position";
        exampleDescription.textContent = "";
        fenInput.value = randomPracticalFen();
        demoStartFen = fenInput.value;
        demoLine = [];
        demoIndex = 0;
        triggerEvaluate(false);
      });
      dotRow.appendChild(randomBtn);
    }

    function markTurn(turn, isCheck, isOver) {
      turnBadge.textContent = `Turn · ${turn}`;
      stateBadge.textContent = isCheck ? "Check" : (isOver ? "Game over" : "Live");
    }

    function setLoading(loading) {
      if (loading) {
        stateBadge.textContent = "Evaluating…";
      }
      evalBtn.disabled = loading;
    }

    async function evaluateCurrent() {
      const id = ++requestSeq;
      setLoading(true);
      setStatus("Evaluating...");
      try {
        const response = await fetch("/api/evaluate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ fen: currentFen, stockfish_depth: 14, stockfish_time: 0.25 }),
        });
        const payload = await response.json();
        if (id !== requestSeq) return;
        if (!response.ok) {
          setStatus(payload.error || "Evaluation failed", true);
          return;
        }

        const results = Object.fromEntries((payload.evaluations || []).map((item) => [item.name, item]));
        currentFen = payload.fen || currentFen;
        syncGameFromFen(currentFen);
        boardFromFen(payload.fen || currentFen);
        markTurn(payload.turn || "White", payload.is_check, payload.is_game_over);

        const material = results.Material || null;
        const cnn = results.CNN || null;
        const custom = results["Custom Engine"] || null;
        const stockfish = results.Stockfish || null;
        updateModel("mat", material);
        updateModel("cnn", cnn);
        updateModel("custom", custom);
        updateModel("sf", stockfish);

        const anchor = Number.isFinite(stockfish?.cp) ? stockfish.cp : Number.isFinite(cnn?.cp) ? cnn.cp : 0;
        updateMaster(anchor, stockfish?.display_value || null);
        updateBestMove(stockfish, payload.turn);

        fenInput.value = currentFen;
        setStatus("Updated");
        setTimeout(() => setStatus("Type or pick a position below"), 1200);
      } finally {
        if (id === requestSeq) {
          setLoading(false);
        }
      }
    }

    function triggerEvaluate(clearDemo = true) {
      const nextFen = String(fenInput.value || START_FEN).trim();
      currentFen = nextFen;
      selectedSquare = null;
      syncGameFromFen(nextFen);
      if (clearDemo) clearDemoLine();
      boardFromFen(nextFen);
      evaluateCurrent();
    }

    function replayDemoTo(index) {
      const replay = new Chess(demoStartFen);
      for (let i = 0; i < index; i++) {
        if (!replay.move(demoLine[i], { sloppy: true })) break;
      }
      demoIndex = index;
      game = replay;
      currentFen = game.fen();
      fenInput.value = currentFen;
      selectedSquare = null;
      boardFromFen(currentFen);
      evaluateCurrent();
    }

    function stepDemo(direction) {
      if (!demoLine.length) return;
      if (direction > 0 && demoIndex < demoLine.length) {
        replayDemoTo(demoIndex + 1);
      } else if (direction < 0 && demoIndex > 0) {
        replayDemoTo(demoIndex - 1);
      }
    }

    function pieceAt(square) {
      return game && square ? game.get(square) : null;
    }

    function handleBoardClick(event) {
      const squareEl = event.target.closest(".square");
      if (!squareEl || !game) {
        fenInput.focus();
        return;
      }

      const target = squareEl.dataset.square;
      const piece = pieceAt(target);
      if (!selectedSquare) {
        if (!piece || piece.color !== game.turn()) return;
        selectedSquare = target;
        boardFromFen(game.fen());
        return;
      }

      const move = game.move({ from: selectedSquare, to: target, promotion: "q" });
      if (move) {
        clearDemoLine();
        selectedSquare = null;
        currentFen = game.fen();
        fenInput.value = currentFen;
        dotRow.querySelectorAll("button").forEach((el) => el.classList.remove("active"));
        exampleTitle.textContent = "Custom position";
        exampleDescription.textContent = `You played ${move.san}. The dashboard is evaluating the new board.`;
        boardFromFen(currentFen);
        evaluateCurrent();
        return;
      }

      if (piece && piece.color === game.turn()) {
        selectedSquare = target;
      } else {
        selectedSquare = null;
      }
      boardFromFen(game.fen());
    }

    window.addEventListener("DOMContentLoaded", () => {
      boardFromFen(START_FEN);
      fenInput.value = START_FEN;
      syncGameFromFen(START_FEN);
      renderExamples();
      const first = dotRow.querySelector("button");
      if (first) {
        first.click();
      } else {
        triggerEvaluate();
      }
      evalBtn.addEventListener("click", triggerEvaluate);
      prevLineBtn.addEventListener("click", () => stepDemo(-1));
      nextLineBtn.addEventListener("click", () => stepDemo(1));
      fenInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          triggerEvaluate();
        }
      });
      boardEl.addEventListener("click", handleBoardClick);
    });
  </script>
</body>
</html>
"""


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template_string(
            HTML,
            start_fen=START_FEN,
            examples=EXAMPLE_FENS,
            common_opening_lines=COMMON_OPENING_LINES,
        )

    @app.post("/api/evaluate")
    def api_evaluate():
        data = request.get_json(silent=True) or {}
        fen = str(data.get("fen", "")).strip()
        try:
            depth = int(data.get("stockfish_depth", 14))
            depth = max(1, min(20, depth))
            time_limit = float(data.get("stockfish_time", 0.25))
            time_limit = max(0.05, min(2.0, time_limit))
            return jsonify(evaluate_position(fen, depth, time_limit))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/health")
    def api_health():
        return jsonify(
            {
                "ok": True,
                "model_exists": MODEL_PATH.exists(),
                "custom_engine": find_custom_engine(),
                "stockfish": find_stockfish(),
                "stockfish_nets": find_stockfish_nets(),
                "stockfish_engine_options": stockfish_engine_options(),
                "custom_stockfish_eval_file": find_custom_stockfish_eval_file(),
                "stockfish_net_diagnostics": stockfish_net_diagnostics(),
                "stockfish_setup_error": stockfish_setup_error(find_stockfish()) if find_stockfish() else None,
            }
        )

    return app


app = create_app()


def main() -> None:
    port = int(os.environ.get("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
