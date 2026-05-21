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
    "Starting position": START_FEN,
    "Italian opening": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 3",
    "Fried Liver-ish": "r1bqkb1r/ppp2ppp/2n5/3np1N1/2B5/8/PPPP1PPP/RNBQK2R w KQkq - 0 6",
    "White up a queen": "rnb1kbnr/pppp1ppp/8/4p3/4Q3/8/PPPP1PPP/RNB1KBNR b KQkq - 0 3",
    "Black material edge": "rnbqkbnr/pppppppp/8/8/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
}
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


def result_payload(name: str, cp: float | None, label: str, note: str) -> dict[str, Any]:
    return {
        "name": name,
        "cp": None if cp is None else round(float(cp), 1),
        "label": label,
        "note": note,
    }


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
        for move in info.get("pv", [])[:5]:
            pv_moves.append(pv_board.san(move))
            pv_board.push(move)
        note = f"Local Stockfish depth {reached_depth}, nodes {nodes:,}" + (f"; PV: {' '.join(pv_moves)}" if pv_moves else "")
        return result_payload("Stockfish", float(cp), pretty_label_from_eval(float(cp)), note)
    except Exception as exc:
        return result_payload("Stockfish", None, "Unavailable", f"Stockfish failed: {exc}")


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
            stockfish_result(board.fen(), stockfish_depth, stockfish_time),
        ],
    }


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Chess Evaluation — Minimal View</title>
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
      --bg: #000000;
      --wire: rgba(255, 255, 255, 0.1);
      --ink: #ffffff;
      --muted: rgba(255, 255, 255, 0.5);
      --charcoal: #111111;
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
      transition: all 0.3s ease-in-out;
      animation: breathe 10s ease-in-out infinite alternate;
    }
    @keyframes breathe {
      from { background: #000000; }
      to { background: #060606; }
    }
    .shell {
      width: min(1400px, 100%);
      margin: 0 auto;
      padding: 2rem 1.05rem 8rem;
    }
    .headline {
      max-width: 980px;
      margin: 0 auto 1rem;
      display: flex;
      flex-wrap: wrap;
      align-items: baseline;
      justify-content: space-between;
      gap: 0.75rem;
    }
    .headline h1 {
      margin: 0;
      font-weight: 700;
      letter-spacing: -0.03em;
      font-size: clamp(2rem, 4vw, 3rem);
      line-height: 1.05;
    }
    .headline p {
      margin: 0;
      color: var(--muted);
      font-size: 0.86rem;
      letter-spacing: 0.01em;
    }
    .panel {
      border: 1px solid var(--wire);
      border-radius: 1.2rem;
      padding: 1rem;
      transition: all 0.3s ease-in-out;
    }
    .workspace {
      margin-top: 1.1rem;
      display: grid;
      gap: 1rem;
      grid-template-columns: 1.25fr 1.35fr 0.95fr;
      align-items: start;
    }
    @media (max-width: 1100px) {
      .workspace {
        grid-template-columns: 1fr;
      }
    }
    .board-wrap {
      display: grid;
      place-items: center;
      min-height: 26rem;
    }
    .chess-board {
      width: min(540px, 100%);
      aspect-ratio: 1 / 1;
    }
    .chess-grid {
      width: 100%;
      height: 100%;
      display: grid;
      grid-template-columns: repeat(8, minmax(0, 1fr));
      grid-template-rows: repeat(8, minmax(0, 1fr));
      overflow: hidden;
      border: 1px solid var(--wire);
    }
    .square {
      display: flex;
      align-items: center;
      justify-content: center;
      user-select: none;
      font-size: clamp(1.4rem, 4vw, 2.65rem);
      line-height: 1;
      letter-spacing: 0;
      font-weight: 500;
      transition: transform 0.2s ease-in-out;
    }
    .light-square {
      background: #ffffff;
      color: #111111;
    }
    .dark-square {
      background: #111111;
      color: #f8f8f8;
    }
    .piece.white {
      color: #ffffff;
      text-shadow: 0 2px 0 rgba(18, 18, 18, 0.4);
    }
    .piece.black {
      color: #111111;
      text-shadow: 0 0 5px rgba(255, 255, 255, 0.5);
    }
    .board-meta {
      margin-top: 0.8rem;
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .dot-row {
      margin-top: 0.7rem;
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }
    .dot-row button {
      appearance: none;
      border: 1px solid var(--wire);
      background: transparent;
      color: var(--ink);
      width: 2.05rem;
      height: 2.05rem;
      border-radius: 9999px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 0.75rem;
      transition: all 0.3s ease-in-out;
    }
    .dot-row button.active,
    .dot-row button:hover {
      color: #000000;
      background: #ffffff;
      border-color: #ffffff;
      box-shadow: 0 0 0 1px rgba(255,255,255,0.2) inset;
    }
    .comparison-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.8rem;
      margin-bottom: 0.9rem;
    }
    .model-card {
      border: 1px solid var(--wire);
      border-radius: 1rem;
      padding: 0.95rem;
      min-height: 13.8rem;
      transition: all 0.3s ease-in-out;
    }
    .model-title {
      font-size: 0.72rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .model-value {
      margin-top: 0.45rem;
      font-size: clamp(2.3rem, 4vw, 3.4rem);
      line-height: 1;
      font-weight: 700;
      letter-spacing: -0.03em;
      transition: color 0.3s ease-in-out;
      white-space: nowrap;
    }
    .sub-tag {
      margin-top: 0.2rem;
      color: var(--muted);
      font-size: 0.75rem;
      letter-spacing: 0.07em;
      text-transform: uppercase;
    }
    .sparkline {
      margin-top: 0.7rem;
      height: 56px;
      border-top: 1px solid var(--wire);
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
      gap: 0.85rem;
      align-content: start;
    }
    .master-caption {
      font-size: 0.72rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .master-bar {
      position: relative;
      height: 1.15rem;
      border: 1px solid var(--wire);
      border-radius: 999px;
      overflow: hidden;
      background: #050505;
    }
    .master-fill {
      position: absolute;
      top: 0;
      bottom: 0;
      border-radius: 999px;
      transition: all 0.3s ease-in-out;
      transform-origin: center;
      box-shadow: 0 0 22px rgba(255, 255, 255, 0.15);
    }
    .master-value {
      font-size: clamp(2rem, 4vw, 3rem);
      line-height: 1;
      font-weight: 700;
      letter-spacing: -0.03em;
    }
    .adv-ring {
      width: 14rem;
      aspect-ratio: 1;
      margin: 0.4rem auto 0;
      border-radius: 9999px;
      display: grid;
      place-items: center;
      border: 2px solid;
      transition: all 0.3s ease-in-out;
    }
    .adv-ring span {
      width: 0.8rem;
      aspect-ratio: 1;
      border-radius: 9999px;
      background: currentColor;
      opacity: 0.86;
      transition: all 0.3s ease-in-out;
    }
    .adv-ring.white {
      color: #ffffff;
      background: rgba(255, 255, 255, 0.18);
      box-shadow: 0 0 40px rgba(255, 255, 255, 0.45);
    }
    .adv-ring.black {
      color: #2f2f2f;
      background: rgba(255, 255, 255, 0.03);
      border-color: #2a2a2a;
      box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.11), 0 0 28px rgba(255, 255, 255, 0.16), inset 0 0 24px rgba(255, 255, 255, 0.08);
      animation: pulseBlack 2.4s ease-in-out infinite;
    }
    .adv-ring.equal {
      color: #666666;
      background: rgba(255, 255, 255, 0.06);
      border-color: #666666;
    }
    @keyframes pulseBlack {
      0%, 100% {
        box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.11), 0 0 22px rgba(255, 255, 255, 0.2), inset 0 0 22px rgba(255, 255, 255, 0.08);
      }
      50% {
        box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.18), 0 0 34px rgba(255, 255, 255, 0.35), inset 0 0 30px rgba(255, 255, 255, 0.12);
      }
    }
    .ring-text {
      text-align: center;
      margin-top: 0.4rem;
      font-size: 0.72rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .fen-shell {
      position: fixed;
      left: 0;
      right: 0;
      bottom: 1rem;
      z-index: 20;
      display: flex;
      justify-content: center;
      pointer-events: none;
    }
    .fen-row {
      width: min(90vw, 48rem);
      display: flex;
      align-items: center;
      gap: 0.65rem;
      pointer-events: auto;
    }
    #fenInput {
      width: 13rem;
      max-width: 100%;
      background: transparent;
      color: #ffffff;
      border: none;
      border-bottom: 1px solid var(--wire);
      padding: 0.7rem 0.1rem;
      outline: none;
      font-size: 0.96rem;
      font-family: inherit;
      transition: width 0.3s ease-in-out, border-color 0.3s ease-in-out;
    }
    #fenInput:focus {
      width: 100%;
      border-color: #ffffff;
    }
    #evalBtn {
      border: 1px solid var(--wire);
      border-radius: 999px;
      background: transparent;
      color: #ffffff;
      padding: 0.52rem 0.88rem;
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      opacity: 0.35;
      pointer-events: none;
      transition: all 0.3s ease-in-out;
    }
    .fen-row:focus-within #evalBtn {
      opacity: 1;
      pointer-events: auto;
    }
    #status {
      color: rgba(255,255,255,0.5);
      font-size: 0.72rem;
      margin-top: 0.65rem;
      min-height: 0.95rem;
    }
    #status.error {
      color: #ff8a80;
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="headline">
      <h1>Chess Evaluation Space</h1>
      <p>Minimal, visual, immediate</p>
    </header>

    <section class="workspace">
      <section class="panel board-panel">
        <div class="board-wrap">
          <div id="board" class="chess-board" aria-label="Chess board"></div>
        </div>
        <div class="board-meta">
          <span id="turnBadge">Turn · White</span>
          <span id="stateBadge">Live</span>
        </div>
        <div id="exampleDots" class="dot-row"></div>
        <div id="status">Type or pick a position below</div>
      </section>

      <section class="panel compare-panel">
        <div class="comparison-grid">
          <article class="model-card">
            <div class="model-title">AI Intuition</div>
            <div id="cnnValue" class="model-value">—</div>
            <div class="sub-tag">CNN</div>
            <div id="cnnSpark" class="sparkline" aria-hidden="true"></div>
          </article>
          <article class="model-card">
            <div class="model-title">Calculated Truth</div>
            <div id="sfValue" class="model-value">—</div>
            <div class="sub-tag">Stockfish</div>
            <div id="sfSpark" class="sparkline" aria-hidden="true"></div>
          </article>
        </div>
        <div id="statusLine" class="board-meta" style="margin:0; text-transform:none; letter-spacing:0.02em;">Comparing two minds, frame by frame.</div>
      </section>

      <section class="panel">
        <div class="master-block">
          <div class="master-caption">Master Eval Bar</div>
          <div class="master-bar">
            <div id="masterFill" class="master-fill"></div>
          </div>
          <div id="masterValue" class="master-value">—</div>
          <div id="advRing" class="adv-ring equal"><span></span></div>
          <div id="ringLabel" class="ring-text">Equal</div>
        </div>
      </section>
    </section>
  </div>

  <div class="fen-shell">
    <div class="fen-row">
      <input id="fenInput" type="text" spellcheck="false" autocomplete="off" />
      <button id="evalBtn" type="button">Evaluate</button>
    </div>
  </div>

  <script>
    const START_FEN = {{ start_fen|tojson }};
    const EXAMPLES = {{ examples|tojson }};
    const MAX_SPARK = 8;
    const boardEl = document.getElementById("board");
    const fenInput = document.getElementById("fenInput");
    const turnBadge = document.getElementById("turnBadge");
    const stateBadge = document.getElementById("stateBadge");
    const statusEl = document.getElementById("status");
    const statusLine = document.getElementById("statusLine");
    const dotRow = document.getElementById("exampleDots");
    const masterFill = document.getElementById("masterFill");
    const masterValueEl = document.getElementById("masterValue");
    const advRing = document.getElementById("advRing");
    const ringLabel = document.getElementById("ringLabel");
    const evalBtn = document.getElementById("evalBtn");

    const pieceGlyphs = {
      K: "♔",
      Q: "♕",
      R: "♖",
      B: "♗",
      N: "♘",
      P: "♙",
      k: "♚",
      q: "♛",
      r: "♜",
      b: "♝",
      n: "♞",
      p: "♟",
    };

    let requestSeq = 0;
    let currentFen = START_FEN;
    const histories = {
      cnn: [],
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

      for (let r = 0; r < 8; r++) {
        const row = rows[r] || "";
        let file = 0;
        for (const ch of row) {
          if (file >= 8) break;
          if (/\d/.test(ch)) {
            const count = Number(ch);
            for (let i = 0; i < count; i++) {
              const square = document.createElement("div");
              square.className = `square ${((r + file) % 2 === 0) ? "light-square" : "dark-square"}`;
              grid.appendChild(square);
              file++;
            }
            continue;
          }
          const square = document.createElement("div");
          square.className = `square ${((r + file) % 2 === 0) ? "light-square" : "dark-square"}`;
          const glyph = pieceGlyphs[ch];
          if (glyph) {
            const p = document.createElement("span");
            p.className = `piece ${ch === ch.toUpperCase() ? "white" : "black"}`;
            p.textContent = glyph;
            square.appendChild(p);
          }
          grid.appendChild(square);
          file++;
        }
        while (file < 8) {
          const square = document.createElement("div");
          square.className = `square ${((r + file) % 2 === 0) ? "light-square" : "dark-square"}`;
          grid.appendChild(square);
          file++;
        }
      }

      while (grid.childElementCount < 64) {
        const square = document.createElement("div");
        const rowIndex = Math.floor(grid.childElementCount / 8);
        const column = grid.childElementCount % 8;
        square.className = `square ${((rowIndex + column) % 2 === 0) ? "light-square" : "dark-square"}`;
        grid.appendChild(square);
      }
      boardEl.replaceChildren(grid);
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

    function updateMaster(cp) {
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
        masterFill.style.background = "#ffffff";
        masterFill.style.boxShadow = "0 0 28px rgba(255,255,255,0.55)";
      } else if (clamped < -60) {
        masterFill.style.background = "#111111";
        masterFill.style.boxShadow = "0 0 24px rgba(255,255,255,0.2)";
      } else {
        masterFill.style.background = "#444444";
        masterFill.style.boxShadow = "0 0 20px rgba(255,255,255,0.22)";
      }

      masterValueEl.textContent = cpToLabel(clamped);
      animateRing(clamped);
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
        sparkEl.innerHTML = buildSpark([0, 0], "#888888");
        return;
      }

      const cp = Number(payload.cp);
      if (Number.isFinite(cp)) {
        histories[side].push(cp);
        if (histories[side].length > MAX_SPARK) histories[side].shift();
        animateValue(valueEl, cp);
        sparkEl.innerHTML = buildSpark(histories[side], side === "cnn" ? "#ffffff" : "#cccccc");
      }
    }

    function renderExamples() {
      const names = Object.entries(EXAMPLES);
      names.forEach(([label, fen], index) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = String(index + 1);
        btn.title = label;
        btn.addEventListener("click", () => {
          dotRow.querySelectorAll("button").forEach((el) => el.classList.remove("active"));
          btn.classList.add("active");
          fenInput.value = fen;
          triggerEvaluate();
        });
        dotRow.appendChild(btn);
      });
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
        boardFromFen(payload.fen || currentFen);
        markTurn(payload.turn || "White", payload.is_check, payload.is_game_over);

        const cnn = results.CNN || null;
        const stockfish = results.Stockfish || null;
        updateModel("cnn", cnn);
        updateModel("sf", stockfish);

        const anchor = Number.isFinite(stockfish?.cp) ? stockfish.cp : Number.isFinite(cnn?.cp) ? cnn.cp : 0;
        updateMaster(anchor);

        currentFen = payload.fen || currentFen;
        fenInput.value = currentFen;
        setStatus("Updated");
        setTimeout(() => setStatus("Type or pick a position below"), 1200);
        statusLine.textContent = `${cnn ? "AI Intuition" : "—"}  ·  ${stockfish ? "Stockfish" : "—"}  ·  two-column momentum`;
      } finally {
        if (id === requestSeq) {
          setLoading(false);
        }
      }
    }

    function triggerEvaluate() {
      const nextFen = String(fenInput.value || START_FEN).trim();
      currentFen = nextFen;
      boardFromFen(nextFen);
      evaluateCurrent();
    }

    window.addEventListener("DOMContentLoaded", () => {
      boardFromFen(START_FEN);
      fenInput.value = START_FEN;
      renderExamples();
      const first = dotRow.querySelector("button");
      if (first) first.classList.add("active");
      triggerEvaluate();
      evalBtn.addEventListener("click", triggerEvaluate);
      fenInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          triggerEvaluate();
        }
      });
      boardEl.addEventListener("click", () => fenInput.focus());
    });
  </script>
</body>
</html>
"""


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template_string(HTML, start_fen=START_FEN, examples=EXAMPLE_FENS)

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
    app.run(host="0.0.0.0", port=8501, debug=False)


if __name__ == "__main__":
    main()
