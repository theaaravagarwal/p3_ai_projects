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
  <title>Chess Evaluation AI Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          colors: {
            bg: "#030712",
            panel: "#0B0F19",
            glass: "rgba(17,24,39,0.7)",
            emerald: "#10B981",
            indigo: "#6366F1",
            silver: "#9CA3AF"
          },
          fontFamily: {
            sans: ["Inter","ui-sans-serif","system-ui","sans-serif"],
            mono: ["JetBrains Mono","ui-monospace","SFMono-Regular","Consolas","monospace"]
          },
          boxShadow: {
            glass: "0 26px 70px -45px rgba(15,23,42,.9), inset 0 1px 0 rgba(255,255,255,.04)",
            glow: "0 0 0 1px rgba(16,185,129,.28), 0 0 26px rgba(16,185,129,.16)"
          }
        }
      }
    };
  </script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.css" />
  <style>
    :root {
      --line: rgba(255, 255, 255, 0.06);
      --muted: #9ca3af;
      --bg-a: #030712;
      --bg-b: #0B0F19;
    }
    html, body {
      background: radial-gradient(circle at 12% 8%, #111827 0%, #0B0F19 38%, #060A14 75%, #030712 100%);
      min-height: 100%;
      margin: 0;
      color: #e5e7eb;
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    }
    .sheet {
      background: linear-gradient(170deg, rgba(17,24,39,0.78), rgba(15,23,42,0.78));
      border: 1px solid var(--line);
      box-shadow: 0 26px 70px -45px rgba(15,23,42,.9), inset 0 1px 0 rgba(255,255,255,.04);
      backdrop-filter: blur(12px);
    }
    .chip {
      transition: all .25s ease;
      border: 1px solid rgba(255,255,255,0.15);
      background: rgba(15,23,42,.55);
    }
    .chip:hover {
      transform: translateY(-1px);
      border-color: rgba(16,185,129,.8);
      color: #6ee7b7;
    }
    .chip.active {
      border-color: rgba(16,185,129,0.95);
      color: #6ee7b7;
      background: rgba(16,185,129,0.1);
      box-shadow: 0 0 0 1px rgba(16,185,129,.28), 0 0 22px rgba(16,185,129,.2);
    }
    .icon-btn {
      width: 2.55rem;
      height: 2.55rem;
      border-radius: 0.8rem;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: 1px solid rgba(148,163,184,.28);
      background: rgba(15,23,42,.6);
      transition: all .2s ease;
    }
    .icon-btn svg {
      width: 1rem;
      height: 1rem;
    }
    .icon-btn:hover {
      border-color: rgba(16,185,129,.7);
      box-shadow: 0 0 0 1px rgba(16,185,129,.2);
    }
    #boardWrap {
      position: relative;
      border-radius: 1rem;
      padding: 0.75rem;
      background: rgba(2,6,23,.45);
      border: 1px solid rgba(148,163,184,.16);
      box-shadow: 0 24px 58px -32px rgba(2,6,23,.85);
    }
    #board {
      width: min(100%, 540px);
      margin: 0 auto;
      filter: drop-shadow(0 18px 28px rgba(2, 6, 23, .7));
    }
    .chessboard-board {
      border-radius: 0.6rem;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,.08);
    }
    .board-sheen::after {
      content: "";
      pointer-events: none;
      position: absolute;
      inset: 0.85rem;
      border-radius: 0.6rem;
      box-shadow: inset 0 0 34px rgba(0,0,0,.28), inset 0 0 0 1px rgba(255,255,255,.04);
    }
    .eval-card {
      position: relative;
      overflow: hidden;
      border: 1px solid var(--line);
      transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
    }
    .eval-card:hover {
      transform: translateY(-1px);
      border-color: rgba(255,255,255,.14);
    }
    .eval-card::before {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      background: linear-gradient(125deg, rgba(255,255,255,0.02), rgba(255,255,255,0), rgba(255,255,255,0));
    }
    .eval-card.loading {
      animation: shimmer 1.1s linear infinite;
      border-color: rgba(16,185,129,.36);
      box-shadow: 0 0 0 1px rgba(16,185,129,.27), 0 0 30px rgba(16,185,129,.2);
    }
    .loading::before {
      content: "";
      position: absolute;
      inset: -1px;
      background: linear-gradient(100deg, transparent, rgba(16,185,129,.18), transparent);
      animation: shimmer-sweep 1.2s linear infinite;
      transform: translateX(-100%);
      z-index: 0;
    }
    @keyframes shimmer {
      0%,100% { filter: saturate(1); }
      50% { filter: saturate(1.15); }
    }
    @keyframes shimmer-sweep {
      0% { transform: translateX(-120%); }
      100% { transform: translateX(120%); }
    }
    .badge {
      border-radius: 999px;
      border: 1px solid;
      padding: 0.2rem 0.7rem;
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.01em;
      line-height: 1.4;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      position: relative;
      z-index: 1;
    }
    .badge.white {
      color: #6ee7b7;
      border-color: rgba(16,185,129,.45);
      background: rgba(16,185,129,.11);
      box-shadow: 0 0 0 1px rgba(16,185,129,.24);
    }
    .badge.black {
      color: #a5b4fc;
      border-color: rgba(99,102,241,.45);
      background: rgba(99,102,241,.12);
      box-shadow: 0 0 0 1px rgba(99,102,241,.24);
    }
    .badge.equal {
      color: #d1d5db;
      border-color: rgba(148,163,184,.45);
      background: rgba(148,163,184,.14);
      box-shadow: 0 0 0 1px rgba(148,163,184,.24);
    }
    .vertical-gauge {
      width: 1.15rem;
      border-radius: 999px;
      min-height: 16rem;
      position: relative;
      border: 1px solid rgba(148,163,184,.18);
      background: linear-gradient(180deg, rgba(30,41,59,.78), rgba(15,23,42,.78));
      overflow: hidden;
    }
    .vertical-gauge::before {
      content: "";
      position: absolute;
      left: 48%;
      top: 0.5rem;
      bottom: 0.5rem;
      width: 0.35rem;
      border-radius: 1rem;
      background: linear-gradient(180deg, #10B981, rgba(16,185,129,.16), #6366F1);
+    }
    .vertical-gauge::after {
      content: "";
      position: absolute;
      left: calc(50% - 2px);
      top: 46%;
      width: 0.55rem;
      height: 0.55rem;
      border-radius: 999px;
      background: rgba(156,163,175,.6);
      border: 1px solid rgba(255,255,255,.4);
    }
    .gauge-fill {
      position: absolute;
      inset: 0.38rem;
      border-radius: 1rem;
      overflow: hidden;
    }
    .gauge-fill::before {
      content: "";
      position: absolute;
      inset: -8px 0;
      background: linear-gradient(180deg, rgba(99,102,241,.9), rgba(16,185,129,.75), rgba(6,182,212,.3));
      transition: all .6s cubic-bezier(0.25, 1, 0.5, 1);
      will-change: transform;
      border-radius: 1rem;
      box-shadow: 0 0 34px rgba(16,185,129,.3);
    }
    .gauge-marker {
      position: absolute;
      left: 50%;
      width: 1.65rem;
      height: 1.65rem;
      border-radius: 999px;
      transform: translateX(-50%);
      border: 1px solid rgba(15,23,42,.85);
      background: radial-gradient(circle at 35% 30%, #f8fafc, #10B981);
      box-shadow: 0 0 24px rgba(16,185,129,.45);
      transition: all .6s cubic-bezier(0.25, 1, 0.5, 1);
      will-change: top;
    }
    .sparkline-wrap {
      height: 45px;
      margin-top: 0.35rem;
    }
    .sparkline-wrap svg {
      width: 100%;
      height: 100%;
    }
    .sparkline-path {
      fill: none;
      stroke-width: 1.6;
      stroke-linecap: round;
      stroke-linejoin: round;
      transition: d .6s cubic-bezier(0.25, 1, 0.5, 1);
    }
    .spec-panel {
      max-height: 0;
      opacity: 0;
      overflow: hidden;
      transition: max-height .35s cubic-bezier(0.16,1,0.3,1), opacity .25s ease;
    }
    .spec-open .spec-panel {
      max-height: 120px;
      opacity: 1;
      margin-top: 0.55rem;
    }
    details {
      border-top: 1px solid rgba(148,163,184,.18);
      margin-top: 0.65rem;
      padding-top: 0.55rem;
    }
    summary {
      list-style: none;
      cursor: pointer;
      color: #86efac;
      font-size: .75rem;
    }
    summary::-webkit-details-marker { display:none; }
    .mono {
      font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Consolas, monospace;
    }
    .diag-dot {
      width: 0.5rem;
      height: 0.5rem;
      border-radius: 999px;
      background: #10B981;
      box-shadow: 0 0 0 4px rgba(16,185,129,.18);
    }
    .diag-dot.off {
      background: #ef4444;
      box-shadow: 0 0 0 4px rgba(239,68,68,.16);
    }
    .floating-tip {
      position: absolute;
      inset: auto 1rem 1rem auto;
      background: rgba(15,23,42,.7);
      border: 1px solid rgba(148,163,184,.22);
      border-radius: .8rem;
      padding: .45rem .6rem;
      font-size: .7rem;
      color: #cbd5e1;
      backdrop-filter: blur(8px);
    }
    @media (max-width: 1024px) {
      .dash-shell {
        grid-template-columns: 1fr !important;
      }
      .mobile-stack {
        display: flex;
        flex-direction: column;
      }
      .mobile-stack > *:first-child {
        order: 1;
      }
      .mobile-stack > *:nth-child(2) {
        order: 2;
      }
    }
  </style>
</head>
<body class="min-h-screen text-slate-200">
  <main class="mx-auto w-full max-w-7xl px-4 py-5 sm:px-6 lg:px-8">
    <header class="relative rounded-2xl sheet p-4 md:p-5 mb-4">
      <div class="floating-tip">Static Evaluator v1.0 • No Search Tree</div>
      <div class="mb-3 flex flex-wrap items-start justify-between gap-3">
        <div class="space-y-1.5">
          <p class="text-[11px] uppercase tracking-[0.22em] text-cyan-300 font-semibold">Chess Evaluation AI Dashboard</p>
          <h1 class="text-2xl md:text-3xl font-semibold tracking-tight">Chess Position Intelligence Console</h1>
          <p class="text-sm text-slate-400 max-w-2xl leading-6">Compare a custom-trained ResNet34 CNN versus Stockfish on static position evaluation. Built as a premium analytics workflow.</p>
        </div>
        <div class="inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs sheet border border-slate-500/20">
          CNN • MAE 125.22 cp • Bucket Accuracy 89.01% (Clean 95.12%)
        </div>
      </div>
      <div class="text-xs text-slate-400">Current model: <span class="mono text-slate-200">models/best_model.pt</span></div>
    </header>

    <section class="dash-shell mobile-stack grid gap-4 lg:grid-cols-[1.25fr,1fr]">
      <section class="sheet rounded-2xl p-4 md:p-5 space-y-4">
        <div class="flex items-center justify-between">
          <h2 class="text-lg font-semibold tracking-wide">Board Zone</h2>
          <div class="text-xs text-slate-400">Interactive drag & drop</div>
        </div>
        <div id="boardWrap" class="board-sheen">
          <div id="board"></div>
        </div>
        <p class="mono text-[11px] text-slate-400 break-all" id="currentFen"></p>

        <div class="space-y-3">
          <label class="block text-[11px] uppercase tracking-[0.2em] text-slate-300">FEN command bar</label>
          <div class="flex items-stretch gap-2">
            <div class="relative flex-1">
              <textarea id="fenInput" rows="2" class="w-full mono text-sm rounded-xl sheet border border-slate-500/20 p-3 pr-24 resize-none focus:outline-none focus:border-emerald-400/60 focus:shadow-[0_0_0_2px_rgba(16,185,129,0.22)]"></textarea>
              <div class="absolute right-2 top-2 flex gap-2">
                <button id="copyFenBtn" class="icon-btn" title="Copy FEN" aria-label="Copy FEN">
                  <svg viewBox="0 0 24 24" fill="none"><path d="M8 9h8v8H8V9Z" stroke="currentColor" stroke-width="1.6"/><path d="M12 4v2M12 18v2M6 7H5a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h2m10-14h1a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-2" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" /></svg>
                </button>
                <button id="pasteFenBtn" class="icon-btn" title="Paste FEN" aria-label="Paste FEN">
                  <svg viewBox="0 0 24 24" fill="none"><path d="M9 4h6a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2Z" stroke="currentColor" stroke-width="1.6"/><path d="M7 8h10M7 11h8" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path d="M11 15h6" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
                </button>
              </div>
            </div>
            <button id="loadFenBtn" class="whitespace-nowrap rounded-xl px-4 text-sm font-semibold tracking-wide bg-emerald-500 text-slate-950 hover:bg-emerald-400 transition">Evaluate</button>
          </div>
          <div class="space-y-2">
            <div class="flex flex-wrap items-center justify-between gap-2">
              <span class="text-[11px] uppercase tracking-[0.2em] text-slate-300">Example positions</span>
              <div class="flex items-center gap-2 text-xs text-slate-400">
                <span>Depth</span>
                <input id="depthInput" type="number" min="6" max="20" value="14" class="w-20 rounded-lg border border-slate-500/25 sheet p-1.5 text-sm focus:outline-none focus:border-emerald-400/60" />
              </div>
            </div>
            <div id="exampleButtons" class="flex flex-wrap gap-2"></div>
          </div>
          <div class="flex flex-wrap items-center gap-2">
            <button id="resetBtn" class="rounded-xl border border-slate-500/20 px-4 py-2 text-sm hover:border-emerald-300/70">Reset</button>
            <button id="flipBtn" class="rounded-xl border border-slate-500/20 px-4 py-2 text-sm hover:border-indigo-300/70">Flip board</button>
          </div>
          <div id="errorBox" class="text-sm rounded-xl bg-red-500/10 border border-red-400/40 text-red-200 p-3 hidden"></div>
        </div>
      </section>

      <section class="space-y-4" id="analyticsPanel">
        <section class="sheet rounded-2xl p-4 md:p-5">
          <h2 class="text-lg font-semibold tracking-wide mb-3">Evaluation Analytics</h2>
          <div class="grid grid-cols-2 gap-2 text-sm">
            <div class="sheet rounded-lg p-3 border border-slate-500/20">
              <p class="text-[11px] uppercase tracking-[0.2em] text-slate-400">Turn</p>
              <p class="mt-1 text-lg font-semibold" id="turnValue">White</p>
            </div>
            <div class="sheet rounded-lg p-3 border border-slate-500/20">
              <p class="text-[11px] uppercase tracking-[0.2em] text-slate-400">Check</p>
              <p class="mt-1 text-lg font-semibold" id="checkValue">No</p>
            </div>
            <div class="sheet rounded-lg p-3 border border-slate-500/20">
              <p class="text-[11px] uppercase tracking-[0.2em] text-slate-400">State</p>
              <p class="mt-1 text-lg font-semibold" id="statusValue">Live</p>
            </div>
            <div class="sheet rounded-lg p-3 border border-slate-500/20">
              <p class="text-[11px] uppercase tracking-[0.2em] text-slate-400">Move</p>
              <p class="mt-1 text-lg font-semibold" id="moveValue">White</p>
            </div>
          </div>

          <div class="mt-4 flex items-stretch gap-3">
            <div class="flex-1">
              <p class="mb-2 text-xs uppercase tracking-[0.2em] text-slate-300">Live Eval (vertical gauge)</p>
              <div class="vertical-gauge">
                <div class="gauge-fill"><span id="gaugeFill"></span></div>
                <div id="gaugeMarker" class="gauge-marker"></div>
              </div>
            </div>
            <div class="flex-1">
              <p class="text-xs uppercase tracking-[0.2em] text-slate-300">Quick readout</p>
              <div class="mt-2 space-y-1 text-sm text-slate-300">
                <p>White zone: <span class="text-emerald-300">+</span> advantage</p>
                <p>Balanced zone: near center</p>
                <p>Black zone: <span class="text-indigo-300">-</span> advantage</p>
              </div>
            </div>
          </div>

          <div id="evalGrid" class="space-y-3 mt-3"></div>
        </section>

        <section class="sheet rounded-2xl p-3 md:p-4 border border-slate-500/20">
          <div class="grid gap-2 text-xs mono text-slate-300">
            <div class="flex items-center gap-2"><span id="statusDot" class="diag-dot animate-pulse"></span><span id="diagStatusLine">Backend Status: Online (Flask API)</span></div>
            <div class="flex items-center gap-2"><span id="sfDot" class="diag-dot animate-pulse"></span><span id="diagBinaryLine">Stockfish: Connected</span></div>
            <div class="flex items-center gap-2"><span id="nnueDot" class="diag-dot animate-pulse"></span><span id="diagNnueLine">NNUE: Active</span></div>
          </div>
        </section>
      </section>
    </section>
  </main>

  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.js"></script>
  <script>
    const START_FEN = {{ start_fen|tojson }};
    const EXAMPLES = {{ examples|tojson }};
    const MAX_SPARK = 10;
    let game = new Chess(START_FEN);
    let board = null;
    let requestId = 0;
    const sparkHistories = { CNN: [], Stockfish: [] };

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function formatCp(value) {
      if (value === null || value === undefined) {
        return "Unavailable";
      }
      const n = Number(value);
      if (!Number.isFinite(n)) {
        return "Unavailable";
      }
      return `${n > 0 ? "+" : ""}${n.toFixed(1)} cp`;
    }

    function bucketClass(label) {
      if (label === "White advantage") return "white";
      if (label === "Black advantage") return "black";
      return "equal";
    }

    function cpToY(value) {
      const clamped = clamp(Number(value || 0), -1000, 1000);
      return 98 - ((clamped + 1000) / 2000) * 96;
    }

    function setError(message) {
      const box = document.getElementById("errorBox");
      if (!message) {
        box.classList.add("hidden");
        box.textContent = "";
        return;
      }
      box.classList.remove("hidden");
      box.textContent = message;
    }

    function syncFenText() {
      const fen = game.fen();
      document.getElementById("fenInput").value = fen;
      document.getElementById("currentFen").textContent = fen;
      document.getElementById("moveValue").textContent = game.turn() === "w" ? "White" : "Black";
    }

    function renderSparkline(values, colorFrom, colorTo) {
      const width = 340;
      const height = 43;
      if (!values.length) {
        return `
          <svg viewBox="0 0 ${width} ${height}" aria-hidden="true">
            <defs>
              <linearGradient id="g" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stop-color="${colorFrom}" stop-opacity="0.22" />
                <stop offset="100%" stop-color="${colorTo}" stop-opacity="0.02" />
              </linearGradient>
            </defs>
            <path d="M0,${height / 2} L${width},${height / 2}" class="sparkline-path" stroke="url(#g)"></path>
          </svg>`;
      }
      const min = Math.min(...values, -1000);
      const max = Math.max(...values, 1000);
      const range = Math.max(Math.abs(min), Math.abs(max), 200);
      const stepX = width / (MAX_SPARK - 1);
      let d = "";
      values.forEach((val, index) => {
        const x = index * stepX;
        const y = ((range - clamp(val, -range, range)) / (2 * range)) * (height - 4) + 2;
        d += `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)} `;
      });
      return `
        <svg viewBox="0 0 ${width} ${height}" aria-hidden="true">
          <defs>
            <linearGradient id="g" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stop-color="${colorFrom}" stop-opacity="0.46" />
              <stop offset="100%" stop-color="${colorTo}" stop-opacity="0.06" />
            </linearGradient>
          </defs>
          <path d="${d}" class="sparkline-path" stroke="url(#g)"></path>
        </svg>`;
    }

    function updateSparkline(elementId, values, colorFrom, colorTo) {
      const element = document.getElementById(elementId);
      element.innerHTML = renderSparkline(values, colorFrom, colorTo);
      const path = element.querySelector("path");
      if (!path) return;
      const length = path.getTotalLength ? path.getTotalLength() : 0;
      path.style.strokeDasharray = `${length}`;
      path.style.strokeDashoffset = `${length}`;
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          path.style.transition = "stroke-dashoffset .65s cubic-bezier(0.25, 1, 0.5, 1)";
          path.style.strokeDashoffset = "0";
        });
      });
    }

    function animateButtonSuccess(button) {
      const icon = button.querySelector("svg");
      if (!icon) return;
      icon.innerHTML = `<path d="M6 12l4 4 8-8" stroke="#10B981" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>`;
      button.classList.add("text-emerald-300");
      setTimeout(() => {
        icon.innerHTML = button.id === "copyFenBtn" ?
          `<path d="M8 9h8v8H8V9Z" stroke="currentColor" stroke-width="1.6"/><path d="M12 4v2M12 18v2M6 7H5a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h2m10-14h1a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-2" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" />` :
          `<path d="M9 4h6a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2Z" stroke="currentColor" stroke-width="1.6"/><path d="M7 8h10M7 11h8" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><path d="M11 15h6" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>`;
        button.classList.remove("text-emerald-300");
      }, 620);
    }

    function renderCards(payload) {
      const evaluations = payload.evaluations || [];
      const map = new Map(evaluations.map((item) => [item.name, item]));
      const material = map.get("Material") || { cp: null, label: "Unavailable", note: "" };
      const cnn = map.get("CNN") || { cp: null, label: "Unavailable", note: "" };
      const stockfish = map.get("Stockfish") || { cp: null, label: "Unavailable", note: "" };

      if (cnn.cp !== null) {
        sparkHistories.CNN.push(Number(cnn.cp));
        if (sparkHistories.CNN.length > MAX_SPARK) sparkHistories.CNN.shift();
      }
      if (stockfish.cp !== null) {
        sparkHistories.Stockfish.push(Number(stockfish.cp));
        if (sparkHistories.Stockfish.length > MAX_SPARK) sparkHistories.Stockfish.shift();
      }

      const cnnMarker = clamp(Number(cnn.cp || 0), -1000, 1000);
      const sfMarker = clamp(Number(stockfish.cp || 0), -1000, 1000);
      const container = document.getElementById("evalGrid");
      container.innerHTML = `
        <article id="cnnCard" class="eval-card sheet rounded-xl p-4">
          <div class="flex items-start justify-between gap-3">
            <div>
              <p class="text-xs uppercase tracking-[0.15em] text-slate-400">CNN Predictor</p>
              <h3 class="mt-1 text-lg font-semibold tracking-wide">ResNet34 Static Evaluator</h3>
              <span class="mt-2 inline-flex badge ${bucketClass(cnn.label)}">${cnn.label}</span>
            </div>
            <p class="mono text-2xl font-semibold text-emerald-300">${formatCp(cnn.cp)}</p>
          </div>
          <div class="mt-3 text-sm text-slate-300">${cnn.note || "No message."}</div>
          <div class="sparkline-wrap" id="cnnSparkline"></div>
          <details class="spec-open">
            <summary>Model specifications</summary>
            <div class="spec-panel">
              <div class="mt-2 grid grid-cols-2 gap-2 text-xs text-slate-300">
                <div class="sheet rounded-lg p-2">MAE <span class="float-right mono text-slate-200">125.22 cp</span></div>
                <div class="sheet rounded-lg p-2">Bucket Accuracy <span class="float-right mono text-slate-200">89.01%</span></div>
                <div class="sheet rounded-lg p-2 col-span-2">Clean Bucket Accuracy <span class="float-right mono text-slate-200">95.12%</span></div>
              </div>
            </div>
          </details>
        </article>
        <article id="sfCard" class="eval-card sheet rounded-xl p-4">
          <div class="flex items-start justify-between gap-3">
            <div>
              <p class="text-xs uppercase tracking-[0.15em] text-slate-400">Stockfish</p>
              <h3 class="mt-1 text-lg font-semibold tracking-wide">Engine Ground-Truth Snapshot</h3>
              <span class="mt-2 inline-flex badge white">${stockfish.label}</span>
            </div>
            <p class="mono text-2xl font-semibold text-indigo-300">${formatCp(stockfish.cp)}</p>
          </div>
          <div class="mt-3 text-sm text-slate-300">Material score: ${
            material.cp === null ? "Unavailable" : `${formatCp(material.cp)} · ${Number(material.cp / 100).toFixed(2)} pawns`
          }</div>
          <div class="sparkline-wrap" id="sfSparkline"></div>
          <div class="text-xs text-slate-400 mt-1">${stockfish.note || "No message."}</div>
        </article>
      `;
      updateSparkline("cnnSparkline", sparkHistories.CNN, "#10b981", "#06b6d4");
      updateSparkline("sfSparkline", sparkHistories.Stockfish, "#6366f1", "#38bdf8");
      updateGauge(Math.abs(cnnMarker) > 0 ? cnnMarker : 0);
    }

    function updateGauge(cp) {
      const marker = document.getElementById("gaugeMarker");
      const fill = document.getElementById("gaugeFill");
      const fillPct = clamp(((Number(cp) + 1000) / 2000) * 100, 0, 100);
      const top = cpToY(cp);
      marker.style.top = `${top}%`;
      fill.style.height = `${100 - fillPct}%`;
      fill.style.top = `${fillPct}%`;
      fill.firstElementChild.style.transform = `translateY(${fillPct - 100}%)`;
    }

    function setupExamples() {
      const mount = document.getElementById("exampleButtons");
      Object.entries(EXAMPLES).forEach(([label, fen]) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "chip rounded-full px-3 py-1.5 text-xs";
        chip.textContent = label;
        chip.addEventListener("click", () => {
          document.querySelectorAll(".chip.active").forEach((item) => item.classList.remove("active"));
          chip.classList.add("active");
          loadFen(fen);
        });
        mount.appendChild(chip);
      });
    }

    async function fetchHealth() {
      try {
        const response = await fetch("/api/health");
        const health = await response.json();
        document.getElementById("diagStatusLine").textContent = `Backend Status: ${response.ok ? "Online (Flask API)" : "Offline"}`;
        const stockfish = health.stockfish || "Unavailable";
        const hasNnue = (health.stockfish_nets && Object.keys(health.stockfish_nets).length) || health.custom_stockfish_eval_file;
        document.getElementById("diagBinaryLine").textContent = `Stockfish: ${stockfish}`;
        document.getElementById("diagNnueLine").textContent = `NNUE: ${hasNnue ? "Active" : "Not found"}`;
        document.getElementById("statusDot").classList.toggle("off", !response.ok);
        const sfConnected = stockfish !== "Unavailable" && !stockfish.includes("None") && response.ok;
        document.getElementById("sfDot").classList.toggle("off", !sfConnected);
        document.getElementById("nnueDot").classList.toggle("off", !hasNnue);
      } catch {
        document.getElementById("diagStatusLine").textContent = "Backend Status: Offline";
        document.getElementById("diagBinaryLine").textContent = "Stockfish: Unreachable";
        document.getElementById("diagNnueLine").textContent = "NNUE: Unknown";
        document.getElementById("statusDot").classList.add("off");
        document.getElementById("sfDot").classList.add("off");
        document.getElementById("nnueDot").classList.add("off");
      }
    }

    async function evaluateCurrentPosition() {
      const id = ++requestId;
      syncFenText();
      setError("");
      const cnnCard = document.getElementById("cnnCard");
      const sfCard = document.getElementById("sfCard");
      const panel = document.getElementById("analyticsPanel");
      if (cnnCard) cnnCard.classList.add("loading");
      if (sfCard) sfCard.classList.add("loading");
      panel.classList.add("opacity-95");
      const depth = Number(document.getElementById("depthInput").value || 14);

      try {
        const response = await fetch("/api/evaluate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ fen: game.fen(), stockfish_depth: depth, stockfish_time: 0.25 })
        });
        const payload = await response.json();
        if (id !== requestId) return;
        if (!response.ok) {
          setError(payload.error || "Evaluation failed");
          return;
        }
        document.getElementById("turnValue").textContent = payload.turn;
        document.getElementById("checkValue").textContent = payload.is_check ? "Yes" : "No";
        document.getElementById("statusValue").textContent = payload.is_game_over ? "Game over" : "Live";
        renderCards(payload);
      } catch (error) {
        if (id === requestId) setError(`Server request failed: ${error}`);
      } finally {
        if (id === requestId) {
          if (cnnCard) cnnCard.classList.remove("loading");
          if (sfCard) sfCard.classList.remove("loading");
          panel.classList.remove("opacity-95");
        }
      }
    }

    function onDragStart(source, piece) {
      if (game.game_over()) return false;
      if (game.turn() === "w" && piece.search(/^b/) !== -1) return false;
      if (game.turn() === "b" && piece.search(/^w/) !== -1) return false;
    }

    function onDrop(source, target) {
      const move = game.move({ from: source, to: target, promotion: "q" });
      if (move === null) return "snapback";
      evaluateCurrentPosition();
    }

    function onSnapEnd() {
      board.position(game.fen());
    }

    function loadFen(fen) {
      const next = new Chess();
      const ok = next.load(fen.trim());
      if (!ok) {
        setError("Invalid FEN");
        return;
      }
      game = next;
      board.position(game.fen(), false);
      evaluateCurrentPosition();
    }

    async function pasteFromClipboard() {
      try {
        const text = await navigator.clipboard.readText();
        document.getElementById("fenInput").value = text || "";
        if (text) loadFen(text);
      } catch {
        setError("Clipboard read denied");
      }
    }

    async function copyToClipboard() {
      try {
        await navigator.clipboard.writeText(document.getElementById("fenInput").value.trim());
        animateButtonSuccess(document.getElementById("copyFenBtn"));
      } catch {
        setError("Clipboard write denied");
      }
    }

    function initBoard() {
      board = Chessboard("board", {
        draggable: true,
        position: "start",
        pieceTheme: "https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png",
        moveSpeed: 220,
        showNotation: false,
        onDragStart,
        onDrop,
        onSnapEnd
      });
    }

    window.addEventListener("DOMContentLoaded", () => {
      initBoard();
      setupExamples();
      syncFenText();
      evaluateCurrentPosition();
      fetchHealth();
      document.getElementById("loadFenBtn").addEventListener("click", () => loadFen(document.getElementById("fenInput").value));
      document.getElementById("resetBtn").addEventListener("click", () => loadFen(START_FEN));
      document.getElementById("flipBtn").addEventListener("click", () => board.flip());
      document.getElementById("depthInput").addEventListener("change", evaluateCurrentPosition);
      document.getElementById("copyFenBtn").addEventListener("click", copyToClipboard);
      document.getElementById("pasteFenBtn").addEventListener("click", () => {
        pasteFromClipboard();
        animateButtonSuccess(document.getElementById("pasteFenBtn"));
      });
      window.addEventListener("resize", board.resize);
      const info = document.querySelector("details");
      if (info) {
        info.addEventListener("toggle", (ev) => {
          ev.currentTarget.classList.toggle("spec-open", ev.currentTarget.open);
        });
      }
      setTimeout(fetchHealth, 1200);
      setInterval(fetchHealth, 9000);
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
