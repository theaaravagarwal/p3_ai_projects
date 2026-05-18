from __future__ import annotations

import sys
from pathlib import Path

import chess

from app import (
    MODEL_PATH,
    START_FEN,
    app,
    create_app,
    evaluate_position,
    find_stockfish,
    find_stockfish_nets,
    material_evaluation_cp,
    stockfish_setup_error,
    stockfish_result,
)


def check(name: str, fn) -> bool:
    try:
        fn()
        print(f"PASS {name}")
        return True
    except Exception as exc:
        print(f"FAIL {name}: {exc}")
        return False


def test_model_exists() -> None:
    assert Path(MODEL_PATH).exists(), "models/best_model.pt is missing"


def test_material_eval() -> None:
    assert material_evaluation_cp(chess.Board(START_FEN)) == 0


def test_evaluate_position() -> None:
    payload = evaluate_position(START_FEN, stockfish_depth=1, stockfish_time=0.05)
    assert payload["fen"] == START_FEN
    assert payload["turn"] == "White"
    assert len(payload["evaluations"]) == 3
    names = {item["name"] for item in payload["evaluations"]}
    assert names == {"Material", "CNN", "Stockfish"}
    cnn = next(item for item in payload["evaluations"] if item["name"] == "CNN")
    assert cnn["label"] in {"White advantage", "Equal", "Black advantage", "Unavailable"}


def test_invalid_fen() -> None:
    try:
        evaluate_position("not a fen")
    except ValueError:
        return
    raise AssertionError("invalid FEN should raise ValueError")


def test_flask_routes() -> None:
    client = create_app().test_client()
    page = client.get("/")
    assert page.status_code == 200
    assert b"Chess Evaluation" in page.data
    assert b"Chessboard" in page.data

    response = client.post("/api/evaluate", json={"fen": START_FEN, "stockfish_depth": 1, "stockfish_time": 0.05})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["fen"] == START_FEN
    assert len(payload["evaluations"]) == 3

    bad = client.post("/api/evaluate", json={"fen": "bad fen"})
    assert bad.status_code == 400


def test_stockfish_optional() -> None:
    path = find_stockfish()
    assert path is None or Path(path).exists()
    nets = find_stockfish_nets()
    assert isinstance(nets, dict)


def test_stockfish_no_fake_zero() -> None:
    fen = "rnb1kbnr/pppp1ppp/8/4p3/4Q3/8/PPPP1PPP/RNB1KBNR b KQkq - 0 3"
    result = stockfish_result(fen, depth=8, time_limit=0.1)
    path = find_stockfish()
    if path and stockfish_setup_error(path):
        assert result["cp"] is None
        assert result["label"] == "Unavailable"
    else:
        assert result["cp"] is None or abs(float(result["cp"])) >= 200


def main() -> None:
    assert app is not None
    tests = [
        ("model artifact exists", test_model_exists),
        ("material heuristic", test_material_eval),
        ("three-way evaluation payload", test_evaluate_position),
        ("invalid FEN handling", test_invalid_fen),
        ("Flask routes", test_flask_routes),
        ("stockfish optional path", test_stockfish_optional),
        ("stockfish no fake zero", test_stockfish_no_fake_zero),
    ]
    ok = all(check(name, fn) for name, fn in tests)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
