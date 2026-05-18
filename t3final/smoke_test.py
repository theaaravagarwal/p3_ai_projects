from __future__ import annotations

import sys

import torch

from src.board_encoding import EXTRA_FEATURES_DIM, encode_fen, mirror_encoded_position, validate_fen
from src.dataset import parse_evaluation_detail
from src.model import create_model


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def check(name: str, fn) -> bool:
    try:
        fn()
        print(f"PASS {name}")
        return True
    except Exception as exc:
        print(f"FAIL {name}: {exc}")
        return False


def test_encoding() -> None:
    board, extras = encode_fen(START_FEN)
    assert board.shape == (12, 8, 8)
    assert extras.shape == (EXTRA_FEATURES_DIM,)
    assert board.dtype.name == "float32"
    assert float(board.sum()) == 32.0


def test_model_forward() -> None:
    model = create_model(model_name="resnet18", extra_features_dim=EXTRA_FEATURES_DIM)
    model.eval()
    board, extras = encode_fen(START_FEN)
    with torch.no_grad():
        pred = model(torch.from_numpy(board).unsqueeze(0), torch.from_numpy(extras).unsqueeze(0))
    assert pred.shape == (1,)


def test_multitask_model_forward() -> None:
    model = create_model(model_name="resnet18", extra_features_dim=EXTRA_FEATURES_DIM, num_buckets=3)
    model.eval()
    board, extras = encode_fen(START_FEN)
    with torch.no_grad():
        pred, bucket_logits = model(torch.from_numpy(board).unsqueeze(0), torch.from_numpy(extras).unsqueeze(0))
    assert pred.shape == (1,)
    assert bucket_logits.shape == (1, 3)


def test_mirror_encoding() -> None:
    board, extras = encode_fen(START_FEN)
    mirrored_board, mirrored_extras, mirrored_target = mirror_encoded_position(board, extras, 0.25)
    assert mirrored_board.shape == board.shape
    assert mirrored_extras.shape == extras.shape
    assert float(mirrored_board.sum()) == 32.0
    assert mirrored_target == -0.25
    assert mirrored_extras[0] == 0.0


def test_invalid_fen() -> None:
    assert not validate_fen("not a fen")
    try:
        encode_fen("not a fen")
    except ValueError:
        return
    raise AssertionError("encode_fen should reject invalid FEN")


def test_parse_mate_labels() -> None:
    mate = parse_evaluation_detail("#3")
    assert mate.is_mate
    assert mate.cp > 0
    cp = parse_evaluation_detail("-42")
    assert not cp.is_mate
    assert cp.cp == -42


def main() -> None:
    tests = [
        ("imports and FEN encoding", test_encoding),
        ("model fake forward pass", test_model_forward),
        ("multitask model fake forward pass", test_multitask_model_forward),
        ("mirror augmentation encoding", test_mirror_encoding),
        ("invalid FEN handling", test_invalid_fen),
        ("mate label parsing", test_parse_mate_labels),
    ]
    ok = all(check(name, fn) for name, fn in tests)
    if ok:
        print("Smoke test complete.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
