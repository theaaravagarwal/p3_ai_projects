from __future__ import annotations

from typing import Tuple

import chess
import numpy as np


PIECE_TO_CHANNEL = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

EXTRA_FEATURES_DIM = 16
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


def validate_fen(fen: str) -> bool:
    if not isinstance(fen, str) or not fen.strip():
        return False
    try:
        chess.Board(fen.strip())
        return True
    except ValueError:
        return False


def encode_fen(fen: str, include_extras: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Encode FEN as 12x8x8 piece planes plus scalar game-state features."""
    try:
        board = chess.Board(fen.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid FEN: {fen}") from exc

    planes = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        channel = PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]
        rank = 7 - chess.square_rank(square)
        file = chess.square_file(square)
        planes[channel, rank, file] = 1.0

    if not include_extras:
        return planes, np.zeros((0,), dtype=np.float32)

    white_material = 0.0
    black_material = 0.0
    piece_count = 0.0
    pawn_count = 0.0
    queen_count = 0.0
    for piece in board.piece_map().values():
        value = PIECE_VALUES[piece.piece_type]
        if piece.color == chess.WHITE:
            white_material += value
        else:
            black_material += value
        piece_count += 1.0
        pawn_count += 1.0 if piece.piece_type == chess.PAWN else 0.0
        queen_count += 1.0 if piece.piece_type == chess.QUEEN else 0.0

    ep_available = 1.0 if board.ep_square is not None else 0.0
    halfmove = min(float(board.halfmove_clock), 100.0) / 100.0
    fullmove = min(float(board.fullmove_number), 200.0) / 200.0
    white_king_safety = _king_safety_proxy(board, chess.WHITE)
    black_king_safety = _king_safety_proxy(board, chess.BLACK)
    extras = np.array(
        [
            1.0 if board.turn == chess.WHITE else 0.0,
            1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
            ep_available,
            halfmove,
            fullmove,
            (white_material - black_material) / 39.0,
            white_material / 39.0,
            black_material / 39.0,
            piece_count / 32.0,
            pawn_count / 16.0,
            queen_count / 2.0,
            white_king_safety,
            black_king_safety,
        ],
        dtype=np.float32,
    )
    return planes, extras


def _king_safety_proxy(board: chess.Board, color: chess.Color) -> float:
    king_square = board.king(color)
    if king_square is None:
        return 0.0
    friendly_nearby = 0
    total_nearby = 0
    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            file = king_file + df
            rank = king_rank + dr
            if 0 <= file < 8 and 0 <= rank < 8:
                total_nearby += 1
                piece = board.piece_at(chess.square(file, rank))
                if piece is not None and piece.color == color:
                    friendly_nearby += 1
    return friendly_nearby / max(total_nearby, 1)


def mirror_encoded_position(
    board: np.ndarray,
    extras: np.ndarray,
    target: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    mirrored_board = np.empty_like(board)
    mirrored_board[:6] = board[6:12, ::-1, :]
    mirrored_board[6:12] = board[:6, ::-1, :]

    mirrored_extras = extras.copy()
    if len(mirrored_extras) >= 8:
        mirrored_extras[0] = 1.0 - mirrored_extras[0]
        mirrored_extras[1], mirrored_extras[2], mirrored_extras[3], mirrored_extras[4] = (
            extras[3],
            extras[4],
            extras[1],
            extras[2],
        )
    if len(mirrored_extras) >= EXTRA_FEATURES_DIM:
        mirrored_extras[8] = -extras[8]
        mirrored_extras[9], mirrored_extras[10] = extras[10], extras[9]
        mirrored_extras[14], mirrored_extras[15] = extras[15], extras[14]

    mirrored_target = -target if target is not None else None
    return np.ascontiguousarray(mirrored_board), mirrored_extras, mirrored_target


def pretty_label_from_eval(cp: float) -> str:
    if cp > 150:
        return "White advantage"
    if cp < -150:
        return "Black advantage"
    return "Equal"
