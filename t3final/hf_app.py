from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chess
import gradio as gr

from src.board_encoding import validate_fen
from src.predict import ChessEvaluationPredictor


MODEL_PATH = Path("models/best_model.pt")
MODEL_NAME = "CNN"
EXAMPLES = [
    ["Starting position", chess.STARTING_FEN],
    ["Italian opening", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 3"],
    ["White up a queen", "rnb1kbnr/pppp1ppp/8/4p3/4Q3/8/PPPP1PPP/RNB1KBNR b KQkq - 0 3"],
    ["Black material edge", "rnbqkbnr/pppppppp/8/8/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"],
    ["Bare kings", "8/8/8/4k3/8/8/8/4K3 w - - 0 1"],
]
PIECE_VALUES_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


@lru_cache(maxsize=1)
def predictor() -> ChessEvaluationPredictor:
    return ChessEvaluationPredictor(MODEL_PATH, device="cpu")


def material_cp(board: chess.Board) -> float:
    score = 0
    for piece in board.piece_map().values():
        value = PIECE_VALUES_CP[piece.piece_type]
        score += value if piece.color == chess.WHITE else -value
    return float(score)


def label_from_cp(cp: float | None) -> str:
    if cp is None:
        return "Unavailable"
    if cp > 150:
        return "White advantage"
    if cp < -150:
        return "Black advantage"
    return "Equal"


def format_cp(cp: float | None) -> str:
    if cp is None:
        return "Unavailable"
    sign = "+" if cp > 0 else ""
    return f"{sign}{cp:.1f} cp"


def evaluate_fen(fen: str) -> tuple[str, str, dict[str, float], str]:
    fen = fen.strip()
    if not validate_fen(fen):
        raise gr.Error("Invalid FEN. Paste a legal chess FEN.")
    if not MODEL_PATH.exists():
        raise gr.Error("models/best_model.pt is missing. Upload the trained checkpoint to the Space.")

    board = chess.Board(fen)
    cnn = predictor().predict(board.fen())
    cnn_cp = float(cnn["centipawns"])
    mat_cp = material_cp(board)
    rows = [
        ["Material", format_cp(mat_cp), label_from_cp(mat_cp)],
        [MODEL_NAME, format_cp(cnn_cp), str(cnn["label"])],
    ]
    probabilities = cnn.get("bucket_probabilities") or {}
    prob_chart = {key.replace(" advantage", ""): float(value) for key, value in probabilities.items()}
    status = "Game over" if board.is_game_over() else ("Check" if board.is_check() else "Live")
    summary = (
        f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}\n"
        f"Status: {status}\n"
        f"Normalized model output: {float(cnn['normalized']):.4f}"
    )
    return board.fen(), summary, prob_chart, rows


def fill_example(name: str) -> str:
    for example_name, fen in EXAMPLES:
        if example_name == name:
            return fen
    return chess.STARTING_FEN


with gr.Blocks(title="Chess Evaluation") as demo:
    gr.Markdown(
        """
        # Chess Evaluation
        Enter a chess FEN and compare a material baseline with the trained CNN.
        Positive centipawns mean White is better; negative centipawns mean Black is better.
        """
    )
    with gr.Row():
        example = gr.Dropdown([name for name, _ in EXAMPLES], value="Starting position", label="Example")
        load_example = gr.Button("Load Example", variant="secondary")
    fen_input = gr.Textbox(value=chess.STARTING_FEN, label="FEN", lines=2)
    evaluate = gr.Button("Evaluate", variant="primary")
    with gr.Row():
        normalized_fen = gr.Textbox(label="Normalized FEN", lines=2)
        position_info = gr.Textbox(label="Position", lines=4)
    probabilities = gr.Label(label="CNN bucket probabilities")
    results = gr.Dataframe(
        headers=["Evaluator", "Centipawns", "Label"],
        datatype=["str", "str", "str"],
        label="Evaluation Results",
        interactive=False,
    )

    load_example.click(fill_example, inputs=example, outputs=fen_input)
    evaluate.click(evaluate_fen, inputs=fen_input, outputs=[normalized_fen, position_info, probabilities, results])
    fen_input.submit(evaluate_fen, inputs=fen_input, outputs=[normalized_fen, position_info, probabilities, results])


if __name__ == "__main__":
    demo.launch()
