---
title: Chess Evaluation CNN
emoji: ♟️
colorFrom: green
colorTo: blue
sdk: docker
python_version: 3.11
app_port: 7860
pinned: false
---

# Chess Evaluation CNN

This Space predicts a static chess position evaluation from FEN notation with an interactive chessboard.

The demo compares:

- `Material baseline`: simple piece-count centipawn score
- `CNN`: trained ResNet34-based neural network checkpoint in `models/best_model.pt`
- optional Stockfish if a binary is available in the container

Positive centipawns mean White is better. Negative centipawns mean Black is better.

## Model

- Input: `12 x 8 x 8` board planes plus 16 scalar chess-state features
- Backbone: ResNet34 adapted for chess boards
- Outputs: centipawn regression head and Black/Equal/White bucket head
- Test MAE: `125.22` cp
- Test direct bucket accuracy: `89.01%`
- Clean-margin direct bucket accuracy: `95.12%`

## Limitations

The CNN is a static evaluator. It does not search legal move trees like a chess engine, so it can miss tactics, forced mates, and long-term strategic ideas. Stockfish was used to create labels and as a local comparison tool, but the deployed CNN does not call Stockfish during inference.
