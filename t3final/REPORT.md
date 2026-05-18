# Chess Evaluation CNN Project Report

## Project Overview

This project builds a machine learning product that evaluates chess positions from FEN notation. Given a board position, the model predicts a centipawn score and a simpler advantage label: `Black advantage`, `Equal`, or `White advantage`.

The final demo lets a user enter or choose a chess position and compare two approaches:

- a material-only baseline
- a trained CNN model

Positive centipawn values mean White is better. Negative values mean Black is better.

## Dataset Used

The project uses public chess evaluation datasets containing FEN positions and engine evaluations. The main supported sources are:

- Kaggle Chess Evaluations: https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations
- Lichess Chess Evaluations: https://www.kaggle.com/datasets/lichess/chess-evaluations
- Optional extra FEN/evaluation data: https://www.kaggle.com/datasets/dev102/chess-fens-evaluations-dataset

The large training run validated `23,978,399` FEN rows. The training split used `19,182,719` positions and the validation split used `2,397,839` positions. The final reported test metrics were measured on a heldout `100,000`-position test sample.

Input features:

- FEN string representing the chess position
- Encoded board as `12 x 8 x 8` piece planes
- 16 scalar features including side to move, castling rights, en passant availability, clocks, material balance, piece counts, queen count, and simple king-safety proxies

Target values:

- Stockfish-style centipawn evaluation
- Derived advantage bucket: Black / Equal / White

## Data Cleanup And Preprocessing

The dataset needed meaningful preprocessing before training:

- auto-detected FEN and evaluation columns across downloaded CSV files
- parsed centipawn and mate-style labels
- dropped forced-mate labels in the strongest training runs because mate labels are not directly comparable to centipawns
- validated FEN strings with `python-chess`
- removed invalid or unparsable rows
- clipped extreme evaluations to reduce instability from outliers
- transformed target centipawns with a `tanh` target transform
- encoded each board into 12 piece planes
- added scalar chess-state features
- split data into train, validation, and test sets
- cached encoded features using compressed arrays or packed memory-mapped bitboards
- used mirror augmentation by swapping colors and negating the target evaluation

## Model

The final model is a CNN based on ResNet34 adapted for chess boards.

Architecture summary:

```text
FEN
-> board encoder
-> 12x8x8 piece planes + 16 scalar features
-> ResNet34 visual backbone
-> dense head
-> centipawn regression output
-> Black/Equal/White bucket output
-> calibrated bucket decision
```

The model was trained from scratch, not from ImageNet weights. The ResNet stem was changed to accept 12 input channels and to preserve the small `8 x 8` board resolution.

Parameter count: `21,685,380`.

Important hyperparameters:

| Hyperparameter | Value |
|---|---:|
| Backbone | ResNet34 |
| Target transform | tanh |
| Target scale | 900 |
| Evaluation clip | 3000 cp |
| Dropout | 0.15 |
| Bucket loss weight | 0.8 |
| Bucket margin | 75 cp |
| Bucket class balancing | enabled |
| Mirror augmentation | enabled |

## Evaluation Metrics

This project uses both regression and classification metrics.

- MAE in centipawns: average size of the evaluation error
- RMSE in centipawns: penalizes large mistakes more strongly
- Pearson correlation: measures whether the model ranks positions similarly to the target engine
- Bucket accuracy: percentage of positions classified as Black / Equal / White correctly
- Clean bucket accuracy: bucket accuracy excluding ambiguous positions close to the +/-150 cp thresholds

Final CNN test metrics:

| Metric | Value |
|---|---:|
| MAE | 125.22 cp |
| RMSE | 277.30 cp |
| Pearson correlation | 0.873 |
| Bucket accuracy from regression | 88.78% |
| Direct bucket accuracy | 89.01% |
| Clean direct bucket accuracy | 95.12% |

Baseline comparison artifact:

| Model | MAE cp | RMSE cp | Pearson | Bucket Accuracy |
|---|---:|---:|---:|---:|
| Material baseline | 270.87 | 441.26 | 0.461 | 63.49% |
| CNN analysis checkpoint | 151.63 | 284.79 | 0.811 | 82.47% |

The final calibrated CNN improves beyond the analysis checkpoint and is the model used in the app.

## Performance Analysis

The CNN strongly outperforms the material-only baseline because it learns positional information beyond piece count. For example, it can use side to move, king safety, castling rights, pawn structure proxies, and board patterns.

The model performs best when positions are clearly better for one side or clearly equal. The hardest examples are positions near the classification boundaries around `-150 cp` and `+150 cp`, where even small centipawn errors can flip the bucket label. This is why clean bucket accuracy is much higher than full bucket accuracy.

The final calibration step improved direct bucket accuracy by adjusting the classifier's Black/Equal/White decision bias using validation predictions.

## Deployment

The local product app is a Flask app in `app.py`.

The Hugging Face deployment entrypoint is `hf_app.py`, a Gradio app designed for Spaces. The online demo avoids relying on a local Stockfish binary and compares the CNN with the material baseline.

To run locally:

```bash
uv run python app.py
```

To test the Hugging Face app locally after installing Gradio:

```bash
uv run python hf_app.py
```

## Limitations And Ethics

Limitations:

- The CNN is a static evaluator, not a chess engine.
- It does not search future moves.
- It can miss tactics, checkmates, and long forcing sequences.
- Labels come from engine evaluations, so the model learns to approximate the dataset rather than true perfect chess understanding.
- Positions near the Black/Equal/White thresholds are noisy and difficult to classify.

Ethics:

- The project uses public chess datasets.
- It does not use private or sensitive personal data.
- The model should be presented as an educational evaluator, not as a replacement for a real chess engine.

## Reflection

This project showed the full process of taking a model from data to product. The most important lesson was that model accuracy is not only about training longer. Data cleaning, target design, evaluation metrics, calibration, deployment, and clear reporting all mattered. The final product is useful because it connects the trained model to an interactive app where users can inspect real chess positions.

