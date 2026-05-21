# Chess Evaluation CNN Project Report

## Project Overview

This project predicts chess position evaluations from FEN notation. The final demo in `hft3final` uses a trained CNN to output a centipawn score and an advantage label: `Black advantage`, `Equal`, or `White advantage`.

The app also compares the CNN with a simple material baseline and optional Stockfish. The CNN is a static evaluator, so it does not search future moves like a chess engine.

## Dataset Used

The project uses public chess evaluation data with FEN positions and engine scores.

| Item | Value |
|---|---:|
| Raw rows found | 24,507,255 |
| Clean rows used in reported run | 1,000,000 |
| Train samples | 800,000 |
| Validation samples | 100,000 |
| Test samples | 100,000 |

Inputs:

| Input | Description |
|---|---|
| Board planes | `12 x 8 x 8` piece encoding |
| Extra features | 16 chess-state features |

Targets:

| Target | Description |
|---|---|
| Centipawn score | Engine evaluation, positive for White and negative for Black |
| Bucket label | Black advantage, Equal, or White advantage |

The test data is somewhat imbalanced:

| Class | Share |
|---|---:|
| Equal | 47.38% |
| White advantage | 28.77% |
| Black advantage | 23.84% |

This matters because a model can look better if it mostly predicts the largest class, so bucket accuracy should be compared with other metrics too.

## Data Cleanup And Preprocessing

The main cleanup steps were:

| Step | Purpose |
|---|---|
| FEN validation | Removed invalid chess positions |
| Evaluation parsing | Converted engine labels into numeric scores |
| Mate-label filtering | Removed forced-mate labels from the stronger runs |
| Clipping | Limited extreme scores to `+/-3000 cp` |
| Normalization | Used a `tanh` target transform with scale `900` |
| Encoding | Converted FENs into board planes and scalar features |
| Augmentation | Mirrored positions by swapping colors and negating the score |

The active run dropped `528,856` mate-label rows. This improved label quality because mate scores are harder to compare directly with normal centipawn scores.

## Model Information

The final deployed model is the `hft3final/models/best_model.pt` checkpoint.

| Item | Value |
|---|---:|
| Model | ResNet34-based CNN |
| Training | From scratch |
| Parameters | 21,702,440 |
| Input | `12 x 8 x 8` board planes + 16 features |
| Outputs | Centipawn regression + 3-class bucket head |

Key hyperparameters:

| Hyperparameter | Value |
|---|---:|
| Batch size | 4096 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Dropout | 0.15 |
| Huber beta | 0.25 |
| Bucket loss weight | 0.8 |
| Bucket margin | 75 cp |
| Gradient clipping | 1.0 |

## Architecture

The model uses ResNet34, but it is adapted for chess instead of images. The first layer accepts 12 board channels instead of RGB channels. The early max-pooling layer is removed because a chess board is only `8 x 8`, so aggressive downsampling would lose important square-level information.

The final layers combine board features with 16 scalar features, then produce both a numeric evaluation and a bucket label. This is useful because the model must estimate both score size and practical advantage category.

## Metrics Of Evaluation

Metrics were chosen to measure both numeric accuracy and label quality.

| Metric | Why Used |
|---|---|
| MAE | Simple average centipawn error |
| RMSE | Shows large mistakes more clearly |
| Bucket accuracy | Measures Black/Equal/White correctness |
| Clean bucket accuracy | Measures less ambiguous positions away from bucket edges |

Final deployed demo stats reported in `hft3final`:

| Metric | Value |
|---|---:|
| Test MAE | 125.22 cp |
| Direct bucket accuracy | 89.01% |
| Clean-margin direct bucket accuracy | 95.12% |

Baseline comparison:

| Model | MAE cp | Bucket Accuracy |
|---|---:|---:|
| Material baseline | 270.87 | 63.49% |
| CNN checkpoint | 151.63 | 82.47% |

The CNN is much stronger than the material baseline, which shows it learned more than piece counting.

## Analysis Of Model Performance

The model performs well for a static evaluator. A `125.22 cp` MAE means the average error is a little over one pawn. The clean bucket accuracy of `95.12%` shows the model is strong when positions are clearly better for one side or clearly equal.

The weakest area is positions near the `+/-150 cp` bucket boundary. Small score errors can flip those labels, so full bucket accuracy is naturally lower than clean-margin accuracy.

Overall quality is strongest for fast evaluation and broad position understanding. It is weaker for tactics, forced mates, and positions that require calculating several moves ahead.

## Limitations And Ethics

Limitations:

| Limitation | Explanation |
|---|---|
| No search | The model only evaluates the current board |
| Tactical weakness | It may miss forcing lines and checkmates |
| Engine-label dependence | It learns from engine-generated labels |
| Class imbalance | Equal positions are the largest class |

Ethics:

| Topic | Explanation |
|---|---|
| Privacy | Uses public chess-position data |
| Use | Best for education and experimentation |
| Honesty | Should not be presented as a full chess engine |

## Reflection

This project showed that data quality and evaluation design matter as much as model architecture. Cleaning FENs, removing noisy mate labels, choosing useful features, and comparing against a baseline made the final result easier to understand.

The main lesson is that a CNN can learn useful chess evaluation patterns, but search is still necessary for engine-level play.

