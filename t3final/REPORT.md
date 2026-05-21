# Chess Evaluation CNN Project Report

## Project Overview

This project trains a neural network to evaluate chess positions from FEN notation. The model predicts a centipawn score, where positive values favor White and negative values favor Black. It also predicts a simple label: `Black advantage`, `Equal`, or `White advantage`.

The project includes training code, evaluation scripts, prediction scripts, a Flask web app, and a Hugging Face/Gradio app. The model is a static evaluator, not a full chess engine, because it does not search future moves.

## Dataset Used

The project uses public chess evaluation datasets made of chess positions and engine scores.

Main sources:

| Dataset | Use |
|---|---|
| Kaggle Chess Evaluations | Main FEN/evaluation source |
| Dev102 Chess FENs Evaluations | Extra FEN/evaluation rows |
| Random and tactic evaluation CSVs | Extra local coverage |

The active reported run found `24,507,255` raw rows and used `1,000,000` cleaned rows.

| Split | Samples |
|---|---:|
| Train | 800,000 |
| Validation | 100,000 |
| Test | 100,000 |

Input features:

| Feature type | Description |
|---|---|
| Board planes | `12 x 8 x 8` binary planes, one plane for each piece type and color |
| Extra features | 16 scalar features |
| FEN metadata | Turn, castling rights, en passant, halfmove clock, and fullmove number |
| Chess summaries | Material balance, piece counts, queen count, and simple king-safety features |

Target values:

| Target | Description |
|---|---|
| Centipawn score | Engine evaluation clipped to a fixed range |
| Advantage bucket | Black advantage, Equal, or White advantage |

The test set is not perfectly balanced.

| Class | Test Share |
|---|---:|
| Equal | 47.38% |
| White advantage | 28.77% |
| Black advantage | 23.84% |

This means accuracy should be interpreted carefully because the `Equal` class is the largest class.

## Data Cleanup And Preprocessing

The project performed these cleanup steps:

| Step | Explanation |
|---|---|
| Column detection | The loader auto-detects FEN and evaluation columns across CSV files |
| Missing value removal | Rows with missing FENs or evaluations are removed |
| FEN validation | Invalid chess positions are removed using `python-chess` |
| Evaluation parsing | Numeric scores and mate-style labels are parsed |
| Mate label filtering | Forced-mate labels are dropped in the stronger training runs |
| Clipping | Extreme evaluations are clipped to reduce outlier impact |
| Target transform | Centipawn values are transformed with `tanh` |
| Board encoding | FENs are converted into `12 x 8 x 8` piece planes |
| Feature scaling | Extra scalar features are normalized into small numeric ranges |
| Data split | Data is split into train, validation, and test sets |
| Caching | Encoded features are cached to speed up later runs |
| Mirror augmentation | Some positions are color-swapped and their target score is negated |

The active run dropped `528,856` mate-label rows before training/evaluation.

## Model Information

The final model is based on ResNet34. It was trained from scratch, not from ImageNet pretrained weights.

Parameter count: `21,685,380`.

Important hyperparameters:

| Hyperparameter | Value |
|---|---:|
| Backbone | ResNet34 |
| Batch size | 4096 |
| Epochs configured | 40 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Dropout | 0.15 |
| Head hidden size | 512 |
| Target transform | tanh |
| Target scale | 900 |
| Evaluation clip | 3000 cp |
| Huber beta | 0.25 |
| Advantage loss weight | 0.5 |
| Bucket loss weight | 1.0 |
| Gradient clipping | 1.0 |
| Mirror augmentation | Enabled |

## Architecture

The architecture is:

```text
FEN
-> board encoder
-> 12x8x8 piece planes
-> 16 extra scalar features
-> modified ResNet34 backbone
-> dense prediction head
-> centipawn regression output
-> 3-class bucket output
```

The ResNet34 architecture was changed for chess boards. The first convolution accepts 12 input channels instead of 3 image channels. The initial max-pooling layer is removed because an `8 x 8` chess board is very small, and early downsampling would lose too much board information.

The extra scalar features are joined with the ResNet output before the final prediction layers. This helps the model use information that is easier to express as numbers, such as castling rights, side to move, and material balance.

## Metrics Of Evaluation

The project uses both regression and classification metrics.

| Metric | Why It Was Used |
|---|---|
| MAE | Easy to understand average centipawn error |
| RMSE | Penalizes large mistakes more strongly |
| Pearson correlation | Measures whether predictions follow the same trend as engine scores |
| Bucket accuracy | Measures Black/Equal/White classification quality |
| Direct bucket accuracy | Measures the direct 3-class classifier head |

Final test metrics from `runs/evaluation_metrics.json`:

| Metric | Value |
|---|---:|
| MAE | 130.44 cp |
| RMSE | 284.04 cp |
| Pearson correlation | 0.860 |
| Bucket accuracy from regression | 86.87% |
| Direct bucket accuracy | 87.19% |
| Test samples | 100,000 |

Baseline comparison artifact:

| Model | MAE cp | RMSE cp | Pearson | Bucket Accuracy |
|---|---:|---:|---:|---:|
| Material baseline | 270.87 | 441.26 | 0.461 | 63.49% |
| CNN checkpoint | 151.63 | 284.79 | 0.811 | 82.47% |

The CNN performs much better than the material-only baseline because it can learn board patterns, not only piece values.

## Analysis Of Model Performance

The model gives useful static evaluations. Its average error is about `130 cp`, which is roughly a little more than one pawn. Its correlation of `0.860` shows that it usually ranks positions in the same direction as the engine labels.

The bucket accuracy is also strong. The model is best when a position is clearly winning, losing, or equal. It is weaker near the `+/-150 cp` bucket boundary, where a small centipawn error can change the class label.

The CNN is much stronger than a material baseline. This shows that the model learned positional patterns such as activity, king safety, side to move, and board structure.

## Limitations And Ethics

Limitations:

| Limitation | Explanation |
|---|---|
| No move search | The model evaluates only the current position |
| Tactical mistakes | It can miss tactics, checkmates, and forcing lines |
| Label dependency | It learns from engine labels, so errors or bias in labels affect the model |
| Boundary noise | Positions near `+/-150 cp` are hard to bucket correctly |
| Not a chess engine | It should not be treated as a replacement for Stockfish |

Ethics:

| Topic | Explanation |
|---|---|
| Data privacy | The project uses public chess-position data, not personal user data |
| Intended use | The model is best used for learning, experimentation, and demonstration |
| Misuse risk | It should not be presented as perfect chess understanding |

## Reflection

This project showed that building a useful model is more than training a network. Data cleaning, FEN validation, target design, feature encoding, evaluation metrics, and deployment all affected the final result.

The biggest lesson was that a static model can learn a lot about chess positions, but it still cannot replace search. The model is useful because it is fast and visual, while Stockfish is stronger because it calculates future moves.

