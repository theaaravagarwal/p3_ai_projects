# Chess Evaluation with a Convolutional Neural Network

This project trains a supervised convolutional neural network to estimate the static evaluation of a chess position from FEN notation. It is not a chess engine: it does not search moves and it does not call Stockfish during inference.

Positive centipawn predictions mean White is better, negative predictions mean Black is better, and values near zero are approximately equal. The demo app also maps scores to `White advantage`, `Equal`, or `Black advantage`.

## Dataset

The project expects the Kaggle Chess Evaluations dataset:

https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations

The dataset contains FEN positions with Stockfish depth-22 evaluations. CSV file names can vary, so the loader searches all CSVs under `data/chess_evaluations/` and auto-detects common FEN and evaluation column names.

An optional extra Kaggle dataset is also supported:

https://www.kaggle.com/datasets/dev102/chess-fens-evaluations-dataset

This adds roughly 8M more `FEN`/`Evaluation` rows and is public-domain according to its Kaggle card. Its labels are Stockfish depth 0, so treat it as extra coverage rather than a clean replacement for the depth-22 dataset.

For much larger experiments, the Lichess chess evaluations corpus is also supported:

https://www.kaggle.com/datasets/lichess/chess-evaluations

This dataset is far larger than the default downloads, so fetch it explicitly with `--dataset lichess` or `--dataset mega`.

## Project Structure

```text
t3final/
  data/                 # local dataset files, ignored by git
  models/               # trained checkpoints, ignored by git
  runs/                 # logs, plots, metrics, ignored by git
  outputs/              # optional generated outputs, ignored by git
  src/
    board_encoding.py
    dataset.py
    model.py
    train.py
    evaluate.py
    predict.py
    visualization.py
    utils.py
  scripts/download_dataset.py
  app.py
  smoke_test.py
```

## Setup

```bash
uv sync
uv run python smoke_test.py
```

On Linux, `uv sync` installs `torch==2.10.0` and `torchvision==0.25.0` from PyTorch's CUDA 12.8 wheel index so the training machine can use an NVIDIA driver that reports CUDA 12.8. If the environment already has the wrong Torch wheel, refresh it with:

```bash
uv sync --reinstall-package torch
```

`requirements.txt` is also provided for pip-based environments:

```bash
pip install -r requirements.txt
```

Python `>=3.11,<3.13` is required. `.python-version` is set to `3.11`.

## Dataset Download

With Kaggle API credentials configured:

```bash
uv run python scripts/download_dataset.py
```

Download only the extra dataset:

```bash
uv run python scripts/download_dataset.py --dataset extra
```

Download the huge Lichess evaluations corpus:

```bash
uv run python scripts/download_dataset.py --dataset lichess
```

If credentials are missing, the script prints manual setup instructions. Place downloaded CSV files under:

```text
data/chess_evaluations/
```

## Training

Recommended presets:

```bash
uv run python -m src.train --preset smoke --fresh
uv run python -m src.train --preset quick --fresh
uv run python -m src.train --preset strong --fresh
uv run python -m src.train --preset max --fresh
```

`--preset max` uses every valid row it can find under `data/chess_evaluations/`. For manual runs, `--max-samples 0` also means no row cap.

Remote high-throughput run:

```bash
uv run python -m src.train --preset max --model-name resnet34 --preprocess-workers 16 --num-workers 16 --fresh
```

Supported model backbones:

- `--model-name resnet18` default
- `--model-name resnet34`
- `--model-name efficientnet_b0`

Useful capacity knobs for the custom `cnn` option are:

- `--model-width`: convolution channels, default `192`
- `--model-depth`: residual block count, default `8`
- `--dropout`: regularization, default `0.15`
- `--head-hidden`: first dense layer width, default `512`
- `--huber-beta`: SmoothL1 transition point on normalized evals, default `0.25`

The default target transform is `--target-transform tanh` with `--target-scale 600` and `--eval-clip 1500`. The stronger presets use a wider `--eval-clip 3000`, `--target-scale 900`, drop forced-mate labels, and apply a small advantage-weighted regression loss so decisive positions are not compressed as aggressively. Training uses extra scalar features by default; pass `--no-extra-features` to disable them. Training also uses random mirror augmentation by default: board ranks are flipped, colors and castling rights are swapped, and the target evaluation is negated. Pass `--no-mirror-augment` to disable it.

The `strong` and `max` presets also train a direct 3-class bucket head for `Black advantage`, `Equal`, and `White advantage`. This is controlled by `--bucket-loss-weight`; set it to `0` for regression-only training. The bucket loss is margin-aware by default: ambiguous labels close to the `+/-150cp` thresholds are ignored for classification while still contributing to regression. `--bucket-margin-cp` controls that ignored band, and `--bucket-class-balance` enables inverse-frequency class weights for the bucket loss. When the bucket head is enabled, evaluation reports both regression-derived `bucket_accuracy` and direct classifier `bucket_accuracy_direct`, plus clean-margin metrics such as `bucket_accuracy_direct_clean`. Use `--warm-start models/best_model.pt` when adding the bucket head to an existing regression checkpoint so the weights load but the optimizer and learning-rate schedule restart cleanly.

If a run is interrupted, restart from the last complete checkpoint:

```bash
uv run python -m src.train --preset max --model-name resnet34 --batch-size 4096 --preprocess-workers 16 --num-workers 16 --resume models/latest_model.pt
```

`--compile` is optional. Use it only if the local GPU/PyTorch stack is stable with TorchInductor; otherwise eager training is usually more reliable.
ResNet18/34 are adapted for the 8x8 board input with a 3x3 stride-1 stem and no initial maxpool. Plain ImageNet-style downsampling is too destructive for chess boards.

Quick verification run:

```bash
uv run python -m src.train --preset smoke --fresh
```

Training saves:

- `models/best_model.pt`
- `models/final_model.pt`
- `runs/training_log.csv`
- `runs/training_plot.png`
- `runs/training_loss.png`
- `runs/validation_metrics.png`
- `runs/bucket_accuracy.png`
- `runs/training_config.json`
- `runs/data_summary.json`

Validation and FEN encoding are cached under `data/cache/` by default. The first run for a given dataset/config still validates and encodes, but later runs with the same CSV files, seed, sample count, and eval clip reuse cached data. Small encoded caches are saved as compressed `.npz` files. Large encoded caches are saved as memory-mapped packed bitboards so training can random-access them without loading everything into RAM; this is much smaller than full `12x8x8` board-plane `.npy` caches. Pass `--no-cache` to force a rebuild.

Cache files can get large because each sample-size/config split may cache pre-encoded board tensors. New encoded caches are compressed `.npz` files. To inspect and clean cache storage:

```bash
uv run python scripts/manage_cache.py --summary
uv run python scripts/manage_cache.py --compress --delete-legacy
uv run python scripts/manage_cache.py --prune-keep-latest 6 --dry-run
uv run python scripts/manage_cache.py --prune-keep-latest 6
```

Use prune only when you are fine rebuilding older sample/config caches later.

## Evaluation

```bash
uv run python -m src.evaluate --model models/best_model.pt
```

Evaluation reports MAE, MSE, RMSE, Pearson correlation, bucket accuracy, and a confusion matrix for `Black advantage`, `Equal`, and `White advantage`.

Compare the model against a material-only baseline from an existing predictions artifact:

```bash
uv run python scripts/baseline_comparison.py \
  --predictions runs_eval_best_model_calibrated/predictions.csv \
  --output-dir outputs/baseline_comparison
```

If the raw dataset is available locally, the same script can compute the baseline directly from the selected split:

```bash
uv run python scripts/baseline_comparison.py \
  --max-samples 1000000 \
  --split test \
  --cnn-predictions runs_eval_best_model_calibrated/predictions.csv
```

## Analysis Report

Generate report-ready metrics, plots, worst examples, best examples, and board images:

```bash
uv run python -m src.analyze_results --model models/best_model.pt
```

Outputs are saved under `outputs/analysis/`, including:

- `metrics_summary.json`
- `metrics_summary.txt`
- `predicted_vs_actual.png`
- `error_histogram.png`
- `residual_plot.png`
- `confusion_matrix.png`
- `bucket_accuracy_bar.png`
- `error_by_eval_range.png`
- `worst_predictions.csv`
- `best_predictions.csv`
- `sample_prediction_boards/`

Compare checkpoints on the same test split:

```bash
uv run python -m src.compare_models --models models/best_model.pt models/final_model.pt
```

This saves `outputs/model_comparison.csv` and `outputs/model_comparison.png`.

Run an overnight bucket-head sweep:

```bash
uv run python scripts/overnight_bucket_sweep.py --epochs 24 --patience 5
```

The sweep trains several `--bucket-loss-weight` / `--bucket-margin-cp` combinations from `models/best_model.pt`, evaluates each best checkpoint, and writes `sweeps/bucket/leaderboard.json`.

## Prediction

```bash
uv run python -m src.predict --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --model models/best_model.pt
```

The prediction command encodes the FEN, runs the CNN once, converts the normalized output back to centipawns, and prints the advantage label.

## Phase 2 Flask App

Run the product smoke test:

```bash
uv run python product_smoke_test.py
```

Run the app:

```bash
uv run python app.py
```

Open the printed local URL, usually `http://127.0.0.1:8501`.

The app uses a Flask Python backend and a JavaScript chessboard frontend. You play on one draggable board, paste/load FENs, and the server returns three evaluations for the same current position:

- material-only heuristic
- the trained CNN in `models/best_model.pt`
- optional local Stockfish UCI reference

The CNN does not use Stockfish during inference. Stockfish is only used as a separate comparison result when a binary is available, such as `/opt/homebrew/bin/stockfish`, `/usr/local/bin/stockfish`, or `stockfish` on `PATH`.

Current demo model:

- `models/best_model.pt`
- CNN
- ResNet34 backbone
- MAE `125.22` cp
- Direct bucket accuracy `89.01%`
- Clean direct bucket accuracy `95.12%`

## Hugging Face Spaces

`hf_app.py` is a lightweight Gradio deployment entrypoint for Hugging Face Spaces. It compares the trained CNN with a material-only baseline and avoids requiring a Stockfish binary in the hosted environment.

Local check:

```bash
uv run python hf_app.py
```

When creating the Space, use:

- SDK: `Gradio`
- App file: `hf_app.py`
- Model artifact: upload `models/best_model.pt`

`README_HF.md` contains Space-ready metadata and a shorter model card. Copy it to `README.md` in the Space repo if deploying the app in a separate Hugging Face repository.

## Model

The default backbone is torchvision ResNet18 adapted to 12 board planes shaped `12 x 8 x 8`, one channel per piece type and color. ResNet34 and EfficientNet-B0 are also available. The model concatenates optional scalar features after the visual backbone: side to move, castling rights, en passant availability, clocks, material balance, material totals, piece counts, pawn/queen counts, and simple king-safety proxies.

The custom `cnn` option is still available and uses:

- Conv2d `12 -> width`, BatchNorm, SiLU
- `depth` residual Conv-BatchNorm-SiLU blocks
- Flatten and concatenate scalar features
- Linear `-> head_hidden`, SiLU, Dropout
- Linear `-> head_hidden / 2`, SiLU, Dropout
- Linear `-> 1`

The default target is `tanh(cp / 600)` after clipping to `[-1500, 1500]`; evaluation and prediction invert this back to centipawns. Mate scores such as `#3` and `#-2` are converted to large centipawn equivalents before clipping.

## Remote Training Sync

Push source/config/docs to the remote machine, excluding generated artifacts:

```bash
./push.sh
```

Train remotely, then pull generated artifacts only:

```bash
./pull.sh
```

Remote:

```text
software@100.64.0.25:/home/software/Documents/utils/aarav/t3final/
```

## Limitations

- The model predicts static evaluations only. It does not choose moves.
- It learns from Stockfish labels but does not run Stockfish at inference time.
- Accuracy depends heavily on dataset size, training time, and hardware.
- Scores are clipped by default to `[-1500, 1500]` centipawns, so decisive positions are compressed.
- FEN validity is checked, but unusual dataset formats may require `--fen-column` or `--eval-column`.
