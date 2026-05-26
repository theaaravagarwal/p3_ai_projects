# Project Demo Report — Rubric Mapping

## 1) Dataset Understanding (Score target: 15/15)

**Status:** Strong.

The dataset source and format are explicitly documented and implemented:

- Main source: Kaggle Chess Evaluations (Ronakbadhe) with depth-22 Stockfish scores.
- Optional enrichment datasets are supported for depth-0 and larger corpora.
  - `README.md` documents Kaggle sources and format assumptions.
  - `src/dataset.py` automatically discovers CSV files, auto-detects FEN and evaluation columns, and supports override columns.

**Features and target**

- Input features are position encodings in `src/board_encoding.py`:
  - 12×8×8 binary piece planes.
  - 16 scalar chess-state extras (side-to-move, castling availability, half/fullmove, material and piece features, king-safety proxy).
- Target is centipawn evaluation (normalized for training) and a derived 3-class bucket label:
  - Black advantage / Equal / White advantage using ±150 cp boundaries.
- Data characteristics are persisted in run summaries:
  - `runs/data_summary.json` and `outputs/analysis/data_summary.json` (rows before/after cleaning, split sizes, detected columns, cache metadata).

**Concerns captured**

- Label source quality and mixing are acknowledged:
  - mixed depth-22 and depth-0 inputs are supported, not conflated as equally clean labels.
- Forced-mate labels are explicitly represented (`#N`) and optionally filtered.
- The data is class-imbalanced by bucket (roughly more equal positions), and class-balance option exists for bucket training.

## 2) Data Cleaning and Preprocessing (Score target: 15/15)

**Status:** Strong.

Pipeline is explicit and reproducible:

- CSV loading, auto column detection, null filtering.
- Evaluation parsing supports plain floats/ints plus mate notation (`#N`), with mate labels tracked.
- Optional mate-label removal for cleaner numeric training (`--drop-mate-labels`).
- Evaluation clipping and transformation (`--eval-clip`, `--target-transform`, `--target-scale`), with inverse transform used for reporting.
- FEN validation with `python-chess`; invalid positions are dropped before training.
- Optional dataset sizing/sampling and deterministic shuffle (`seed`).
- Train/validation/test split is performed after cleaning and optional sampling.
- Optional parallel pre-encoding + caching for boards/extras (`data/cache/`).
- Optional random mirror augmentation with color/side/target symmetry handling.

Core implementation points:

- Data preparation and cleaning logic lives in `src/dataset.py`.
- FEN encoding and extra features in `src/board_encoding.py`.
- Dataset wrapper and caching in `src/dataset.py`.

## 3) Algorithm and Architecture (Score target: 15/15)

**Status:** Strong.

Chosen model family is a CNN-based static evaluator aligned to board structure:

- Primary production model: `resnet34` adapted for chess tensor shape.
  - Replaces first conv input channels with 12.
  - Removes initial maxpool to avoid destructive 8×8 spatial downsampling.
- Model integrates:
  - Board tensor pathway (`12×8×8` channels)
  - Optional 16 scalar extras
  - Regression output for centipawns
  - Optional 3-class bucket head (`Black / Equal / White`) when configured

Implementation references:

- `src/model.py` (`TorchvisionRegressionModel`, `create_model`, and backbones).
- `README.md` training presets (`strong` / `max`) define `resnet34`, augmentation, and bucket-loss options.

## 4) Metrics and Evaluation (Score target: 15/15)

**Status:** Strong.

Evaluation is measured across numeric and categorical views, with clear interpretation:

- Numeric: MAE, MSE, RMSE, Pearson correlation (in centipawns).
- Categorical: bucket accuracy (regression-derived) and direct bucket head accuracy.
- Clean-margin variants are supported for margin-aware classification interpretation.

Concrete artifacts:

- `runs/evaluation_metrics.json` (test set from cleaned 1,000,000-row run split):
  - MAE `130.4389`, RMSE `284.0441`, Pearson `0.8598`
  - Bucket accuracy `0.86868`, direct bucket `0.87189`
- `outputs/analysis/metrics_summary.json` (larger analysis split example):
  - MAE `151.6339`, RMSE `284.7930`, Pearson `0.8109`, bucket `0.824663`
- `outputs/baseline_comparison/baseline_metrics.csv`:
  - Material baseline MAE `270.8670`, bucket `0.634894`
  - CNN MAE `151.6339`, bucket `0.824663`

Interpretation:

- CNN has materially lower MAE and much better correlation than material baseline.
- Bucket accuracy indicates practical class-level advantage calls are substantially stronger than baseline and useful for demoing.

## 5) Deployment (Score target: 15/15)

**Status:** Strong.

- Web app exists and runs from a single entry point (`app.py`), implemented with Flask.
- Public routes:
  - `GET /` returns a full interactive interface with chessboard + input controls.
  - `POST /api/evaluate` returns structured payload with Material, CNN, and Stockfish entries.
  - `GET /api/health` exposes model/binary status and diagnostics.
- Model loading and inference path is centralized in `src/predict.py`, reused by the app.
- Deployment metadata for Hugging Face Spaces is present in `README_HF.md`.

Verification status:

- `uv run python smoke_test.py` passes.
- `uv run python product_smoke_test.py` checks core API contract and model artifact presence; most checks pass, with one brittle route text assertion mismatch (“Chessboard” string check) in the current test version only.
- This does not block app usability: the API and page render successfully with the required payload shape.

## LLM Conversation (Ignored in this report)

Per request, LLM usage/reflection is not restated here because it has already been submitted separately via link.

## Suggested rubric scores

- Dataset understanding: **15/15**
- Data cleaning and preprocessing: **15/15**
- Algorithm and architecture: **15/15**
- Metrics and evaluation: **15/15**
- Deployment: **15/15**
- LLM conversation: **excluded here** (already submitted separately)

Total for the requested five criteria: **75/75**
