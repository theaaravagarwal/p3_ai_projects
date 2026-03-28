# ASL CNN Project Documentation (`aslcnn` v1 -> `aslproj` v2)

## Project Overview
This repository contains two generations of an American Sign Language (ASL) image classifier focused on letters **A-E**:

- `aslcnn` = first working prototype (v1).
- `aslproj` = optimized final system (v2) with stronger data handling, faster training throughput, richer metrics, and more evaluation utilities.

Both versions are now documented around a **local folder dataset workflow** (no external dataset API dependency in `aslcnn/main.py` or `aslcnn/main-modified.py`).

---

## Assignment Alignment (Step-by-Step)

### Step 1: Data Collection
Recommended data strategy implemented/documented across this repo:

- Collect at least **50 images per class** for `A`, `B`, `C`, `D`, `E`.
- Use webcam capture (OpenCV) and/or Teachable Machine export.
- Store data as one folder per class.
- Include variation:
  - hand orientation changes
  - multiple lighting conditions
  - background diversity
  - multiple participants
  - both left and right hands
- The evaluation/"secret" set can arrive at **300x300** resolution.

Expected directory layout:

```text
<dataset_root>/
  A/
    img001.jpg
    ...
  B/
  C/
  D/
  E/
```

Alternative explicit split supported in v2 (and partially in v1 local loader):

```text
<dataset_root>/
  train/
    A/ ...
    B/ ...
    C/ ...
    D/ ...
    E/ ...
  test/
    A/ ...
    B/ ...
    C/ ...
    D/ ...
    E/ ...
```

### Step 2: Data Preprocessing
Implemented behaviors across versions:

- Input images are loaded as **RGB** (3 channels).
- `aslcnn` (v1): train-time resize/crop/augment pipeline at **128x128**.
- `aslproj` (v2): hand-focused preprocessing via MediaPipe crop (or raw-mode fallback), then fixed-size resizing (default **128x128**).
- Normalization uses ImageNet-style channel stats:
  - mean `(0.485, 0.456, 0.406)`
  - std `(0.229, 0.224, 0.225)`

For 300x300 evaluation data:

- The pipeline ingests larger images and converts them into model input size during preprocessing.
- This keeps inference shape stable without requiring dataset-side manual resizing.

### Step 3: Model Design
- Framework: **PyTorch**.
- Output classes: 5 (`A-E`) with logits passed through `softmax` at inference/evaluation time.
- v1 uses a custom residual CNN.
- v2 uses an EfficientNet-B0 transfer-learning classifier head.

### Step 4: Training
- Train/validation split target is **80/20**.
- v1 now performs class-aware local split from folder data.
- v2 uses stratified split via `train_test_split(..., stratify=labels)` when explicit `train/test` folders are not provided.
- Metrics tracked in v2 include:
  - training/validation loss
  - training/validation accuracy
  - training/validation macro F1
  - learning rate and system-throughput telemetry

### Step 5: Testing
- Real-time and offline evaluation scripts are included under `aslproj/stor`.
- Multiple scripts support webcam testing, ensemble-style model stacking, and per-class score reports.

---

## v1: `aslcnn` (Initial Prototype)

### Purpose
First attempt at a robust A-E ASL classifier with custom residual blocks and aggressive augmentation.

### Refactor Completed
`aslcnn/main.py` and `aslcnn/main-modified.py` were updated to use **local directory-based loading** instead of remote dataset download APIs.

Local training entry now expects:

- `--data-dir <path>` where path contains class folders `A-E` (or `train/test` subfolders with `A-E`).

### Data Loading Flow (Current v1)
- Discover class folders recursively (`A-E`).
- If class folders are absent, fallback to filename-label inference.
- Build class-indexed sample list.
- Create train/validation split with class-balanced logic (approximately 80/20, minimum 1 validation sample per class).

### Preprocessing and Augmentation (v1)
Training transform:
- Resize to `140x140`
- Random crop to `128x128`
- Random affine (`degrees=12`, translation, scale, shear)
- Random perspective
- Color jitter
- Random erasing
- Normalize with channel mean/std

Validation transform:
- Resize to `128x128`
- Normalize with same mean/std

### Model Architecture (v1)
Defined in `aslcnn/main.py`:

- Stem: `Conv(3->32, 3x3, s1)` + BN + GELU, repeated conv.
- Residual stack with SE + spatial attention blocks:
  - `32->64` (stride 2), then residual
  - `64->128` (stride 2), then residual
  - `128->192` (stride 2), then residual
  - `192->256` (stride 1), then residual
- Dropout2d at intermediate stages: `0.08`, `0.12`, `0.16`
- Head:
  - AdaptiveAvgPool
  - LayerNorm
  - FC `256->192` + GELU + Dropout `0.35`
  - FC `192->5`

### Training Hyperparameters (v1)
- Epochs: `75`
- Optimizer: `AdamW(lr=7.5e-4, weight_decay=3e-4)`
- Loss: `CrossEntropyLoss(label_smoothing=0.03)`
- Scheduler: `ReduceLROnPlateau`
- Early stopping patience: `8` epochs without validation improvement

### v1 Limitations
- Single-script training stack with fewer operational utilities.
- Less explicit artifact/version tracking than v2.
- Metric scope focused mostly on accuracy (v2 adds robust F1 tracking and richer telemetry).

---

## v2: `aslproj` (Optimized Final Version)

### Core Improvements Over v1
- More robust sample discovery and split handling.
- Optional MediaPipe hand-landmark cropping and preprocessing cache.
- Transfer learning backbone (EfficientNet-B0).
- Dynamic batch/worker sizing based on compute and memory budgets.
- GPU prefetch + aggressive throughput tuning.
- Detailed per-epoch telemetry with macro F1 as a first-class metric.
- Structured run artifacts (`config.json`, `metrics.csv`, checkpoint, plots).

### Data Loading and Split Logic (v2)
From `aslproj/main.py`:
- `--data-dir` local root.
- Supports either:
  - explicit `train/` and `test/` folders, or
  - one root pool split into train/val using stratified 80/20.
- Accepts class labeling by folder name (`A-E`) and can fallback to filename label parsing.

### Preprocessing (v2)
- `LandmarkCropper` uses MediaPipe hand landmarks to isolate hand ROI.
- Optional `--raw-train` bypasses landmark crop and uses full-frame resized input.
- Batch-level transforms:
  - `ToDtype(float32, scale=True)` => normalizes to `[0,1]`
  - training augmentations: random rotation + color jitter
  - optional horizontal flip
  - channel normalization with ImageNet stats
- Optional offline preprocessing cache for speed/repeatability.

### Model (v2)
`ASLModel` wraps `torchvision.models.efficientnet_b0`:
- optional ImageNet pretrained weights
- classifier head replaced with `Linear(in_features, 5)`

### Training Hyperparameters / Ranges Observed in Artifacts
From `aslproj/stor/*/config.json` snapshots:
- Image size: `128`
- Batch size used across runs: `4`, `16`, `128`, `256`, `384`, `512`
- Epochs: `10-50`
- LR values used: `1e-4`, `1.125e-4`, `1.5e-4`, `3e-4`
- Weight decay: `1e-4`
- Validation ratio: `0.2`
- Worker counts observed: `1`, `6-10`, `14`, `16`
- Prefetch factor: `2` or `10`
- GPU prefetch batches: `6`

### Observed Results (from stored metrics)
Top runs in `aslproj/stor/*/metrics.csv` reached:
- validation macro F1 up to **0.999951**
- validation accuracy up to **0.999945**

Notes:
- Most high-performing runs are near-saturated on available validation sets.
- One run (`17`) shows anomalous metric behavior (`val_f1` very low with perfect accuracy), which suggests either data/label mismatch or metric/reporting inconsistency in that run.

---

## Architecture Comparison

| Aspect | v1 `aslcnn` | v2 `aslproj` |
|---|---|---|
| Backbone | Custom residual CNN + SE + spatial attention | EfficientNet-B0 |
| Conv depth style | Manual stage design (`32->64->128->192->256`) | MBConv-based pretrained backbone |
| Pooling | AdaptiveAvgPool2d(1) | EfficientNet internal + classifier head |
| Activations | GELU (mostly) | SiLU/ReLU internals from EfficientNet |
| Dropout | Dropout2d `0.08/0.12/0.16` + FC dropout `0.35` | Backbone defaults + head replacement (no heavy custom dropout stack in main trainer) |
| Input preprocessing | torchvision augment chain, direct image pipeline | MediaPipe ROI crop + cache + batch transforms |
| Split method | class-aware local split in updated script | explicit train/test or stratified 80/20 |
| Metrics emphasis | Loss + accuracy | Loss + accuracy + macro F1 + throughput/resource telemetry |

---

## `aslproj/stor` Utilities and Subfolders

### Primary Utility Scripts
- `aslproj/stor/eval.py`
  - Model architecture inspector for assignment reporting.
  - Extracts conv layers, kernel sizes, FC layer layout, and parameter counts.
- `aslproj/stor/test.py`
  - Live webcam inference pipeline with optional hand crop + smoothing.
- `aslproj/stor/test-stack.py`
  - Multi-model/ensemble-style inference and auto-model selection from run artifacts.
- `aslproj/stor/secrettest.py`
  - Evaluation-focused script for held-out/secret-style testing with class-level reporting.
- `aslproj/stor/webcam_asl_hf.py`
  - Webcam inference path using a Hugging Face-hosted classifier artifact.
- `aslproj/stor/transfer4060.sh`
  - Artifact transfer helper script.

### Artifact / Data Subfolders
- `aslproj/stor/<run_id>/`
  - Per-run outputs (`config.json`, `metrics.csv`, `label_map.json`, model checkpoints where applicable).
- `aslproj/stor/final_data/`
  - Curated A-E dataset samples.
- `aslproj/stor/secretdata/`
  - Secret/evaluation-style sample structure.
- `aslproj/stor/stor/`
  - Archived experimental models/scripts (legacy snapshots and alternative architectures).

---

## Practical Training Usage

### v1 Prototype Training
```bash
python3 aslcnn/main.py --data-dir /path/to/asl_data
```

### v2 Optimized Training
```bash
python3 aslproj/main.py --data-dir /path/to/asl_data --output-dir artifacts
```

Example local data root (required class names):

```text
/path/to/asl_data/
  A/
  B/
  C/
  D/
  E/
```

---

## Challenges, Pitfalls, and Mitigations

Common ASL classification issues and how this project handles them:

- Lighting variability
  - mitigated with augmentation and normalization
- Hand orientation/viewpoint drift
  - mitigated with affine/rotation transforms and landmark-based ROI extraction
- Background clutter
  - mitigated in v2 via hand-centric cropping
- Overfitting on limited participants
  - mitigated by multi-participant collection and split discipline
- Train/val leakage risk
  - mitigated with explicit split logic and reproducible seeds

---

## Submission Checklist Mapping
This repository now contains enough material to produce the assignment report sections:

- Data collection strategy
- Model architecture details
- Training approach + hyperparameters
- Observed results + challenges
- Testing/evaluation utilities

For final submission, include links to your code host and team LLM conversations as requested by the assignment.
