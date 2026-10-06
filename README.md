# Traffic Sign Recognition & Benchmark Robustness

A deep learning framework for traffic sign classification and corruption robustness benchmarking across three international datasets:
- 🇩🇪 **German Traffic Sign Recognition Benchmark (GTSRB)** — 43 classes (51,839 images)
- 🇧🇪 **Belgium Traffic Sign Classification Benchmark (BelgiumTSC)** — 62 classes (7,125 images)
- 🇨🇳 **Chinese Traffic Sign Dataset (CTSD)** — 58 classes (6,164 images)

---

## 📁 Repository Structure

```
ML_Project_RegSwaTan/
├── config.py                 # Global paths, dataset class counts, hyperparameters, and device config
├── data/
│   ├── dataset.py            # Multi-dataset loader, GTSRB/Belgian/Chinese preprocessing, DataLoaders
│   ├── __init__.py           # Data module exports
│   ├── German Dataset/       # GTSRB annotations and images
│   ├── Belgian Dataset/      # BelgiumTSC annotations and PPM images
│   └── Chinese Dataset/      # CTSD annotations.csv and images/
├── models/
│   ├── RawCNN.py             # Basic Convolutional Neural Network baseline (BasicSignClassifier)
│   ├── vgg19_model.py        # VGG19 / VGG19-BN architectures with compact head option
│   └── __init__.py           # Model factory and exports
├── corruption/               # Image corruption robustness suite (14 Hendrycks & Dietterich corruptions)
│   ├── gaussian.py, shot.py, impulse.py        # Noise corruptions
│   ├── defocus_blur.py, motion_blur.py, etc.   # Blur corruptions
│   ├── fog.py, frost.py, snow.py               # Weather corruptions
│   ├── brightness.py, contrast.py, etc.        # Digital/photometric corruptions
│   └── corrupt_images_test.py                  # Standalone corruption test script
├── training/
│   ├── train.py              # Training pipeline with upfront time estimation & checkpointing
│   └── __init__.py           # Training module exports
├── evaluation/
│   ├── evaluate.py           # Standalone checkpoint evaluation CLI & metric exporter
│   ├── confusion_matrix.py   # Confusion matrix pipeline, heatmaps, auto-metadata detection
│   ├── metrics.py            # Accuracy, Precision, Recall, Macro/Weighted F1 functions
│   └── __init__.py           # Evaluation module exports
├── checkpoints/              # Saved model weights (.pth)
├── results/
│   ├── confusion_matrices/   # Confusion matrix CSVs, JSON reports, PNG heatmaps
│   ├── csv/                  # Per-class metrics and training history tables
│   └── plots/                # Training loss/accuracy curves
├── requirements.txt          # Python package dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Environment Setup & Installation

### 1. Prerequisites
- **Python**: 3.10+ (Recommended: Python 3.11 for pre-compiled PyTorch / NumPy binaries)
- **GPU Acceleration**: CUDA-compatible GPU (optional, CPU supported automatically)

### 2. Virtual Environment Setup (Windows PowerShell)

```powershell
# 1. Create virtual environment using Python 3.11 (or system default)
py -3.11 -m venv .venv

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Upgrade core packaging tools
pip install --upgrade pip setuptools wheel

# 4. Install project dependencies
pip install -r requirements.txt
```

*Note: If you run into PowerShell script execution policy restrictions, activate using `.venv\Scripts\activate` or run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first.*

---

## 🚀 Model Training

Train models using [`training/train.py`](file:///c:/Users/Regen/Machine%20Learning/ML_Project_RegSwaTan/training/train.py). The training pipeline automatically benchmarks device performance across sample batches to estimate training time prior to starting.

### 1. Training Baseline CNN (`raw_cnn`)

```powershell
# German Dataset (GTSRB - 43 Classes)
python training/train.py --dataset german --model_type raw_cnn --epochs 30 --batch_size 64 --lr 0.001

# Belgian Dataset (BelgiumTSC - 62 Classes)
python training/train.py --dataset belgian --model_type raw_cnn --epochs 30 --batch_size 64 --lr 0.001

# Chinese Dataset (CTSD - 58 Classes)
python training/train.py --dataset chinese --model_type raw_cnn --epochs 30 --batch_size 64 --lr 0.001
```

---

### 2. Training VGG19 & VGG19-BN

Supported architectures include standard `vgg19` and Batch-Normalized `vgg19_bn` with optional ImageNet pre-training (`--pretrained`) and compact classification heads (`--compact_head`).

> [!TIP]
> Add `-y` or `--yes` to skip the interactive confirmation prompt during automated or script-driven runs.

```powershell
# Train VGG19 with Batch Normalization on German Dataset (Recommended)
python training/train.py --model_type vgg19_bn --dataset german --epochs 30 --batch_size 64 -y

# Train Standard VGG19 (without BatchNorm)
python training/train.py --model_type vgg19 --dataset german --epochs 30 --batch_size 64 -y

# Transfer Learning with Pre-trained ImageNet Weights
python training/train.py --model_type vgg19_bn --dataset german --pretrained --epochs 25 --batch_size 64 --lr 0.0001 -y

# VGG19-BN on Belgian Benchmark
python training/train.py --model_type vgg19_bn --dataset belgian --epochs 30 --batch_size 64 -y

# VGG19-BN on Chinese Benchmark
python training/train.py --model_type vgg19_bn --dataset chinese --epochs 30 --batch_size 64 -y
```

---

### 3. Key Training CLI Flags

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--dataset` | `german` | Target dataset (`german`, `belgian`, `chinese`). |
| `--model_type` | `raw_cnn` | Model architecture (`raw_cnn`, `basic_cnn`, `vgg19`, `vgg19_bn`). |
| `--epochs` | `30` | Number of training epochs. |
| `--batch_size` | `64` | Training batch size. |
| `--lr` | `0.001` | Initial learning rate (Cosine Annealing scheduler). |
| `--weight_decay`| `0.0001`| L2 weight regularization penalty. |
| `--img_size` | `32 32` | Input image dimensions `(Height Width)`. |
| `--num_workers` | `0` | DataLoader worker threads (`0` recommended for Windows). |
| `--pretrained` | `False` | Initialize VGG backbone with ImageNet weights. |
| `--compact_head`| `True` | Light classifier head (~21M params vs ~140M params). |
| `-y`, `--yes` | `False` | Skip interactive prompt and start training immediately. |

---

## 📊 Model Testing & Evaluation

Test any trained `.pth` checkpoint using [`evaluation/confusion_matrix.py`](file:///c:/Users/Regen/Machine%20Learning/ML_Project_RegSwaTan/evaluation/confusion_matrix.py) or [`evaluation/evaluate.py`](file:///c:/Users/Regen/Machine%20Learning/ML_Project_RegSwaTan/evaluation/evaluate.py).

> [!NOTE]
> **Automatic Architecture Detection**: The evaluation module reads embedded metadata (`model_type`, `num_classes`) directly from the `.pth` file dictionary. If you pass `--model_type vgg19` for a `vgg19_bn` checkpoint, the pipeline will automatically detect the mismatch and load the correct model parameters.

### 1. Generating Confusion Matrices & Full Evaluation Reports

```powershell
# Evaluate VGG19-BN Checkpoint on Test Set (German Dataset)
python -m evaluation.confusion_matrix --checkpoint "checkpoints/vgg19_bn_german_best.pth" --dataset german --model_type vgg19_bn --split test

# Evaluate Baseline RawCNN Checkpoint
python -m evaluation.confusion_matrix --checkpoint "checkpoints/raw_cnn_german_best.pth" --dataset german --model_type raw_cnn --split test

# Evaluate Belgian / Chinese Checkpoints
python -m evaluation.confusion_matrix --checkpoint "checkpoints/vgg19_bn_belgian_best.pth" --dataset belgian --model_type vgg19_bn --split test
```

### 2. Standalone Evaluator CLI (`evaluate.py`)

```powershell
python evaluation/evaluate.py --checkpoint "checkpoints/vgg19_bn_german_best.pth" --dataset german --model_type vgg19_bn --split test
```

---

## 📈 Calculated Metrics & Generated Artifacts

Every evaluation run automatically generates complete classification reports saved to `results/`:

### 1. Summary Metrics & Reports (`results/confusion_matrices/`)
* **`*_metrics.json`**: Complete metrics summary containing:
  * **Overall Accuracy**
  * **Macro-Average F1-Score**
  * **Weighted-Average F1-Score**
  * **Top-15 Most Confused Class Pairs** (with misclassification counts and class error rates)
* **`*_cm.png`**: High-resolution (300 DPI) raw count confusion matrix heatmap.
* **`*_cm_normalized.png`**: High-resolution row-normalized (recall) heatmap.
* **`*_cm_raw.csv`**: Raw confusion count matrix table.
* **`*_cm_norm.csv`**: Row-normalized recall matrix table.

### 2. Detailed Per-Class Metrics (`results/csv/`)
* **`*_per_class_metrics.csv`**: Class-by-class table containing **Precision**, **Recall**, **F1-Score**, and **Support** for all classes.

---

## 🌧️ Image Corruption & Robustness Benchmarking

The [`corruption/`](file:///c:/Users/Regen/Machine%20Learning/ML_Project_RegSwaTan/corruption) module implements the **14 standard Hendrycks & Dietterich benchmark corruptions** across 5 severity levels:

* **Noise**: Gaussian Noise, Shot Noise, Impulse Noise
* **Blur**: Defocus Blur, Motion Blur, Zoom Blur
* **Weather**: Fog, Frost, Snow
* **Digital / Photometric**: Brightness, Contrast, Pixelate, Elastic Transform, JPEG Compression

### Python Usage Example

```python
from PIL import Image
from corruption import apply_corruption, CORRUPTIONS_DICT

# Load sample image
img = Image.open("path/to/traffic_sign.png").convert("RGB")

# Apply Level 3 Fog corruption
corrupted_img = apply_corruption(img, corruption_type="fog", severity=3)

# Apply Level 5 Gaussian Noise
noisy_img = apply_corruption(img, corruption_type="gaussian_noise", severity=5)
```

---

## 🐍 Python API Reference

### Load DataLoaders
```python
from data import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(
    dataset_name="german",  # "german" | "belgian" | "chinese"
    batch_size=64,
    img_size=(32, 32),
    val_split=0.2,
)
```

### Model Forward Pass
```python
from models.RawCNN import get_model
from config import DATASET_NUM_CLASSES, DEVICE

model = get_model(num_classes=DATASET_NUM_CLASSES["german"], model_type="vgg19_bn").to(DEVICE)
images, labels = next(iter(train_loader))
outputs = model(images.to(DEVICE))  # Shape: [64, 43]
```

### Programmatic Checkpoint Evaluation
```python
from evaluation import compute_and_save_confusion_matrix, evaluate_checkpoint_and_generate_cm

evaluate_checkpoint_and_generate_cm(
    checkpoint_path="checkpoints/vgg19_bn_german_best.pth",
    dataset_name="german",
    model_type="vgg19_bn",
    split="test",
)
```