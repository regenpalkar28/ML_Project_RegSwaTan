# Traffic Sign Recognition & Benchmark Robustness

A deep learning framework for traffic sign classification and corruption robustness benchmarking across three international datasets:
- 🇩🇪 **German Traffic Sign Recognition Benchmark (GTSRB)** — 43 classes (51,839 images)
- 🇧🇪 **Belgium Traffic Sign Classification Benchmark (BelgiumTSC)** — 62 classes (7,125 images)
- 🇨🇳 **Chinese Traffic Sign Dataset (CTSD)** — 58 classes (6,164 images)

---

## 📁 Repository Structure

```
ML_Project_RegSwaTan/
├── config.py                 # Global paths, class counts, and hyperparameter configs
├── data/
│   ├── dataset.py            # Multi-dataset loader, augmentations, and DataLoader factory
│   ├── __init__.py           # Data module exports
│   ├── German Dataset/       # GTSRB annotations and images
│   ├── Belgian Dataset/      # BelgiumTSC annotations and PPM images
│   └── Chinese Dataset/      # CTSD annotations.csv and images/
├── models/
│   ├── RawCNN.py             # Basic Convolutional Neural Network baseline
│   ├── vgg19_model.py        # VGG19 / VGG19-BN architectures
│   └── __init__.py           # Model factory and exports
├── training/
│   ├── train.py              # Training loop with time estimation and checkpointing
│   └── __init__.py           # Training module exports
├── evaluation/
│   ├── evaluate.py           # Standalone checkpoint evaluation CLI
│   ├── confusion_matrix.py   # Confusion matrix computation & heatmap plotting
│   ├── metrics.py            # Accuracy, Precision, Recall, Macro/Weighted F1
│   └── __init__.py           # Evaluation module exports
├── checkpoints/              # Saved model weights (.pth)
├── results/
│   ├── confusion_matrices/   # Confusion matrix CSVs, JSON reports, PNG heatmaps
│   ├── csv/                  # Per-class metrics and training history tables
│   └── plots/                # Training loss/accuracy curves
├── Documentation/            # Architecture specifications and planning docs
├── requirements.txt          # Python package dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Environment Setup & Installation

### 1. Prerequisites
- Python 3.10+ (Recommended: Python 3.11)
- CUDA-compatible GPU (optional, CPU execution supported)

### 2. Setup Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

---

## 🚀 Model Training

Use [`training/train.py`](file:///c:/Users/Regen/Machine%20Learning/ML_Project_RegSwaTan/training/train.py) to train models. The script automatically estimates total training time before execution and prompts for confirmation.

### 1. Training Baseline CNN (`raw_cnn`)

#### A. Train on German Dataset (GTSRB — 43 Classes)
```powershell
python training/train.py --dataset german --model_type raw_cnn --epochs 30 --batch_size 64 --lr 0.001
```

#### B. Train on Belgian Dataset (BelgiumTSC — 62 Classes)
```powershell
python training/train.py --dataset belgian --model_type raw_cnn --epochs 30 --batch_size 64 --lr 0.001
```

#### C. Train on Chinese Dataset (CTSD — 58 Classes)
```powershell
python training/train.py --dataset chinese --model_type raw_cnn --epochs 30 --batch_size 64 --lr 0.001
```

---

### 2. Training VGG19 & VGG19-BN

The VGG19 pipeline in [`models/vgg19_model.py`](file:///c:/Users/Regen/Machine%20Learning/ML_Project_RegSwaTan/models/vgg19_model.py) supports deep 19-layer architectures with Batch Normalization (`vgg19_bn`), optional ImageNet pre-training (`--pretrained`), and optimized classification heads (`--compact_head`).

> [!NOTE]
> Before launching the full training loop, `train.py` runs benchmark sample batches on your compute device (CPU/GPU) to estimate the time required per epoch and total training duration, then prompts:
> ```
> Do you wish to proceed with training? [Y/n]: 
> ```
> Press **Enter** or type `y` to start training, or pass `-y` / `--yes` to skip the prompt for automated runs.

#### A. Recommended: Train VGG19 with Batch Normalization (German GTSRB)
```powershell
python training/train.py --model_type vgg19_bn --dataset german --epochs 30 --batch_size 64
```

#### B. Train Standard VGG19 (without BatchNorm)
```powershell
python training/train.py --model_type vgg19 --dataset german --epochs 30 --batch_size 64
```

#### C. Transfer Learning with Pretrained ImageNet Weights
```powershell
python training/train.py --model_type vgg19_bn --dataset german --pretrained --epochs 25 --batch_size 64 --lr 0.0001
```

#### D. Train VGG19 on Belgian and Chinese Benchmarks
```powershell
# Belgian Dataset (62 classes)
python training/train.py --model_type vgg19_bn --dataset belgian --epochs 30 --batch_size 64

# Chinese Dataset (58 classes)
python training/train.py --model_type vgg19_bn --dataset chinese --epochs 30 --batch_size 64
```

---

### 3. Key Training Flags

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--dataset` | `german` | Benchmark dataset (`german`, `belgian`, `chinese`). |
| `--model_type` | `raw_cnn` | Architecture (`raw_cnn`, `basic_cnn`, `vgg19`, `vgg19_bn`). |
| `--epochs` | `30` | Number of training epochs. |
| `--batch_size` | `64` | Mini-batch size. |
| `--lr` | `0.001` | Initial learning rate (Cosine Annealing scheduler). |
| `--weight_decay`| `0.0001`| L2 weight regularization penalty. |
| `--img_size` | `32 32` | Input image dimensions (Height Width). |
| `--num_workers` | `0` | Number of DataLoader workers (`0` recommended on Windows). |
| `--pretrained` | `False` | Initialize backbone with ImageNet pre-trained weights. |
| `--compact_head`| `True` | Use efficient classifier head (~21M params vs ~140M params). |
| `-y`, `--yes` | `False` | Skip interactive prompt and proceed immediately with training. |

---

## 📊 Evaluation & Testing

Evaluate any saved `.pth` checkpoint using [`evaluation/evaluate.py`](file:///c:/Users/Regen/Machine%20Learning/ML_Project_RegSwaTan/evaluation/evaluate.py) or [`evaluation/confusion_matrix.py`](file:///c:/Users/Regen/Machine%20Learning/ML_Project_RegSwaTan/evaluation/confusion_matrix.py).

### 1. Evaluate Saved Checkpoints on Test Set

```powershell
# Evaluate Baseline CNN Checkpoint
python evaluation/evaluate.py --checkpoint checkpoints/raw_cnn_german_best.pth --dataset german --model_type raw_cnn --split test

# Evaluate VGG19-BN Checkpoint
python evaluation/evaluate.py --checkpoint checkpoints/vgg19_bn_german_best.pth --dataset german --model_type vgg19_bn --split test

# Evaluate Belgian Checkpoint
python evaluation/evaluate.py --checkpoint checkpoints/vgg19_bn_belgian_best.pth --dataset belgian --model_type vgg19_bn --split test

# Evaluate Chinese Checkpoint
python evaluation/evaluate.py --checkpoint checkpoints/vgg19_bn_chinese_best.pth --dataset chinese --model_type vgg19_bn --split test
```

### 2. Standalone Confusion Matrix Generator

```powershell
python evaluation/confusion_matrix.py --checkpoint checkpoints/vgg19_bn_german_best.pth --dataset german --model_type vgg19_bn --split test
```

---

## 📈 Generated Artifacts & Storage

All training and evaluation runs store outputs automatically in `results/`:

1. **Model Checkpoints** (`checkpoints/`):
   - Best validation model weights: `{model}_{dataset}_best.pth`
2. **Confusion Matrices** (`results/confusion_matrices/`):
   - Raw count table: `{model}_{dataset}_{split}_cm_raw.csv`
   - Normalized recall matrix: `{model}_{dataset}_{split}_cm_norm.csv`
   - High-resolution heatmap plots (300 DPI): `{model}_{dataset}_{split}_cm.png` & `{model}_{dataset}_{split}_cm_normalized.png`
   - Detailed JSON report with Top-15 confused pairs: `{model}_{dataset}_{split}_metrics.json`
3. **Per-Class Metrics** (`results/csv/`):
   - Table with per-class Precision, Recall, F1, Support: `{model}_{dataset}_{split}_per_class_metrics.csv`

---

## 🐍 Python API Examples

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

model = get_model(num_classes=DATASET_NUM_CLASSES["german"]).to(DEVICE)
images, labels = next(iter(train_loader))
outputs = model(images.to(DEVICE))  # Shape: [64, 43]
```

### Evaluate Checkpoint Programmatically
```python
from evaluation import evaluate_model

results = evaluate_model(
    checkpoint_path="checkpoints/raw_cnn_german_best.pth",
    dataset_name="german",
    split="test",
)

metrics = results["metrics"]
print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
print(f"Macro F1:      {metrics['macro_f1']:.4f}")
```