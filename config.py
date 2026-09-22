"""
Global Configuration for Traffic Sign Recognition Project.
Defines directory paths, hyperparameters, dataset parameters, and hardware settings.
"""

from pathlib import Path
import torch

# ==========================================
# Paths Configuration
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

# Dataset Directories
GERMAN_DATASET_DIR = DATA_DIR / "German Dataset"
BELGIAN_DATASET_DIR = DATA_DIR / "Belgian Dataset"
CHINESE_DATASET_DIR = DATA_DIR / "Chinese Dataset"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
CSV_RESULTS_DIR = RESULTS_DIR / "csv"
PLOTS_RESULTS_DIR = RESULTS_DIR / "plots"
CONFUSION_MATRICES_DIR = RESULTS_DIR / "confusion_matrices"

# Ensure essential output directories exist
for directory in [CHECKPOINTS_DIR, RESULTS_DIR, CSV_RESULTS_DIR, PLOTS_RESULTS_DIR, CONFUSION_MATRICES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================
# Dataset & Preprocessing Hyperparameters
# ==========================================
NUM_CLASSES_GERMAN = 43
NUM_CLASSES_BELGIAN = 62
NUM_CLASSES_CHINESE = 58

DATASET_NUM_CLASSES = {
    "german": NUM_CLASSES_GERMAN,
    "gtsrb": NUM_CLASSES_GERMAN,
    "belgian": NUM_CLASSES_BELGIAN,
    "btsc": NUM_CLASSES_BELGIAN,
    "chinese": NUM_CLASSES_CHINESE,
    "ctsd": NUM_CLASSES_CHINESE,
}

# Default dataset and classes
DEFAULT_DATASET = "german"
NUM_CLASSES = NUM_CLASSES_GERMAN
IMAGE_SIZE = (32, 32)  # (Height, Width)
IN_CHANNELS = 3

# Normalization constants (GTSRB RGB statistics or ImageNet defaults)
# GTSRB specific: mean=[0.3403, 0.3121, 0.3214], std=[0.2724, 0.2608, 0.2669]
NORM_MEAN = [0.3403, 0.3121, 0.3214]
NORM_STD = [0.2724, 0.2608, 0.2669]

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 64
VAL_SPLIT = 0.2
NUM_WORKERS = 0  # 0 recommended for Windows stability, 2-4 on Linux
PIN_MEMORY = torch.cuda.is_available()
SEED = 42

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30

# ==========================================
# Hardware / Device
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
