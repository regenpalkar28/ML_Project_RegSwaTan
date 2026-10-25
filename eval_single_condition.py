"""
Simple Single-Condition Test Script for Traffic Sign Recognition (GTSRB).

Evaluates a specific model ('baseline', 'msff', or 'msff_attn') on the GTSRB test set
under either clean conditions or any of the 14 standard corruptions at a selected severity.

Usage:
------
Edit the CONFIGURATION block below directly in your IDE, then run:
    python eval_single_condition.py
"""

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from corruption import apply_corruption
from data.dataset import GTSRBDataset
from models.vgg19_model import get_vgg19_model
from models.vgg_msff_attn import (
    VGG19BN_MSFF,
    VGG19BN_MSFF_Attn,
    get_vgg19_msff_attn_model,
    get_vgg19_msff_model,
)

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_TYPE = "msff_attn"  # Choices: 'baseline', 'msff', 'msff_attn'
CHECKPOINT_PATH = "checkpoints/vgg19_msff_attn_clean.pth"
CORRUPTION_NAME = "motion_blur"  # Choice of 14 ImageNet-C/GTSRB-C corruptions or 'clean'
SEVERITY = 5  # Integer [1-5]
DATA_DIR = "./data/GTSRB"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================

# Dataset normalization constants (GTSRB)
NORM_MEAN = [0.3403, 0.3121, 0.3214]
NORM_STD = [0.2724, 0.2608, 0.2669]
IMAGE_SIZE = (32, 32)
NUM_CLASSES = 43


class SingleConditionTransform:
    """
    Applies optional corruption to PIL image, converts to tensor, and normalizes.
    """

    def __init__(
        self,
        corruption: Optional[str] = None,
        severity: int = 1,
        img_size: Tuple[int, int] = IMAGE_SIZE,
    ):
        self.corruption = corruption.lower() if corruption else "clean"
        self.severity = int(severity)
        self.resize = transforms.Resize(img_size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)

    def __call__(self, img):
        img = self.resize(img)
        if self.corruption != "clean":
            img = apply_corruption(img, self.corruption, severity=self.severity)
        tensor = self.to_tensor(img)
        return self.normalize(tensor)


def load_selected_model(model_type: str, device: torch.device) -> nn.Module:
    """Instantiate model based on MODEL_TYPE configuration."""
    model_type = model_type.lower()
    if model_type in ("baseline", "vgg19", "vgg19_bn"):
        model = get_vgg19_model(
            num_classes=NUM_CLASSES,
            compact_head=True,
            use_batch_norm=True,
        )
    elif model_type in ("msff", "vgg19_bn_msff"):
        model = get_vgg19_msff_model(num_classes=NUM_CLASSES)
    elif model_type in ("msff_attn", "vgg19_bn_msff_attn"):
        model = get_vgg19_msff_attn_model(num_classes=NUM_CLASSES)
    else:
        raise ValueError(
            f"Unknown MODEL_TYPE: '{model_type}'. Supported: 'baseline', 'msff', 'msff_attn'"
        )

    model = model.to(device)
    return model


def load_weights(model: nn.Module, checkpoint_path: str, device: torch.device) -> bool:
    """Load model weights from checkpoint path if present."""
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.is_file():
        print(f"\n[WARNING] Checkpoint not found at: '{ckpt_p.resolve()}'")
        print("          Proceeding with randomly initialized weights for verification/testing.\n")
        return False

    checkpoint = torch.load(ckpt_p, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strip potential 'module.' prefix from DataParallel training
    cleaned_dict = {}
    for k, v in state_dict.items():
        new_k = k[7:] if k.startswith("module.") else k
        cleaned_dict[new_k] = v

    try:
        model.load_state_dict(cleaned_dict, strict=True)
        print(f"[INFO] Successfully loaded model weights from: '{ckpt_p.resolve()}'")
        return True
    except RuntimeError as err:
        print(f"[WARN] Strict loading failed: {err}")
        model.load_state_dict(cleaned_dict, strict=False)
        print(f"[INFO] Non-strict loading succeeded.")
        return True


def evaluate():
    device = torch.device(DEVICE)
    print("=" * 65)
    print("GTSRB SINGLE-CONDITION EVALUATION")
    print("=" * 65)
    print(f"Model Architecture : {MODEL_TYPE.upper()}")
    print(f"Checkpoint Path    : {CHECKPOINT_PATH}")
    print(f"Corruption         : {CORRUPTION_NAME} (Severity {SEVERITY})")
    print(f"Dataset Root       : {DATA_DIR}")
    print(f"Device             : {device}")
    print(f"Batch Size         : {BATCH_SIZE}")
    print("=" * 65)

    # 1. Load Model
    model = load_selected_model(MODEL_TYPE, device)
    load_weights(model, CHECKPOINT_PATH, device)
    model.eval()

    # 2. Setup Dataset & DataLoader
    transform = SingleConditionTransform(
        corruption=CORRUPTION_NAME,
        severity=SEVERITY,
        img_size=IMAGE_SIZE,
    )

    try:
        test_dataset = GTSRBDataset(
            root_dir=DATA_DIR,
            split="test",
            transform=transform,
        )
    except FileNotFoundError:
        # Fallback to local default data directory
        test_dataset = GTSRBDataset(
            root_dir=None,
            split="test",
            transform=transform,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print(f"[INFO] Test samples loaded: {len(test_dataset)}")
    criterion = nn.CrossEntropyLoss()

    # 3. Evaluation Loop
    running_loss = 0.0
    all_preds = []
    all_targets = []

    start_time = time.time()
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc=f"Evaluating {CORRUPTION_NAME} (Sev {SEVERITY})"):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    total_time = time.time() - start_time
    total_samples = len(all_targets)

    # 4. Metrics
    mean_loss = running_loss / total_samples
    accuracy = accuracy_score(all_targets, all_preds) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0) * 100.0
    fps = total_samples / total_time if total_time > 0 else 0.0

    # 5. Display Results
    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)
    print(f"Condition          : {CORRUPTION_NAME} (Severity {SEVERITY})")
    print(f"Evaluated Samples  : {total_samples}")
    print(f"Cross Entropy Loss : {mean_loss:.4f}")
    print(f"Top-1 Accuracy     : {accuracy:.2f}%")
    print(f"Macro F1-Score     : {macro_f1:.2f}%")
    print(f"Weighted F1-Score  : {weighted_f1:.2f}%")
    print(f"Total Time Taken   : {total_time:.2f}s ({fps:.1f} images/sec)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    evaluate()
