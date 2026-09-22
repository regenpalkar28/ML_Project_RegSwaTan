"""
Confusion Matrix Pipeline for Traffic Sign Recognition.
Computes, visualizes, and exports raw and normalized confusion matrices,
classification reports, and top-confused class pairs across German, Belgian, and Chinese datasets.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn

from config import (
    CONFUSION_MATRICES_DIR,
    DATASET_NUM_CLASSES,
    DEVICE,
    IMAGE_SIZE,
)
from data.dataset import CLASS_NAMES as GTSRB_CLASS_NAMES, get_dataloaders
from models.RawCNN import get_model


# ==============================================================================
# Helper Functions
# ==============================================================================
def get_class_labels(dataset_name: str, num_classes: int) -> List[str]:
    """Generates human-readable or numbered class labels for the dataset."""
    name = dataset_name.lower().strip()
    if name in ["german", "gtsrb"]:
        return [f"{i}: {GTSRB_CLASS_NAMES.get(i, f'Class {i}')}" for i in range(num_classes)]
    elif name in ["belgian", "btsc"]:
        return [f"B_{i:02d}" for i in range(num_classes)]
    elif name in ["chinese", "ctsd"]:
        return [f"C_{i:02d}" for i in range(num_classes)]
    else:
        return [f"Class_{i}" for i in range(num_classes)]


def generate_confusion_matrix(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes raw and row-normalized (recall) confusion matrices.

    Args:
        y_true: Ground truth class indices.
        y_pred: Predicted class indices.
        num_classes: Total number of classes.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (raw_cm, normalized_cm)
    """
    labels = list(range(num_classes))
    raw_cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Compute row-normalized confusion matrix (safe division against zero-count classes)
    row_sums = raw_cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(
        raw_cm.astype("float"),
        row_sums,
        out=np.zeros_like(raw_cm, dtype=float),
        where=(row_sums != 0),
    )

    return raw_cm, norm_cm


def save_confusion_matrix_csv(
    cm: np.ndarray,
    save_path: Union[str, Path],
    class_labels: Optional[List[str]] = None,
    is_normalized: bool = False,
) -> Path:
    """
    Saves a confusion matrix as a formatted CSV file with labeled rows and columns.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    num_classes = cm.shape[0]
    if class_labels is None or len(class_labels) != num_classes:
        class_labels = [f"Class_{i}" for i in range(num_classes)]

    df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    df.index.name = "True_Class"

    if is_normalized:
        df.to_csv(save_path, float_format="%.4f")
    else:
        df.to_csv(save_path, float_format="%d")

    return save_path


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Union[str, Path],
    class_labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "Blues",
) -> Path:
    """
    Renders and saves a publication-quality confusion matrix heatmap.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    num_classes = cm.shape[0]
    if class_labels is None or len(class_labels) != num_classes:
        class_labels = [str(i) for i in range(num_classes)]

    # Dynamic figure sizing based on class count
    if figsize is None:
        if num_classes <= 20:
            figsize = (10, 8)
        elif num_classes <= 45:
            figsize = (16, 14)
        else:
            figsize = (20, 18)

    plt.figure(figsize=figsize, dpi=300)

    # For large matrices (>30 classes), suppress numerical annotations inside cells to avoid clutter
    show_annot = num_classes <= 25
    fmt = ".2f" if normalize else "d"

    ax = sns.heatmap(
        cm,
        annot=show_annot,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar=True,
        square=True,
        linewidths=0.2 if num_classes <= 45 else 0.0,
        cbar_kws={"shrink": 0.8, "label": "Recall Rate" if normalize else "Sample Count"},
    )

    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("True Class", fontsize=12, fontweight="bold", labelpad=10)

    # Set tick labels rotation
    tick_fontsize = 8 if num_classes > 30 else 10
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right", fontsize=tick_fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tick_fontsize)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return save_path


def find_top_confused_pairs(
    raw_cm: np.ndarray,
    class_labels: List[str],
    top_k: int = 10,
) -> List[Dict[str, Union[int, str, float]]]:
    """
    Identifies the top-k most frequent misclassification pairs (excluding the diagonal).
    """
    num_classes = raw_cm.shape[0]
    confused_pairs = []

    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and raw_cm[i, j] > 0:
                true_count = raw_cm[i].sum()
                rate = float(raw_cm[i, j]) / true_count if true_count > 0 else 0.0
                confused_pairs.append(
                    {
                        "true_class_id": int(i),
                        "true_class_label": class_labels[i],
                        "pred_class_id": int(j),
                        "pred_class_label": class_labels[j],
                        "count": int(raw_cm[i, j]),
                        "error_rate_for_class": round(rate, 4),
                    }
                )

    confused_pairs.sort(key=lambda x: x["count"], reverse=True)
    return confused_pairs[:top_k]


# ==============================================================================
# End-to-End Pipeline Execution
# ==============================================================================
def compute_and_save_confusion_matrix(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    model_name: str,
    dataset_name: str,
    split: str = "test",
    save_dir: Optional[Union[str, Path]] = None,
    class_labels: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """
    End-to-end confusion matrix pipeline:
    1. Computes raw and normalized confusion matrices.
    2. Saves raw matrix CSV table.
    3. Saves normalized recall CSV table.
    4. Renders and saves high-resolution visual heatmap (PNG).
    5. Saves detailed summary metrics JSON with top-confused pairs.

    Returns:
        Dict[str, Path]: Dictionary of generated output file paths.
    """
    out_dir = Path(save_dir) if save_dir is not None else CONFUSION_MATRICES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)

    num_classes = DATASET_NUM_CLASSES.get(dataset_name.lower(), int(max(y_true_arr.max(), y_pred_arr.max()) + 1))

    if class_labels is None:
        class_labels = get_class_labels(dataset_name, num_classes)

    # 1. Compute matrices
    raw_cm, norm_cm = generate_confusion_matrix(y_true_arr, y_pred_arr, num_classes)

    # Base filenames
    prefix = f"{model_name.lower()}_{dataset_name.lower()}_{split.lower()}"
    raw_csv_path = out_dir / f"{prefix}_cm_raw.csv"
    norm_csv_path = out_dir / f"{prefix}_cm_norm.csv"
    heatmap_path = out_dir / f"{prefix}_cm.png"
    norm_heatmap_path = out_dir / f"{prefix}_cm_normalized.png"
    metrics_json_path = out_dir / f"{prefix}_metrics.json"

    # 2. Save CSVs
    save_confusion_matrix_csv(raw_cm, raw_csv_path, class_labels, is_normalized=False)
    save_confusion_matrix_csv(norm_cm, norm_csv_path, class_labels, is_normalized=True)

    # 3. Save Heatmap plots
    plot_confusion_matrix(
        raw_cm,
        heatmap_path,
        class_labels=class_labels,
        title=f"Confusion Matrix ({model_name.upper()} on {dataset_name.upper()} {split.capitalize()} Set)",
        normalize=False,
    )
    plot_confusion_matrix(
        norm_cm,
        norm_heatmap_path,
        class_labels=class_labels,
        title=f"Normalized Confusion Matrix ({model_name.upper()} on {dataset_name.upper()} {split.capitalize()} Set)",
        normalize=True,
    )

    # 4. Generate JSON summary and classification metrics
    report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=list(range(num_classes)),
        target_names=[f"Class_{i}" for i in range(num_classes)],
        output_dict=True,
        zero_division=0,
    )
    top_confused = find_top_confused_pairs(raw_cm, class_labels, top_k=15)

    summary_metrics = {
        "model": model_name,
        "dataset": dataset_name,
        "split": split,
        "num_classes": num_classes,
        "total_samples": len(y_true_arr),
        "overall_accuracy": round(float(report.get("accuracy", 0.0)), 4),
        "macro_avg_f1": round(float(report.get("macro avg", {}).get("f1-score", 0.0)), 4),
        "weighted_avg_f1": round(float(report.get("weighted avg", {}).get("f1-score", 0.0)), 4),
        "top_confused_pairs": top_confused,
        "saved_files": {
            "raw_csv": str(raw_csv_path.name),
            "normalized_csv": str(norm_csv_path.name),
            "raw_heatmap_png": str(heatmap_path.name),
            "normalized_heatmap_png": str(norm_heatmap_path.name),
        },
    }

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=4)

    print(f"\n[Confusion Matrix Pipeline] Successfully generated and stored outputs in: {out_dir}")
    print(f"  - Raw Count CSV:       {raw_csv_path.name}")
    print(f"  - Normalized CSV:      {norm_csv_path.name}")
    print(f"  - Visual Heatmap:      {heatmap_path.name}")
    print(f"  - Normalized Heatmap:  {norm_heatmap_path.name}")
    print(f"  - Metrics & Errors:    {metrics_json_path.name}")

    return {
        "raw_csv": raw_csv_path,
        "norm_csv": norm_csv_path,
        "heatmap": heatmap_path,
        "norm_heatmap": norm_heatmap_path,
        "metrics_json": metrics_json_path,
    }


# ==============================================================================
# Standalone CLI Interface
# ==============================================================================
def evaluate_checkpoint_and_generate_cm(
    checkpoint_path: Union[str, Path],
    dataset_name: str = "german",
    model_type: str = "raw_cnn",
    split: str = "test",
    batch_size: int = 64,
):
    """Loads a saved checkpoint and runs evaluation to generate confusion matrix."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    num_classes = DATASET_NUM_CLASSES.get(dataset_name.lower(), 43)

    # DataLoaders
    _, val_loader, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        img_size=IMAGE_SIZE,
        num_workers=0,
    )
    eval_loader = test_loader if split.lower() == "test" else val_loader

    # Load Model
    model = get_model(num_classes=num_classes, model_type=model_type).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    y_true_list = []
    y_pred_list = []

    print(f"Evaluating {checkpoint_path.name} on {dataset_name} ({split} split)...")
    with torch.no_grad():
        for images, targets in eval_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred_list.extend(preds)
            y_true_list.extend(targets.numpy())

    compute_and_save_confusion_matrix(
        y_true=y_true_list,
        y_pred=y_pred_list,
        model_name=model_type,
        dataset_name=dataset_name,
        split=split,
    )


def main():
    parser = argparse.ArgumentParser(description="Traffic Sign Confusion Matrix Generator")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="german",
        choices=["german", "belgian", "chinese"],
        help="Dataset benchmark",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="raw_cnn",
        choices=["raw_cnn", "basic_cnn", "vgg19", "vgg19_bn"],
        help="Model architecture",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Evaluation split",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    evaluate_checkpoint_and_generate_cm(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        model_type=args.model_type,
        split=args.split,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
