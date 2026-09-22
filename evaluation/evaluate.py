"""
Model Evaluation CLI Script for Traffic Sign Recognition.
Loads any saved PyTorch checkpoint (.pth) and computes:
- Overall Accuracy & Loss
- Macro, Weighted, and Micro F1-Scores, Precision, Recall
- Confusion Matrix (Raw CSV, Normalized CSV, Heatmap PNG)
- Per-Class Performance Breakdown (CSV & JSON)
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from tqdm import tqdm

from config import (
    CSV_RESULTS_DIR,
    DATASET_NUM_CLASSES,
    DEVICE,
    IMAGE_SIZE,
)
from data.dataset import get_dataloaders
from evaluation.confusion_matrix import compute_and_save_confusion_matrix, get_class_labels
from evaluation.metrics import compute_metrics, print_metrics_summary
from models.RawCNN import get_model


def evaluate_model(
    checkpoint_path: str,
    dataset_name: str = "german",
    model_type: str = "raw_cnn",
    split: str = "test",
    batch_size: int = 64,
    img_size: tuple = IMAGE_SIZE,
    data_dir: str = None,
):
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    num_classes = DATASET_NUM_CLASSES.get(dataset_name.lower(), 43)

    print("=" * 65)
    print(f"EVALUATION: {model_type.upper()} on {dataset_name.upper()} ({split.upper()} SET)")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Device:     {DEVICE}")
    print("=" * 65)

    # 1. Load DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        root_dir=data_dir,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=0,
    )
    eval_loader = test_loader if split.lower() == "test" else val_loader
    print(f"Evaluating {len(eval_loader.dataset):,} samples across {num_classes} classes...\n")

    # 2. Instantiate and load model
    model = get_model(num_classes=num_classes, model_type=model_type).to(DEVICE)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    y_true_list = []
    y_pred_list = []

    # 3. Inference loop
    with torch.no_grad():
        pbar = tqdm(eval_loader, desc=f"Evaluating ({split})", leave=False)
        for images, targets in pbar:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)

            y_pred_list.extend(preds.cpu().tolist())
            y_true_list.extend(targets.cpu().tolist())

    total_samples = len(y_true_list)
    avg_loss = running_loss / total_samples

    # 4. Compute Metrics (Accuracy, F1, Precision, Recall)
    class_labels = get_class_labels(dataset_name, num_classes)
    metrics = compute_metrics(y_true_list, y_pred_list, num_classes, class_labels)

    # Print summary
    print_metrics_summary(metrics)
    print(f"  Average Loss:       {avg_loss:.4f}")
    print("=" * 65)

    # 5. Save per-class metrics CSV
    prefix = f"{model_type.lower()}_{dataset_name.lower()}_{split.lower()}"
    per_class_csv_path = CSV_RESULTS_DIR / f"{prefix}_per_class_metrics.csv"
    metrics["per_class_df"].to_csv(per_class_csv_path, index=False)
    print(f"\nSaved per-class metrics table to: {per_class_csv_path.name}")

    # 6. Generate & Store Confusion Matrices
    print("\nGenerating and storing confusion matrix artifacts...")
    cm_files = compute_and_save_confusion_matrix(
        y_true=y_true_list,
        y_pred=y_pred_list,
        model_name=model_type,
        dataset_name=dataset_name,
        split=split,
        class_labels=class_labels,
    )

    return {
        "metrics": metrics,
        "loss": avg_loss,
        "per_class_csv": per_class_csv_path,
        "confusion_matrix_files": cm_files,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate any saved .pth checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved PyTorch model checkpoint (.pth)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="german",
        choices=["german", "belgian", "chinese"],
        help="Target dataset benchmark ('german', 'belgian', 'chinese')",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="raw_cnn",
        choices=["raw_cnn", "basic_cnn", "vgg19", "vgg19_bn"],
        help="Model architecture ('raw_cnn', 'vgg19', etc.)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Dataset split to evaluate ('test' or 'val')",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--data_dir", type=str, default=None, help="Custom dataset path")
    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        model_type=args.model_type,
        split=args.split,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
