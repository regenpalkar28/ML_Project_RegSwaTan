"""
Automated Robustness & Corruption Benchmark Runner for Traffic Sign Recognition.
Evaluates trained models across Clean Baseline and 14 Hendrycks & Dietterich Corruptions
at 5 Severity Levels (70 Corrupted Test Conditions).
Exports tabular CSVs, computes mCE (mean Corruption Error), and generates degradation curves.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import (
    CHECKPOINTS_DIR,
    CSV_RESULTS_DIR,
    DATASET_NUM_CLASSES,
    DEVICE,
    IMAGE_SIZE,
    NORM_MEAN,
    NORM_STD,
    PLOTS_RESULTS_DIR,
)
from corruption import (
    BENCHMARK_CORRUPTIONS,
    CORRUPTION_CATEGORIES,
    apply_corruption,
)
from data.dataset import get_dataset
from evaluation.metrics import compute_metrics
from models.RawCNN import get_model


class CorruptedTransform:
    """
    Applies resolution-appropriate corruption on unnormalized PIL image,
    then converts to tensor and applies dataset normalization.
    """

    def __init__(
        self,
        corruption_type: Optional[str] = None,
        severity: int = 1,
        img_size: Tuple[int, int] = IMAGE_SIZE,
        mean: List[float] = NORM_MEAN,
        std: List[float] = NORM_STD,
    ):
        self.corruption_type = corruption_type
        self.severity = severity
        self.img_size = img_size
        self.resize = transforms.Resize(img_size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, img):
        img = self.resize(img)
        if self.corruption_type is not None:
            img = apply_corruption(img, self.corruption_type, severity=self.severity)
        tensor = self.to_tensor(img)
        return self.normalize(tensor)


@torch.no_grad()
def evaluate_condition(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Evaluating",
) -> Tuple[float, float, float, float]:
    """
    Evaluates model on a specific loader and computes:
    loss, accuracy, macro_f1, weighted_f1
    """
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []

    for images, targets in tqdm(loader, desc=desc, leave=False):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)

        y_true.extend(targets.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    total = len(y_true)
    avg_loss = running_loss / max(total, 1)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    acc = 100.0 * np.mean(y_true_arr == y_pred_arr)

    # Use scikit-learn metrics
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)

    return avg_loss, acc, macro_f1, weighted_f1


def plot_degradation_curves(
    df_results: pd.DataFrame,
    clean_acc: float,
    model_name: str,
    dataset_name: str,
    save_dir: Path,
):
    """
    Generates 4-panel publication-ready line plots grouped by corruption category:
    Noise, Blur, Weather, Digital.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    category_names = ["Noise", "Blur", "Weather", "Digital"]
    colors = sns.color_palette("tab10", 6)

    for idx, cat in enumerate(category_names):
        ax = axes[idx]
        corr_list = CORRUPTION_CATEGORIES.get(cat, [])
        cat_df = df_results[df_results["Corruption"].isin(corr_list)]

        for c_idx, c_name in enumerate(corr_list):
            subset = cat_df[cat_df["Corruption"] == c_name].sort_values("Severity")
            if not subset.empty:
                # Prepend severity 0 as Clean baseline
                severities = [0] + subset["Severity"].tolist()
                accuracies = [clean_acc] + subset["Accuracy"].tolist()
                clean_label = c_name.replace("_", " ").title()
                ax.plot(
                    severities,
                    accuracies,
                    marker="o",
                    linewidth=2.2,
                    markersize=6,
                    label=clean_label,
                    color=colors[c_idx % len(colors)],
                )

        ax.axhline(clean_acc, color="gray", linestyle="--", alpha=0.6, label="Clean Baseline")
        ax.set_title(f"{cat} Corruptions", fontsize=13, fontweight="bold")
        ax.set_xlabel("Severity Level (0 = Clean, 1-5 = Corrupted)", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_ylim(-2, 102)
        ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    plt.suptitle(
        f"Robustness Degradation Across 14 Corruptions: {model_name.upper()} on {dataset_name.upper()}",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()

    out_file = save_dir / f"degradation_curves_{model_name}_{dataset_name}.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  --> Saved degradation curves plot to: {out_file.name}")


def run_robustness_experiment(args):
    print("=" * 70)
    print("  TRAFFIC SIGN RECOGNITION: ROBUSTNESS BENCHMARK (70 CONDITIONS)")
    print("=" * 70)
    print(f"Model Architecture: {args.model_type.upper()}")
    print(f"Target Dataset:     {args.dataset.upper()}")
    print(f"Device:             {DEVICE}")
    print(f"Batch Size:         {args.batch_size}")
    print("=" * 70)

    # 1. Resolve Checkpoint
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
    else:
        # Auto-detect default best checkpoint
        candidates = [
            CHECKPOINTS_DIR / f"{args.model_type}_{args.dataset}_best.pth",
            CHECKPOINTS_DIR / f"{args.model_type}_german_best.pth",
            CHECKPOINTS_DIR / f"{args.model_type}_best.pth",
        ]
        ckpt_path = next((c for c in candidates if c.exists()), None)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"No checkpoint found. Searched:\n"
                + "\n".join(f" - {c}" for c in candidates)
                + "\nPlease specify --checkpoint <path>."
            )

    print(f"Loading checkpoint: {ckpt_path.name}")
    num_classes = DATASET_NUM_CLASSES.get(args.dataset.lower(), 43)

    # Instantiate model
    model = get_model(
        num_classes=num_classes,
        model_type=args.model_type,
        compact_head=args.compact_head,
    ).to(DEVICE)

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # 2. Evaluate Clean Baseline
    print("\n[Phase 1/2] Evaluating Clean Baseline Test Set...")
    clean_transform = CorruptedTransform(
        corruption_type=None,
        img_size=args.img_size,
    )
    clean_dataset = get_dataset(
        dataset_name=args.dataset,
        split="test",
        root_dir=args.data_dir,
        transform=clean_transform,
    )
    clean_loader = DataLoader(
        clean_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    clean_loss, clean_acc, clean_macro_f1, clean_weighted_f1 = evaluate_condition(
        model, clean_loader, criterion, DEVICE, desc="Clean Baseline"
    )

    print("-" * 70)
    print(f"CLEAN BASELINE PERFORMANCE:")
    print(f"  Accuracy:    {clean_acc:.2f}%")
    print(f"  Macro F1:    {clean_macro_f1:.4f}")
    print(f"  Weighted F1: {clean_weighted_f1:.4f}")
    print(f"  Loss:        {clean_loss:.4f}")
    print("-" * 70)

    # 3. Evaluate Corrupted Test Sets (14 types x 5 severities = 70 conditions)
    print("\n[Phase 2/2] Running 70 Corrupted Test Conditions (14 Corruptions x 5 Severities)...")
    records = []
    severities = args.severities
    corruptions_to_test = args.corruptions if args.corruptions else BENCHMARK_CORRUPTIONS

    total_conditions = len(corruptions_to_test) * len(severities)
    current_cond = 0
    start_time = time.time()

    for corr in corruptions_to_test:
        print(f"\n>>> Corruption: {corr.upper()}")
        for sev in severities:
            current_cond += 1
            cond_desc = f"[{current_cond:02d}/{total_conditions:02d}] {corr} (Sev {sev})"

            # Build on-the-fly corrupted test loader
            c_transform = CorruptedTransform(
                corruption_type=corr,
                severity=sev,
                img_size=args.img_size,
            )
            c_dataset = get_dataset(
                dataset_name=args.dataset,
                split="test",
                root_dir=args.data_dir,
                transform=c_transform,
            )
            c_loader = DataLoader(
                c_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=False,
            )

            c_loss, c_acc, c_macro_f1, c_weighted_f1 = evaluate_condition(
                model, c_loader, criterion, DEVICE, desc=cond_desc
            )

            acc_drop = clean_acc - c_acc
            rel_drop = (acc_drop / clean_acc * 100.0) if clean_acc > 0 else 0.0

            # Find category
            category = "Other"
            for cat, members in CORRUPTION_CATEGORIES.items():
                if corr in members:
                    category = cat
                    break

            records.append(
                {
                    "Model": args.model_type,
                    "Dataset": args.dataset,
                    "Category": category,
                    "Corruption": corr,
                    "Severity": sev,
                    "Accuracy": round(c_acc, 2),
                    "Macro_F1": round(c_macro_f1, 4),
                    "Weighted_F1": round(c_weighted_f1, 4),
                    "Loss": round(c_loss, 4),
                    "Accuracy_Drop": round(acc_drop, 2),
                    "Relative_Drop_Pct": round(rel_drop, 2),
                }
            )

            print(
                f"  Sev {sev}: Acc={c_acc:5.2f}% (Drop: -{acc_drop:5.2f}%) | "
                f"Macro F1={c_macro_f1:.4f} | Loss={c_loss:.4f}"
            )

    elapsed_min = (time.time() - start_time) / 60.0
    print("=" * 70)
    print(f"Robustness Benchmark Finished in {elapsed_min:.2f} minutes.")

    # 4. Save Tabular CSV Results
    df_results = pd.DataFrame(records)

    # Detailed 70-row CSV
    detailed_csv = CSV_RESULTS_DIR / f"robustness_{args.model_type}_{args.dataset}_detailed.csv"
    df_results.to_csv(detailed_csv, index=False)
    print(f"Saved detailed 70-condition CSV to: {detailed_csv.name}")

    # Matrix Table (14 rows x 5 severities + Mean Accuracy)
    pivot_acc = df_results.pivot(index="Corruption", columns="Severity", values="Accuracy")
    pivot_acc["Mean_Acc"] = pivot_acc.mean(axis=1).round(2)
    pivot_acc["Clean_Acc"] = round(clean_acc, 2)
    matrix_csv = CSV_RESULTS_DIR / f"robustness_{args.model_type}_{args.dataset}_matrix.csv"
    pivot_acc.to_csv(matrix_csv)
    print(f"Saved summary matrix CSV to:       {matrix_csv.name}")

    # 5. Category-wise Summary & mCE (Mean Corruption Error)
    cat_summary = (
        df_results.groupby("Category")["Accuracy"]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={"Accuracy": "Mean_Corrupted_Acc"})
    )
    cat_summary["Clean_Baseline"] = round(clean_acc, 2)
    cat_summary["Mean_Drop"] = (clean_acc - cat_summary["Mean_Corrupted_Acc"]).round(2)
    print("\n" + "=" * 55)
    print("  CATEGORY ROBUSTNESS SUMMARY")
    print("=" * 55)
    print(cat_summary.to_string(index=False))

    # Overall Mean Corruption Accuracy
    mean_corrupted_acc = df_results["Accuracy"].mean()
    print("-" * 55)
    print(f"Clean Baseline Accuracy:          {clean_acc:.2f}%")
    print(f"Mean Corrupted Accuracy (All 70): {mean_corrupted_acc:.2f}%")
    print(f"Average Accuracy Degradation:     -{clean_acc - mean_corrupted_acc:.2f}%")
    print("=" * 55)

    # 6. Generate Publication Degradation Plots
    if args.save_plots:
        print("\nGenerating publication figures...")
        plot_degradation_curves(
            df_results=df_results,
            clean_acc=clean_acc,
            model_name=args.model_type,
            dataset_name=args.dataset,
            save_dir=PLOTS_RESULTS_DIR,
        )
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Sign Robustness & Corruption Benchmark")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model .pth checkpoint (auto-detects if None)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="vgg19_bn",
        choices=["raw_cnn", "basic_cnn", "cnn", "vgg19", "vgg19_bn"],
        help="Model architecture",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="german",
        choices=["german", "belgian", "chinese"],
        help="Dataset benchmark",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Custom dataset root directory",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--img_size", type=int, nargs=2, default=IMAGE_SIZE, help="Image size (H W)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--compact_head",
        action="store_true",
        default=True,
        help="Use compact classification head for VGG",
    )
    parser.add_argument(
        "--severities",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Severity levels to evaluate",
    )
    parser.add_argument(
        "--corruptions",
        type=str,
        nargs="+",
        default=None,
        help="Subset of corruptions to evaluate (defaults to all 14)",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        default=True,
        help="Generate degradation curve figures into results/plots/",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_robustness_experiment(args)
