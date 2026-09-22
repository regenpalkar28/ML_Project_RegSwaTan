"""
Metrics calculation module for Traffic Sign Recognition.
Calculates Accuracy, Precision, Recall, F1-Scores (Macro, Weighted, Micro),
and per-class performance tables.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    num_classes: int,
    class_labels: Optional[List[str]] = None,
) -> Dict[str, Union[float, Dict, pd.DataFrame]]:
    """
    Computes comprehensive classification metrics.

    Args:
        y_true: Ground truth class indices.
        y_pred: Predicted class indices.
        num_classes: Total number of classes.
        class_labels: Optional human-readable class names.

    Returns:
        Dict with overall accuracy, macro/weighted F1, and per-class metrics DataFrame.
    """
    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)

    if class_labels is None or len(class_labels) != num_classes:
        class_labels = [f"Class_{i}" for i in range(num_classes)]

    acc = accuracy_score(y_true_arr, y_pred_arr)
    macro_f1 = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
    micro_f1 = f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)

    macro_precision = precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    macro_recall = recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)

    # Per-class metrics
    per_class_report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=list(range(num_classes)),
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )

    per_class_rows = []
    for i, name in enumerate(class_labels):
        metrics = per_class_report.get(name, {})
        per_class_rows.append(
            {
                "Class_ID": i,
                "Class_Name": name,
                "Precision": round(metrics.get("precision", 0.0), 4),
                "Recall": round(metrics.get("recall", 0.0), 4),
                "F1_Score": round(metrics.get("f1-score", 0.0), 4),
                "Support": int(metrics.get("support", 0)),
            }
        )

    per_class_df = pd.DataFrame(per_class_rows)

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "micro_f1": round(float(micro_f1), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "per_class_df": per_class_df,
        "raw_report_dict": per_class_report,
    }


def print_metrics_summary(metrics: Dict):
    """Prints a clean, formatted metrics summary to stdout."""
    print("=" * 65)
    print("                EVALUATION METRICS SUMMARY")
    print("=" * 65)
    print(f"  Overall Accuracy:   {metrics['accuracy'] * 100:.2f}%")
    print(f"  Macro F1-Score:     {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1-Score:  {metrics['weighted_f1']:.4f}")
    print(f"  Macro Precision:    {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:       {metrics['macro_recall']:.4f}")
    print("=" * 65)
