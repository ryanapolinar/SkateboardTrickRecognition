"""Baseline evaluators and metrics calculation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


class MajorityClassBaseline:
    """B0: Always predicts the most frequent class seen during fit."""

    def __init__(self) -> None:
        self.majority_class: Optional[str] = None
        self.classes_: List[str] = []

    def fit(self, y: List[str] | pd.Series | np.ndarray) -> MajorityClassBaseline:
        s = pd.Series(y)
        self.majority_class = str(s.mode()[0])
        self.classes_ = sorted(s.unique().tolist())
        return self

    def predict(self, n_samples: int) -> List[str]:
        if self.majority_class is None:
            raise RuntimeError("Baseline is not fitted yet.")
        return [self.majority_class] * n_samples


def evaluate_predictions(
    y_true: List[str] | pd.Series | np.ndarray,
    y_pred: List[str] | pd.Series | np.ndarray,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute accuracy, macro F1, and confusion matrix dict."""
    y_true_list = [str(x) for x in y_true]
    y_pred_list = [str(x) for x in y_pred]

    if labels is None:
        labels = sorted(list(set(y_true_list) | set(y_pred_list)))

    acc = float(accuracy_score(y_true_list, y_pred_list))
    macro_f1 = float(f1_score(y_true_list, y_pred_list, average="macro", zero_division=0))
    cm = confusion_matrix(y_true_list, y_pred_list, labels=labels)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "report": classification_report(y_true_list, y_pred_list, labels=labels, zero_division=0),
    }


def format_confusion_matrix(cm: List[List[int]], labels: List[str]) -> str:
    """Format confusion matrix as a clean ASCII table."""
    col_width = max(max(len(label) for label in labels), 8) + 2
    title_cell = "True \\ Pred"
    header = f"{title_cell:<{col_width}}" + "".join(f"{lbl:>{col_width}}" for lbl in labels)
    lines = [header, "-" * len(header)]
    for i, row in enumerate(cm):
        row_str = f"{labels[i]:<{col_width}}" + "".join(f"{val:>{col_width}}" for val in row)
        lines.append(row_str)
    return "\n".join(lines)
