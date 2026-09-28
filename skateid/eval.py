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


def format_confusion_matrix(
    cm: List[List[int]],
    labels: List[str],
    max_labels: Optional[int] = 12,
) -> str:
    """Format a confusion matrix as a readable ASCII table.

    A full 23-class matrix is ~576 characters wide, which no terminal shows in
    one piece, and M0's exit-gate deliverable is precisely "prints a confusion
    matrix". So beyond ``max_labels`` classes the table keeps the
    highest-support ones and collapses the remainder into a single ``(other)``
    row and column, stating how many were folded in. Nothing is hidden silently:
    the per-class numbers for *every* class are still in the classification
    report printed alongside, and the full matrix is written to the metrics JSON
    when ``--out`` is used.

    ``max_labels=None`` prints every class, which is what you want for a 2-class
    problem or when dumping to a file.
    """
    support = [sum(row) for row in cm]
    total = sum(support)

    if max_labels is None or len(labels) <= max_labels:
        keep = list(range(len(labels)))
        folded = 0
    else:
        # Highest support first, then re-sort ascending for a stable display order.
        order = sorted(range(len(labels)), key=lambda i: (-support[i], labels[i]))
        keep = sorted(order[:max_labels])
        folded = len(labels) - len(keep)

    kept_set = set(keep)
    dropped = [i for i in range(len(labels)) if i not in kept_set]

    # Cells for the kept rows/columns come straight from the matrix. The (other)
    # row and column are computed by subtraction, so the grand total is preserved
    # and the table still adds up.
    rows = []
    for i in keep:
        kept_cells = [cm[i][j] for j in keep]
        rest = (sum(cm[i]) - sum(kept_cells)) if dropped else None
        rows.append(kept_cells + ([rest] if dropped else []))

    if dropped:
        other_row = [sum(cm[i][j] for i in dropped) for j in keep]
        other_row.append(sum(sum(cm[i]) for i in dropped) - sum(other_row))
        rows.append(other_row)

    shown = [labels[i] for i in keep]
    other_label = f"(other x{folded})" if folded else None
    row_labels = list(shown) + ([other_label] if folded else [])
    column_labels = list(shown) + ([other_label] if folded else [])

    width = max(max(len(label) for label in column_labels + row_labels), 8) + 2
    header = f"{'True \\ Pred':<{width}}" + "".join(f"{lbl:>{width}}" for lbl in column_labels)
    lines = [header, "-" * len(header)]
    for label, row in zip(row_labels, rows):
        lines.append(f"{label:<{width}}" + "".join(f"{val:>{width}}" for val in row))

    if folded:
        lines.append(
            f"({folded} lower-support class(es) folded into (other); "
            f"{total} samples total, all classes in the report below)"
        )
    return "\n".join(lines)
