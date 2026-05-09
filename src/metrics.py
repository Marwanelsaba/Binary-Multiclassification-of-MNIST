"""Evaluation metrics implemented from scratch using NumPy only."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[label_to_index[true_label], label_to_index[pred_label]] += 1
    return cm


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n_classes = cm.shape[0]
    precisions = np.zeros(n_classes)
    recalls = np.zeros(n_classes)
    f1s = np.zeros(n_classes)
    supports = np.sum(cm, axis=1)

    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        precisions[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recalls[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if (precisions[i] + recalls[i]) > 0:
            f1s[i] = 2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i])
        else:
            f1s[i] = 0.0

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    weighted_precision = float(np.sum(precisions * supports) / np.sum(supports))
    weighted_recall = float(np.sum(recalls * supports) / np.sum(supports))
    weighted_f1 = float(np.sum(f1s * supports) / np.sum(supports))

    return {
        "per_class_precision": precisions,
        "per_class_recall": recalls,
        "per_class_f1": f1s,
        "support": supports,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
    }


def classification_report_text(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> str:
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    metrics = precision_recall_f1(y_true, y_pred, labels=labels)
    lines: List[str] = []
    lines.append(f"{'Class':<10}{'Precision':>12}{'Recall':>12}{'F1-score':>12}{'Support':>12}")
    for idx, label in enumerate(labels):
        lines.append(
            f"{str(label):<10}"
            f"{metrics['per_class_precision'][idx]:>12.4f}"
            f"{metrics['per_class_recall'][idx]:>12.4f}"
            f"{metrics['per_class_f1'][idx]:>12.4f}"
            f"{int(metrics['support'][idx]):>12d}"
        )
    lines.append("")
    lines.append(f"{'Accuracy':<10}{accuracy_score(y_true, y_pred):>12.4f}")
    lines.append(f"{'Macro F1':<10}{metrics['macro_f1']:>12.4f}")
    lines.append(f"{'Weighted F1':<10}{metrics['weighted_f1']:>12.4f}")
    return "\n".join(lines)
