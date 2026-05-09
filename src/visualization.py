"""Visualization helpers using matplotlib."""
from __future__ import annotations

import os
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence, title: str, save_path: str) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    threshold = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_loss_curve(losses: Sequence[float], title: str, save_path: str) -> None:
    if not losses:
        return
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_learning_curve(curve: List[dict], title: str, save_path: str) -> None:
    train_sizes = [item["train_size"] for item in curve]
    train_acc = [item["train_accuracy"] for item in curve]
    val_acc = [item["val_accuracy"] for item in curve]

    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_acc, marker="o", label="Train accuracy")
    plt.plot(train_sizes, val_acc, marker="s", label="Validation accuracy")
    plt.title(title)
    plt.xlabel("Training samples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()




def plot_input_samples(
    images: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    max_per_class: int = 8,
    classes: Sequence | None = None,
) -> None:
    if classes is None:
        classes = np.unique(labels)
    classes = list(classes)
    rows = len(classes)
    cols = max_per_class
    plt.figure(figsize=(1.5 * cols, 1.8 * rows))
    plot_idx = 1
    for cls in classes:
        cls_images = images[labels == cls][:max_per_class]
        for img in cls_images:
            plt.subplot(rows, cols, plot_idx)
            plt.imshow(img, cmap="gray")
            plt.title(f"Class {cls}")
            plt.axis("off")
            plot_idx += 1
        while (plot_idx - 1) % cols != 0:
            plt.subplot(rows, cols, plot_idx)
            plt.axis("off")
            plot_idx += 1
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_sample_predictions(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    max_items: int = 16,
) -> None:
    n = min(max_items, len(images))
    plt.figure(figsize=(10, 10))
    for i in range(n):
        plt.subplot(int(np.ceil(np.sqrt(max_items))), int(np.ceil(np.sqrt(max_items))), i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(f"T:{y_true[i]} P:{y_pred[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
