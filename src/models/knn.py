"""KNN implemented manually with optimized low-level NumPy distance computation.

The classifier logic is still handwritten:
- store training set manually
- compute distances manually
- choose k nearest manually
- vote manually

No ML library is used.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class KNNClassifier:
    def __init__(self, k: int = 3, distance: str = "euclidean", batch_size: int = 256) -> None:
        self.k = k
        self.distance = distance
        self.batch_size = batch_size
        self.x_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self.x_train_ = np.asarray(x, dtype=np.float64)
        self.y_train_ = np.asarray(y)
        self.classes_ = np.unique(self.y_train_)
        return self

    def _distances_to_one(self, row: np.ndarray) -> np.ndarray:
        assert self.x_train_ is not None
        if self.distance == "euclidean":
            diff = self.x_train_ - row
            return np.sqrt(np.sum(diff * diff, axis=1))
        if self.distance == "manhattan":
            return np.sum(np.abs(self.x_train_ - row), axis=1)
        raise ValueError(f"Unsupported distance metric: {self.distance}")

    def _predict_one(self, row: np.ndarray):
        if self.x_train_ is None or self.y_train_ is None:
            raise ValueError("KNN must be fitted before prediction.")
        distances = self._distances_to_one(row)
        k = min(self.k, len(distances))
        nearest_idx = np.argpartition(distances, k - 1)[:k]
        nearest_labels = self.y_train_[nearest_idx]

        counts = {}
        for label in nearest_labels:
            label = int(label)
            counts[label] = counts.get(label, 0) + 1

        best_label = None
        best_count = -1
        for label in sorted(counts.keys()):
            count = counts[label]
            if count > best_count:
                best_count = count
                best_label = label
        return best_label

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.x_train_ is None or self.y_train_ is None:
            raise ValueError("KNN must be fitted before prediction.")
        x = np.asarray(x, dtype=np.float64)
        preds = np.zeros(len(x), dtype=self.y_train_.dtype)
        for sample_index in range(len(x)):
            preds[sample_index] = self._predict_one(x[sample_index])
        return preds
