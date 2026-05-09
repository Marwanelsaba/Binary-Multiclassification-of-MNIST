"""CART-style decision tree classifier implemented from scratch.

This implementation intentionally limits the number of threshold candidates per feature
so it remains practical on MNIST-sized experiments without scikit-learn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TreeNode:
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    value: Optional[int] = None
    class_counts: Optional[np.ndarray] = None


class DecisionTreeClassifierScratch:
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[str | int] = None,
        n_thresholds: int = 10,
        random_state: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.random_state = random_state

        self.root_: Optional[TreeNode] = None
        self.n_classes_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.rng_ = np.random.default_rng(random_state)

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[Optional[int], Optional[float], float]:
        n_samples, n_features = x.shape
        best_feature = None
        best_threshold = None
        best_gain = -np.inf
        parent_impurity = self._gini(y)

        if self.max_features is None:
            feature_indices = np.arange(n_features)
        elif self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
            feature_indices = self.rng_.choice(n_features, size=k, replace=False)
        elif isinstance(self.max_features, int):
            k = min(self.max_features, n_features)
            feature_indices = self.rng_.choice(n_features, size=k, replace=False)
        else:
            raise ValueError(f"Unsupported max_features value: {self.max_features}")

        for feature in feature_indices:
            values = x[:, feature]
            unique_vals = np.unique(values)
            if len(unique_vals) <= 1:
                continue

            if len(unique_vals) <= self.n_thresholds:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            else:
                percentiles = np.linspace(5, 95, self.n_thresholds)
                thresholds = np.percentile(values, percentiles)
                thresholds = np.unique(thresholds)

            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                left_impurity = self._gini(y[left_mask])
                right_impurity = self._gini(y[right_mask])
                weighted_impurity = (
                    np.sum(left_mask) / n_samples * left_impurity
                    + np.sum(right_mask) / n_samples * right_impurity
                )
                gain = parent_impurity - weighted_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature)
                    best_threshold = float(threshold)
        return best_feature, best_threshold, best_gain

    def _build_tree(self, x: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        values, counts = np.unique(y, return_counts=True)
        predicted_class = int(values[np.argmax(counts)])
        class_counts = np.zeros(self.n_classes_, dtype=np.int64)
        assert self.classes_ is not None and self.n_classes_ is not None
        for cls, count in zip(values, counts):
            class_index = np.where(self.classes_ == cls)[0][0]
            class_counts[class_index] = count

        node = TreeNode(value=predicted_class, class_counts=class_counts)

        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(values) == 1
        ):
            return node

        feature, threshold, gain = self._best_split(x, y)
        if feature is None or threshold is None or gain <= 0:
            return node

        left_mask = x[:, feature] <= threshold
        right_mask = ~left_mask
        node.feature_index = feature
        node.threshold = threshold
        node.left = self._build_tree(x[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(x[right_mask], y[right_mask], depth + 1)
        return node

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifierScratch":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.root_ = self._build_tree(x, y, depth=0)
        return self

    def _predict_one(self, row: np.ndarray, node: TreeNode) -> int:
        while node.feature_index is not None and node.threshold is not None and node.left is not None and node.right is not None:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        assert node.value is not None
        return node.value

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise ValueError("Decision tree must be fitted before prediction.")
        x = np.asarray(x, dtype=np.float64)
        return np.array([self._predict_one(row, self.root_) for row in x])
