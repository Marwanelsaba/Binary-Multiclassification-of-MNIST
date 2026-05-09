"""Random forest classifier implemented from scratch using the custom decision tree."""
from __future__ import annotations

from typing import List

import numpy as np

from .decision_tree import DecisionTreeClassifierScratch


class RandomForestClassifierScratch:
    def __init__(
        self,
        n_estimators: int = 15,
        max_depth: int = 12,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | int | None = "sqrt",
        n_thresholds: int = 10,
        bootstrap: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees_: List[DecisionTreeClassifierScratch] = []
        self.classes_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForestClassifierScratch":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []

        for i in range(self.n_estimators):
            if self.bootstrap:
                indices = rng.choice(len(x), size=len(x), replace=True)
            else:
                indices = np.arange(len(x))
            x_sample = x[indices]
            y_sample = y[indices]

            tree = DecisionTreeClassifierScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                n_thresholds=self.n_thresholds,
                random_state=self.random_state + i,
            )
            tree.fit(x_sample, y_sample)
            self.trees_.append(tree)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise ValueError("Random forest must be fitted before prediction.")
        all_preds = np.column_stack([tree.predict(x) for tree in self.trees_])
        final_preds = []
        for row in all_preds:
            values, counts = np.unique(row, return_counts=True)
            max_count = np.max(counts)
            winners = values[counts == max_count]
            final_preds.append(np.min(winners))
        return np.asarray(final_preds)
