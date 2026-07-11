"""Gaussian Naive Bayes implemented manually with explicit loops.

The statistical formulas are implemented directly without any ML library.
"""
from __future__ import annotations
import math
import numpy as np

class GaussianNaiveBayes:
    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing = var_smoothing
        self.classes_: np.ndarray | None = None
        self.class_priors_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.vars_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        n_classes = len(self.classes_)
        n_features = x.shape[1]

        self.class_priors_ = np.zeros(n_classes, dtype=np.float64)
        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.vars_ = np.zeros((n_classes, n_features), dtype=np.float64)

        for class_index in range(n_classes):
            cls = self.classes_[class_index]
            class_rows = []
            for sample_index in range(len(y)):
                if y[sample_index] == cls:
                    class_rows.append(x[sample_index])

            if len(class_rows) == 0:
                continue

            x_cls = np.asarray(class_rows, dtype=np.float64)
            self.class_priors_[class_index] = len(x_cls) / len(x)

            for feature_index in range(n_features):
                total = 0.0
                for sample_index in range(len(x_cls)):
                    total += float(x_cls[sample_index, feature_index])
                mean_value = total / len(x_cls)
                self.means_[class_index, feature_index] = mean_value

                var_total = 0.0
                for sample_index in range(len(x_cls)):
                    diff = float(x_cls[sample_index, feature_index]) - mean_value
                    var_total += diff * diff
                self.vars_[class_index, feature_index] = var_total / len(x_cls) + self.var_smoothing
        return self

    def _joint_log_likelihood_one(self, row: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.class_priors_ is None or self.means_ is None or self.vars_ is None:
            raise ValueError("Model must be fitted before prediction.")

        scores = np.zeros(len(self.classes_), dtype=np.float64)
        for class_index in range(len(self.classes_)):
            score = math.log(float(self.class_priors_[class_index]) + 1e-15)
            for feature_index in range(len(row)):
                mean_value = float(self.means_[class_index, feature_index])
                var_value = float(self.vars_[class_index, feature_index])
                x_value = float(row[feature_index])
                score += -0.5 * math.log(2.0 * math.pi * var_value)
                score += -0.5 * ((x_value - mean_value) ** 2) / var_value
            scores[class_index] = score
        return scores

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("Model must be fitted before prediction.")
        x = np.asarray(x, dtype=np.float64)
        preds = np.zeros(len(x), dtype=self.classes_.dtype)

        for sample_index in range(len(x)):
            scores = self._joint_log_likelihood_one(x[sample_index])
            best_index = 0
            best_score = scores[0]
            for class_index in range(1, len(scores)):
                if scores[class_index] > best_score:
                    best_score = scores[class_index]
                    best_index = class_index
            preds[sample_index] = self.classes_[best_index]
        return preds
