"""Manual logistic regression optimized with NumPy math.

The algorithm is still implemented from scratch:
- custom training loop
- custom gradients
- custom sigmoid / softmax / loss
- no scikit-learn / no ready-made optimizer

Only low-level NumPy array math is used to speed up execution.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class LogisticRegressionScratch:
    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 30,
        batch_size: int = 128,
        reg_type: Optional[str] = None,
        reg_strength: float = 0.0,
        random_state: int = 42,
        verbose: bool = False,
        loss_eval_interval: int = 5,
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg_type = reg_type
        self.reg_strength = reg_strength
        self.random_state = random_state
        self.verbose = verbose
        self.loss_eval_interval = max(1, int(loss_eval_interval))

        self.weights_: np.ndarray | None = None
        self.bias_: np.ndarray | float | None = None
        self.classes_: np.ndarray | None = None
        self.loss_history_: List[float] = []
        self.mode_: str | None = None

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        sums = np.sum(exp_values, axis=1, keepdims=True)
        sums = np.where(sums == 0.0, 1.0, sums)
        return exp_values / sums

    @staticmethod
    def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
        result = np.zeros((len(y), n_classes), dtype=np.float64)
        result[np.arange(len(y)), y.astype(int)] = 1.0
        return result

    def _regularization_loss(self, weights: np.ndarray) -> float:
        if self.reg_type is None or self.reg_strength <= 0:
            return 0.0
        reg = self.reg_type.upper()
        if reg == "L2":
            return 0.5 * self.reg_strength * float(np.sum(weights * weights))
        if reg == "L1":
            return self.reg_strength * float(np.sum(np.abs(weights)))
        raise ValueError(f"Unsupported regularization type: {self.reg_type}")

    def _regularization_grad(self, weights: np.ndarray) -> np.ndarray:
        if self.reg_type is None or self.reg_strength <= 0:
            return np.zeros_like(weights)
        reg = self.reg_type.upper()
        if reg == "L2":
            return self.reg_strength * weights
        if reg == "L1":
            return self.reg_strength * np.sign(weights)
        raise ValueError(f"Unsupported regularization type: {self.reg_type}")

    def _binary_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        assert self.weights_ is not None and self.bias_ is not None
        eps = 1e-12
        logits = x @ self.weights_ + float(self.bias_)
        probs = self._sigmoid(logits)
        loss = -np.mean(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))
        return float(loss + self._regularization_loss(self.weights_))

    def _multiclass_loss(self, x: np.ndarray, y_one_hot: np.ndarray) -> float:
        assert self.weights_ is not None and self.bias_ is not None
        eps = 1e-12
        logits = x @ self.weights_ + np.asarray(self.bias_, dtype=np.float64)
        probs = self._softmax(logits)
        loss = -np.mean(np.sum(y_one_hot * np.log(probs + eps), axis=1))
        return float(loss + self._regularization_loss(self.weights_))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = x.shape
        n_classes = len(self.classes_)
        y_encoded = np.searchsorted(self.classes_, y)
        rng = np.random.default_rng(self.random_state)
        self.loss_history_ = []

        if n_classes == 2:
            self.mode_ = "binary"
            self.weights_ = rng.normal(0.0, 0.01, size=n_features)
            self.bias_ = 0.0
            y_float = y_encoded.astype(np.float64)

            for epoch in range(self.epochs):
                perm = rng.permutation(n_samples)
                x_shuffled = x[perm]
                y_shuffled = y_float[perm]

                for start in range(0, n_samples, self.batch_size):
                    end = min(start + self.batch_size, n_samples)
                    xb = x_shuffled[start:end]
                    yb = y_shuffled[start:end]

                    logits = xb @ self.weights_ + float(self.bias_)
                    probs = self._sigmoid(logits)
                    errors = probs - yb

                    grad_w = (xb.T @ errors) / len(xb)
                    grad_w += self._regularization_grad(self.weights_)
                    grad_b = float(np.mean(errors))

                    self.weights_ -= self.learning_rate * grad_w
                    self.bias_ = float(self.bias_) - self.learning_rate * grad_b

                if epoch == 0 or (epoch + 1) % self.loss_eval_interval == 0 or epoch == self.epochs - 1:
                    loss = self._binary_loss(x, y_float)
                    self.loss_history_.append(loss)
                    if self.verbose:
                        print(f"[Binary LR] Epoch {epoch + 1}/{self.epochs} - loss={loss:.6f}")
        else:
            self.mode_ = "multiclass"
            self.weights_ = rng.normal(0.0, 0.01, size=(n_features, n_classes))
            self.bias_ = np.zeros(n_classes, dtype=np.float64)
            y_one_hot = self._one_hot(y_encoded, n_classes)

            for epoch in range(self.epochs):
                perm = rng.permutation(n_samples)
                x_shuffled = x[perm]
                y_shuffled = y_one_hot[perm]

                for start in range(0, n_samples, self.batch_size):
                    end = min(start + self.batch_size, n_samples)
                    xb = x_shuffled[start:end]
                    yb = y_shuffled[start:end]

                    logits = xb @ self.weights_ + self.bias_
                    probs = self._softmax(logits)
                    errors = probs - yb

                    grad_w = (xb.T @ errors) / len(xb)
                    grad_w += self._regularization_grad(self.weights_)
                    grad_b = np.mean(errors, axis=0)

                    self.weights_ -= self.learning_rate * grad_w
                    self.bias_ = np.asarray(self.bias_, dtype=np.float64) - self.learning_rate * grad_b

                if epoch == 0 or (epoch + 1) % self.loss_eval_interval == 0 or epoch == self.epochs - 1:
                    loss = self._multiclass_loss(x, y_one_hot)
                    self.loss_history_.append(loss)
                    if self.verbose:
                        print(f"[Multiclass LR] Epoch {epoch + 1}/{self.epochs} - loss={loss:.6f}")
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights_ is None or self.bias_ is None or self.classes_ is None:
            raise ValueError("Model must be fitted before prediction.")
        x = np.asarray(x, dtype=np.float64)

        if self.mode_ == "binary":
            logits = x @ self.weights_ + float(self.bias_)
            probs_pos = self._sigmoid(logits)
            return np.column_stack([1.0 - probs_pos, probs_pos])

        logits = x @ self.weights_ + np.asarray(self.bias_, dtype=np.float64)
        return self._softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("Model must be fitted before prediction.")
        probs = self.predict_proba(x)
        pred_idx = np.argmax(probs, axis=1)
        return self.classes_[pred_idx]
