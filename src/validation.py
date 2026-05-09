"""Cross-validation, grid search, and learning curve utilities."""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple, Type

import numpy as np

from .metrics import accuracy_score
from .utils import flatten_dict_grid


ModelFactory = Callable[[], object]


def stratified_kfold_indices(y: np.ndarray, k: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    folds_per_class: Dict[int, List[np.ndarray]] = {}
    y = np.asarray(y)

    for cls in np.unique(y):
        indices = np.where(y == cls)[0]
        rng.shuffle(indices)
        folds_per_class[int(cls)] = np.array_split(indices, k)

    folds = []
    for fold_idx in range(k):
        val_parts = []
        train_parts = []
        for cls in folds_per_class.keys():
            val_parts.append(folds_per_class[cls][fold_idx])
            train_parts.extend([folds_per_class[cls][i] for i in range(k) if i != fold_idx])
        val_idx = np.concatenate(val_parts)
        train_idx = np.concatenate(train_parts)
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        folds.append((train_idx, val_idx))
    return folds


def grid_search_cv(
    model_class,
    param_grid: Dict[str, Sequence],
    x: np.ndarray,
    y: np.ndarray,
    k: int = 3,
    seed: int = 42,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    combinations = flatten_dict_grid(param_grid)
    folds = stratified_kfold_indices(y, k=k, seed=seed)
    results: List[Dict[str, object]] = []

    best_score = -np.inf
    best_params: Dict[str, object] = {}

    for params in combinations:
        fold_scores = []
        for train_idx, val_idx in folds:
            model = model_class(**params)
            model.fit(x[train_idx], y[train_idx])
            preds = model.predict(x[val_idx])
            fold_scores.append(accuracy_score(y[val_idx], preds))
        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        record = {
            "params": params,
            "mean_accuracy": mean_score,
            "std_accuracy": std_score,
            "fold_scores": fold_scores,
        }
        results.append(record)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    results.sort(key=lambda item: item["mean_accuracy"], reverse=True)
    return best_params, results


def compute_learning_curve(
    model_factory: Callable[[], object],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    train_sizes: Sequence[float],
    seed: int = 42,
) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    n_samples = len(x_train)
    indices = rng.permutation(n_samples)
    x_shuffled = x_train[indices]
    y_shuffled = y_train[indices]

    curve = []
    for frac in train_sizes:
        subset_size = max(2, int(frac * n_samples)) if frac <= 1 else int(frac)
        subset_size = min(subset_size, n_samples)
        model = model_factory()
        model.fit(x_shuffled[:subset_size], y_shuffled[:subset_size])
        train_pred = model.predict(x_shuffled[:subset_size])
        val_pred = model.predict(x_val)
        curve.append(
            {
                "train_size": int(subset_size),
                "train_accuracy": float(accuracy_score(y_shuffled[:subset_size], train_pred)),
                "val_accuracy": float(accuracy_score(y_val, val_pred)),
            }
        )
    return curve
