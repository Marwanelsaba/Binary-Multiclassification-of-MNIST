"""Preprocessing helpers for images and tabular features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


class MinMaxNormalizer:
    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        self.feature_range = feature_range
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "MinMaxNormalizer":
        self.data_min_ = np.min(x, axis=0)
        self.data_max_ = np.max(x, axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.data_min_ is None or self.data_max_ is None:
            raise ValueError("MinMaxNormalizer must be fitted before transform.")
        low, high = self.feature_range
        denom = self.data_max_ - self.data_min_
        denom = np.where(denom == 0, 1.0, denom)
        scaled = (x - self.data_min_) / denom
        return scaled * (high - low) + low

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


class StandardScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("StandardScaler must be fitted before transform.")
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


class FlattenTransformer:
    def fit(self, x: np.ndarray) -> "FlattenTransformer":
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(x.shape[0], -1).astype(np.float64)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.transform(x)


def normalize_images_to_unit_range(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float64) / 255.0


def resize_images_nearest(images: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
    """Resize images using nearest-neighbor interpolation implemented with NumPy only."""
    n, old_h, old_w = images.shape
    new_h, new_w = new_size
    row_idx = np.round(np.linspace(0, old_h - 1, new_h)).astype(int)
    col_idx = np.round(np.linspace(0, old_w - 1, new_w)).astype(int)
    resized = images[:, row_idx][:, :, col_idx]
    return resized
