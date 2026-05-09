"""Feature extraction methods implemented from scratch using NumPy only.

Notes for the project:
- Flatten: manual reshape.
- PCA: custom PCA pipeline (mean centering + covariance + eigen decomposition).
  No sklearn PCA is used.
- HOG: fully manual implementation with gradient, cell histograms, and block normalization.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


class PCA:
    """Principal Component Analysis without sklearn.

    Uses a dual-covariance trick when the number of samples is smaller than the
    number of features. This keeps the implementation manual but much faster on
    MNIST-sized classroom experiments.
    """

    def __init__(self, n_components: int = 50) -> None:
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "PCA":
        x = np.asarray(x, dtype=np.float64)
        n_samples, n_features = x.shape

        self.mean_ = np.mean(x, axis=0)
        x_centered = x - self.mean_

        if n_samples <= n_features:
            gram = (x_centered @ x_centered.T) / max(n_samples - 1, 1)
            eigenvalues, eigenvectors_small = np.linalg.eigh(gram)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors_small = eigenvectors_small[:, order]

            positive = eigenvalues > 1e-12
            eigenvalues = eigenvalues[positive]
            eigenvectors_small = eigenvectors_small[:, positive]

            components = []
            for idx in range(len(eigenvalues)):
                direction = x_centered.T @ eigenvectors_small[:, idx]
                denom = np.sqrt(max(eigenvalues[idx], 1e-12) * max(n_samples - 1, 1))
                direction = direction / denom
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                components.append(direction)

            if components:
                components = np.column_stack(components)
            else:
                components = np.zeros((n_features, 0), dtype=np.float64)
        else:
            cov = (x_centered.T @ x_centered) / max(n_samples - 1, 1)
            eigenvalues, components = np.linalg.eigh(cov)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            components = components[:, order]

        n_components = min(self.n_components, components.shape[1])
        self.components_ = components[:, :n_components]
        self.explained_variance_ = np.clip(eigenvalues[:n_components], a_min=0.0, a_max=None)

        total_variance = float(np.sum(np.clip(eigenvalues, a_min=0.0, a_max=None)))
        if total_variance == 0.0:
            self.explained_variance_ratio_ = np.zeros_like(self.explained_variance_)
        else:
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise ValueError("PCA must be fitted before transform.")
        x_centered = np.asarray(x, dtype=np.float64) - self.mean_
        return x_centered @ self.components_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


class HOGExtractor:
    """Histogram of Oriented Gradients (HOG) implemented from scratch.

    Notes:
    - Uses unsigned orientations in [0, 180).
    - Uses bilinear interpolation between neighboring bins.
    - Performs L2 block normalization.
    """

    def __init__(self, cell_size: int = 4, block_size: int = 2, bins: int = 9, epsilon: float = 1e-6) -> None:
        self.cell_size = cell_size
        self.block_size = block_size
        self.bins = bins
        self.epsilon = epsilon

    @staticmethod
    def _gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gx = np.zeros_like(image, dtype=np.float64)
        gy = np.zeros_like(image, dtype=np.float64)

        gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
        gx[:, 0] = image[:, 1] - image[:, 0]
        gx[:, -1] = image[:, -1] - image[:, -2]

        gy[1:-1, :] = image[2:, :] - image[:-2, :]
        gy[0, :] = image[1, :] - image[0, :]
        gy[-1, :] = image[-1, :] - image[-2, :]
        return gx, gy

    def _cell_histograms(self, image: np.ndarray) -> np.ndarray:
        gx, gy = self._gradients(image)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        orientation = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0

        h, w = image.shape
        n_cells_y = h // self.cell_size
        n_cells_x = w // self.cell_size
        hist = np.zeros((n_cells_y, n_cells_x, self.bins), dtype=np.float64)
        bin_width = 180.0 / self.bins

        for cy in range(n_cells_y):
            for cx in range(n_cells_x):
                y0 = cy * self.cell_size
                y1 = y0 + self.cell_size
                x0 = cx * self.cell_size
                x1 = x0 + self.cell_size

                cell_mag = magnitude[y0:y1, x0:x1].ravel()
                cell_ori = orientation[y0:y1, x0:x1].ravel()

                for mag, ang in zip(cell_mag, cell_ori):
                    bin_position = ang / bin_width
                    left_bin = int(np.floor(bin_position)) % self.bins
                    right_bin = (left_bin + 1) % self.bins
                    right_weight = bin_position - np.floor(bin_position)
                    left_weight = 1.0 - right_weight
                    hist[cy, cx, left_bin] += mag * left_weight
                    hist[cy, cx, right_bin] += mag * right_weight
        return hist

    def _block_normalize(self, hist: np.ndarray) -> np.ndarray:
        n_cells_y, n_cells_x, _ = hist.shape
        by = n_cells_y - self.block_size + 1
        bx = n_cells_x - self.block_size + 1
        features: List[np.ndarray] = []
        for y in range(by):
            for x in range(bx):
                block = hist[y:y + self.block_size, x:x + self.block_size, :].ravel()
                norm = np.sqrt(np.sum(block ** 2) + self.epsilon ** 2)
                features.append(block / norm)
        if not features:
            return hist.ravel()
        return np.concatenate(features, axis=0)

    def extract_one(self, image: np.ndarray) -> np.ndarray:
        hist = self._cell_histograms(image)
        return self._block_normalize(hist)

    def fit(self, x: np.ndarray) -> "HOGExtractor":
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        feats = [self.extract_one(img) for img in x]
        return np.asarray(feats, dtype=np.float64)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.transform(x)
