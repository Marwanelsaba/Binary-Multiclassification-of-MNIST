"""Data loading and splitting utilities.

The project implements the machine-learning pipeline from scratch, but the user
requested that the MNIST dataset itself be imported using TensorFlow/Keras.
This loader therefore supports TensorFlow first, with an IDX fallback only when
TensorFlow is unavailable.
"""
from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .utils import class_counts, ensure_dir


MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


class MNISTLoader:
    def __init__(self, data_dir: str = "data", source: str = "tensorflow") -> None:
        self.data_dir = ensure_dir(data_dir)
        self.source = source

    def _download_if_missing(self, key: str) -> str:
        url = MNIST_URLS[key]
        filename = os.path.join(self.data_dir, os.path.basename(url))
        if not os.path.exists(filename):
            print(f"Downloading {url} -> {filename}")
            urllib.request.urlretrieve(url, filename)
        return filename

    @staticmethod
    def _read_idx_images(gz_path: str) -> np.ndarray:
        with gzip.open(gz_path, "rb") as f:
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(f"Invalid image file magic number: {magic}")
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows, cols)

    @staticmethod
    def _read_idx_labels(gz_path: str) -> np.ndarray:
        with gzip.open(gz_path, "rb") as f:
            magic, num_labels = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError(f"Invalid label file magic number: {magic}")
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_labels)

    def _load_from_tensorflow(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        try:
            from tensorflow.keras.datasets import mnist  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "TensorFlow is not available. Install it with 'pip install tensorflow'."
            ) from exc

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return (
            np.asarray(x_train, dtype=np.uint8),
            np.asarray(y_train, dtype=np.uint8),
            np.asarray(x_test, dtype=np.uint8),
            np.asarray(y_test, dtype=np.uint8),
        )

    def _load_from_idx(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_images_path = self._download_if_missing("train_images")
        train_labels_path = self._download_if_missing("train_labels")
        test_images_path = self._download_if_missing("test_images")
        test_labels_path = self._download_if_missing("test_labels")

        x_train = self._read_idx_images(train_images_path)
        y_train = self._read_idx_labels(train_labels_path)
        x_test = self._read_idx_images(test_images_path)
        y_test = self._read_idx_labels(test_labels_path)
        return x_train, y_train, x_test, y_test

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.source == "tensorflow":
            try:
                print("Loading MNIST using tensorflow.keras.datasets.mnist ...")
                return self._load_from_tensorflow()
            except ImportError:
                print("TensorFlow loader unavailable. Falling back to raw IDX files.")
                return self._load_from_idx()
        if self.source == "idx":
            print("Loading MNIST from raw IDX files ...")
            return self._load_from_idx()
        raise ValueError(f"Unsupported MNIST source: {self.source}")


def filter_classes(
    x: np.ndarray,
    y: np.ndarray,
    selected_classes: Sequence[int],
    remap_to_binary: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isin(y, selected_classes)
    x_filtered = x[mask]
    y_filtered = y[mask]
    if remap_to_binary:
        class_to_index = {int(cls): idx for idx, cls in enumerate(selected_classes)}
        y_filtered = np.array([class_to_index[int(label)] for label in y_filtered], dtype=np.int64)
    return x_filtered, y_filtered


def subsample_stratified(
    x: np.ndarray,
    y: np.ndarray,
    per_class: Optional[int] = None,
    total_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified subsampling.

    Use `per_class` when you want an equal number from each class.
    Use `total_samples` for approximate global size while preserving class ratios.
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)

    indices = []
    if per_class is not None:
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            take = min(per_class, len(cls_idx))
            indices.append(cls_idx[:take])
    elif total_samples is not None:
        total_count = len(y)
        for cls, count in zip(classes, counts):
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            take = max(1, int(round((count / total_count) * total_samples)))
            take = min(take, len(cls_idx))
            indices.append(cls_idx[:take])
    else:
        return x.copy(), y.copy()

    chosen = np.concatenate(indices)
    rng.shuffle(chosen)
    return x[chosen], y[chosen]


def stratified_train_val_test_split(
    x: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Manual stratified split into train/val/test."""
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1:
        raise ValueError("val_ratio + test_ratio must be in [0, 1).")

    rng = np.random.default_rng(seed)
    train_idx_list = []
    val_idx_list = []
    test_idx_list = []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_test = int(np.floor(test_ratio * n))
        n_val = int(np.floor(val_ratio * n))
        test_idx_list.append(cls_idx[:n_test])
        val_idx_list.append(cls_idx[n_test:n_test + n_val])
        train_idx_list.append(cls_idx[n_test + n_val:])

    train_idx = np.concatenate(train_idx_list)
    val_idx = np.concatenate(val_idx_list)
    test_idx = np.concatenate(test_idx_list)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return {
        "x_train": x[train_idx],
        "y_train": y[train_idx],
        "x_val": x[val_idx],
        "y_val": y[val_idx],
        "x_test": x[test_idx],
        "y_test": y[test_idx],
    }


def oversample_minority_classes(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple random oversampling to balance the classes."""
    rng = np.random.default_rng(seed)
    counts = class_counts(y)
    target_count = max(counts.values())

    xs = []
    ys = []
    for cls in sorted(counts.keys()):
        cls_x = x[y == cls]
        cls_y = y[y == cls]
        if len(cls_x) < target_count:
            extra_idx = rng.choice(len(cls_x), size=target_count - len(cls_x), replace=True)
            cls_x = np.concatenate([cls_x, cls_x[extra_idx]], axis=0)
            cls_y = np.concatenate([cls_y, cls_y[extra_idx]], axis=0)
        xs.append(cls_x)
        ys.append(cls_y)

    x_balanced = np.concatenate(xs, axis=0)
    y_balanced = np.concatenate(ys, axis=0)
    perm = rng.permutation(len(y_balanced))
    return x_balanced[perm], y_balanced[perm]
