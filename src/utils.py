"""Utility helpers for the MNIST image classification project."""
from __future__ import annotations

import itertools
import json
import os
import random
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


ArrayLike = np.ndarray


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def flatten_dict_grid(param_grid: Dict[str, Sequence]) -> List[Dict[str, object]]:
    """Convert a parameter grid dictionary into a list of combinations."""
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos = []
    for combination in itertools.product(*values):
        combos.append(dict(zip(keys, combination)))
    return combos


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_text(text: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def class_counts(y: ArrayLike) -> Dict[int, int]:
    classes, counts = np.unique(y, return_counts=True)
    return {int(c): int(n) for c, n in zip(classes, counts)}
