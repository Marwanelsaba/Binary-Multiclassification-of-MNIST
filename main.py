from __future__ import annotations

import argparse
import json
import os
from pprint import pprint

from src.experiments import run_phase, run_phase2_with_improvements


DEFAULT_CONFIG = {
    "data_dir": "data",
    "output_dir": "outputs",
    "random_state": 42,
    "resize_to": [28, 28],
    "val_ratio": 0.10,
    "test_ratio": 0.20,
    "balance_train": False,
    "features": ["flatten", "pca", "hog"],
    "models": ["logreg", "knn", "gnb"],
    "pca_components": 50,
    "hog_cell_size": 4,
    "hog_block_size": 2,
    "hog_bins": 9,
    "class_a": 3,
    "class_b": 8,
    "subsample_total": None,
    "subsample_per_class": None,
    "phase2_feature": "pca",
    "rf_n_estimators": 15,
    "rf_max_depth": 12,
    "mnist_loader": "tensorflow",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MNIST image classification project from scratch.")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2], help="Project phase to run.")
    parser.add_argument("--improved", action="store_true", help="Run the phase 2 improved pipeline.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
    parser.add_argument("--class-a", type=int, default=None, help="First class for binary phase.")
    parser.add_argument("--class-b", type=int, default=None, help="Second class for binary phase.")
    parser.add_argument("--subsample-total", type=int, default=None, help="Optional stratified total sample cap.")
    parser.add_argument("--subsample-per-class", type=int, default=None, help="Optional equal samples per class.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory.")
    parser.add_argument("--mnist-loader", type=str, default=None, choices=["tensorflow", "idx"], help="How to import MNIST.")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)

    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            config.update(json.load(f))

    config["phase"] = args.phase
    if args.class_a is not None:
        config["class_a"] = args.class_a
    if args.class_b is not None:
        config["class_b"] = args.class_b
    if args.subsample_total is not None:
        config["subsample_total"] = args.subsample_total
    if args.subsample_per_class is not None:
        config["subsample_per_class"] = args.subsample_per_class
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.data_dir is not None:
        config["data_dir"] = args.data_dir
    if args.mnist_loader is not None:
        config["mnist_loader"] = args.mnist_loader

    print("Running with configuration:")
    pprint(config)

    if args.improved or args.phase == 2:
        summary = run_phase2_with_improvements(config) if args.improved else run_phase(config)
    else:
        summary = run_phase(config)

    print("\nTop experiments:")
    for item in summary["experiments"][:5]:
        print(
            f"- {item['experiment_name'] if 'experiment_name' in item else item['model_name']}: "
            f"test_accuracy={item['test_accuracy']:.4f}, test_macro_f1={item['test_macro_f1']:.4f}"
        )
