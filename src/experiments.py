"""Experiment orchestration for phase 1 and phase 2."""
from __future__ import annotations

import os
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .data import (
    MNISTLoader,
    filter_classes,
    oversample_minority_classes,
    stratified_train_val_test_split,
    subsample_stratified,
)
from .features import HOGExtractor, PCA
from .metrics import accuracy_score, classification_report_text, precision_recall_f1
from .models.gaussian_nb import GaussianNaiveBayes
from .models.knn import KNNClassifier
from .models.logistic_regression import LogisticRegressionScratch
from .models.random_forest import RandomForestClassifierScratch
from .preprocess import FlattenTransformer, StandardScaler, normalize_images_to_unit_range, resize_images_nearest
from .utils import class_counts, ensure_dir, save_json, save_text, set_seed, timestamp
from .validation import compute_learning_curve, grid_search_cv
from .visualization import plot_confusion_matrix, plot_input_samples, plot_learning_curve, plot_loss_curve, plot_sample_predictions


MODEL_REGISTRY = {
    "logreg": LogisticRegressionScratch,
    "knn": KNNClassifier,
    "gnb": GaussianNaiveBayes,
    "rf": RandomForestClassifierScratch,
}


def default_model_params(random_state: int = 42) -> Dict[str, Dict[str, object]]:
    return {
        "logreg": {
            "learning_rate": 0.08,
            "epochs": 25,
            "batch_size": 128,
            "reg_type": None,
            "reg_strength": 0.0,
            "random_state": random_state,
            "verbose": False,
            "loss_eval_interval": 5,
        },
        "knn": {"k": 3, "distance": "euclidean", "batch_size": 256},
        "gnb": {"var_smoothing": 1e-3},
        "rf": {
            "n_estimators": 11,
            "max_depth": 12,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "n_thresholds": 8,
            "bootstrap": True,
            "random_state": random_state,
        },
    }


def extract_features(
    train_images: np.ndarray,
    val_images: np.ndarray,
    test_images: np.ndarray,
    feature_type: str,
    pca_components: int = 50,
    hog_cell_size: int = 4,
    hog_block_size: int = 2,
    hog_bins: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    metadata: Dict[str, object] = {"feature_type": feature_type}

    if feature_type == "flatten":
        transformer = FlattenTransformer()
        x_train = transformer.fit_transform(train_images)
        x_val = transformer.transform(val_images)
        x_test = transformer.transform(test_images)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
        metadata["output_dim"] = int(x_train.shape[1])
        return x_train, x_val, x_test, metadata

    if feature_type == "pca":
        flatten = FlattenTransformer()
        train_flat = flatten.fit_transform(train_images)
        val_flat = flatten.transform(val_images)
        test_flat = flatten.transform(test_images)
        scaler = StandardScaler()
        train_flat = scaler.fit_transform(train_flat)
        val_flat = scaler.transform(val_flat)
        test_flat = scaler.transform(test_flat)
        pca = PCA(n_components=pca_components)
        x_train = pca.fit_transform(train_flat)
        x_val = pca.transform(val_flat)
        x_test = pca.transform(test_flat)
        metadata["output_dim"] = int(x_train.shape[1])
        metadata["explained_variance_ratio_sum"] = float(np.sum(pca.explained_variance_ratio_))
        return x_train, x_val, x_test, metadata

    if feature_type == "hog":
        hog = HOGExtractor(cell_size=hog_cell_size, block_size=hog_block_size, bins=hog_bins)
        x_train = hog.fit_transform(train_images)
        x_val = hog.transform(val_images)
        x_test = hog.transform(test_images)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
        metadata["output_dim"] = int(x_train.shape[1])
        metadata["cell_size"] = hog_cell_size
        metadata["block_size"] = hog_block_size
        metadata["bins"] = hog_bins
        return x_train, x_val, x_test, metadata

    raise ValueError(f"Unsupported feature type: {feature_type}")


def evaluate_model(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    experiment_name: str,
    original_test_images: np.ndarray | None = None,
    display_label_map: Dict[int, int] | None = None,
) -> Dict[str, object]:
    start_time = time.time()
    model.fit(x_train, y_train)
    fit_seconds = time.time() - start_time

    y_val_pred = model.predict(x_val)
    y_test_pred = model.predict(x_test)

    y_val_display = y_val.copy()
    y_test_display = y_test.copy()
    y_val_pred_display = y_val_pred.copy()
    y_test_pred_display = y_test_pred.copy()
    display_labels = labels.copy()

    if display_label_map is not None:
        remap = np.vectorize(lambda value: display_label_map[int(value)])
        y_val_display = remap(y_val)
        y_test_display = remap(y_test)
        y_val_pred_display = remap(y_val_pred)
        y_test_pred_display = remap(y_test_pred)
        display_labels = np.array([display_label_map[int(label)] for label in labels], dtype=np.int64)

    val_metrics = precision_recall_f1(y_val, y_val_pred, labels=labels)
    test_metrics = precision_recall_f1(y_test, y_test_pred, labels=labels)

    report_text = (
        "Validation Report\n"
        + classification_report_text(y_val_display, y_val_pred_display, labels=display_labels)
        + "\n\n"
        + "Test Report\n"
        + classification_report_text(y_test_display, y_test_pred_display, labels=display_labels)
    )
    save_text(report_text, os.path.join(output_dir, f"{experiment_name}_report.txt"))

    cm_for_plot = test_metrics["confusion_matrix"]
    plot_confusion_matrix(
        cm_for_plot,
        labels=display_labels,
        title=f"Confusion Matrix - {experiment_name}",
        save_path=os.path.join(output_dir, f"{experiment_name}_confusion_matrix.png"),
    )

    if hasattr(model, "loss_history_"):
        plot_loss_curve(
            getattr(model, "loss_history_"),
            title=f"Loss Curve - {experiment_name}",
            save_path=os.path.join(output_dir, f"{experiment_name}_loss_curve.png"),
        )

    if original_test_images is not None:
        plot_sample_predictions(
            images=original_test_images[:16],
            y_true=y_test_display[:16],
            y_pred=y_test_pred_display[:16],
            save_path=os.path.join(output_dir, f"{experiment_name}_sample_predictions.png"),
            max_items=16,
        )

    result = {
        "experiment_name": experiment_name,
        "fit_time_seconds": float(fit_seconds),
        "val_accuracy": float(accuracy_score(y_val, y_val_pred)),
        "val_macro_f1": float(val_metrics["macro_f1"]),
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_weighted_f1": float(test_metrics["weighted_f1"]),
        "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
    }
    return result


def prepare_phase_data(config: Dict[str, object]) -> Dict[str, np.ndarray]:
    loader = MNISTLoader(
        data_dir=str(config.get("data_dir", "data")),
        source=str(config.get("mnist_loader", "tensorflow")),
    )
    x_train_full, y_train_full, x_test_full, y_test_full = loader.load()

    x_all = np.concatenate([x_train_full, x_test_full], axis=0)
    y_all = np.concatenate([y_train_full, y_test_full], axis=0)

    x_all = resize_images_nearest(x_all, tuple(config.get("resize_to", (28, 28))))
    x_all = normalize_images_to_unit_range(x_all)

    label_map = None
    if int(config["phase"]) == 1:
        selected_classes = [int(config.get("class_a", 3)), int(config.get("class_b", 8))]
        label_map = {0: selected_classes[0], 1: selected_classes[1]}
        x_all, y_all = filter_classes(x_all, y_all, selected_classes, remap_to_binary=True)
    else:
        selected_classes = list(range(10))

    if config.get("subsample_total") is not None:
        x_all, y_all = subsample_stratified(
            x_all,
            y_all,
            total_samples=int(config["subsample_total"]),
            seed=int(config.get("random_state", 42)),
        )

    if config.get("subsample_per_class") is not None:
        x_all, y_all = subsample_stratified(
            x_all,
            y_all,
            per_class=int(config["subsample_per_class"]),
            seed=int(config.get("random_state", 42)),
        )

    splits = stratified_train_val_test_split(
        x_all,
        y_all,
        val_ratio=float(config.get("val_ratio", 0.1)),
        test_ratio=float(config.get("test_ratio", 0.2)),
        seed=int(config.get("random_state", 42)),
    )

    if bool(config.get("balance_train", False)):
        splits["x_train"], splits["y_train"] = oversample_minority_classes(
            splits["x_train"],
            splits["y_train"],
            seed=int(config.get("random_state", 42)),
        )

    splits["selected_classes"] = np.array(selected_classes, dtype=np.int64)
    if label_map is not None:
        splits["display_label_map"] = label_map
    return splits


def run_phase(config: Dict[str, object]) -> Dict[str, object]:
    set_seed(int(config.get("random_state", 42)))
    phase = int(config["phase"])
    output_root = ensure_dir(str(config.get("output_dir", "outputs")))
    run_name = f"phase{phase}_{timestamp()}"
    run_dir = ensure_dir(os.path.join(output_root, run_name))

    splits = prepare_phase_data(config)
    labels = np.unique(splits["y_train"])
    summary = {
        "phase": phase,
        "train_class_counts": class_counts(splits["y_train"]),
        "val_class_counts": class_counts(splits["y_val"]),
        "test_class_counts": class_counts(splits["y_test"]),
        "selected_classes": splits["selected_classes"].tolist(),
        "mnist_loader": str(config.get("mnist_loader", "tensorflow")),
        "experiments": [],
    }

    features = list(config.get("features", ["flatten", "pca", "hog"]))
    models = list(config.get("models", ["logreg", "knn", "gnb"]))
    params = default_model_params(random_state=int(config.get("random_state", 42)))

    for feature_type in features:
        print(f"\n=== Extracting features: {feature_type} ===")
        feature_start = time.time()
        x_train_feat, x_val_feat, x_test_feat, feature_meta = extract_features(
            splits["x_train"],
            splits["x_val"],
            splits["x_test"],
            feature_type=feature_type,
            pca_components=int(config.get("pca_components", 50)),
            hog_cell_size=int(config.get("hog_cell_size", 4)),
            hog_block_size=int(config.get("hog_block_size", 2)),
            hog_bins=int(config.get("hog_bins", 9)),
        )

        print(f"Feature extraction finished in {time.time() - feature_start:.2f} s")
        feature_dir = ensure_dir(os.path.join(run_dir, feature_type))
        for model_name in models:
            print(f"Running model: {model_name} on {feature_type}")
            model_start = time.time()
            model_class = MODEL_REGISTRY[model_name]
            model = model_class(**params[model_name])
            experiment_name = f"{feature_type}_{model_name}"
            result = evaluate_model(
                model=model,
                x_train=x_train_feat,
                y_train=splits["y_train"],
                x_val=x_val_feat,
                y_val=splits["y_val"],
                x_test=x_test_feat,
                y_test=splits["y_test"],
                labels=labels,
                output_dir=feature_dir,
                experiment_name=experiment_name,
                original_test_images=splits["x_test"],
                display_label_map=splits.get("display_label_map"),
            )
            result.update(feature_meta)
            result["model_name"] = model_name
            summary["experiments"].append(result)
            print(f"Finished {model_name} on {feature_type} in {time.time() - model_start:.2f} s")

    summary["experiments"].sort(key=lambda item: item["test_accuracy"], reverse=True)
    save_json(summary, os.path.join(run_dir, "summary.json"))
    return summary


def run_phase2_with_improvements(config: Dict[str, object]) -> Dict[str, object]:
    """Runs phase 2 plus three improvement strategies:
    1) Hyperparameter tuning with cross-validation.
    2) Regularization + bias/variance diagnosis via learning curves.
    3) Ensemble model (random forest).
    """
    config = dict(config)
    config["phase"] = 2
    output_root = ensure_dir(str(config.get("output_dir", "outputs")))
    run_name = f"phase2_improved_{timestamp()}"
    run_dir = ensure_dir(os.path.join(output_root, run_name))

    set_seed(int(config.get("random_state", 42)))
    splits = prepare_phase_data(config)
    labels = np.unique(splits["y_train"])

    # Use PCA as the default improved classical representation for multiclass experiments.
    x_train_feat, x_val_feat, x_test_feat, feature_meta = extract_features(
        splits["x_train"],
        splits["x_val"],
        splits["x_test"],
        feature_type=str(config.get("phase2_feature", "pca")),
        pca_components=int(config.get("pca_components", 60)),
        hog_cell_size=int(config.get("hog_cell_size", 4)),
        hog_block_size=int(config.get("hog_block_size", 2)),
        hog_bins=int(config.get("hog_bins", 9)),
    )

    summary = {
        "phase": 2,
        "improvements": [
            "hyperparameter_tuning_cv",
            "regularization_and_learning_curves",
            "ensemble_random_forest",
        ],
        "feature_metadata": feature_meta,
        "train_class_counts": class_counts(splits["y_train"]),
        "val_class_counts": class_counts(splits["y_val"]),
        "test_class_counts": class_counts(splits["y_test"]),
        "selected_classes": splits["selected_classes"].tolist(),
        "mnist_loader": str(config.get("mnist_loader", "tensorflow")),
        "experiments": [],
    }

    # 1) Hyperparameter tuning for KNN.
    knn_dir = ensure_dir(os.path.join(run_dir, "knn_tuning"))
    best_knn_params, knn_cv_results = grid_search_cv(
        KNNClassifier,
        param_grid={"k": [3, 5], "distance": ["euclidean"], "batch_size": [256]},
        x=x_train_feat,
        y=splits["y_train"],
        k=2,
        seed=int(config.get("random_state", 42)),
    )
    save_json({"best_params": best_knn_params, "cv_results": knn_cv_results}, os.path.join(knn_dir, "grid_search.json"))
    knn_model = KNNClassifier(**best_knn_params)
    knn_result = evaluate_model(
        knn_model,
        x_train_feat,
        splits["y_train"],
        x_val_feat,
        splits["y_val"],
        x_test_feat,
        splits["y_test"],
        labels,
        knn_dir,
        "phase2_knn_tuned",
        original_test_images=splits["x_test"],
        display_label_map=splits.get("display_label_map"),
    )
    knn_result["model_name"] = "knn"
    knn_result["best_params"] = best_knn_params
    summary["experiments"].append(knn_result)

    # 2) Logistic regression with L2 regularization + learning curves.
    logreg_dir = ensure_dir(os.path.join(run_dir, "logreg_regularized"))
    best_lr_params, lr_cv_results = grid_search_cv(
        LogisticRegressionScratch,
        param_grid={
            "learning_rate": [0.05],
            "epochs": [20],
            "batch_size": [128],
            "reg_type": ["L2"],
            "reg_strength": [1e-4, 1e-3],
            "random_state": [int(config.get("random_state", 42))],
            "verbose": [False],
            "loss_eval_interval": [5],
        },
        x=x_train_feat,
        y=splits["y_train"],
        k=2,
        seed=int(config.get("random_state", 42)),
    )
    save_json({"best_params": best_lr_params, "cv_results": lr_cv_results}, os.path.join(logreg_dir, "grid_search.json"))
    lr_model = LogisticRegressionScratch(**best_lr_params)
    lr_result = evaluate_model(
        lr_model,
        x_train_feat,
        splits["y_train"],
        x_val_feat,
        splits["y_val"],
        x_test_feat,
        splits["y_test"],
        labels,
        logreg_dir,
        "phase2_logreg_regularized",
        original_test_images=splits["x_test"],
        display_label_map=splits.get("display_label_map"),
    )
    learning_curve = compute_learning_curve(
        model_factory=lambda: LogisticRegressionScratch(**best_lr_params),
        x_train=x_train_feat,
        y_train=splits["y_train"],
        x_val=x_val_feat,
        y_val=splits["y_val"],
        train_sizes=[0.2, 0.5, 1.0],
        seed=int(config.get("random_state", 42)),
    )
    plot_learning_curve(
        learning_curve,
        title="Learning Curve - Regularized Logistic Regression",
        save_path=os.path.join(logreg_dir, "learning_curve.png"),
    )
    save_json({"learning_curve": learning_curve}, os.path.join(logreg_dir, "learning_curve.json"))
    lr_result["model_name"] = "logreg"
    lr_result["best_params"] = best_lr_params
    summary["experiments"].append(lr_result)

    # 3) Ensemble method: Random Forest.
    rf_dir = ensure_dir(os.path.join(run_dir, "random_forest"))
    rf_model = RandomForestClassifierScratch(
        n_estimators=int(config.get("rf_n_estimators", 5)),
        max_depth=int(config.get("rf_max_depth", 8)),
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        n_thresholds=8,
        bootstrap=True,
        random_state=int(config.get("random_state", 42)),
    )
    rf_result = evaluate_model(
        rf_model,
        x_train_feat,
        splits["y_train"],
        x_val_feat,
        splits["y_val"],
        x_test_feat,
        splits["y_test"],
        labels,
        rf_dir,
        "phase2_random_forest",
        original_test_images=splits["x_test"],
        display_label_map=splits.get("display_label_map"),
    )
    rf_result["model_name"] = "rf"
    summary["experiments"].append(rf_result)

    summary["experiments"].sort(key=lambda item: item["test_accuracy"], reverse=True)
    save_json(summary, os.path.join(run_dir, "summary.json"))
    return summary
