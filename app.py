from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageOps
import streamlit as st

from src.data import MNISTLoader, filter_classes, stratified_train_val_test_split, subsample_stratified
from src.features import HOGExtractor, PCA
from src.metrics import accuracy_score, precision_recall_f1
from src.models.gaussian_nb import GaussianNaiveBayes
from src.models.knn import KNNClassifier
from src.models.logistic_regression import LogisticRegressionScratch
from src.models.random_forest import RandomForestClassifierScratch
from src.preprocess import FlattenTransformer, StandardScaler, normalize_images_to_unit_range, resize_images_nearest


st.set_page_config(page_title="MNIST Project Demo", page_icon="🔢", layout="wide")


@dataclass
class FeaturePipeline:
    feature_type: str
    scaler: StandardScaler | None = None
    pca: PCA | None = None
    hog: HOGExtractor | None = None
    flatten: FlattenTransformer | None = None

    def fit(self, images: np.ndarray, pca_components: int, hog_cell_size: int, hog_block_size: int, hog_bins: int) -> "FeaturePipeline":
        if self.feature_type == "flatten":
            self.flatten = FlattenTransformer()
            train_feat = self.flatten.fit_transform(images)
            self.scaler = StandardScaler()
            self.scaler.fit(train_feat)
            return self

        if self.feature_type == "pca":
            self.flatten = FlattenTransformer()
            train_flat = self.flatten.fit_transform(images)
            self.scaler = StandardScaler()
            train_flat = self.scaler.fit_transform(train_flat)
            self.pca = PCA(n_components=pca_components)
            self.pca.fit(train_flat)
            return self

        if self.feature_type == "hog":
            self.hog = HOGExtractor(cell_size=hog_cell_size, block_size=hog_block_size, bins=hog_bins)
            train_feat = self.hog.fit_transform(images)
            self.scaler = StandardScaler()
            self.scaler.fit(train_feat)
            return self

        raise ValueError(f"Unsupported feature type: {self.feature_type}")

    def transform(self, images: np.ndarray) -> np.ndarray:
        if self.feature_type == "flatten":
            assert self.flatten is not None and self.scaler is not None
            feat = self.flatten.transform(images)
            return self.scaler.transform(feat)

        if self.feature_type == "pca":
            assert self.flatten is not None and self.scaler is not None and self.pca is not None
            flat = self.flatten.transform(images)
            flat = self.scaler.transform(flat)
            return self.pca.transform(flat)

        if self.feature_type == "hog":
            assert self.hog is not None and self.scaler is not None
            feat = self.hog.transform(images)
            return self.scaler.transform(feat)

        raise ValueError(f"Unsupported feature type: {self.feature_type}")


@st.cache_data(show_spinner=False)
def load_all_mnist() -> Tuple[np.ndarray, np.ndarray]:
    loader = MNISTLoader(data_dir="data", source="tensorflow")
    x_train, y_train, x_test, y_test = loader.load()
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    x_all = resize_images_nearest(x_all, (28, 28))
    x_all = normalize_images_to_unit_range(x_all)
    return x_all, y_all


def prepare_phase_dataset(phase: int, class_a: int, class_b: int, subsample_per_class: int, random_state: int):
    x_all, y_all = load_all_mnist()
    if phase == 1:
        selected = [class_a, class_b]
        display_map = {0: class_a, 1: class_b}
        x_all, y_all = filter_classes(x_all, y_all, selected, remap_to_binary=True)
    else:
        selected = list(range(10))
        display_map = {i: i for i in range(10)}

    if subsample_per_class is not None:
        x_all, y_all = subsample_stratified(
            x_all,
            y_all,
            per_class=subsample_per_class,
            seed=random_state,
        )

    splits = stratified_train_val_test_split(x_all, y_all, val_ratio=0.1, test_ratio=0.2, seed=random_state)
    return splits, display_map, selected


def build_model(phase: int, model_name: str, random_state: int):
    if phase == 1:
        if model_name == "logreg":
            return LogisticRegressionScratch(
                learning_rate=0.08,
                epochs=20,
                batch_size=64,
                reg_type=None,
                reg_strength=0.0,
                random_state=random_state,
                verbose=False,
                loss_eval_interval=5,
            )
        if model_name == "knn":
            return KNNClassifier(k=3, distance="euclidean")
        if model_name == "gnb":
            return GaussianNaiveBayes(var_smoothing=1e-3)
    else:
        if model_name == "logreg_regularized":
            return LogisticRegressionScratch(
                learning_rate=0.08,
                epochs=20,
                batch_size=64,
                reg_type="L2",
                reg_strength=1e-3,
                random_state=random_state,
                verbose=False,
                loss_eval_interval=5,
            )
        if model_name == "knn_tuned":
            return KNNClassifier(k=5, distance="euclidean")
        if model_name == "random_forest":
            return RandomForestClassifierScratch(
                n_estimators=11,
                max_depth=10,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features="sqrt",
                n_thresholds=8,
                bootstrap=True,
                random_state=random_state,
            )
    raise ValueError(f"Unsupported model: {model_name}")


@st.cache_resource(show_spinner=False)
def train_bundle(
    phase: int,
    feature_type: str,
    model_name: str,
    subsample_per_class: int,
    class_a: int,
    class_b: int,
    pca_components: int,
    hog_cell_size: int,
    hog_block_size: int,
    hog_bins: int,
    random_state: int,
):
    splits, display_map, selected = prepare_phase_dataset(phase, class_a, class_b, subsample_per_class, random_state)

    pipeline = FeaturePipeline(feature_type=feature_type)
    pipeline.fit(
        splits["x_train"],
        pca_components=pca_components,
        hog_cell_size=hog_cell_size,
        hog_block_size=hog_block_size,
        hog_bins=hog_bins,
    )

    x_train_feat = pipeline.transform(splits["x_train"])
    x_val_feat = pipeline.transform(splits["x_val"])
    x_test_feat = pipeline.transform(splits["x_test"])

    model = build_model(phase, model_name, random_state)
    model.fit(x_train_feat, splits["y_train"])

    y_val_pred = model.predict(x_val_feat)
    y_test_pred = model.predict(x_test_feat)

    labels = np.unique(splits["y_test"])
    val_metrics = precision_recall_f1(splits["y_val"], y_val_pred, labels=np.unique(splits["y_val"]))
    test_metrics = precision_recall_f1(splits["y_test"], y_test_pred, labels=labels)

    return {
        "phase": phase,
        "feature_type": feature_type,
        "model_name": model_name,
        "pipeline": pipeline,
        "model": model,
        "display_map": display_map,
        "selected_classes": selected,
        "x_test_raw": splits["x_test"],
        "y_test": splits["y_test"],
        "val_accuracy": float(accuracy_score(splits["y_val"], y_val_pred)),
        "test_accuracy": float(accuracy_score(splits["y_test"], y_test_pred)),
        "val_macro_f1": float(val_metrics["macro_f1"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_weighted_f1": float(test_metrics["weighted_f1"]),
        "train_size": int(len(splits["x_train"])),
        "val_size": int(len(splits["x_val"])),
        "test_size": int(len(splits["x_test"])),
    }


def preprocess_uploaded_image(file_bytes: bytes, invert_colors: bool) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("L")
    image = ImageOps.contain(image, (28, 28))
    canvas = Image.new("L", (28, 28), color=0)
    left = (28 - image.width) // 2
    top = (28 - image.height) // 2
    canvas.paste(image, (left, top))
    array = np.asarray(canvas, dtype=np.uint8)
    if invert_colors:
        array = 255 - array
    array = resize_images_nearest(array[None, :, :], (28, 28))[0]
    array = normalize_images_to_unit_range(array[None, :, :])[0]
    return array


def predict_single(bundle: Dict[str, object], image_28: np.ndarray) -> Tuple[int, np.ndarray | None]:
    pipeline: FeaturePipeline = bundle["pipeline"]
    model = bundle["model"]
    features = pipeline.transform(image_28[None, :, :])
    pred_internal = int(model.predict(features)[0])
    pred_display = int(bundle["display_map"][pred_internal])
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            raw_probs = model.predict_proba(features)[0]
            if bundle["phase"] == 1:
                probs = np.asarray([raw_probs[0], raw_probs[1]], dtype=np.float64)
            else:
                probs = np.asarray(raw_probs, dtype=np.float64)
        except Exception:
            probs = None
    return pred_display, probs


def model_label_options(phase: int):
    if phase == 1:
        return {
            "Logistic Regression": "logreg",
            "K-Nearest Neighbors": "knn",
            "Gaussian Naive Bayes": "gnb",
        }
    return {
        "Regularized Logistic Regression": "logreg_regularized",
        "Tuned KNN": "knn_tuned",
        "Random Forest": "random_forest",
    }


def feature_options(phase: int):
    if phase == 1:
        return {"Flatten": "flatten", "PCA": "pca", "HOG": "hog"}
    return {"PCA": "pca", "HOG": "hog", "Flatten": "flatten"}


st.title("🔢 MNIST Project Demo Website")
st.caption("A simple deployment interface to try the manual classifiers from your machine-learning project.")

with st.sidebar:
    st.header("Demo settings")
    phase = st.radio("Choose phase", [1, 2], format_func=lambda x: f"Phase {x}")
    phase_explain = "Binary classification (3 vs 8 by default)" if phase == 1 else "10-class improved phase"
    st.caption(phase_explain)

    feature_label = st.selectbox("Feature extraction", list(feature_options(phase).keys()))
    feature_type = feature_options(phase)[feature_label]

    model_label = st.selectbox("Model", list(model_label_options(phase).keys()))
    model_name = model_label_options(phase)[model_label]

    if phase == 1:
        class_a, class_b = st.columns(2)
        with class_a:
            digit_a = st.number_input("Digit A", min_value=0, max_value=9, value=3, step=1)
        with class_b:
            digit_b = st.number_input("Digit B", min_value=0, max_value=9, value=8, step=1)
        if digit_a == digit_b:
            st.error("Choose two different digits for Phase 1.")
        default_subsample = 200
    else:
        digit_a, digit_b = 3, 8
        default_subsample = 120

    subsample_per_class = st.slider("Samples per class for training demo", min_value=40, max_value=400, value=default_subsample, step=20)
    pca_components = st.slider("PCA components", min_value=10, max_value=80, value=40, step=5)
    hog_cell_size = st.select_slider("HOG cell size", options=[2, 4, 7], value=4)
    hog_block_size = st.select_slider("HOG block size", options=[1, 2, 3], value=2)
    hog_bins = st.select_slider("HOG bins", options=[6, 9, 12], value=9)
    random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

    load_button = st.button("Train / Load model", use_container_width=True)

st.markdown(
    """
    This demo uses the same project pipeline steps:
    **MNIST loading → resize and normalization → feature extraction (Flatten/PCA/HOG) → manual classifier → prediction**.
    """
)

if load_button:
    if phase == 1 and digit_a == digit_b:
        st.error("Phase 1 needs two different digits.")
    else:
        with st.spinner("Training the selected manual model ..."):
            bundle = train_bundle(
                phase=phase,
                feature_type=feature_type,
                model_name=model_name,
                subsample_per_class=subsample_per_class,
                class_a=int(digit_a),
                class_b=int(digit_b),
                pca_components=pca_components,
                hog_cell_size=int(hog_cell_size),
                hog_block_size=int(hog_block_size),
                hog_bins=int(hog_bins),
                random_state=int(random_state),
            )
        st.session_state["trained_bundle"] = bundle
        st.success("Model is ready.")

bundle = st.session_state.get("trained_bundle")

if bundle is None:
    st.info("Choose the settings from the sidebar, then click **Train / Load model**.")
else:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Validation accuracy", f"{bundle['val_accuracy']:.3f}")
    col2.metric("Test accuracy", f"{bundle['test_accuracy']:.3f}")
    col3.metric("Macro F1", f"{bundle['test_macro_f1']:.3f}")
    col4.metric("Weighted F1", f"{bundle['test_weighted_f1']:.3f}")

    st.write(
        f"**Train / Val / Test sizes:** {bundle['train_size']} / {bundle['val_size']} / {bundle['test_size']}"
    )
    if bundle["phase"] == 1:
        st.write(f"**Phase 1 digits:** {bundle['selected_classes'][0]} vs {bundle['selected_classes'][1]}")
    else:
        st.write("**Phase 2 classes:** 0 to 9")

    tab1, tab2 = st.tabs(["Try a held-out MNIST sample", "Upload your own image"])

    with tab1:
        if "demo_index" not in st.session_state:
            st.session_state["demo_index"] = 0
        if st.button("Pick another random test sample"):
            st.session_state["demo_index"] = int(np.random.randint(0, len(bundle["x_test_raw"])))

        idx = int(st.session_state["demo_index"])
        image = bundle["x_test_raw"][idx]
        true_internal = int(bundle["y_test"][idx])
        true_label = int(bundle["display_map"][true_internal])
        pred_label, probs = predict_single(bundle, image)

        left, right = st.columns([1, 2])
        with left:
            st.image((image * 255).astype(np.uint8), caption="Held-out test image", width=220, clamp=True)
        with right:
            st.write(f"**True label:** {true_label}")
            st.write(f"**Predicted label:** {pred_label}")
            if probs is not None:
                if bundle["phase"] == 1:
                    st.write("**Probabilities**")
                    probs_map = {
                        int(bundle["selected_classes"][0]): float(probs[0]),
                        int(bundle["selected_classes"][1]): float(probs[1]),
                    }
                    st.json(probs_map)
                else:
                    prob_map = {int(i): float(probs[i]) for i in range(len(probs))}
                    st.write("**Class probabilities**")
                    st.json(prob_map)

    with tab2:
        uploaded_file = st.file_uploader("Upload a digit image (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])
        invert = st.checkbox("Invert colors", value=True, help="Useful when your uploaded image has a black digit on a white background.")

        if uploaded_file is not None:
            processed = preprocess_uploaded_image(uploaded_file.read(), invert_colors=invert)
            pred_label, probs = predict_single(bundle, processed)

            left, right = st.columns([1, 2])
            with left:
                st.image((processed * 255).astype(np.uint8), caption="Processed 28×28 image", width=220, clamp=True)
            with right:
                st.write(f"**Predicted label:** {pred_label}")
                st.caption("Tip: for better results, use a centered handwritten digit with a dark background or enable invert colors.")
                if probs is not None:
                    if bundle["phase"] == 1:
                        probs_map = {
                            int(bundle["selected_classes"][0]): float(probs[0]),
                            int(bundle["selected_classes"][1]): float(probs[1]),
                        }
                        st.write("**Probabilities**")
                        st.json(probs_map)
                    else:
                        prob_map = {int(i): float(probs[i]) for i in range(len(probs))}
                        st.write("**Class probabilities**")
                        st.json(prob_map)

st.markdown("---")
st.caption("Deployment note: this is a lightweight local demo website for the optional simple deployment part of Phase 2.")
