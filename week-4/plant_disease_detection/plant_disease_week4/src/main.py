"""
main.py
========
Entry point that runs the full Week 4 pipeline end-to-end:

    1. Dataset discovery + EDA (class distribution, dimensions, duplicates,
       corrupt-file detection, sample grid)
    2. Preprocessing (RGB, resize, denoise, normalize) + before/after figure
    3. Stratified train/val/test split (leak-free) + imbalance analysis
    4. Training-only augmentation + example figure
    5. Feature engineering (color/HSV/histogram/texture/edge/shape) -> CSV
    6. Feature-distribution visualizations
    7. Model training (CNN via TensorFlow/Keras if available, otherwise an
       automatic scikit-learn baseline on the engineered features)
    8. Evaluation: accuracy/precision/recall/F1 + confusion matrix

Run with:
    python main.py
    python main.py --dataset_root dataset --img_size 224 --max_per_class 60

If dataset/ has no real images, a small labeled-as-synthetic demo dataset
is generated automatically so every stage can be verified to actually run
(see src/demo_dataset.py for details and README.md for how to plug in the
real PlantVillage dataset).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

# Make sure the project root (parent of src/) is importable as a package
# root, so "from src.xxx import yyy" works no matter where this script is
# launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import scan_dataset, find_exact_duplicates, flatten_index
from src.data_analysis import (
    plot_class_distribution, analyze_image_dimensions, plot_image_dimensions,
    plot_sample_grid, print_dataset_statistics,
)
from src.preprocessing import (
    preprocess_image, resize_image, load_image_rgb, normalize_pixels,
    stratified_split, analyze_class_imbalance, compute_balanced_class_weights,
    DEFAULT_IMG_SIZE,
)
from src.augmentation import augment_image, augment_training_set
from src.feature_extraction import extract_features, build_feature_dataframe
from src.model import (
    TENSORFLOW_AVAILABLE, build_cnn_model, train_cnn,
    build_baseline_classifier, train_baseline, predict_baseline,
)
from src.evaluation import compute_metrics, plot_confusion_matrix, save_full_report
from src.demo_dataset import generate_demo_dataset, dataset_is_empty


SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def limit_per_class(filepaths, labels, max_per_class: int, seed: int = SEED):
    """Optionally cap the number of images used per class (for fast demo runs
    or machines that can't fit the full PlantVillage dataset in memory)."""
    if max_per_class is None or max_per_class <= 0:
        return filepaths, labels
    rng = random.Random(seed)
    by_class = {}
    for fp, lb in zip(filepaths, labels):
        by_class.setdefault(lb, []).append(fp)
    out_fp, out_lb = [], []
    for lb, files in by_class.items():
        chosen = files if len(files) <= max_per_class else rng.sample(files, max_per_class)
        out_fp.extend(chosen)
        out_lb.extend([lb] * len(chosen))
    return out_fp, out_lb


def plot_before_after(sample_paths, img_size, out_path):
    n = min(4, len(sample_paths))
    if n == 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(n * 2.6, 5.2))
    for i in range(n):
        raw = load_image_rgb(sample_paths[i])
        processed = preprocess_image(sample_paths[i], size=img_size)
        ax0 = axes[0][i] if n > 1 else axes[0]
        ax1 = axes[1][i] if n > 1 else axes[1]
        ax0.imshow(raw)
        ax0.set_title(f"Original\n{raw.shape[1]}x{raw.shape[0]}", fontsize=8)
        ax0.axis("off")
        ax1.imshow(processed)
        ax1.set_title(f"Preprocessed\n{img_size[0]}x{img_size[1]}", fontsize=8)
        ax1.axis("off")
    fig.suptitle("Before vs After Preprocessing")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[main] Saved before/after preprocessing figure -> {out_path}")


def plot_augmentation_examples(sample_path, img_size, out_path, n_examples=5):
    base = preprocess_image(sample_path, size=img_size)
    fig, axes = plt.subplots(1, n_examples + 1, figsize=((n_examples + 1) * 2.4, 2.8))
    axes[0].imshow(base)
    axes[0].set_title("Original\n(preprocessed)", fontsize=8)
    axes[0].axis("off")
    for i in range(n_examples):
        aug = augment_image(base, n_ops=2, seed=100 + i)
        axes[i + 1].imshow(np.clip(aug, 0, 1))
        axes[i + 1].set_title(f"Augmented #{i+1}", fontsize=8)
        axes[i + 1].axis("off")
    fig.suptitle("Training-Only Data Augmentation Examples")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[main] Saved augmentation examples figure -> {out_path}")


def plot_color_analysis(images, labels, out_path, max_classes=6):
    """Average color histogram per class, to visually confirm classes differ in color."""
    classes = sorted(set(labels))[:max_classes]
    fig, axes = plt.subplots(1, len(classes), figsize=(len(classes) * 3, 3))
    if len(classes) == 1:
        axes = [axes]
    for ax, cls in zip(axes, classes):
        cls_imgs = [img for img, lbl in zip(images, labels) if lbl == cls]
        if not cls_imgs:
            continue
        stacked = np.stack(cls_imgs)
        for ci, cname, color in zip(range(3), ["R", "G", "B"], ["red", "green", "blue"]):
            channel_vals = stacked[:, :, :, ci].flatten()
            ax.hist(channel_vals, bins=30, alpha=0.5, label=cname, color=color, density=True)
        ax.set_title(cls, fontsize=7)
        ax.legend(fontsize=6)
    fig.suptitle("Per-Class Color Channel Distributions")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[main] Saved color analysis figure -> {out_path}")


def plot_feature_distributions(feature_df: pd.DataFrame, out_path: str):
    numeric_cols = [c for c in ["rgb_g_mean", "hsv_h_mean", "texture_contrast",
                                 "edge_density", "shape_area_ratio"] if c in feature_df.columns]
    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(len(numeric_cols) * 3.2, 3.2))
    if len(numeric_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, numeric_cols):
        for cls in sorted(feature_df["label"].unique()):
            vals = feature_df.loc[feature_df["label"] == cls, col]
            ax.hist(vals, bins=15, alpha=0.5, label=cls, density=True)
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=6)
    axes[0].legend(fontsize=5, loc="upper right")
    fig.suptitle("Feature Distributions by Class")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[main] Saved feature distribution figure -> {out_path}")


def plot_texture_edge_examples(sample_paths, img_size, out_path):
    import cv2
    n = min(4, len(sample_paths))
    if n == 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(n * 2.6, 5.2))
    for i in range(n):
        img = preprocess_image(sample_paths[i], size=img_size)
        img_u8 = (img * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        ax0 = axes[0][i] if n > 1 else axes[0]
        ax1 = axes[1][i] if n > 1 else axes[1]
        ax0.imshow(gray, cmap="gray")
        ax0.set_title("Grayscale (texture basis)", fontsize=7)
        ax0.axis("off")
        ax1.imshow(edges, cmap="gray")
        ax1.set_title("Canny edges", fontsize=7)
        ax1.axis("off")
    fig.suptitle("Texture / Edge Feature Examples")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[main] Saved texture/edge example figure -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plant Disease Detection - Week 4 pipeline")
    parser.add_argument("--dataset_root", type=str, default=None,
                         help="Defaults to <project_root>/dataset")
    parser.add_argument("--img_size", type=int, default=224, help="Square input size (e.g. 224)")
    parser.add_argument("--max_per_class", type=int, default=40,
                         help="Cap images per class for a fast demo run. Use 0 for no cap "
                              "(full dataset).")
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--aug_multiplier", type=int, default=1,
                         help="How many augmented copies to add per training image.")
    parser.add_argument("--cnn_epochs", type=int, default=8)
    args = parser.parse_args()

    set_global_seed(SEED)
    img_size = (args.img_size, args.img_size)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.dataset_root is None:
        args.dataset_root = os.path.join(project_root, "dataset")
    out_fig = os.path.join(project_root, "outputs", "figures")
    out_feat = os.path.join(project_root, "outputs", "features")
    out_res = os.path.join(project_root, "outputs", "results")
    for d in (out_fig, out_feat, out_res):
        os.makedirs(d, exist_ok=True)

    t0 = time.time()

    # ---------------------------------------------------------------
    # 0. Ensure there is *something* in dataset/ to run against
    # ---------------------------------------------------------------
    if dataset_is_empty(args.dataset_root):
        generate_demo_dataset(args.dataset_root, images_per_class=40)

    # ---------------------------------------------------------------
    # 1. Dataset discovery + EDA
    # ---------------------------------------------------------------
    print("\n########## STEP 1: DATASET ANALYSIS ##########")
    index = scan_dataset(args.dataset_root)
    duplicates = find_exact_duplicates(index)
    dims = analyze_image_dimensions(index)

    plot_class_distribution(index, os.path.join(out_fig, "01_class_distribution.png"))
    plot_image_dimensions(dims, os.path.join(out_fig, "02_image_dimensions.png"))
    plot_sample_grid(index, os.path.join(out_fig, "03_sample_images.png"))
    stats = print_dataset_statistics(index, dims, duplicates)

    filepaths, labels = flatten_index(index)
    filepaths, labels = limit_per_class(filepaths, labels, args.max_per_class)
    print(f"[main] Using {len(filepaths)} images after applying max_per_class="
          f"{args.max_per_class or 'ALL'} cap.")

    if len(filepaths) == 0:
        print("[main] No valid images found - aborting.")
        return

    # ---------------------------------------------------------------
    # 2. Preprocessing figures + leak-free split
    # ---------------------------------------------------------------
    print("\n########## STEP 2: PREPROCESSING ##########")
    plot_before_after(filepaths[:4], img_size, os.path.join(out_fig, "04_before_after_preprocessing.png"))

    splits = stratified_split(filepaths, labels, val_size=args.val_size, test_size=args.test_size, seed=SEED)
    train_fp, train_lb = splits["train"]
    val_fp, val_lb = splits["val"]
    test_fp, test_lb = splits["test"]

    imbalance_info = analyze_class_imbalance(train_lb)
    class_weights = compute_balanced_class_weights(train_lb)

    # Actually preprocess every image in every split (load -> RGB -> denoise -> resize -> normalize)
    def preprocess_set(fps):
        return [preprocess_image(fp, size=img_size) for fp in fps]

    print("[main] Preprocessing train/val/test images (this may take a moment)...")
    X_train_imgs = preprocess_set(train_fp)
    X_val_imgs = preprocess_set(val_fp)
    X_test_imgs = preprocess_set(test_fp)

    plot_color_analysis(X_train_imgs, train_lb, os.path.join(out_fig, "06_color_analysis.png"))
    plot_texture_edge_examples(train_fp[:4], img_size, os.path.join(out_fig, "08_texture_edge_examples.png"))

    # ---------------------------------------------------------------
    # 3. Training-only augmentation
    # ---------------------------------------------------------------
    print("\n########## STEP 3: AUGMENTATION (TRAIN SPLIT ONLY) ##########")
    plot_augmentation_examples(train_fp[0], img_size, os.path.join(out_fig, "05_augmentation_examples.png"))
    X_train_aug_imgs, y_train_aug = augment_training_set(
        X_train_imgs, train_lb, multiplier=args.aug_multiplier, seed=SEED
    )

    # ---------------------------------------------------------------
    # 4. Feature engineering
    # ---------------------------------------------------------------
    print("\n########## STEP 4: FEATURE ENGINEERING ##########")
    train_feat_df = build_feature_dataframe(X_train_aug_imgs, y_train_aug)
    val_feat_df = build_feature_dataframe(X_val_imgs, val_lb)
    test_feat_df = build_feature_dataframe(X_test_imgs, test_lb)

    train_feat_df.to_csv(os.path.join(out_feat, "train_features.csv"), index=False)
    val_feat_df.to_csv(os.path.join(out_feat, "val_features.csv"), index=False)
    test_feat_df.to_csv(os.path.join(out_feat, "test_features.csv"), index=False)
    np.savez_compressed(
        os.path.join(out_feat, "preprocessed_arrays.npz"),
        X_train=np.array(X_train_imgs, dtype=np.float32),
        y_train=np.array(train_lb),
        X_val=np.array(X_val_imgs, dtype=np.float32),
        y_val=np.array(val_lb),
        X_test=np.array(X_test_imgs, dtype=np.float32),
        y_test=np.array(test_lb),
    )
    print(f"[main] Saved feature CSVs and preprocessed_arrays.npz -> {out_feat}")

    plot_feature_distributions(train_feat_df, os.path.join(out_fig, "07_feature_distributions.png"))

    # ---------------------------------------------------------------
    # 5. Model-ready pipeline: train + evaluate
    # ---------------------------------------------------------------
    print("\n########## STEP 5: MODEL TRAINING & EVALUATION ##########")
    class_names = sorted(set(labels))

    used_cnn = False
    if TENSORFLOW_AVAILABLE:
        try:
            print("[main] TensorFlow detected - training CNN on preprocessed image tensors.")
            label_to_idx = {c: i for i, c in enumerate(class_names)}
            X_tr = np.array(X_train_aug_imgs, dtype=np.float32)
            y_tr = np.array([label_to_idx[l] for l in y_train_aug])
            X_va = np.array(X_val_imgs, dtype=np.float32)
            y_va = np.array([label_to_idx[l] for l in val_lb])
            X_te = np.array(X_test_imgs, dtype=np.float32)
            y_te_idx = np.array([label_to_idx[l] for l in test_lb])

            cnn_class_weight = {label_to_idx[c]: w for c, w in class_weights.items()}
            model = build_cnn_model((img_size[1], img_size[0], 3), len(class_names))
            train_cnn(model, X_tr, y_tr, X_va, y_va, class_weight=cnn_class_weight,
                      epochs=args.cnn_epochs, seed=SEED)
            probs = model.predict(X_te, verbose=0)
            pred_idx = probs.argmax(axis=1)
            y_pred = [class_names[i] for i in pred_idx]
            model.save(os.path.join(out_res, "cnn_model.keras"))
            used_cnn = True
        except Exception as e:
            print(f"[main] CNN training failed ({e}); falling back to classical-ML baseline.")

    if not used_cnn:
        print("[main] TensorFlow/PyTorch not available in this sandboxed execution "
              "environment (no internet access to install them), so the pipeline "
              "automatically falls back to the scikit-learn baseline classifier "
              "(RandomForest) trained on the engineered features from Step 4. "
              "model.py also contains a full, ready-to-run tf.keras CNN "
              "(build_cnn_model/train_cnn) that will be used automatically instead "
              "the moment TensorFlow is installed - no code changes needed.")
        feature_cols = [c for c in train_feat_df.columns if c not in ("label", "filepath")]
        X_tr = train_feat_df[feature_cols].values
        y_tr = train_feat_df["label"].values
        X_te = test_feat_df[feature_cols].values
        y_te = test_feat_df["label"].values

        clf = build_baseline_classifier(seed=SEED)
        clf, scaler = train_baseline(clf, X_tr, y_tr)
        y_pred = predict_baseline(clf, scaler, X_te)
        y_te_idx = None  # not used in this branch

    y_true = test_lb if used_cnn is False else [class_names[i] for i in y_te_idx]

    metrics = compute_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, class_names, os.path.join(out_fig, "09_confusion_matrix.png"))
    save_full_report(y_true, y_pred, class_names, metrics, out_res)

    # ---------------------------------------------------------------
    # 6. Persist the trained model (+ scaler/feature schema) for inference
    #    so the Flask app (src/app.py) can load it and score uploaded
    #    images through the "Scan Image" upload form.
    # ---------------------------------------------------------------
    print("\n########## STEP 6: SAVING MODEL FOR INFERENCE ##########")
    inference_meta = {
        "model_type": "cnn" if used_cnn else "baseline",
        "class_names": class_names,
        "img_size": [img_size[0], img_size[1]],
        "apply_denoise": True,
    }
    if used_cnn:
        # model.save(...) already ran above; just record how to use it.
        inference_meta["cnn_model_path"] = "cnn_model.keras"
    else:
        import joblib
        joblib.dump(clf, os.path.join(out_res, "baseline_model.joblib"))
        joblib.dump(scaler, os.path.join(out_res, "baseline_scaler.joblib"))
        inference_meta["baseline_model_path"] = "baseline_model.joblib"
        inference_meta["baseline_scaler_path"] = "baseline_scaler.joblib"
        inference_meta["feature_columns"] = feature_cols
        print(f"[main] Saved baseline_model.joblib + baseline_scaler.joblib -> {out_res}")

    with open(os.path.join(out_res, "inference_meta.json"), "w") as f:
        json.dump(inference_meta, f, indent=2)
    print(f"[main] Saved inference_meta.json -> {out_res} "
          f"(used by src/app.py for the image-upload prediction feature)")

    elapsed = time.time() - t0
    print(f"\n[main] Pipeline finished in {elapsed:.1f}s. "
          f"Model used: {'CNN (TensorFlow/Keras)' if used_cnn else 'RandomForest baseline (scikit-learn)'}.")
    print(f"[main] All figures saved under {out_fig}/")
    print(f"[main] All features saved under {out_feat}/")
    print(f"[main] All results saved under {out_res}/")


if __name__ == "__main__":
    main()
