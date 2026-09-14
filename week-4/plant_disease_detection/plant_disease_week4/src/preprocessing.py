"""
preprocessing.py
==================
Core image preprocessing pipeline:
  - load image (OpenCV) + RGB conversion
  - resize to a fixed input size (default 224x224)
  - denoising for low-quality images
  - pixel normalization to [0, 1]
  - stratified train/validation/test split done on FILE PATHS
    (before any augmentation) to prevent data leakage
  - class-imbalance analysis and class-weight computation
"""

from __future__ import annotations

import os
from typing import List, Tuple, Dict

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

DEFAULT_IMG_SIZE = (224, 224)  # (width, height)


def load_image_rgb(path: str) -> np.ndarray:
    """Load an image from disk with OpenCV and convert BGR -> RGB."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def denoise(img: np.ndarray) -> np.ndarray:
    """
    Light denoising step to handle sensor/JPEG-compression noise commonly
    present in field-collected leaf photos. A small-kernel median blur
    removes salt-and-pepper style noise while preserving lesion edges
    (unlike a stronger Gaussian blur, which would smear disease spots).
    """
    return cv2.medianBlur(img, 3)


def resize_image(img: np.ndarray, size: Tuple[int, int] = DEFAULT_IMG_SIZE) -> np.ndarray:
    """Resize to (width, height) using area interpolation (good for down-sizing photos)."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def normalize_pixels(img: np.ndarray) -> np.ndarray:
    """Scale uint8 [0,255] image to float32 [0,1]."""
    return img.astype(np.float32) / 255.0


def preprocess_image(path: str, size: Tuple[int, int] = DEFAULT_IMG_SIZE,
                      apply_denoise: bool = True) -> np.ndarray:
    """Full single-image preprocessing pipeline: load -> RGB -> denoise -> resize -> normalize."""
    img = load_image_rgb(path)
    if apply_denoise:
        img = denoise(img)
    img = resize_image(img, size)
    img = normalize_pixels(img)
    return img


def stratified_split(
    filepaths: List[str],
    labels: List[str],
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Split (filepaths, labels) into train/val/test using stratified sampling
    so class proportions are preserved in every split.

    IMPORTANT (data-leakage prevention): this split happens on the raw file
    list BEFORE any augmentation is applied. Augmented copies of an image
    are only ever generated from within the training split later in the
    pipeline, so no augmented version of a validation/test image can leak
    into training, and no image (or its near-duplicate) appears in more
    than one split.
    """
    train_fp, temp_fp, train_lb, temp_lb = train_test_split(
        filepaths, labels,
        test_size=(val_size + test_size),
        random_state=seed,
        stratify=labels,
    )
    relative_test = test_size / (val_size + test_size)
    val_fp, test_fp, val_lb, test_lb = train_test_split(
        temp_fp, temp_lb,
        test_size=relative_test,
        random_state=seed,
        stratify=temp_lb,
    )
    print(f"[preprocessing] Split sizes -> train: {len(train_fp)}, "
          f"val: {len(val_fp)}, test: {len(test_fp)}")
    return {
        "train": (train_fp, train_lb),
        "val": (val_fp, val_lb),
        "test": (test_fp, test_lb),
    }


def analyze_class_imbalance(labels: List[str]) -> Dict[str, float]:
    """Return per-class sample counts and an overall imbalance ratio (max/min)."""
    classes, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(classes.tolist(), counts.tolist()))
    ratio = float(counts.max() / counts.min()) if counts.min() > 0 else float("inf")
    print(f"[preprocessing] Class imbalance ratio (max/min) = {ratio:.2f}")
    return {"distribution": dist, "imbalance_ratio": ratio}


def compute_balanced_class_weights(labels: List[str]) -> Dict[str, float]:
    """
    Compute per-class weights inversely proportional to class frequency
    (sklearn 'balanced' scheme). These weights are passed into the model's
    loss function so the model isn't biased toward majority classes,
    rather than physically duplicating minority-class images (which would
    risk overfitting to a handful of samples).
    """
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.array(labels))
    class_weights = {cls: float(w) for cls, w in zip(classes, weights)}
    print(f"[preprocessing] Computed balanced class weights: {class_weights}")
    return class_weights
