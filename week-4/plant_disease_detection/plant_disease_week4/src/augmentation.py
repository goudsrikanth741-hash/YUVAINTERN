"""
augmentation.py
=================
Training-only data augmentation implemented directly with OpenCV/NumPy
(no dependency on a specific deep-learning framework's ImageDataGenerator,
so it works whether the downstream model is Keras, PyTorch, or the
scikit-learn feature-based baseline).

Augmentation must ONLY ever be applied to the training split - never to
validation or test images - to keep evaluation numbers honest.
"""

from __future__ import annotations

import random
from typing import Callable, List

import cv2
import numpy as np


def random_horizontal_flip(img: np.ndarray, p: float = 0.5) -> np.ndarray:
    if random.random() < p:
        return np.fliplr(img).copy()
    return img


def random_rotation(img: np.ndarray, max_angle: float = 25.0) -> np.ndarray:
    angle = random.uniform(-max_angle, max_angle)
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def random_brightness_contrast(img: np.ndarray, brightness: float = 0.2, contrast: float = 0.2) -> np.ndarray:
    """img expected as float32 in [0,1]."""
    b = 1.0 + random.uniform(-brightness, brightness)
    c = 1.0 + random.uniform(-contrast, contrast)
    mean = img.mean()
    out = (img - mean) * c + mean * b
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def random_zoom(img: np.ndarray, max_zoom: float = 0.15) -> np.ndarray:
    h, w = img.shape[:2]
    zoom = 1.0 + random.uniform(0, max_zoom)
    nh, nw = int(h * zoom), int(w * zoom)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (nh - h) // 2
    left = (nw - w) // 2
    return resized[top:top + h, left:left + w]


def random_gaussian_noise(img: np.ndarray, std: float = 0.02) -> np.ndarray:
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0).astype(np.float32)


AUGMENTATIONS: List[Callable[[np.ndarray], np.ndarray]] = [
    random_horizontal_flip,
    random_rotation,
    random_brightness_contrast,
    random_zoom,
    random_gaussian_noise,
]


def augment_image(img: np.ndarray, n_ops: int = 2, seed: int | None = None) -> np.ndarray:
    """
    Apply a random subset of `n_ops` augmentation operations to a single
    already-preprocessed (resized, normalized, float32 [0,1]) image.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    ops = random.sample(AUGMENTATIONS, k=min(n_ops, len(AUGMENTATIONS)))
    out = img.copy()
    for op in ops:
        out = op(out)
        # random_zoom/random_rotation can change dtype/shape edge cases; guard shape
        if out.shape[:2] != img.shape[:2]:
            out = cv2.resize(out, (img.shape[1], img.shape[0]))
    return out.astype(np.float32)


def augment_training_set(images: List[np.ndarray], labels: List[str],
                          multiplier: int = 1, seed: int = 42) -> tuple:
    """
    Expand a training set by generating `multiplier` augmented copies of
    every image (in addition to the original). Only meant to be called on
    the TRAIN split.
    """
    rng = random.Random(seed)
    out_images, out_labels = list(images), list(labels)
    for i, (img, lbl) in enumerate(zip(images, labels)):
        for k in range(multiplier):
            aug = augment_image(img, n_ops=2, seed=seed + i * 10 + k)
            out_images.append(aug)
            out_labels.append(lbl)
    print(f"[augmentation] Training set expanded from {len(images)} to {len(out_images)} images "
          f"(multiplier={multiplier}).")
    return out_images, out_labels
