"""
feature_extraction.py
=======================
Hand-crafted feature engineering for leaf images using OpenCV / scikit-image.
These features feed the classical-ML baseline model (see model.py) and are
also useful on their own for exploratory analysis (feature_distributions
figure in main.py).

Feature groups extracted per image:
  1. RGB color statistics       (mean/std per channel)
  2. HSV color statistics       (mean/std per channel - Hue is more robust
                                  to lighting changes than RGB, useful for
                                  distinguishing disease discoloration)
  3. Color histograms           (coarse-binned RGB histogram, flattened)
  4. Texture features           (GLCM: contrast, homogeneity, energy,
                                  correlation - captures lesion/spot texture)
  5. Edge features               (Canny edge density - disease lesions and
                                  necrotic regions increase local edge density)
  6. Shape / leaf-region features (foreground mask via Otsu threshold on
                                  the green/HSV channel: area ratio,
                                  perimeter, extent, solidity - captures
                                  leaf shape irregularity/damage)

All features are simple, interpretable, and directly motivated by what
visually distinguishes a healthy leaf from a diseased one - no filler
features were added purely to inflate the feature count.
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

FEATURE_NAMES: List[str] = []  # populated the first time extract_features runs


def _rgb_stats(img_float: np.ndarray) -> Dict[str, float]:
    feats = {}
    names = ["r", "g", "b"]
    for i, n in enumerate(names):
        channel = img_float[:, :, i]
        feats[f"rgb_{n}_mean"] = float(channel.mean())
        feats[f"rgb_{n}_std"] = float(channel.std())
    return feats


def _hsv_stats(img_uint8_rgb: np.ndarray) -> Dict[str, float]:
    hsv = cv2.cvtColor(img_uint8_rgb, cv2.COLOR_RGB2HSV)
    feats = {}
    names = ["h", "s", "v"]
    for i, n in enumerate(names):
        channel = hsv[:, :, i].astype(np.float32)
        feats[f"hsv_{n}_mean"] = float(channel.mean())
        feats[f"hsv_{n}_std"] = float(channel.std())
    return feats


def _color_histogram(img_uint8_rgb: np.ndarray, bins: int = 8) -> Dict[str, float]:
    feats = {}
    for i, n in enumerate(["r", "g", "b"]):
        hist = cv2.calcHist([img_uint8_rgb], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        for b in range(bins):
            feats[f"hist_{n}_{b}"] = float(hist[b])
    return feats


def _texture_features(gray_uint8: np.ndarray) -> Dict[str, float]:
    # Quantize to fewer gray levels for a faster/more stable GLCM
    levels = 32
    quant = (gray_uint8.astype(np.float32) / 256.0 * levels).astype(np.uint8)
    quant = np.clip(quant, 0, levels - 1)
    glcm = graycomatrix(quant, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                         levels=levels, symmetric=True, normed=True)
    feats = {}
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        vals = graycoprops(glcm, prop)
        feats[f"texture_{prop}"] = float(vals.mean())
    return feats


def _edge_features(gray_uint8: np.ndarray) -> Dict[str, float]:
    edges = cv2.Canny(gray_uint8, 100, 200)
    density = float((edges > 0).mean())
    return {"edge_density": density}


def _shape_features(img_uint8_rgb: np.ndarray) -> Dict[str, float]:
    hsv = cv2.cvtColor(img_uint8_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    # Otsu threshold on saturation channel tends to separate the leaf
    # (higher saturation) from a plain/backdrop background.
    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = mask.shape
    area_ratio = float((mask > 0).sum() / (h * w))

    feats = {"shape_area_ratio": area_ratio}
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        x, y, bw, bh = cv2.boundingRect(largest)
        extent = float(area / (bw * bh)) if bw * bh > 0 else 0.0
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        solidity = float(area / hull_area) if hull_area > 0 else 0.0
        feats.update({
            "shape_perimeter": float(perimeter),
            "shape_extent": extent,
            "shape_solidity": solidity,
        })
    else:
        feats.update({"shape_perimeter": 0.0, "shape_extent": 0.0, "shape_solidity": 0.0})
    return feats


def extract_features(img_float01: np.ndarray) -> Dict[str, float]:
    """
    Extract the full feature vector for one preprocessed image.
    `img_float01` must be RGB, float32, normalized to [0, 1] (i.e. the
    output of preprocessing.preprocess_image).
    """
    img_uint8 = (img_float01 * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    feats: Dict[str, float] = {}
    feats.update(_rgb_stats(img_float01))
    feats.update(_hsv_stats(img_uint8))
    feats.update(_color_histogram(img_uint8))
    feats.update(_texture_features(gray))
    feats.update(_edge_features(gray))
    feats.update(_shape_features(img_uint8))

    global FEATURE_NAMES
    if not FEATURE_NAMES:
        FEATURE_NAMES = list(feats.keys())
    return feats


def build_feature_dataframe(images: List[np.ndarray], labels: List[str],
                             filepaths: List[str] = None) -> pd.DataFrame:
    """Extract features for a list of preprocessed images and return a tidy DataFrame."""
    rows = []
    for i, (img, lbl) in enumerate(zip(images, labels)):
        feats = extract_features(img)
        feats["label"] = lbl
        if filepaths is not None:
            feats["filepath"] = filepaths[i]
        rows.append(feats)
    df = pd.DataFrame(rows)
    return df
