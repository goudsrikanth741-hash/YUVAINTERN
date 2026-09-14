"""
data_analysis.py
==================
Exploratory Data Analysis (EDA) for the plant-disease dataset:
- Class distribution (counts + bar chart)
- Image dimension analysis (width/height distribution)
- Sample image grid per class
- Printed dataset statistics summary

All figures are saved under outputs/figures/.
"""

from __future__ import annotations

import os
import random
from collections import Counter
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt
from PIL import Image

from src.data_loader import DatasetIndex


def class_distribution(index: DatasetIndex) -> Dict[str, int]:
    """Return {class_name: image_count} sorted by class name."""
    return {cls: len(files) for cls, files in sorted(index.class_to_files.items())}


def plot_class_distribution(index: DatasetIndex, out_path: str) -> Dict[str, int]:
    dist = class_distribution(index)
    if not dist:
        print("[data_analysis] No classes found - skipping class distribution plot.")
        return dist

    classes = list(dist.keys())
    counts = list(dist.values())

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.6), 6))
    bars = ax.bar(classes, counts, color="#4C8C4A")
    ax.set_title("Class Distribution - Images per Class")
    ax.set_ylabel("Number of Images")
    ax.set_xlabel("Class")
    plt.xticks(rotation=60, ha="right")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(c),
                 ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[data_analysis] Saved class distribution plot -> {out_path}")
    return dist


def analyze_image_dimensions(index: DatasetIndex, sample_per_class: int = 25) -> List[tuple]:
    """
    Sample up to `sample_per_class` images per class and record (width, height, mode).
    Sampling (rather than reading every file) keeps this fast on large datasets while
    still giving a representative picture of image size/format variability.
    """
    dims = []
    rng = random.Random(42)
    for cls, files in index.class_to_files.items():
        sample = files if len(files) <= sample_per_class else rng.sample(files, sample_per_class)
        for fpath in sample:
            try:
                with Image.open(fpath) as img:
                    dims.append((img.width, img.height, img.mode))
            except OSError:
                continue
    return dims


def plot_image_dimensions(dims: List[tuple], out_path: str) -> None:
    if not dims:
        print("[data_analysis] No dimension data - skipping dimension plot.")
        return
    widths = [d[0] for d in dims]
    heights = [d[1] for d in dims]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(widths, bins=20, color="#4C8C4A", alpha=0.8)
    axes[0].set_title("Image Width Distribution")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Count")

    axes[1].hist(heights, bins=20, color="#8C4C4C", alpha=0.8)
    axes[1].set_title("Image Height Distribution")
    axes[1].set_xlabel("Height (px)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[data_analysis] Saved image dimension plot -> {out_path}")


def plot_sample_grid(index: DatasetIndex, out_path: str, n_classes: int = 6, per_class: int = 4) -> None:
    """Show a grid of raw sample images, a few per class, for visual sanity-checking."""
    rng = random.Random(7)
    classes = list(index.class_to_files.keys())
    if not classes:
        print("[data_analysis] No classes found - skipping sample grid.")
        return
    chosen_classes = classes if len(classes) <= n_classes else rng.sample(classes, n_classes)

    fig, axes = plt.subplots(len(chosen_classes), per_class,
                              figsize=(per_class * 2.4, len(chosen_classes) * 2.4))
    if len(chosen_classes) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, cls in enumerate(chosen_classes):
        files = index.class_to_files[cls]
        sample = files if len(files) <= per_class else rng.sample(files, per_class)
        for col in range(per_class):
            ax = axes[row][col]
            ax.axis("off")
            if col < len(sample):
                try:
                    with Image.open(sample[col]) as img:
                        ax.imshow(img.convert("RGB"))
                except OSError:
                    pass
            if col == 0:
                ax.set_ylabel(cls, fontsize=8)
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
        axes[row][0].set_title("")
    fig.suptitle("Sample Images per Class (raw, unprocessed)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[data_analysis] Saved sample image grid -> {out_path}")


def print_dataset_statistics(index: DatasetIndex, dims: List[tuple], duplicates: Dict) -> Dict:
    """Print a human-readable statistics summary and return it as a dict for logging."""
    dist = class_distribution(index)
    counts = np.array(list(dist.values())) if dist else np.array([0])
    widths = np.array([d[0] for d in dims]) if dims else np.array([0])
    heights = np.array([d[1] for d in dims]) if dims else np.array([0])
    modes = Counter([d[2] for d in dims])

    stats = {
        "num_classes": index.num_classes,
        "total_valid_images": index.total_valid_images,
        "corrupt_files": len(index.corrupt_files),
        "non_image_files_skipped": len(index.non_image_files_skipped),
        "min_images_in_a_class": int(counts.min()) if dist else 0,
        "max_images_in_a_class": int(counts.max()) if dist else 0,
        "mean_images_per_class": float(counts.mean()) if dist else 0.0,
        "imbalance_ratio(max/min)": float(counts.max() / max(counts.min(), 1)) if dist else 0.0,
        "sampled_width_mean": float(widths.mean()) if dims else 0.0,
        "sampled_height_mean": float(heights.mean()) if dims else 0.0,
        "sampled_width_range": (int(widths.min()), int(widths.max())) if dims else (0, 0),
        "sampled_height_range": (int(heights.min()), int(heights.max())) if dims else (0, 0),
        "color_mode_counts": dict(modes),
        "duplicate_groups": len(duplicates),
        "duplicate_redundant_files": sum(len(v) - 1 for v in duplicates.values()) if duplicates else 0,
    }

    print("\n===== DATASET STATISTICS =====")
    for k, v in stats.items():
        print(f"{k:32s}: {v}")
    print("===============================\n")
    return stats
