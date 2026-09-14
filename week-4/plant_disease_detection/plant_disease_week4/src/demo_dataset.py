"""
demo_dataset.py
=================
*** FOR TESTING / DEMONSTRATION ONLY - NOT REAL PLANTVILLAGE DATA ***

The real PlantVillage dataset is not present in this environment and
cannot be downloaded here (no internet access). To let every stage of
the pipeline (loading, EDA, preprocessing, augmentation, feature
extraction, model training, evaluation) actually run and be verified
end-to-end, this module generates a small SYNTHETIC placeholder dataset
that mimics the PlantVillage folder structure:

    dataset/
        Apple___healthy/
        Apple___Black_rot/
        Potato___Early_blight/
        Potato___Late_blight/
        Tomato___healthy/

Each synthetic "leaf" image is a simple procedurally-generated image
(a leaf-shaped blob of green shades with class-specific colored blotches
to simulate lesions, plus random noise). These images are NOT real leaf
photographs and any statistics computed from them describe only this
synthetic demo data, NOT the actual PlantVillage dataset.

When the real dataset is placed under dataset/ (see README.md for the
expected structure), this generator is skipped automatically and the
real images are used instead.
"""

from __future__ import annotations

import os
import random

import numpy as np
from PIL import Image, ImageDraw

DEMO_CLASSES = {
    "Apple___healthy": {"leaf": (60, 140, 60), "spot": None},
    "Apple___Black_rot": {"leaf": (70, 130, 55), "spot": (40, 25, 15)},
    "Potato___Early_blight": {"leaf": (80, 150, 70), "spot": (110, 80, 30)},
    "Potato___Late_blight": {"leaf": (65, 120, 60), "spot": (60, 45, 25)},
    "Tomato___healthy": {"leaf": (55, 150, 65), "spot": None},
}


def _make_leaf_image(size: int, leaf_color: tuple, spot_color, rng: random.Random,
                      corrupt: bool = False) -> Image.Image:
    img = Image.new("RGB", (size, size), (235, 235, 225))  # plain background
    draw = ImageDraw.Draw(img)

    # Simple leaf silhouette: an ellipse with a bit of random jitter.
    cx, cy = size // 2, size // 2
    rx, ry = int(size * 0.38), int(size * 0.44)
    jitter = int(size * 0.03)
    bbox = [cx - rx + rng.randint(-jitter, jitter), cy - ry + rng.randint(-jitter, jitter),
            cx + rx + rng.randint(-jitter, jitter), cy + ry + rng.randint(-jitter, jitter)]
    shade = tuple(max(0, min(255, c + rng.randint(-15, 15))) for c in leaf_color)
    draw.ellipse(bbox, fill=shade)

    # A simple midrib line.
    draw.line([(cx, bbox[1] + 5), (cx, bbox[3] - 5)], fill=tuple(max(0, c - 30) for c in shade), width=2)

    # Disease "lesions" as small random blotches, only for diseased classes.
    if spot_color is not None:
        n_spots = rng.randint(4, 10)
        for _ in range(n_spots):
            sx = rng.randint(bbox[0] + 10, bbox[2] - 10)
            sy = rng.randint(bbox[1] + 10, bbox[3] - 10)
            r = rng.randint(3, int(size * 0.05))
            spot_shade = tuple(max(0, min(255, c + rng.randint(-10, 10))) for c in spot_color)
            draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=spot_shade)

    arr = np.array(img).astype(np.int16)
    noise = np.random.randint(-6, 6, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    if corrupt:
        return img  # caller will save this truncated on purpose
    return img


def generate_demo_dataset(dataset_root: str, images_per_class: int = 40,
                           image_size: int = 256, seed: int = 42,
                           include_corrupt: int = 2, include_duplicates: int = 3) -> None:
    """
    Populate `dataset_root` with a small synthetic PlantVillage-style
    dataset if it is currently empty. Also intentionally injects a couple
    of corrupt files and duplicate files so the data-quality checks in
    data_loader.py / data_analysis.py have something real to detect.
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    os.makedirs(dataset_root, exist_ok=True)

    print("=" * 70)
    print("[demo_dataset] NOTE: Generating a SYNTHETIC placeholder dataset for")
    print("[demo_dataset] demonstration purposes because no real PlantVillage")
    print("[demo_dataset] data was found under dataset/. These are procedurally")
    print("[demo_dataset] drawn images, NOT real leaf photographs. Replace")
    print("[demo_dataset] dataset/ with the real PlantVillage folders for a")
    print("[demo_dataset] genuine run. See README.md for instructions.")
    print("=" * 70)

    for cls, spec in DEMO_CLASSES.items():
        cls_dir = os.path.join(dataset_root, cls)
        os.makedirs(cls_dir, exist_ok=True)
        saved_paths = []
        for i in range(images_per_class):
            img = _make_leaf_image(image_size, spec["leaf"], spec["spot"], rng)
            fname = f"{cls}_{i:04d}.jpg"
            fpath = os.path.join(cls_dir, fname)
            img.save(fpath, quality=90)
            saved_paths.append(fpath)

        # Inject a couple of intentionally corrupt files to test detection.
        for c in range(include_corrupt):
            bad_path = os.path.join(cls_dir, f"{cls}_corrupt_{c}.jpg")
            with open(bad_path, "wb") as f:
                f.write(b"not a real jpeg file - truncated/corrupt bytes")

        # Inject exact duplicates (copy of an existing valid file) to test
        # duplicate detection.
        for d in range(include_duplicates):
            if saved_paths:
                src = rng.choice(saved_paths)
                dup_path = os.path.join(cls_dir, f"{cls}_dup_{d}.jpg")
                with open(src, "rb") as fsrc, open(dup_path, "wb") as fdst:
                    fdst.write(fsrc.read())

    print(f"[demo_dataset] Done. Wrote {len(DEMO_CLASSES)} classes x "
          f"~{images_per_class} images to '{dataset_root}'.")


def dataset_is_empty(dataset_root: str) -> bool:
    if not os.path.isdir(dataset_root):
        return True
    for entry in os.listdir(dataset_root):
        full = os.path.join(dataset_root, entry)
        if os.path.isdir(full) and any(os.scandir(full)):
            return False
    return True
