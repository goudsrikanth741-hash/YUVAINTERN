"""
data_loader.py
================
Responsible for discovering the PlantVillage-style dataset on disk,
detecting classes, listing image paths per class, and validating
that each image file is actually readable (not corrupt/truncated).

Expected folder layout:

    dataset/
        Apple___healthy/
            image1.jpg
            image2.jpg
        Apple___Black_rot/
            ...
        Potato___Early_blight/
            ...

Each sub-folder of `dataset/` is treated as one class. The class
name is taken directly from the folder name (e.g. "Tomato___healthy").
"""

from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from PIL import Image, UnidentifiedImageError

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class DatasetIndex:
    """Container holding the result of scanning the dataset folder."""

    root: str
    classes: List[str] = field(default_factory=list)
    class_to_files: Dict[str, List[str]] = field(default_factory=dict)
    corrupt_files: List[str] = field(default_factory=list)
    non_image_files_skipped: List[str] = field(default_factory=list)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def total_valid_images(self) -> int:
        return sum(len(v) for v in self.class_to_files.values())


def discover_classes(dataset_root: str) -> List[str]:
    """Return sorted list of class names (sub-folder names) found under dataset_root."""
    if not os.path.isdir(dataset_root):
        return []
    classes = [
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d)) and not d.startswith(".")
    ]
    return sorted(classes)


def _is_readable_image(path: str) -> bool:
    """Try to fully load an image to make sure it is not corrupt/truncated."""
    try:
        with Image.open(path) as img:
            img.verify()  # cheap structural check
        # verify() leaves the file handle unusable for pixel access, so
        # re-open and force a full pixel decode to catch truncated data too.
        with Image.open(path) as img:
            img.load()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def scan_dataset(dataset_root: str, verbose: bool = True) -> DatasetIndex:
    """
    Walk the dataset directory, detect classes, list image files per class,
    and validate that each image is actually readable. Corrupt/unreadable
    files are recorded separately and excluded from class_to_files.
    """
    index = DatasetIndex(root=dataset_root)
    index.classes = discover_classes(dataset_root)

    for cls in index.classes:
        cls_dir = os.path.join(dataset_root, cls)
        valid_files = []
        for fname in sorted(os.listdir(cls_dir)):
            fpath = os.path.join(cls_dir, fname)
            if not os.path.isfile(fpath):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VALID_EXTENSIONS:
                index.non_image_files_skipped.append(fpath)
                continue
            if _is_readable_image(fpath):
                valid_files.append(fpath)
            else:
                index.corrupt_files.append(fpath)
        index.class_to_files[cls] = valid_files

    if verbose:
        print(f"[data_loader] Scanned '{dataset_root}': "
              f"{index.num_classes} classes, "
              f"{index.total_valid_images} valid images, "
              f"{len(index.corrupt_files)} corrupt/unreadable files skipped, "
              f"{len(index.non_image_files_skipped)} non-image files skipped.")

    return index


def file_md5(path: str, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file's bytes (used for exact-duplicate detection)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def find_exact_duplicates(index: DatasetIndex, verbose: bool = True) -> Dict[str, List[str]]:
    """
    Detect exact byte-for-byte duplicate images across the whole dataset
    using MD5 hashing. Returns a dict {hash: [file_paths...]} for hashes
    that appear more than once (i.e. actual duplicate groups).
    """
    hash_to_paths: Dict[str, List[str]] = {}
    for cls, files in index.class_to_files.items():
        for fpath in files:
            try:
                h = file_md5(fpath)
            except OSError:
                continue
            hash_to_paths.setdefault(h, []).append(fpath)

    duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
    if verbose:
        n_dupe_files = sum(len(p) - 1 for p in duplicates.values())
        print(f"[data_loader] Duplicate scan: {len(duplicates)} duplicate groups, "
              f"{n_dupe_files} redundant files.")
    return duplicates


def flatten_index(index: DatasetIndex) -> Tuple[List[str], List[str]]:
    """Flatten class_to_files into parallel (filepaths, labels) lists."""
    filepaths, labels = [], []
    for cls, files in index.class_to_files.items():
        for f in files:
            filepaths.append(f)
            labels.append(cls)
    return filepaths, labels
