"""Data loading, preprocessing, and augmentation utilities for GTSRB.

NOTE ON A PREVIOUSLY-BROKEN BUG
--------------------------------
`torchvision.datasets.GTSRB` does NOT expose `.classes`, `.targets`, or
`._labels` attributes (unlike e.g. ImageFolder). It only stores raw
(path, label) tuples in `._samples`. Earlier versions of this file assumed
those attributes existed, which raised an `AttributeError` the moment
`make_dataloaders()` (or the Streamlit app) touched the dataset. This
version reads labels directly from `._samples`, which is what actually
exists on the object, and no longer crashes.
"""
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image

GTSRB_MEAN = (0.3403, 0.3121, 0.3214)
GTSRB_STD = (0.2724, 0.2608, 0.2669)
NUM_CLASSES = 43

# Human-readable GTSRB class names (official ordering, index == ClassId)
GTSRB_CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 tons",
    "Right-of-way at next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 tons prohibited", "No entry",
    "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 tons",
]


def get_class_name(class_idx: int) -> str:
    if 0 <= class_idx < len(GTSRB_CLASS_NAMES):
        return GTSRB_CLASS_NAMES[class_idx]
    return f"Class {class_idx}"


def _labels_of(dataset) -> list:
    """Safely extract integer labels from a torchvision GTSRB dataset."""
    if hasattr(dataset, "_samples"):
        return [label for _, label in dataset._samples]
    if hasattr(dataset, "_labels"):
        return list(dataset._labels)
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    raise AttributeError("Could not find labels on GTSRB dataset object")


def make_transforms(image_size=64, augmentation=False, rotation=True, affine=True, color_jitter=True):
    """Build train/eval transform pipelines.

    `augmentation` is the master switch. When True, the individual
    `rotation` / `affine` / `color_jitter` flags let the caller (e.g. the
    Streamlit Augmentation page) turn specific augmentation ops on or off
    for experimentation, instead of an all-or-nothing boolean.
    """
    train_ops = [transforms.Resize((image_size, image_size))]
    if augmentation:
        if rotation:
            train_ops.append(transforms.RandomRotation(12))
        if affine:
            train_ops.append(transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)))
        if color_jitter:
            train_ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    train_ops += [transforms.ToTensor(), transforms.Normalize(GTSRB_MEAN, GTSRB_STD)]
    eval_ops = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(GTSRB_MEAN, GTSRB_STD),
    ]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def stratified_indices(targets, val_split=0.15, seed=42):
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(targets):
        ids = np.flatnonzero(targets == cls)
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * val_split)))
        val_idx.extend(ids[:n_val].tolist())
        train_idx.extend(ids[n_val:].tolist())
    return np.array(train_idx), np.array(val_idx)


def make_dataloaders(data_dir, image_size=64, batch_size=64, num_workers=2,
                      val_split=0.15, augmentation=False, seed=42,
                      rotation=True, affine=True, color_jitter=True):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_tf, eval_tf = make_transforms(image_size, augmentation, rotation, affine, color_jitter)

    # torchvision downloads and extracts GTSRB when download=True.
    full_train_aug = datasets.GTSRB(root=str(data_dir), split="train", download=True, transform=train_tf)
    full_train_eval = datasets.GTSRB(root=str(data_dir), split="train", download=True, transform=eval_tf)
    test_ds = datasets.GTSRB(root=str(data_dir), split="test", download=True, transform=eval_tf)

    targets = np.asarray(_labels_of(full_train_eval))
    train_idx, val_idx = stratified_indices(targets, val_split, seed)
    train_ds = Subset(full_train_aug, train_idx)
    val_ds = Subset(full_train_eval, val_idx)

    nclasses = NUM_CLASSES

    generator = torch.Generator().manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader, nclasses


# ---------------------------------------------------------------------------
# Helpers used by the Streamlit "Data Collection" / "Preprocessing" /
# "Augmentation" pipeline pages. These are intentionally lightweight and
# read-only so they can be called from the UI without side effects beyond
# downloading the dataset the first time.
# ---------------------------------------------------------------------------

def collect_dataset(data_dir, download=True):
    """Download (if needed) and summarize the raw GTSRB dataset.

    Returns a dict with counts and the per-class distribution, suitable for
    display on the Data Collection page.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    base_tf = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    train_ds = datasets.GTSRB(root=str(data_dir), split="train", download=download, transform=base_tf)
    test_ds = datasets.GTSRB(root=str(data_dir), split="test", download=download, transform=base_tf)

    train_labels = _labels_of(train_ds)
    test_labels = _labels_of(test_ds)
    class_counts = Counter(train_labels)

    return {
        "num_train": len(train_ds),
        "num_test": len(test_ds),
        "num_classes": NUM_CLASSES,
        "class_distribution": {int(k): int(v) for k, v in sorted(class_counts.items())},
        "data_dir": str(data_dir),
    }


def sample_raw_images(data_dir, n=6, seed=42):
    """Return `n` raw (unprocessed) PIL images + labels from the train split,
    for previewing on the Data Collection / Preprocessing pages."""
    data_dir = Path(data_dir)
    raw_ds = datasets.GTSRB(root=str(data_dir), split="train", download=True, transform=None)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(raw_ds), size=min(n, len(raw_ds)), replace=False)
    samples = []
    for i in idxs:
        img, label = raw_ds[int(i)]
        samples.append((img, int(label)))
    return samples


def preview_preprocessing(image: Image.Image, image_size=64):
    """Apply the eval preprocessing pipeline to a single PIL image and
    return a displayable (denormalized) PIL image alongside the raw resize."""
    resized = image.convert("RGB").resize((image_size, image_size))
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(GTSRB_MEAN, GTSRB_STD),
    ])
    tensor = tf(image.convert("RGB"))
    denorm = tensor.clone()
    for c in range(3):
        denorm[c] = denorm[c] * GTSRB_STD[c] + GTSRB_MEAN[c]
    denorm = torch.clamp(denorm, 0, 1)
    denorm_img = transforms.ToPILImage()(denorm)
    return resized, denorm_img


def preview_augmentations(image: Image.Image, image_size=64, n=6, rotation=True, affine=True, color_jitter=True, seed=None):
    """Generate `n` augmented previews of a single PIL image using the same
    ops as training augmentation, for display on the Augmentation page."""
    if seed is not None:
        torch.manual_seed(seed)
    ops = [transforms.Resize((image_size, image_size))]
    if rotation:
        ops.append(transforms.RandomRotation(12))
    if affine:
        ops.append(transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)))
    if color_jitter:
        ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    display_tf = transforms.Compose(ops)
    image = image.convert("RGB")
    return [display_tf(image) for _ in range(n)]
