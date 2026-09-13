"""Helpers for discovering trained experiments and running predictions on
single images. Used by the Streamlit Prediction page so the user can pick
*which* trained experiment's model to use, instead of always loading
whichever folder happens to sort last.
"""
from pathlib import Path
import json
import torch
from torchvision import transforms

from .models import build_model
from .data import GTSRB_MEAN, GTSRB_STD, get_class_name


def list_trained_experiments(output_dir="outputs"):
    """Return experiment names (sorted) that have a saved checkpoint."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []
    names = []
    for d in sorted(output_dir.iterdir()):
        if d.is_dir() and (d / "best_model.pt").exists():
            names.append(d.name)
    return names


def _experiment_architecture(output_dir, exp_name, default="resnet18"):
    """Figure out which architecture an experiment used, from metrics.json
    if available (falls back to a sensible default for older runs)."""
    metrics_path = Path(output_dir) / exp_name / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "model" in data:
                return data["model"]
        except Exception:
            pass
    return default


def load_experiment_model(exp_name, output_dir="outputs", num_classes=43, device=None):
    """Load the best checkpoint for `exp_name` and return (model, device)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = _experiment_architecture(output_dir, exp_name)
    model = build_model(arch, num_classes, pretrained=False).to(device)
    ckpt_path = Path(output_dir) / exp_name / "best_model.pt"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, device


def predict_image(model, device, image, image_size=64, top_k=5):
    """Run inference on a single PIL image, returning a list of
    (class_idx, class_name, confidence) tuples sorted by confidence desc."""
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(GTSRB_MEAN, GTSRB_STD),
    ])
    image = image.convert("RGB")
    with torch.no_grad():
        input_tensor = tf(image).unsqueeze(0).to(device)
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
    values, indices = torch.topk(probs, min(top_k, probs.shape[0]))
    results = []
    for confidence, class_idx in zip(values.tolist(), indices.tolist()):
        results.append((class_idx, get_class_name(class_idx), confidence))
    return results
