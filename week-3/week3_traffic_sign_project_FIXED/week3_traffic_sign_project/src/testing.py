"""Held-out test evaluation helper."""
import torch
from .engine import run_epoch
from .metrics import compute_metrics


def test_model(model, loader, device, num_classes):
    criterion = torch.nn.CrossEntropyLoss()
    loss, y_true, y_pred, probs = run_epoch(model, loader, criterion, device, train=False)
    metrics = compute_metrics(y_true, y_pred, probs, num_classes)
    metrics["test_loss"] = loss
    return metrics, y_true, y_pred, probs
