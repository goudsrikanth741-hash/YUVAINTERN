"""Validation helper for evaluating a model during experiment development."""
import torch
from .engine import run_epoch


def validate(model, loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    loss, y_true, y_pred, probs = run_epoch(model, loader, criterion, device, train=False)
    accuracy = float((y_true == y_pred).mean())
    return {"loss": loss, "accuracy": accuracy, "y_true": y_true, "y_pred": y_pred, "probs": probs}
