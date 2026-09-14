"""
evaluation.py
==============
Standard classification evaluation: accuracy, precision, recall, F1
(macro-averaged, appropriate given class imbalance), confusion matrix,
and a full text classification report. Results are saved as both a
plot (confusion matrix) and text/CSV files under outputs/results/.
"""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)


def compute_metrics(y_true: List[str], y_pred: List[str]) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    print("\n===== EVALUATION METRICS =====")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.4f}")
    print("===============================\n")
    return metrics


def plot_confusion_matrix(y_true: List[str], y_pred: List[str], class_names: List[str],
                           out_path: str) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.7), max(5, len(class_names) * 0.7)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluation] Saved confusion matrix plot -> {out_path}")
    return cm


def save_full_report(y_true: List[str], y_pred: List[str], class_names: List[str],
                      metrics: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    report_txt = classification_report(y_true, y_pred, labels=class_names, zero_division=0)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report_txt)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluation] Saved classification_report.txt and metrics.json -> {out_dir}")
