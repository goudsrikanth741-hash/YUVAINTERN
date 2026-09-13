from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay


def save_training_curves(history, path):
    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(9, 5)); plt.plot(epochs, [h["train_loss"] for h in history], label="Train loss")
    plt.plot(epochs, [h["val_loss"] for h in history], label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training / Validation Loss"); plt.legend(); plt.tight_layout(); plt.savefig(path); plt.close()


def save_confusion_matrix(cm, path, class_names=None):
    plt.figure(figsize=(12, 10)); plt.imshow(cm, interpolation="nearest", cmap="Blues"); plt.title("Confusion Matrix"); plt.colorbar()
    ticks = np.arange(len(cm)); plt.xticks(ticks, class_names if class_names else ticks, rotation=90, fontsize=6); plt.yticks(ticks, class_names if class_names else ticks, fontsize=6)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()


def save_roc_curve(y_true, probs, path, num_classes):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    yb = label_binarize(y_true, classes=list(range(num_classes)))
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        if yb[:, i].sum() == 0 or yb[:, i].sum() == len(yb):
            continue
        fpr, tpr, _ = roc_curve(yb[:, i], probs[:, i]); plt.plot(fpr, tpr, alpha=0.35)
    plt.plot([0, 1], [0, 1], linestyle="--"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("One-vs-Rest ROC Curves"); plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()


def save_comparison(df, path):
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro_ovr"]
    ax = df.set_index("experiment")[metrics].plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05); ax.set_title("Experiment Comparison"); plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()
