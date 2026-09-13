import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def compute_metrics(y_true, y_pred, probs, num_classes):
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    y_true_oh = np.eye(num_classes)[np.asarray(y_true)]
    try:
        result["roc_auc_macro_ovr"] = float(roc_auc_score(y_true_oh, probs, multi_class="ovr", average="macro"))
    except ValueError:
        result["roc_auc_macro_ovr"] = float("nan")
    result["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()
    return result
