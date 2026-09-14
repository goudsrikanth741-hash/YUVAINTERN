"""
evaluation.py
=============
Metrics, diagnostics, plots and result-table utilities.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, classification_report,
    balanced_accuracy_score, matthews_corrcoef, log_loss, brier_score_loss
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
RES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "results")
sns.set_theme(style="whitegrid")


def ensure_output_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)


def get_probabilities(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        p = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    else:
        p = model.predict(X).astype(float)
    return np.clip(np.asarray(p), 1e-7, 1 - 1e-7)


def compute_metrics(model, X_test, y_test, threshold=0.5):
    y_proba = get_probabilities(model, X_test)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "log_loss": log_loss(y_test, y_proba, labels=[0, 1]),
        "brier_score": brier_score_loss(y_test, y_proba),
        "threshold": float(threshold), "y_pred": y_pred, "y_proba": y_proba,
    }


def metrics_table(all_metrics):
    rows = []
    for name, m in all_metrics.items():
        rows.append({
            "Model": name, "Accuracy": round(m["accuracy"], 4),
            "Balanced Accuracy": round(m["balanced_accuracy"], 4),
            "Precision": round(m["precision"], 4), "Recall": round(m["recall"], 4),
            "F1-score": round(m["f1_score"], 4), "MCC": round(m["mcc"], 4),
            "ROC-AUC": round(m["roc_auc"], 4), "Log Loss": round(m["log_loss"], 4),
            "Brier Score": round(m["brier_score"], 4), "Threshold": m["threshold"],
        })
    return pd.DataFrame(rows).sort_values("F1-score", ascending=False).reset_index(drop=True)


def classification_reports(models, X, y, threshold=0.5):
    rows = []
    for name, model in models.items():
        m = compute_metrics(model, X, y, threshold)
        report = classification_report(y, m["y_pred"], output_dict=True, zero_division=0)
        for label, vals in report.items():
            if isinstance(vals, dict):
                rows.append({"Model": name, "Class": label,
                             "Precision": round(vals["precision"], 4),
                             "Recall": round(vals["recall"], 4),
                             "F1": round(vals["f1-score"], 4),
                             "Support": int(vals["support"])})
    return pd.DataFrame(rows)


def plot_class_distribution(y, save_name="class_distribution.png"):
    ensure_output_dirs()
    plt.figure(figsize=(5, 4))
    counts = pd.Series(y).value_counts().sort_index()
    labels = ["No Churn (0)", "Churn (1)"]
    sns.barplot(x=labels, y=counts.reindex([0, 1], fill_value=0).values, hue=labels, legend=False)
    plt.title("Class Distribution: Customer Churn"); plt.ylabel("Customers")
    for i, v in enumerate(counts.reindex([0, 1], fill_value=0).values):
        plt.text(i, v + max(counts.sum()*0.01, 1), f"{v} ({v/counts.sum():.1%})", ha="center")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, save_name), dpi=150); plt.close()


def plot_confusion_matrix(y_test, y_pred, model_name, save_name):
    ensure_output_dirs()
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title(f"Confusion Matrix - {model_name}"); plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, save_name), dpi=150); plt.close()
    return cm


def plot_roc_curves(all_metrics, y_test, save_name="roc_curves.png"):
    ensure_output_dirs(); plt.figure(figsize=(6, 5))
    for name, m in all_metrics.items():
        fpr, tpr, _ = roc_curve(y_test, m["y_proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={m['roc_auc']:.3f})")
    plt.plot([0,1],[0,1],"--",label="Random")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curves")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, save_name), dpi=150); plt.close()


def plot_pr_curves(all_metrics, y_test, save_name="precision_recall_curves.png"):
    ensure_output_dirs(); plt.figure(figsize=(6, 5))
    for name, m in all_metrics.items():
        p, r, _ = precision_recall_curve(y_test, m["y_proba"])
        plt.plot(r, p, label=name)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curves")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, save_name), dpi=150); plt.close()


def plot_metric_comparison(metrics_df, save_name="metric_comparison.png"):
    ensure_output_dirs()
    cols = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "MCC"]
    melted = metrics_df.melt(id_vars="Model", value_vars=cols, var_name="Metric", value_name="Score")
    plt.figure(figsize=(9, 5)); sns.barplot(data=melted, x="Metric", y="Score", hue="Model")
    plt.ylim(-1,1); plt.title("Model Evaluation Comparison"); plt.xticks(rotation=15)
    plt.legend(bbox_to_anchor=(1.02,1), loc="upper left")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, save_name), dpi=150); plt.close()


def plot_feature_importance(model, feature_names, model_name, save_name, top_n=15):
    ensure_output_dirs()
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False).head(top_n)
    plt.figure(figsize=(7,6)); sns.barplot(data=imp_df, x="importance", y="feature", hue="feature", legend=False)
    plt.title(f"Feature Importance - {model_name}"); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, save_name), dpi=150); plt.close()
    return imp_df


def plot_threshold_curve(sweep_df, save_name="threshold_tradeoff_curve.png"):
    ensure_output_dirs(); plt.figure(figsize=(7,5))
    for col in ["F1", "Precision", "Recall"]:
        plt.plot(sweep_df["threshold"], sweep_df[col], label=col)
    plt.xlabel("Decision Threshold"); plt.ylabel("Score"); plt.ylim(0,1)
    plt.title("Threshold Trade-off"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, save_name), dpi=150); plt.close()


def gap_analysis(metrics_df):
    df = metrics_df.copy()
    df["Precision-Recall Gap"] = (df["Precision"] - df["Recall"]).round(4)
    df["Weakness"] = np.select(
        [df["Recall"] < 0.6, df["Precision"] < 0.6, df["ROC-AUC"] < 0.75],
        ["Low recall: threshold/class-imbalance issue",
         "Low precision: too many false alarms", "Weak separability: improve features/model"],
        default="No major weakness at default threshold")
    return df


def save_table(df, filename):
    ensure_output_dirs(); path = os.path.join(RES_DIR, filename)
    df.to_csv(path, index=False); return path
