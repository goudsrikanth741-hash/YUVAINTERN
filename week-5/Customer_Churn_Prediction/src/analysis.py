"""
analysis.py
===========
Builds the evaluation framework description and the clearly-labelled
HYPOTHETICAL / SIMULATED analysis section that the Week 5 assignment
explicitly permits.

Nothing in this file touches the real train/test data used for the
actual model results - it is a self-contained illustrative example
of how evaluation results are interpreted, using invented "what-if"
numbers.
"""

import numpy as np
import pandas as pd


def evaluation_framework() -> dict:
    """Return the evaluation framework description as structured data,
    written out to outputs/results/evaluation_framework.md by main.py."""
    return {
        "what_will_be_evaluated": [
            "Baseline classification models (Logistic Regression, Random Forest, "
            "Gradient Boosting) trained to predict customer churn (Yes/No).",
            "The impact of optimization strategies (class-imbalance handling, "
            "hyperparameter tuning, decision-threshold adjustment) on the same models.",
        ],
        "performance_indicators": [
            "Accuracy - overall proportion of correct predictions.",
            "Precision - of customers predicted to churn, how many actually churned "
            "(controls the cost of unnecessary retention offers).",
            "Recall - of customers who actually churned, how many were correctly caught "
            "(controls the cost of missed at-risk customers).",
            "F1-score - harmonic mean of precision and recall, useful under class imbalance.",
            "ROC-AUC - threshold-independent measure of class separability.",
            "Confusion matrix - full breakdown of TP/FP/TN/FN for error-type analysis.",
        ],
        "tools_and_libraries": [
            "pandas / NumPy for data handling",
            "scikit-learn for modelling, metrics, cross-validation and grid search",
            "matplotlib / seaborn for visualization",
        ],
        "interpretation_approach": (
            "Metrics are compared across models and against the baseline before/after "
            "optimization. Because churn is imbalanced (churners are the minority class), "
            "accuracy alone is treated as unreliable; F1-score, recall and ROC-AUC are "
            "weighted more heavily since missing a churner (false negative) is typically "
            "more costly to the business than a false retention offer (false positive)."
        ),
        "weakness_identification_approach": (
            "For each model, precision and recall are compared: a large gap where recall "
            "is much lower than precision indicates the model is biased toward the majority "
            "(non-churn) class, a classic symptom of class imbalance. ROC-AUC below ~0.75 is "
            "treated as evidence of weak feature separability requiring feature engineering "
            "or a different model family. Cross-validation score variance is used to detect "
            "overfitting to the training split."
        ),
    }


def simulated_scenarios(seed: int = 7) -> pd.DataFrame:
    """Return a small table of HYPOTHETICAL / SIMULATED scenarios illustrating how
    evaluation results could be interpreted under different business conditions.

    IMPORTANT: These numbers are invented for illustration only. They are
    NOT derived from the real experiment run elsewhere in this project and
    must never be presented as actual experimental results.
    """
    rng = np.random.default_rng(seed)
    scenarios = [
        "Aggressive retention (low threshold)",
        "Conservative retention (high threshold)",
        "Balanced threshold",
        "Severely imbalanced future dataset (5% churn)",
        "More balanced future dataset (40% churn)",
    ]
    rows = []
    base_precision = [0.45, 0.78, 0.62, 0.30, 0.71]
    base_recall = [0.82, 0.38, 0.60, 0.25, 0.68]
    for name, p, r in zip(scenarios, base_precision, base_recall):
        p_sim = float(np.clip(p + rng.normal(0, 0.02), 0, 1))
        r_sim = float(np.clip(r + rng.normal(0, 0.02), 0, 1))
        f1_sim = 2 * p_sim * r_sim / (p_sim + r_sim) if (p_sim + r_sim) > 0 else 0.0
        rows.append({
            "Scenario (HYPOTHETICAL)": name,
            "Simulated Precision": round(p_sim, 3),
            "Simulated Recall": round(r_sim, 3),
            "Simulated F1-score": round(f1_sim, 3),
            "Interpretation": _interpret(p_sim, r_sim),
        })
    df = pd.DataFrame(rows)
    return df


def _interpret(precision: float, recall: float) -> str:
    if recall > precision + 0.15:
        return "Recall-favoured: catches most churners but with more false alarms - suits high retention budgets."
    if precision > recall + 0.15:
        return "Precision-favoured: retention offers rarely wasted, but many churners are missed."
    return "Balanced trade-off between catching churners and avoiding wasted retention offers."
