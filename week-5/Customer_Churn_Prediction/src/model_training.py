"""
model_training.py
==================
Baseline classification models for churn prediction.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

RANDOM_SEED = 42


def get_baseline_models():
    """Return a dict of un-tuned baseline models."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
    }


def train_models(models: dict, X_train, y_train):
    """Fit every model in `models` on the training data. Returns fitted models."""
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted
