"""
optimization.py
===============
Leakage-safe optimization experiments for customer churn prediction.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, precision_score

RANDOM_SEED = 42


def random_oversample(X_train, y_train, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    X_train = X_train.reset_index(drop=True)
    y_train = np.asarray(y_train)
    minority = np.where(y_train == 1)[0]
    majority = np.where(y_train == 0)[0]
    if len(minority) == 0 or len(minority) >= len(majority):
        return X_train, y_train
    extra = rng.choice(minority, size=len(majority)-len(minority), replace=True)
    X_resampled = pd.concat([X_train, X_train.iloc[extra]], ignore_index=True)
    y_resampled = np.concatenate([y_train, y_train[extra]])
    idx = rng.permutation(len(y_resampled))
    return X_resampled.iloc[idx].reset_index(drop=True), y_resampled[idx]


def class_weight_balanced_models():
    return {
        "Logistic Regression (balanced)": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED),
        "Random Forest (balanced)": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
    }


def tune_gradient_boosting(X_train, y_train, cv_folds=5):
    grid = GridSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_SEED),
        {"n_estimators":[100,200], "learning_rate":[0.05,0.1], "max_depth":[2,3]},
        scoring="f1",
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, float(grid.best_score_)


def cross_validate_model(model, X, y, cv_folds=5, scoring="f1"):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    return cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)


def threshold_sweep(y_true, y_proba):
    """Evaluate thresholds on a validation set only."""
    records = []
    for t in np.arange(0.10, 0.91, 0.025):
        pred = (y_proba >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f = f1_score(y_true, pred, zero_division=0)
        records.append({"threshold": round(float(t), 3), "Precision": p, "Recall": r, "F1": f})
    return pd.DataFrame(records)


def find_best_threshold(y_true, y_proba, metric="F1"):
    sweep = threshold_sweep(y_true, y_proba)
    idx = sweep[metric].idxmax()
    return float(sweep.loc[idx, "threshold"]), float(sweep.loc[idx, metric]), sweep


def find_cost_optimal_threshold(y_true, y_proba, fn_cost=5.0, fp_cost=1.0):
    """Choose threshold minimizing business cost on validation data."""
    sweep = threshold_sweep(y_true, y_proba)
    costs = []
    for t in sweep["threshold"]:
        pred = (y_proba >= t).astype(int)
        fn = int(((y_true == 1) & (pred == 0)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        costs.append(fn_cost * fn + fp_cost * fp)
    sweep["BusinessCost"] = costs
    idx = sweep["BusinessCost"].idxmin()
    return float(sweep.loc[idx, "threshold"]), float(sweep.loc[idx, "BusinessCost"]), sweep
