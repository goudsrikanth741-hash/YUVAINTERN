"""
evaluate.py
-----------
Reproducible evaluation and robustness experiments for the fake-news project.

Usage:
    python src/evaluate.py

The script performs:
1. Leakage-safe 3-fold cross-validation using a Pipeline (TF-IDF is fitted
   independently inside each fold).
2. Held-out robustness checks for uppercase, punctuation, URL noise, and
   shortened inputs.
3. Confidence-threshold coverage analysis.

Outputs are written to experiments/results/.
"""
import json, os, re, sys
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed", "cleaned_dataset.csv")
MODEL = os.path.join(ROOT, "models", "fake_news_model.pkl")
VEC = os.path.join(ROOT, "models", "tfidf_vectorizer.pkl")
OUT = os.path.join(ROOT, "experiments", "results")
os.makedirs(OUT, exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    X, y = df["clean_content"], df["label_num"]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=15000, ngram_range=(1,2),
                                  min_df=2, max_df=.95, sublinear_tf=True,
                                  strip_accents="unicode")),
        ("clf", LogisticRegression(max_iter=2000, C=1.5, class_weight="balanced",
                                   solver="liblinear", random_state=42)),
    ])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_validate(pipe, X, y, cv=cv,
                            scoring=["accuracy","precision","recall","f1"])
    cv_summary = {m: {"mean": float(scores[f"test_{m}"].mean()),
                      "std": float(scores[f"test_{m}"].std())}
                  for m in ["accuracy","precision","recall","f1"]}

    model = joblib.load(MODEL)
    vectorizer = joblib.load(VEC)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y
    )

    def metric_row(name, texts):
        pred = model.predict(vectorizer.transform(texts))
        return {
            "scenario": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
        }

    scenarios = [
        ("Baseline", X_test),
        ("Uppercase", X_test.str.upper()),
        ("Punctuation noise", X_test.apply(lambda t: re.sub(r"\s+", " !!! ", t) + " ???")),
        ("URL appended", X_test + " visit https://example.com/news?id=123"),
        ("First 30 tokens", X_test.apply(lambda t: " ".join(t.split()[:30]))),
    ]
    robustness = [metric_row(name, texts) for name, texts in scenarios]

    probs = model.predict_proba(vectorizer.transform(X_test))
    confidence = probs.max(axis=1)
    pred = model.classes_[probs.argmax(axis=1)]
    thresholds = []
    for threshold in [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]:
        mask = confidence >= threshold
        thresholds.append({
            "threshold": threshold,
            "coverage": float(mask.mean()),
            "accuracy_on_covered": float(accuracy_score(y_test[mask], pred[mask])),
            "samples_covered": int(mask.sum()),
        })

    with open(os.path.join(OUT, "cross_validation.json"), "w") as f:
        json.dump(cv_summary, f, indent=2)
    pd.DataFrame(robustness).to_csv(os.path.join(OUT, "robustness.csv"), index=False)
    pd.DataFrame(thresholds).to_csv(os.path.join(OUT, "confidence_thresholds.csv"), index=False)
    print(json.dumps(cv_summary, indent=2))
    print(pd.DataFrame(robustness).to_string(index=False))
    print(pd.DataFrame(thresholds).to_string(index=False))

if __name__ == "__main__":
    main()
