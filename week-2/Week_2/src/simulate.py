"""
simulate.py
-----------
Controlled simulation/experiment harness for Week 3.

It reuses the fixed held-out test split and applies realistic text perturbations
to test whether the classifier is robust to superficial formatting changes and
reduced context. Results are saved under experiments/results/.
"""
import os, re, sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed", "cleaned_dataset.csv")
MODEL = os.path.join(ROOT, "models", "fake_news_model.pkl")
VEC = os.path.join(ROOT, "models", "tfidf_vectorizer.pkl")
OUT = os.path.join(ROOT, "experiments", "results")
os.makedirs(OUT, exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_content"], df["label_num"], test_size=.2,
        random_state=42, stratify=df["label_num"]
    )
    model, vec = joblib.load(MODEL), joblib.load(VEC)

    cases = {
        "Baseline": X_test,
        "Uppercase formatting": X_test.str.upper(),
        "Punctuation noise": X_test.apply(lambda t: re.sub(r"\s+", " !!! ", t) + " ???"),
        "URL appended": X_test + " visit https://example.com/news?id=123",
        "Short context (30 tokens)": X_test.apply(lambda t: " ".join(t.split()[:30])),
    }
    rows = []
    for scenario, texts in cases.items():
        pred = model.predict(vec.transform(texts))
        rows.append({
            "scenario": scenario,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1_score": f1_score(y_test, pred),
            "samples": len(y_test),
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "simulation_results.csv"), index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
