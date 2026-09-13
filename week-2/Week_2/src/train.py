"""
train.py
--------
End-to-end training pipeline for the Fake News Detection project.

Run this file directly (or run the equivalent notebook in notebooks/)
to reproduce the trained model from scratch:

    python src/train.py

What it does, in order:
    1. Load the dataset from data/raw/fake_news_dataset.csv
    2. Explore it (shape, class balance, missing values, duplicates)
    3. Clean the data (drop nulls/duplicates, combine title+text)
    4. Preprocess the text (see src/preprocessing.py)
    5. Split into train/test sets
    6. Vectorize text with TF-IDF
    7. Train TWO candidate models (Multinomial Naive Bayes, Logistic Regression)
    8. Evaluate both on the held-out test set
    9. Save the better model + the fitted vectorizer to models/
   10. Save the comparison metrics to models/metrics.json for transparency

All metrics printed/saved come from actually running this code on the
real dataset - nothing here is a placeholder or invented number.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import clean_text  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "fake_news_dataset.csv")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_dataset.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    """Stage: Load dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Place fake_news_dataset.csv inside data/raw/ before training."
        )
    df = pd.read_csv(path)
    print(f"[LOAD] Loaded dataset with shape {df.shape}")
    return df


def explore_data(df: pd.DataFrame) -> None:
    """Stage: Explore dataset (prints summary only, no side effects)."""
    print("\n[EXPLORE] Columns:", list(df.columns))
    print("[EXPLORE] Class balance:\n", df["label"].value_counts())
    print("[EXPLORE] Missing values:\n", df.isnull().sum())
    print("[EXPLORE] Duplicate rows:", df.duplicated().sum())


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage: Clean data.

    - Drops rows with missing title/text/label
    - Drops exact duplicate rows
    - Combines title + text into a single `content` column (headline often
      carries strong signal on its own, so we keep both)
    - Normalizes the label column to {0, 1} (FAKE=0, REAL=1)
    """
    df = df.copy()

    # Keep only the columns we need, regardless of an extra index column
    # that some CSV exports include (e.g. "Unnamed: 0").
    keep_cols = [c for c in ["title", "text", "label"] if c in df.columns]
    df = df[keep_cols]

    before = len(df)
    df = df.dropna(subset=["title", "text", "label"])
    df = df.drop_duplicates()
    after = len(df)
    print(f"\n[CLEAN] Removed {before - after} rows (nulls/duplicates). Remaining: {after}")

    df["label"] = df["label"].str.upper().str.strip()
    df = df[df["label"].isin(["FAKE", "REAL"])]
    df["label_num"] = (df["label"] == "REAL").astype(int)  # REAL=1, FAKE=0

    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
    return df[["content", "label", "label_num"]]


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Stage: Preprocess text using the shared cleaning pipeline."""
    print("\n[PREPROCESS] Cleaning text (this can take a minute on the full dataset)...")
    start = time.time()
    df = df.copy()
    df["clean_content"] = df["content"].apply(clean_text)
    df = df[df["clean_content"].str.len() > 0]  # drop rows that became empty
    print(f"[PREPROCESS] Done in {time.time() - start:.1f}s. Shape now: {df.shape}")
    return df


def split_data(df: pd.DataFrame):
    """Stage: Split into train/test sets (80/20, stratified by label)."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_content"],
        df["label_num"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label_num"],
    )
    print(f"\n[SPLIT] Train size: {len(X_train)} | Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def vectorize(X_train, X_test):
    """
    Stage: Feature extraction with TF-IDF.

    TF-IDF (Term Frequency - Inverse Document Frequency) turns each cleaned
    article into a numeric vector, weighting words that are frequent in a
    given article but rare across the whole dataset more heavily. This is a
    strong, lightweight baseline for text classification and works well with
    linear models like Logistic Regression and Naive Bayes.
    """
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"\n[VECTORIZE] TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
    return vectorizer, X_train_vec, X_test_vec


def evaluate_model(name, model, X_test_vec, y_test) -> dict:
    """Stage: Evaluate a trained model and return its metrics as a dict."""
    y_pred = model.predict(X_test_vec)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    print(f"\n[EVALUATE] {name}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-score : {metrics['f1_score']:.4f}")
    print("  Confusion matrix [[TN, FP], [FN, TP]]:")
    print(" ", metrics["confusion_matrix"])
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))
    return metrics


def train_and_compare(X_train_vec, y_train, X_test_vec, y_test):
    """
    Stage: Train model.

    We train and compare TWO reasonable baseline approaches:

    1. Multinomial Naive Bayes
       - Very fast, works well with word-count/TF-IDF style features,
         common first baseline for text classification.
       - Assumes feature independence, which is rarely true for language,
         so it can under-perform on nuanced phrasing.

    2. Logistic Regression
       - Also fast to train on TF-IDF features and typically performs at
         least as well as Naive Bayes on this type of dataset.
       - Produces a model confidence estimate and its coefficients directly tell
         us which words push a prediction toward FAKE or REAL, giving us
         a simple, honest explanation for each prediction.

    The model with the higher F1-score on the test set is selected and
    returned, since F1 balances precision and recall and is a fair
    single metric for a roughly balanced two-class dataset like this one.
    """
    results = {}

    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)
    results["Multinomial Naive Bayes"] = {
        "model": nb_model,
        "metrics": evaluate_model("Multinomial Naive Bayes", nb_model, X_test_vec, y_test),
    }

    lr_model = LogisticRegression(
        max_iter=2000,
        C=1.5,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    lr_model.fit(X_train_vec, y_train)
    results["Logistic Regression"] = {
        "model": lr_model,
        "metrics": evaluate_model("Logistic Regression", lr_model, X_test_vec, y_test),
    }

    best_name = max(results, key=lambda k: results[k]["metrics"]["f1_score"])
    print(f"\n[SELECT] Best model based on F1-score: {best_name}")
    return best_name, results


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

    df_raw = load_data(DATA_PATH)
    explore_data(df_raw)

    df_clean = clean_dataframe(df_raw)
    df_processed = preprocess_dataframe(df_clean)
    df_processed.to_csv(PROCESSED_PATH, index=False)
    print(f"\n[SAVE] Cleaned dataset written to {PROCESSED_PATH}")

    X_train, X_test, y_train, y_test = split_data(df_processed)
    vectorizer, X_train_vec, X_test_vec = vectorize(X_train, X_test)

    best_name, results = train_and_compare(X_train_vec, y_train, X_test_vec, y_test)
    best_model = results[best_name]["model"]

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\n[SAVE] Model saved to {MODEL_PATH}")
    print(f"[SAVE] Vectorizer saved to {VECTORIZER_PATH}")

    all_metrics = {
        "best_model": best_name,
        "comparison": {name: r["metrics"] for name, r in results.items()},
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[SAVE] Metrics saved to {METRICS_PATH}")

    print("\n[DONE] Training pipeline finished successfully.")


if __name__ == "__main__":
    main()
