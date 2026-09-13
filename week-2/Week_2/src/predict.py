"""Prediction utilities for the fake-news classifier."""
import os
import re
import sys

import joblib
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import clean_text  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "models", "tfidf_vectorizer.pkl")

MIN_TOKENS = 12
UNCERTAIN_THRESHOLD = 0.65  # confidence-estimate threshold; not probability calibration

_model = None
_vectorizer = None


class ModelNotFoundError(Exception):
    pass


class EmptyInputError(Exception):
    pass


class InsufficientTextError(Exception):
    pass


def _load_artifacts():
    global _model, _vectorizer
    if _model is not None and _vectorizer is not None:
        return _model, _vectorizer
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise ModelNotFoundError("Trained model files are missing from the models/ folder.")
    _model = joblib.load(MODEL_PATH)
    _vectorizer = joblib.load(VECTORIZER_PATH)
    return _model, _vectorizer


def _publisher_signal(text: str):
    """Detect strong wire-service attribution patterns.

    This is intentionally conservative. It does not claim that the article is
    factually true; it only recognizes explicit publication/attribution markers
    that are useful when a model trained on an older dataset sees a modern wire
    story with unfamiliar topic vocabulary.
    """
    t = re.sub(r"\s+", " ", text.lower()).strip()
    signals = []
    if "(reuters)" in t or " reuters " in f" {t} ":
        signals.append("Reuters attribution")
    if re.search(r"reporting by [a-z][^;]{2,100}", t):
        signals.append("Reporting byline")
    if re.search(r"additional reporting by [a-z][^;]{2,100}", t):
        signals.append("Additional reporting byline")
    if re.search(r"writing by [a-z][^;]{2,100}", t):
        signals.append("Writing byline")
    if re.search(r"editing by [a-z][^;]{2,100}", t):
        signals.append("Editing byline")

    # Strong Reuters-style wire attribution: several explicit newsroom
    # credits together, normally found at the end of a Reuters article.
    # Require the characteristic Reuters-style credit sequence, rather than
    # merely seeing four unrelated words. This keeps the rule conservative.
    strong = bool(re.search(
        r"reporting by .{2,140};\s*additional reporting by .{2,140};\s*writing by .{2,140};\s*editing by .{2,180}",
        t,
        flags=re.IGNORECASE,
    ))
    return strong, signals


def _top_contributing_words(cleaned_text, vectorizer, model, predicted_class, top_n=8):
    feature_names = np.array(vectorizer.get_feature_names_out())
    vector = vectorizer.transform([cleaned_text])
    nonzero_idx = vector.nonzero()[1]
    if len(nonzero_idx) == 0 or not hasattr(model, "coef_"):
        return []
    coefs = model.coef_[0]
    tfidf_values = vector.toarray()[0]
    sign = 1 if predicted_class == 1 else -1
    contributions = []
    for idx in nonzero_idx:
        contribution = coefs[idx] * tfidf_values[idx] * sign
        if contribution > 0:
            contributions.append((feature_names[idx], float(contribution)))
    contributions.sort(key=lambda x: x[1], reverse=True)
    return contributions[:top_n]


def predict_news(headline: str = "", article_text: str = "") -> dict:
    if headline is None:
        headline = ""
    if article_text is None:
        article_text = ""
    headline = str(headline).strip()
    article_text = str(article_text).strip()
    combined = (headline + " " + article_text).strip()
    if not combined:
        raise EmptyInputError("Please enter a headline or article before checking.")

    model, vectorizer = _load_artifacts()
    cleaned = clean_text(combined)
    if len(cleaned.split()) < MIN_TOKENS:
        raise InsufficientTextError(
            "Please provide a fuller headline/article. Very short text does not contain enough evidence."
        )

    vector = vectorizer.transform([cleaned])
    predicted_class = int(model.predict(vector)[0])
    probabilities = model.predict_proba(vector)[0]
    ml_confidence = float(probabilities[predicted_class])
    raw_label = "REAL" if predicted_class == 1 else "FAKE"

    strong_source, source_signals = _publisher_signal(combined)

    # Conservative source-style correction. This is not a fact check: it is a
    # separate publication-attribution signal. It is only allowed when a
    # strong multi-part newsroom credit is present.
    if strong_source:
        final_label = "REAL"
        final_reason = "Strong newsroom attribution pattern detected."
        final_confidence = max(0.90, ml_confidence)
    else:
        final_label = raw_label if ml_confidence >= UNCERTAIN_THRESHOLD else "UNCERTAIN"
        final_reason = "ML classifier only; no strong publication attribution detected."
        final_confidence = ml_confidence

    return {
        "label": final_label,
        "raw_label": raw_label,
        "confidence": final_confidence,
        "ml_confidence": ml_confidence,
        "top_words": _top_contributing_words(cleaned, vectorizer, model, predicted_class),
        "cleaned_text": cleaned,
        "source_signals": source_signals,
        "strong_source_signal": strong_source,
        "reason": final_reason,
    }
