"""
preprocessing.py
-----------------
Deterministic text cleaning utilities shared by training and prediction.

The pipeline avoids downloading NLTK corpora at runtime so the same
preprocessing works reliably on local machines and deployment services.
Steps:
    1. Lowercase
    2. Remove URLs and HTML
    3. Keep alphabetic tokens
    4. Remove common English stop words
    5. Apply lightweight stemming
"""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

_STOPWORDS = set(ENGLISH_STOP_WORDS)
_STEMMER = PorterStemmer()
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_HTML_PATTERN = re.compile(r"<.*?>")
_TOKEN_PATTERN = re.compile(r"[a-zA-Z]{3,}")


def clean_text(text: str) -> str:
    """Return normalized, space-separated tokens for model input."""
    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = text.lower()
    text = _URL_PATTERN.sub(" ", text)
    text = _HTML_PATTERN.sub(" ", text)

    tokens = _TOKEN_PATTERN.findall(text)
    tokens = [_STEMMER.stem(tok) for tok in tokens if tok not in _STOPWORDS]
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "BREAKING: Scientists find a MIRACLE cure!!! Visit http://fake-site.com now."
    print("Original:", sample)
    print("Cleaned :", clean_text(sample))
