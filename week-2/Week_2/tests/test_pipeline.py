import os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import clean_text

def test_clean_text_removes_url_and_html():
    out = clean_text("Hello <b>WORLD</b> https://example.com/news")
    assert "http" not in out
    assert "b" not in out
    assert "world" in out

def test_dataset_has_expected_schema():
    root = os.path.join(os.path.dirname(__file__), "..")
    df = pd.read_csv(os.path.join(root, "data", "raw", "fake_news_dataset.csv"))
    assert {"title", "text", "label"}.issubset(df.columns)
    assert set(df["label"].dropna().str.upper().unique()) <= {"FAKE", "REAL"}
