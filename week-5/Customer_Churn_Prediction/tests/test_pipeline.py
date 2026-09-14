import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import _generate_synthetic_telco
from src.preprocessing import clean_data, encode_features, preprocess_pipeline
from src.optimization import threshold_sweep

def test_synthetic_schema_and_target():
    df = _generate_synthetic_telco(200, 42)
    assert df.shape == (200, 21)
    assert set(df["Churn"]) == {"Yes", "No"}

def test_feature_engineering_and_split():
    prep = preprocess_pipeline(_generate_synthetic_telco(500, 42))
    assert len(prep["feature_names"]) > 21
    assert len(prep["y_train"]) + len(prep["y_val"]) + len(prep["y_test"]) == 500
    assert prep["X_train"].shape[1] == prep["X_test"].shape[1]

def test_threshold_sweep():
    y = np.array([0,0,1,1,1,0])
    p = np.array([.1,.2,.4,.7,.8,.9])
    sweep = threshold_sweep(y,p)
    assert {"threshold","Precision","Recall","F1"}.issubset(sweep.columns)
    assert len(sweep) > 10
