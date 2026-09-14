"""
predictor.py
============
Reusable inference helper. The training script saves a complete sklearn
Pipeline-like bundle (model, feature columns, scaler metadata). This module
also validates user input and produces a retention recommendation.
"""
import numpy as np
import pandas as pd


def predict_from_features(model, feature_row: pd.DataFrame, feature_names, threshold=0.5):
    """Align a preprocessed one-row dataframe and return churn probability."""
    row = feature_row.reindex(columns=feature_names, fill_value=0.0)
    probability = float(model.predict_proba(row)[:, 1][0])
    label = "Likely Churn" if probability >= threshold else "Likely Stay"
    risk = "High" if probability >= 0.70 else ("Medium" if probability >= 0.40 else "Low")
    action = ("Prioritize retention call" if risk == "High" else
              "Offer targeted incentive" if risk == "Medium" else "Maintain normal engagement")
    return {"churn_probability": round(probability, 4), "prediction": label,
            "risk_level": risk, "recommended_action": action}


def validate_customer_input(data: dict):
    required = ["tenure", "MonthlyCharges", "TotalCharges", "Contract",
                "InternetService", "PaymentMethod"]
    missing = [c for c in required if c not in data]
    if missing:
        raise ValueError(f"Missing required prediction fields: {missing}")
    if float(data["tenure"]) < 0:
        raise ValueError("tenure cannot be negative")
    if float(data["MonthlyCharges"]) < 0 or float(data["TotalCharges"]) < 0:
        raise ValueError("charges cannot be negative")
    return True


def transform_raw_customer(raw: dict, scaler, feature_names, numeric_feature_names=None):
    """Transform one raw Telco-style customer record exactly like training."""
    from src.preprocessing import engineer_features
    from src.data_loader import CATEGORICAL_COLUMNS
    df = pd.DataFrame([raw])
    engineered = engineer_features(df)
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in engineered.columns] + (
        ["TenureBand"] if "TenureBand" in engineered.columns else []
    )
    X = pd.get_dummies(engineered, columns=cat_cols, drop_first=True, dtype=float)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = X.reindex(columns=feature_names, fill_value=0.0)
    numeric_cols = numeric_feature_names or list(getattr(scaler, "feature_names_in_", []))
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    X[numeric_cols] = scaler.transform(X[numeric_cols])
    return X
