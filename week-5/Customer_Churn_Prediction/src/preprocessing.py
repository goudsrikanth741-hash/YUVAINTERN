"""
preprocessing.py
================
Data cleaning, feature engineering, encoding, and leakage-safe train/
validation/test splitting for customer churn prediction.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN

RANDOM_SEED = 42


def validate_schema(df: pd.DataFrame):
    """Validate the minimum required Telco schema before modelling."""
    required = set(CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + [TARGET_COLUMN])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    if df[TARGET_COLUMN].dropna().nunique() < 2:
        raise ValueError("Target column must contain both churn classes.")
    return {"rows": int(len(df)), "columns": int(df.shape[1]), "missing_required": missing}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean types, duplicates and missing target values."""
    validate_schema(df)
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    for c in ["tenure", "MonthlyCharges", "SeniorCitizen"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    df = df.drop_duplicates().dropna(subset=[TARGET_COLUMN])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add business-relevant features without using the target."""
    df = df.copy()
    service_cols = [
        "PhoneService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    yes = pd.DataFrame({c: (df[c].astype(str) == "Yes").astype(int) for c in service_cols})
    df["ServiceCount"] = yes.sum(axis=1)
    df["HasInternet"] = (df["InternetService"].astype(str) != "No").astype(int)
    df["IsMonthToMonth"] = (df["Contract"].astype(str) == "Month-to-month").astype(int)
    df["IsLongContract"] = df["Contract"].astype(str).isin(["One year", "Two year"]).astype(int)
    df["IsElectronicCheck"] = (df["PaymentMethod"].astype(str) == "Electronic check").astype(int)
    df["IsAutoPay"] = df["PaymentMethod"].astype(str).isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    ).astype(int)
    df["MonthlyChargePerTenure"] = df["MonthlyCharges"] / df["tenure"].replace(0, np.nan)
    df["MonthlyChargePerTenure"] = df["MonthlyChargePerTenure"].replace([np.inf, -np.inf], np.nan).fillna(df["MonthlyCharges"])
    df["AverageHistoricalCharge"] = df["TotalCharges"] / df["tenure"].replace(0, np.nan)
    df["AverageHistoricalCharge"] = df["AverageHistoricalCharge"].replace([np.inf, -np.inf], np.nan).fillna(df["MonthlyCharges"])
    df["TenureBand"] = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, 72], labels=["0-6", "7-12", "13-24", "25-48", "49-72"]
    ).astype(str)
    return df


def encode_features(df: pd.DataFrame):
    """One-hot encode categoricals and map target Yes/No to 1/0."""
    df = engineer_features(df)
    y = (df[TARGET_COLUMN].astype(str).str.strip().str.lower() == "yes").astype(int).to_numpy()
    feature_df = df.drop(columns=[TARGET_COLUMN])

    original_cat = [c for c in CATEGORICAL_COLUMNS if c in feature_df.columns]
    engineered_cat = ["TenureBand"]
    cat_cols = original_cat + [c for c in engineered_cat if c in feature_df.columns]
    X = pd.get_dummies(feature_df, columns=cat_cols, drop_first=True, dtype=float)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    numeric = [c for c in X.columns if c in NUMERIC_COLUMNS or c in [
        "ServiceCount", "HasInternet", "IsMonthToMonth", "IsLongContract",
        "IsElectronicCheck", "IsAutoPay", "MonthlyChargePerTenure", "AverageHistoricalCharge"
    ]]
    return X, y, numeric


def split_and_scale(X, y, numeric_cols, test_size=0.2, validation_size=0.2, seed=RANDOM_SEED):
    """Create leakage-safe stratified train/validation/test sets and scale numeric features."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    val_relative = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, random_state=seed, stratify=y_temp
    )
    scaler = StandardScaler()
    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
    existing_numeric = [c for c in numeric_cols if c in X_train.columns]
    scaler.fit(X_train[existing_numeric])
    X_train[existing_numeric] = scaler.transform(X_train[existing_numeric])
    X_val[existing_numeric] = scaler.transform(X_val[existing_numeric])
    X_test[existing_numeric] = scaler.transform(X_test[existing_numeric])
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def preprocess_pipeline(df: pd.DataFrame):
    """Full cleaning -> feature engineering -> encoding -> leakage-safe split."""
    cleaned = clean_data(df)
    X, y, num_cols = encode_features(cleaned)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(X, y, num_cols)
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "scaler": scaler, "feature_names": list(X.columns),
        "cleaned_rows": len(cleaned),
        "engineered_feature_count": len(X.columns),
        "class_balance": {
            "train_positive_rate": float(np.mean(y_train)),
            "validation_positive_rate": float(np.mean(y_val)),
            "test_positive_rate": float(np.mean(y_test)),
            "overall_positive_rate": float(np.mean(y)),
        },
    }
