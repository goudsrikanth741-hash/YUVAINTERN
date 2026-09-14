"""
predict.py - predict churn for one raw customer record.

Example:
    python predict.py sample_customer.json
"""
import json
import os
import sys
import joblib
from src.predictor import validate_customer_input, transform_raw_customer, predict_from_features

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "sample_customer.json")
    artifact = joblib.load(os.path.join(ROOT, "models", "best_model.joblib"))
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    validate_customer_input(data)
    X = transform_raw_customer(data, artifact["scaler"], artifact["feature_names"], artifact.get("numeric_feature_names")) if "scaler" in artifact else None
    if X is None:
        raise RuntimeError("Model artifact does not contain preprocessing scaler.")
    result = predict_from_features(artifact["model"], X, artifact["feature_names"], artifact["threshold"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
