"""Quick integrity checks for the completed Week 5 project."""
import os, json, joblib, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
required = [
    "main.py","predict.py","requirements.txt","README.md",
    "src/data_loader.py","src/preprocessing.py","src/model_training.py",
    "src/evaluation.py","src/optimization.py","src/analysis.py","src/predictor.py",
    "tests/test_pipeline.py","models/best_model.joblib",
    "outputs/results/baseline_metrics.csv",
    "outputs/results/optimization_comparison.csv",
    "outputs/results/final_test_metrics.csv",
    "outputs/results/run_manifest.json",
]
missing=[p for p in required if not os.path.exists(os.path.join(ROOT,p))]
if missing: raise SystemExit("MISSING: " + ", ".join(missing))
artifact=joblib.load(os.path.join(ROOT,"models/best_model.joblib"))
assert all(k in artifact for k in ["model","feature_names","scaler","threshold"])
metrics=pd.read_csv(os.path.join(ROOT,"outputs/results/final_test_metrics.csv"))
assert len(metrics)==1 and 0 <= float(metrics.iloc[0]["f1_score"]) <= 1
with open(os.path.join(ROOT,"outputs/results/run_manifest.json"),encoding="utf-8") as f: manifest=json.load(f)
assert manifest["test_set_used_for_threshold_selection"] is False
print("INTEGRITY CHECK: PASS")
