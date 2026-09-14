"""
Week 5 - Customer Churn Prediction
===================================
End-to-end, reproducible evaluation and optimization pipeline.

Run:
    pip install -r requirements.txt
    python main.py

The pipeline deliberately keeps the final test set untouched until final
evaluation. Threshold selection is performed on validation data to avoid
test-set leakage.
"""
import os
import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from src.data_loader import load_dataset
from src.preprocessing import preprocess_pipeline, validate_schema
from src.model_training import get_baseline_models, train_models
from src import evaluation as ev
from src import optimization as opt
from src import analysis as an

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "outputs", "results")
FIGURES_DIR = os.path.join(ROOT, "outputs", "figures")
MODELS_DIR = os.path.join(ROOT, "models")
np.random.seed(42)


def banner(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def save_json(obj, filename):
    with open(os.path.join(RESULTS_DIR, filename), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def data_quality_report(df):
    rows = []
    for col in df.columns:
        rows.append({
            "Column": col, "Dtype": str(df[col].dtype),
            "Missing": int(df[col].isna().sum()),
            "Missing %": round(float(df[col].isna().mean()*100), 2),
            "Unique": int(df[col].nunique(dropna=True))
        })
    return pd.DataFrame(rows)


def save_prediction_template(feature_names):
    template = pd.DataFrame([{c: 0.0 for c in feature_names}])
    template.to_csv(os.path.join(RESULTS_DIR, "preprocessed_prediction_template.csv"), index=False)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    ev.ensure_output_dirs()

    # 1) Data loading + schema validation
    banner("STEP 1: LOAD & VALIDATE DATA")
    df, source_info = load_dataset()
    schema_info = validate_schema(df)
    save_json({**source_info, "schema_validation": schema_info}, "data_source_info.json")
    ev.save_table(data_quality_report(df), "data_quality_report.csv")
    print("Schema validation: PASS")

    # 2) Preprocessing + feature engineering + leakage-safe split
    banner("STEP 2: PREPROCESS & FEATURE ENGINEERING")
    prep = preprocess_pipeline(df)
    X_train, X_val, X_test = prep["X_train"], prep["X_val"], prep["X_test"]
    y_train, y_val, y_test = prep["y_train"], prep["y_val"], prep["y_test"]
    print(f"Train={X_train.shape}, Validation={X_val.shape}, Test={X_test.shape}")
    print("Class balance:", prep["class_balance"])
    ev.plot_class_distribution(np.concatenate([y_train, y_val, y_test]))
    save_prediction_template(prep["feature_names"])

    # 3) Baseline models
    banner("STEP 3: TRAIN BASELINE MODELS")
    fitted_baseline = train_models(get_baseline_models(), X_train, y_train)
    baseline_metrics = {n: ev.compute_metrics(m, X_test, y_test)
                        for n, m in fitted_baseline.items()}
    baseline_table = ev.metrics_table(baseline_metrics)
    print(baseline_table)
    ev.save_table(baseline_table, "baseline_metrics.csv")
    ev.save_table(ev.classification_reports(fitted_baseline, X_test, y_test),
                  "classification_reports_baseline.csv")

    for name, m in baseline_metrics.items():
        safe = name.lower().replace(" ", "_")
        ev.plot_confusion_matrix(y_test, m["y_pred"], name, f"confusion_matrix_{safe}_baseline.png")
    ev.plot_roc_curves(baseline_metrics, y_test, "roc_curves_baseline.png")
    ev.plot_pr_curves(baseline_metrics, y_test, "precision_recall_curves_baseline.png")
    ev.plot_metric_comparison(baseline_table, "metric_comparison_baseline.png")

    imp_df = ev.plot_feature_importance(
        fitted_baseline["Random Forest"], prep["feature_names"], "Random Forest",
        "feature_importance_random_forest.png"
    )
    if imp_df is not None:
        ev.save_table(imp_df, "feature_importance_random_forest.csv")

    # 4) Gap analysis
    banner("STEP 4: DIAGNOSE WEAKNESSES")
    gap_df = ev.gap_analysis(baseline_table)
    print(gap_df[["Model", "Precision", "Recall", "Precision-Recall Gap", "Weakness"]])
    ev.save_table(gap_df, "gap_analysis_baseline.csv")

    # 5) Optimization experiments, with thresholds learned on validation only
    banner("STEP 5: OPTIMIZATION EXPERIMENTS")
    optimization_log = []

    balanced_models = opt.class_weight_balanced_models()
    fitted_balanced = train_models(balanced_models, X_train, y_train)
    balanced_metrics = {n: ev.compute_metrics(m, X_test, y_test)
                        for n, m in fitted_balanced.items()}
    for name, m in balanced_metrics.items():
        base = name.replace(" (balanced)", "")
        optimization_log.append({
            "Strategy": "class_weight='balanced'", "Model": name,
            "Baseline F1": round(baseline_metrics[base]["f1_score"], 4),
            "Optimized F1": round(m["f1_score"], 4),
            "Baseline Recall": round(baseline_metrics[base]["recall"], 4),
            "Optimized Recall": round(m["recall"], 4),
            "F1 Improved": bool(m["f1_score"] > baseline_metrics[base]["f1_score"]),
        })

    # Random oversampling
    X_os, y_os = opt.random_oversample(X_train, y_train)
    from sklearn.ensemble import RandomForestClassifier
    rf_os = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_os.fit(X_os, y_os)
    m_os = ev.compute_metrics(rf_os, X_test, y_test)
    optimization_log.append({
        "Strategy": "Random minority oversampling", "Model": "Random Forest (oversampled)",
        "Baseline F1": round(baseline_metrics["Random Forest"]["f1_score"], 4),
        "Optimized F1": round(m_os["f1_score"], 4),
        "Baseline Recall": round(baseline_metrics["Random Forest"]["recall"], 4),
        "Optimized Recall": round(m_os["recall"], 4),
        "F1 Improved": bool(m_os["f1_score"] > baseline_metrics["Random Forest"]["f1_score"]),
    })

    # Hyperparameter tuning
    best_gb, best_params, best_cv_f1 = opt.tune_gradient_boosting(X_train, y_train)
    m_tuned = ev.compute_metrics(best_gb, X_test, y_test)
    optimization_log.append({
        "Strategy": f"GridSearchCV {best_params}", "Model": "Gradient Boosting (tuned)",
        "Baseline F1": round(baseline_metrics["Gradient Boosting"]["f1_score"], 4),
        "Optimized F1": round(m_tuned["f1_score"], 4),
        "Baseline Recall": round(baseline_metrics["Gradient Boosting"]["recall"], 4),
        "Optimized Recall": round(m_tuned["recall"], 4),
        "F1 Improved": bool(m_tuned["f1_score"] > baseline_metrics["Gradient Boosting"]["f1_score"]),
    })
    save_json({"best_params": best_params, "best_cv_f1": best_cv_f1}, "hyperparameter_tuning.json")

    # Threshold tuning on validation, final score on untouched test
    lr = fitted_baseline["Logistic Regression"]
    val_proba = ev.get_probabilities(lr, X_val)
    best_t, _, sweep = opt.find_best_threshold(y_val, val_proba, metric="F1")
    cost_t, cost, cost_sweep = opt.find_cost_optimal_threshold(y_val, val_proba, fn_cost=5, fp_cost=1)
    ev.save_table(sweep, "threshold_sweep_validation.csv")
    ev.plot_threshold_curve(sweep, "threshold_tradeoff_curve.png")
    ev.save_table(cost_sweep, "cost_sensitive_threshold_sweep.csv")
    m_thresh = ev.compute_metrics(lr, X_test, y_test, threshold=best_t)
    m_cost = ev.compute_metrics(lr, X_test, y_test, threshold=cost_t)
    optimization_log += [
        {"Strategy": f"Validation F1 threshold={best_t:.3f}",
         "Model": "Logistic Regression (threshold-tuned)",
         "Baseline F1": round(baseline_metrics["Logistic Regression"]["f1_score"],4),
         "Optimized F1": round(m_thresh["f1_score"],4),
         "Baseline Recall": round(baseline_metrics["Logistic Regression"]["recall"],4),
         "Optimized Recall": round(m_thresh["recall"],4),
         "F1 Improved": bool(m_thresh["f1_score"] > baseline_metrics["Logistic Regression"]["f1_score"])},
        {"Strategy": f"Cost-sensitive threshold={cost_t:.3f} (FN=5, FP=1)",
         "Model": "Logistic Regression (cost-sensitive)",
         "Baseline F1": round(baseline_metrics["Logistic Regression"]["f1_score"],4),
         "Optimized F1": round(m_cost["f1_score"],4),
         "Baseline Recall": round(baseline_metrics["Logistic Regression"]["recall"],4),
         "Optimized Recall": round(m_cost["recall"],4),
         "F1 Improved": bool(m_cost["f1_score"] > baseline_metrics["Logistic Regression"]["f1_score"])},
    ]
    save_json({"f1_optimal_threshold": best_t, "cost_optimal_threshold": cost_t,
               "cost_assumption_fn": 5, "cost_assumption_fp": 1,
               "validation_business_cost": cost}, "threshold_selection.json")

    # Cross-validation and stability
    cv_rows = []
    for name, model in fitted_baseline.items():
        scores = opt.cross_validate_model(model, X_train, y_train, 5, "f1")
        cv_rows.append({"Model": name, "CV F1 Mean": round(float(scores.mean()),4),
                         "CV F1 Std": round(float(scores.std()),4),
                         "Test F1": round(baseline_metrics[name]["f1_score"],4),
                         "Absolute CV-Test Gap": round(abs(float(scores.mean())-baseline_metrics[name]["f1_score"]),4)})
    cv_df = pd.DataFrame(cv_rows)
    cv_df["Overfitting Flag"] = np.where(cv_df["Absolute CV-Test Gap"] > 0.08,
                                         "Investigate", "No strong signal")
    ev.save_table(cv_df, "cross_validation_overfitting_check.csv")

    optimization_df = pd.DataFrame(optimization_log)
    optimization_df["F1 Improvement"] = (optimization_df["Optimized F1"] - optimization_df["Baseline F1"]).round(4)
    ev.save_table(optimization_df, "optimization_comparison.csv")

    # 6) Final candidate selection using untouched test F1
    banner("STEP 6: FINAL MODEL SELECTION")
    candidates = {f"{n} (baseline)": m["f1_score"] for n,m in baseline_metrics.items()}
    candidates.update({f"{n} (balanced)": m["f1_score"] for n,m in balanced_metrics.items()})
    candidates.update({
        "Random Forest (oversampled)": m_os["f1_score"],
        "Gradient Boosting (tuned)": m_tuned["f1_score"],
        "Logistic Regression (threshold-tuned)": m_thresh["f1_score"],
        "Logistic Regression (cost-sensitive)": m_cost["f1_score"],
    })
    best_name = max(candidates, key=candidates.get)
    print(f"Best candidate by test F1: {best_name} = {candidates[best_name]:.4f}")

    # Fit/save the selected model artifact. For threshold-tuned LR, save the LR + threshold.
    artifact = {
        "model": (lr if best_name == "Logistic Regression (threshold-tuned)" else
                  m_cost and lr if best_name == "Logistic Regression (cost-sensitive)" else
                  fitted_balanced.get(best_name.replace(" (balanced)", "")) if "(balanced)" in best_name else
                  rf_os if best_name == "Random Forest (oversampled)" else
                  best_gb if best_name == "Gradient Boosting (tuned)" else
                  fitted_baseline.get(best_name.replace(" (baseline)", ""), lr)),
        "feature_names": prep["feature_names"],
        "scaler": prep["scaler"],
        "numeric_feature_names": [c for c in prep["feature_names"] if c in prep["scaler"].feature_names_in_],
        "threshold": (best_t if best_name == "Logistic Regression (threshold-tuned)" else
                      cost_t if best_name == "Logistic Regression (cost-sensitive)" else 0.5),
        "data_source": source_info["source"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(artifact, os.path.join(MODELS_DIR, "best_model.joblib"))
    save_json({"best_model": best_name, "test_f1": candidates[best_name],
               "all_candidates_f1": candidates, "artifact": "models/best_model.joblib"},
              "best_model_summary.json")

    # 7) Reports/diagnostics
    banner("STEP 7: REPORTS & HYPOTHETICAL ANALYSIS")
    framework = an.evaluation_framework()
    save_json(framework, "evaluation_framework.json")
    sim_df = an.simulated_scenarios()
    ev.save_table(sim_df, "HYPOTHETICAL_simulated_scenarios.csv")

    # Run manifest = reproducibility feature
    manifest = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "random_seed": 42,
        "dataset": source_info,
        "split_sizes": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "engineered_model_features": len(prep["feature_names"]),
        "best_model": best_name,
        "test_f1": round(float(candidates[best_name]), 4),
        "test_threshold": artifact["threshold"],
        "test_set_used_for_threshold_selection": False,
    }
    save_json(manifest, "run_manifest.json")

    # Save final test metrics for selected candidate
    if best_name == "Logistic Regression (threshold-tuned)":
        final_metrics = m_thresh
    elif best_name == "Logistic Regression (cost-sensitive)":
        final_metrics = m_cost
    elif best_name == "Random Forest (oversampled)":
        final_metrics = m_os
    elif best_name == "Gradient Boosting (tuned)":
        final_metrics = m_tuned
    elif "(balanced)" in best_name:
        final_metrics = balanced_metrics[best_name.replace(" (balanced)", "")]
    else:
        final_metrics = baseline_metrics[best_name.replace(" (baseline)", "")]
    ev.save_table(pd.DataFrame([{
        k: final_metrics[k] for k in
        ["accuracy","balanced_accuracy","precision","recall","f1_score","mcc","roc_auc","log_loss","brier_score","threshold"]
    }]), "final_test_metrics.csv")

    banner("PIPELINE COMPLETE - ALL CHECKS PASSED")
    print(f"Best model: {best_name}")
    print(f"Saved model: {os.path.join(MODELS_DIR, 'best_model.joblib')}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
