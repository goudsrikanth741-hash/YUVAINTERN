# Week 5 – Customer Churn Prediction: Evaluation & Optimization

## Objective
Build an end-to-end customer churn prediction experiment that **evaluates multiple baseline models, diagnoses weaknesses, applies targeted optimization strategies, and verifies the final model on an untouched test set**.

> **Data note:** The bundled run uses a seeded synthetic Telco-style dataset because the real IBM Telco CSV is not bundled. It is clearly labelled as synthetic. For real-data results, place the IBM/Kaggle CSV at `data/Telco-Customer-Churn.csv` and run `python main.py`.

## What is included
The project has been upgraded with **11 practical features/improvements**:

1. **Automatic dataset/schema validation** before modelling.
2. **Data-quality report** with dtype, missing-value and cardinality checks.
3. **Business-focused feature engineering**: service count, internet flag, contract flags, payment flags, charge-per-tenure, historical average charge and tenure bands.
4. **Leakage-safe train/validation/test split** (60/20/20) with stratification.
5. **Expanded evaluation metrics**: accuracy, balanced accuracy, precision, recall, F1, MCC, ROC-AUC, log loss and Brier score.
6. **Confusion-matrix, ROC, PR and feature-importance visualizations**.
7. **Class-imbalance optimization** using balanced class weights and random oversampling.
8. **Hyperparameter tuning** with 5-fold GridSearchCV.
9. **Validation-only threshold optimization**, preventing test-set leakage.
10. **Cost-sensitive threshold selection** (configurable FN/FP business-cost assumptions).
11. **Reusable saved model + raw-customer prediction CLI**, plus a reproducibility run manifest.

## Models
Baseline:
- Logistic Regression
- Random Forest
- Gradient Boosting

Optimization experiments:
- `class_weight='balanced'`
- Random minority oversampling
- Gradient Boosting GridSearchCV
- Validation F1 threshold tuning
- Cost-sensitive threshold tuning
- 5-fold cross-validation for stability/overfitting checks

## Current bundled experiment
The generated run completed successfully using **4,000 synthetic customers** with a 26.1% churn rate.

The final bundled experiment selected:

**Logistic Regression (cost-sensitive threshold)**  
**Test F1-score: 0.5729**

Important: the threshold was selected using the validation set, and the final test set was used only for final scoring.

See:
- `outputs/results/final_test_metrics.csv`
- `outputs/results/optimization_comparison.csv`
- `outputs/results/cross_validation_overfitting_check.csv`
- `outputs/results/best_model_summary.json`
- `outputs/results/run_manifest.json`

## Run the project

### Windows
```powershell
cd week5_customer_churn
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Linux/macOS
```bash
cd week5_customer_churn
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Run tests
```bash
python -m pytest -q
```

### Make a sample prediction
After `python main.py` has created `models/best_model.joblib`:

```bash
python predict.py data/sample_customer.json
```

The prediction returns:
- churn probability
- churn/stay prediction
- Low/Medium/High risk
- recommended retention action

## Real IBM Telco dataset
Put the real CSV here:

```text
data/Telco-Customer-Churn.csv
```

Then run:

```bash
python main.py
```

The loader automatically detects the real file. The output metadata records whether the run used real or synthetic data.

## Output structure
```text
week5_customer_churn/
├── data/
│   ├── README.md
│   └── sample_customer.json
├── models/
│   └── best_model.joblib
├── src/
│   ├── analysis.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── model_training.py
│   ├── optimization.py
│   ├── predictor.py
│   └── preprocessing.py
├── tests/
│   └── test_pipeline.py
├── outputs/
│   ├── figures/
│   └── results/
├── main.py
├── predict.py
└── requirements.txt
```

## Academic evaluation framework
The project evaluates:
- **Recall:** ability to catch customers who actually churn.
- **Precision:** how many flagged customers really churn.
- **F1:** balance of precision and recall.
- **ROC-AUC:** ranking/separation quality.
- **MCC:** robust correlation-style classification measure.
- **Balanced accuracy:** useful when classes are imbalanced.
- **Log loss/Brier score:** probability-quality diagnostics.
- **Confusion matrix:** concrete FP/FN business trade-off.

The project also contains a clearly labelled `HYPOTHETICAL_simulated_scenarios.csv`. These values are illustrative only and are never mixed with actual experimental results.

## Reproducibility
All modelling uses seed `42`. `outputs/results/run_manifest.json` records the Python/platform version, dataset source, split sizes, feature count, selected model and threshold-selection policy.
