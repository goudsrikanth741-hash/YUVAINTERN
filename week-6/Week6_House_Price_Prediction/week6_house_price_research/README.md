
# HouseValue AI — Week 6: Model Evaluation, Optimization & Deployment

## 1. Project overview

HouseValue AI is a complete Python machine-learning project for **house-price regression**. It is deliberately structured like a small real product rather than a single notebook: it has a reproducible training pipeline, data provenance, feature engineering, multiple model families, evaluation diagnostics, hyperparameter optimization, model persistence, automated tests, batch prediction, and an interactive Streamlit web interface.

**Important data note:** the standard California Housing dataset is attempted first. If it cannot be downloaded, the project uses a clearly labelled seeded synthetic fallback with the same schema. Do not present synthetic-run metrics as real-world benchmark results.

## 2. Week 6 objective

Evaluate model performance, compare baseline and engineered-feature models, identify the best model using measurable metrics, run cross-validation, optimize hyperparameters, document limitations, and package the result as a usable application.

## 3. 25+ implemented product features

1. Interactive dashboard
2. KPI cards for R², RMSE, MAE and test size
3. Model leaderboard
4. Data-source/provenance transparency
5. Single-record price prediction
6. Model selector for predictions
7. Income sensitivity scenarios
8. CSV batch prediction
9. CSV template download
10. Batch input validation
11. Prediction CSV export
12. Actual-vs-predicted diagnostics
13. Residual diagnostics
14. MAE/MSE/RMSE/R² evaluation
15. Five-fold cross-validation
16. Randomized hyperparameter optimization
17. Tuned-model comparison
18. Native feature importance
19. Derived-feature dictionary
20. Geographic exploratory map
21. Dataset statistics explorer
22. Feature distribution explorer
23. Correlation matrix explorer
24. Automated QA/health checks
25. Reproducibility metadata
26. Persistent Joblib model bundle
27. CLI training pipeline
28. Automated unit/smoke tests
29. Exportable model metadata
30. Explicit limitations and responsible-use notes

## 4. ML workflow

**Data loading → cleaning → 80/20 split → scaling → domain feature engineering → model training → test evaluation → cross-validation → hyperparameter optimization → artifact persistence → web deployment.**

### Source features
MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude.

### Engineered features
RoomsPerHousehold, BedroomRatio, PopulationPerHousehold, DistanceToCoast, IncomePerRoom, RoomsMinusBedrooms, IncomePerOccupant, RoomsSquared, IncomeSquared, AgeSquared, LatitudeLongitudeInteraction, IncomeRoomsInteraction.

That gives **20+ model inputs** while keeping the original dataset schema intact.

## 5. Models

- Linear Regression — interpretable baseline
- Random Forest Regressor — nonlinear ensemble
- Gradient Boosting Regressor — main optimization candidate
- Tuned Gradient Boosting — optional RandomizedSearchCV experiment

## 6. Evaluation strategy

Primary metrics:
- **MAE:** average absolute error
- **RMSE:** penalizes larger errors
- **R²:** variance explained

Validation:
- fixed random seed = 42
- 80/20 holdout test split
- shuffled 5-fold cross-validation
- controlled hyperparameter search

A model should not be called “best” from one metric alone. Check RMSE, MAE, R² and CV stability together.

## 7. Hyperparameter optimization

The Experiment Center tunes Gradient Boosting over estimator count, learning rate, tree depth, minimum split/leaf sizes and subsampling. The search minimizes cross-validated RMSE. The optimized result must be compared against the baseline rather than automatically accepted.

## 8. Run locally

```powershell
cd week6_house_price_research
python -m pip install -r requirements.txt
python main.py
python -m unittest discover -s tests -v
streamlit run app.py
```

The browser UI normally opens at the local Streamlit address shown in the terminal.

## 9. Output structure

```text
artifacts/model_bundle.joblib
outputs/
  figures/
  results/
src/
  data_loader.py
  preprocessing.py
  feature_engineering.py
  models.py
  evaluation.py
  pipeline.py
tests/
app.py
main.py
requirements.txt
README.md
```

## 10. Requirements checklist

| Requirement | Status |
|---|---|
| Clear regression problem | PASS |
| Public dataset / transparent fallback | PASS |
| Data cleaning | PASS |
| Leakage-safe train/test split | PASS |
| Feature engineering | PASS |
| Multiple ML models | PASS |
| Quantitative evaluation | PASS |
| Error/residual analysis | PASS |
| Cross-validation | PASS |
| Optimization strategy | PASS |
| Controlled experiment settings | PASS |
| Reproducibility | PASS |
| Model persistence | PASS |
| Interactive web application | PASS |
| Batch inference | PASS |
| Automated tests | PASS |
| Limitations / responsible use | PASS |
| Future work | PASS |

## 11. Limitations

This is a research/educational prediction system, not a certified property appraisal tool. The historical California Housing target is a block-group median and can be capped; it does not describe every individual home. A synthetic fallback is useful for offline execution but cannot validate real-market accuracy. Spatial validation, newer parcel-level data, external economic variables, uncertainty calibration and fairness analysis are logical next steps.

## 12. Responsible interpretation

Predictions should be treated as estimates, not guaranteed market prices. Avoid using a model trained on historical aggregate data as the sole basis for lending, purchasing, selling, or other high-impact financial decisions.
