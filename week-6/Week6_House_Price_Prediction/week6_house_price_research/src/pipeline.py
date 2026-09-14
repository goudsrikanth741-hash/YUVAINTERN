
"""Reusable training service used by CLI, tests and Streamlit."""
import os, json, joblib
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from src.data_loader import load_dataset, FEATURE_NAMES, TARGET_NAME
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import engineer_features
from src.models import get_models, train_models, tune_gradient_boosting
from src.evaluation import compute_metrics, cross_validate_models

ARTIFACT_DIR=os.path.join(os.path.dirname(__file__),"..","artifacts")
RESULT_DIR=os.path.join(os.path.dirname(__file__),"..","outputs","results")
os.makedirs(ARTIFACT_DIR,exist_ok=True); os.makedirs(RESULT_DIR,exist_ok=True)

def build_training_bundle(tune=False, tuning_iter=8):
    df,source=load_dataset(verbose=False)
    prep=preprocess_pipeline(df)
    Xtr=engineer_features(prep["X_train_raw"]); Xte=engineer_features(prep["X_test_raw"])
    scaler=StandardScaler()
    Xtr_s=pd.DataFrame(scaler.fit_transform(Xtr),columns=Xtr.columns,index=Xtr.index)
    Xte_s=pd.DataFrame(scaler.transform(Xte),columns=Xte.columns,index=Xte.index)
    models=get_models(); fitted=train_models(models,Xtr_s,prep["y_train"])
    if tune:
        tuned,search=tune_gradient_boosting(Xtr_s,prep["y_train"],n_iter=tuning_iter)
        fitted["Tuned Gradient Boosting"]=tuned
    metrics={n:compute_metrics(m,Xte_s,prep["y_test"]) for n,m in fitted.items()}
    table=pd.DataFrame([{"Model":n,"MAE":m["mae"],"RMSE":m["rmse"],"R2":m["r2"]} for n,m in metrics.items()]).sort_values("R2",ascending=False)
    best=table.iloc[0]["Model"]
    bundle={"models":fitted,"scaler":scaler,"feature_names":list(Xtr.columns),
            "raw_features":FEATURE_NAMES,"target":TARGET_NAME,"best_model":best,
            "source_info":source,"metrics":metrics,"train_rows":len(Xtr_s),"test_rows":len(Xte_s)}
    if tune: bundle["tuning_best_params"]=fitted["Tuned Gradient Boosting"].get_params()
    joblib.dump(bundle,os.path.join(ARTIFACT_DIR,"model_bundle.joblib"))
    with open(os.path.join(RESULT_DIR,"production_model_summary.json"),"w",encoding="utf8") as f:
        json.dump({"best_model":best,"metrics":table.to_dict(orient="records"),"source":source,
                   "features":list(Xtr.columns)},f,indent=2,default=str)
    return bundle,df,prep,table

def prepare_input(raw_df):
    return engineer_features(raw_df[FEATURE_NAMES])

def predict(bundle, raw_df, model_name=None):
    model_name=model_name or bundle["best_model"]
    X=prepare_input(raw_df)
    Xs=pd.DataFrame(bundle["scaler"].transform(X),columns=X.columns,index=X.index)
    return bundle["models"][model_name].predict(Xs)

def load_or_train():
    path=os.path.join(ARTIFACT_DIR,"model_bundle.joblib")
    if os.path.exists(path):
        try: return joblib.load(path)
        except Exception: pass
    return build_training_bundle(tune=False)[0]
