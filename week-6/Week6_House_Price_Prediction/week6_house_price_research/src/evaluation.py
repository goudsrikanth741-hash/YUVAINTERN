
"""Evaluation and visualization helpers for the production-style app."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
RES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "results")
RANDOM_SEED = 42

def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "y_pred": y_pred,
    }

def metrics_table(all_metrics):
    return pd.DataFrame([{
        "Model": name, "MAE": round(m["mae"],4), "MSE": round(m["mse"],4),
        "RMSE": round(m["rmse"],4), "R2": round(m["r2"],4)
    } for name,m in all_metrics.items()]).sort_values("R2",ascending=False).reset_index(drop=True)

def cross_validate_models(models, X, y, cv_folds=5):
    cv=KFold(n_splits=cv_folds,shuffle=True,random_state=RANDOM_SEED)
    rows=[]
    for name,model in models.items():
        scores=cross_val_score(model,X,y,cv=cv,scoring="r2",n_jobs=-1)
        rows.append({"Model":name,"CV R2 Mean":scores.mean(),"CV R2 Std":scores.std(),
                     "CV R2 Min":scores.min(),"CV R2 Max":scores.max()})
    return pd.DataFrame(rows)

def plot_actual_vs_predicted(y_true,y_pred,title,path):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    plt.figure(figsize=(6,5))
    plt.scatter(y_true,y_pred,alpha=.35,s=14)
    lo=min(np.min(y_true),np.min(y_pred)); hi=max(np.max(y_true),np.max(y_pred))
    plt.plot([lo,hi],[lo,hi],"--")
    plt.xlabel("Actual value ($100k)")
    plt.ylabel("Predicted value ($100k)")
    plt.title(title); plt.tight_layout(); plt.savefig(path,dpi=150); plt.close()

def plot_residuals(y_true,y_pred,title,path):
    residuals=np.asarray(y_true)-np.asarray(y_pred)
    os.makedirs(os.path.dirname(path),exist_ok=True)
    plt.figure(figsize=(7,4.5))
    plt.scatter(y_pred,residuals,alpha=.35,s=14)
    plt.axhline(0,linestyle="--")
    plt.xlabel("Predicted value ($100k)"); plt.ylabel("Residual")
    plt.title(title); plt.tight_layout(); plt.savefig(path,dpi=150); plt.close()
    return {"mean_residual":float(residuals.mean()),"std_residual":float(residuals.std()),
            "mae_residual":float(np.mean(np.abs(residuals)))}

def plot_feature_importance(model,feature_names,path,top_n=15):
    if not hasattr(model,"feature_importances_"): return None
    imp=pd.DataFrame({"feature":feature_names,"importance":model.feature_importances_})
    imp=imp.sort_values("importance",ascending=False).head(top_n)
    plt.figure(figsize=(7,6)); plt.barh(imp["feature"][::-1],imp["importance"][::-1])
    plt.xlabel("Importance"); plt.title("Top feature importance")
    plt.tight_layout(); plt.savefig(path,dpi=150); plt.close()
    return imp
