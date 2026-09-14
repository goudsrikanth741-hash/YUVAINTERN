
"""Week 6 command-line training and evaluation pipeline.

Run:
    python main.py
or launch the interactive web application:
    streamlit run app.py
"""
import json, os
import numpy as np, pandas as pd
from src.pipeline import build_training_bundle
from src.evaluation import plot_actual_vs_predicted, plot_residuals, plot_feature_importance
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import engineer_features

ROOT=os.path.dirname(__file__)
FIG=os.path.join(ROOT,"outputs","figures")
RES=os.path.join(ROOT,"outputs","results")
os.makedirs(FIG,exist_ok=True); os.makedirs(RES,exist_ok=True)

def main():
    bundle,df,prep,table=build_training_bundle(tune=False)
    Xte=engineer_features(prep["X_test_raw"])
    Xtes=pd.DataFrame(bundle["scaler"].transform(Xte),columns=Xte.columns,index=Xte.index)
    y=prep["y_test"]
    for name,model in bundle["models"].items():
        pred=model.predict(Xtes)
        safe=name.lower().replace(" ","_")
        plot_actual_vs_predicted(y,pred,name,os.path.join(FIG,f"actual_vs_predicted_{safe}.png"))
        plot_residuals(y,pred,name,os.path.join(FIG,f"residuals_{safe}.png"))
        if hasattr(model,"feature_importances_"):
            imp=plot_feature_importance(model,bundle["feature_names"],os.path.join(FIG,f"feature_importance_{safe}.png"))
            if imp is not None: imp.to_csv(os.path.join(RES,f"feature_importance_{safe}.csv"),index=False)
    table.to_csv(os.path.join(RES,"model_comparison_metrics.csv"),index=False)
    with open(os.path.join(RES,"problem_definition.json"),"w") as f:
        json.dump({"task":"Supervised regression","target":"MedHouseVal","objective":"Predict median house value and compare/optimize models.","success_metrics":["MAE","RMSE","R2","5-fold CV R2"]},f,indent=2)
    with open(os.path.join(RES,"limitations.json"),"w") as f:
        json.dump([
            "The built-in dataset may be synthetic when the real California Housing dataset is unavailable.",
            "The source target is a historical block-group median, not an individual property appraisal.",
            "Geographic and economic context is limited; predictions should not be treated as financial advice.",
            "Random train/test splits can overstate spatial generalization; spatial validation is a recommended extension."
        ],f,indent=2)
    print("\nMODEL LEADERBOARD\n",table.to_string(index=False))
    print("\nSaved trained model: artifacts/model_bundle.joblib")
    print("Web app: streamlit run app.py")

if __name__=="__main__":
    main()
