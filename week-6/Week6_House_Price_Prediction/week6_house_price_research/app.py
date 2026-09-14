
"""HouseValue AI — production-style Streamlit interface."""
import io, json, os, sys, time
import numpy as np, pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from src.pipeline import load_or_train, predict, build_training_bundle
from src.data_loader import FEATURE_NAMES, TARGET_NAME, load_dataset
from src.feature_engineering import engineer_features
from src.evaluation import cross_validate_models

st.set_page_config(page_title="HouseValue AI", page_icon="🏠", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.5rem;max-width:1400px}
.hero{padding:1.2rem 1.5rem;border:1px solid #d9dee8;border-radius:16px;background:#f7f9fc}
.card{padding:1rem;border:1px solid #e1e5eb;border-radius:12px;background:white}
.small{color:#64748b;font-size:.88rem}
</style>
""",unsafe_allow_html=True)

@st.cache_resource
def get_bundle(): return load_or_train()

@st.cache_data
def get_data(): return load_dataset(verbose=False)[0]

bundle=get_bundle(); df=get_data()
st.markdown('<div class="hero"><h1>🏠 HouseValue AI</h1><p>End-to-end machine-learning platform for transparent California house-value estimation, evaluation and optimization.</p></div>',unsafe_allow_html=True)
st.caption(f"Data source: {'Synthetic fallback' if bundle['source_info'].get('is_synthetic') else 'Real California Housing'} • Best model: {bundle['best_model']} • 20+ engineered/derived signals")

# Feature 1: global navigation / 2: system status
pages=["🏠 Dashboard","💰 Price Predictor","📂 Batch Prediction","📊 Model Lab","🧪 Experiment Center","🔎 Explainability","🗺️ Geo Explorer","📈 Data Explorer","⚙️ Settings & QA"]
page=st.sidebar.radio("Navigation",pages)
st.sidebar.success("System online")
st.sidebar.write(f"Models loaded: {len(bundle['models'])}")
st.sidebar.write(f"Training rows: {bundle['train_rows']:,}")

# reusable input defaults
defaults={"MedInc":5.0,"HouseAge":25.0,"AveRooms":5.5,"AveBedrms":1.1,"Population":1200.0,"AveOccup":3.0,"Latitude":34.05,"Longitude":-118.25}

if page=="🏠 Dashboard":
    # 3 KPI cards, 4 model leaderboard, 5 data source transparency
    best=bundle["metrics"][bundle["best_model"]]
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Best R²",f"{best['r2']:.3f}")
    c2.metric("RMSE",f"${best['rmse']*100000:,.0f}")
    c3.metric("MAE",f"${best['mae']*100000:,.0f}")
    c4.metric("Test samples",f"{bundle['test_rows']:,}")
    st.subheader("Model leaderboard")
    rows=[]
    for n,m in bundle["metrics"].items(): rows.append({"Model":n,"R²":m["r2"],"RMSE ($)":m["rmse"]*100000,"MAE ($)":m["mae"]*100000})
    st.dataframe(pd.DataFrame(rows).sort_values("R²",ascending=False),use_container_width=True,hide_index=True)
    st.info("Important: the built-in fallback is synthetic when the real California dataset cannot be downloaded. For academic reporting, disclose which source was actually used.")
    st.subheader("What this platform covers")
    cols=st.columns(4)
    for i,t in enumerate(["Prediction","Batch scoring","Model comparison","Cross-validation","Hyperparameter tuning","Feature importance","Residual analysis","Scenario simulation","Data quality","Correlation explorer","Geographic explorer","Exportable results"]):
        cols[i%4].write("✓ "+t)

elif page=="💰 Price Predictor":
    st.subheader("Interactive property/block-group estimator")
    st.write("Enter the eight source variables. The system derives additional interaction and ratio features automatically.")
    cols=st.columns(4); vals={}
    labels={"MedInc":"Median income (10k USD)","HouseAge":"House age (years)","AveRooms":"Average rooms","AveBedrms":"Average bedrooms","Population":"Population","AveOccup":"Average occupants","Latitude":"Latitude","Longitude":"Longitude"}
    for i,f in enumerate(FEATURE_NAMES):
        with cols[i%4]: vals[f]=st.number_input(labels[f],value=float(defaults[f]),key="pred_"+f)
    model_name=st.selectbox("Model",list(bundle["models"].keys()),index=list(bundle["models"].keys()).index(bundle["best_model"]))
    if st.button("Predict house value",type="primary"):
        raw=pd.DataFrame([vals])
        pred=float(predict(bundle,raw,model_name)[0])
        st.success(f"Estimated median value: ${pred*100000:,.0f}")
        st.caption(f"Model: {model_name}. Dataset target is expressed in $100,000 units.")
        # 6 scenario sensitivity
        st.subheader("Quick sensitivity scenarios")
        scenarios=[]
        for factor in [0.85,1.0,1.15]:
            x=raw.copy(); x["MedInc"]*=factor
            p=float(predict(bundle,x,model_name)[0])*100000
            scenarios.append({"Income multiplier":factor,"Estimated value ($)":p})
        st.dataframe(pd.DataFrame(scenarios),hide_index=True,use_container_width=True)

elif page=="📂 Batch Prediction":
    # 7 batch upload, 8 validation, 9 export
    st.subheader("Batch prediction")
    st.write("Upload CSV containing: "+", ".join(FEATURE_NAMES))
    sample=pd.DataFrame([defaults])
    st.download_button("Download CSV template",sample.to_csv(index=False),"house_price_template.csv","text/csv")
    uploaded=st.file_uploader("Upload property/block-group CSV",type=["csv"])
    if uploaded:
        try:
            data=pd.read_csv(uploaded)
            missing=[c for c in FEATURE_NAMES if c not in data.columns]
            if missing: st.error("Missing columns: "+", ".join(missing))
            else:
                work=data.copy()
                work["PredictedValue_100k"]=predict(bundle,work)
                work["PredictedPrice_USD"]=work["PredictedValue_100k"]*100000
                st.success(f"Scored {len(work):,} rows.")
                st.dataframe(work.head(100),use_container_width=True)
                st.download_button("Export predictions",work.to_csv(index=False),"predictions.csv","text/csv")
        except Exception as e: st.error(f"Could not process file: {e}")

elif page=="📊 Model Lab":
    # 10 model metrics, 11 actual/predicted, 12 residuals
    st.subheader("Model evaluation lab")
    metrics=pd.DataFrame([{"Model":n,"MAE":m["mae"],"RMSE":m["rmse"],"R²":m["r2"]} for n,m in bundle["metrics"].items()])
    st.dataframe(metrics.sort_values("R²",ascending=False),use_container_width=True,hide_index=True)
    selected=st.selectbox("Inspect model",list(bundle["models"].keys()))
    m=bundle["metrics"][selected]; ypred=m["y_pred"]; ytrue=np.array(load_dataset(verbose=False)[0].iloc[0:0][TARGET_NAME]) if False else None
    # Recover test data deterministically for plots
    from src.preprocessing import preprocess_pipeline
    prep=preprocess_pipeline(df); Xte=engineer_features(prep["X_test_raw"])
    Xtes=pd.DataFrame(bundle["scaler"].transform(Xte),columns=Xte.columns,index=Xte.index)
    ytrue=prep["y_test"]; ypred=bundle["models"][selected].predict(Xtes)
    c1,c2=st.columns(2)
    with c1:
        fig,ax=plt.subplots(figsize=(6,4)); ax.scatter(ytrue,ypred,alpha=.35,s=12); lo=min(ytrue.min(),ypred.min()); hi=max(ytrue.max(),ypred.max()); ax.plot([lo,hi],[lo,hi],"--"); ax.set(xlabel="Actual ($100k)",ylabel="Predicted ($100k)",title="Actual vs Predicted"); st.pyplot(fig)
    with c2:
        fig,ax=plt.subplots(figsize=(6,4)); ax.scatter(ypred,ytrue-ypred,alpha=.35,s=12); ax.axhline(0,ls="--"); ax.set(xlabel="Predicted",ylabel="Residual",title="Residual diagnostics"); st.pyplot(fig)

elif page=="🧪 Experiment Center":
    # 13 CV, 14 tuning, 15 experiment log, 16 reproducibility
    st.subheader("Optimization & experiment center")
    st.write("Run controlled hyperparameter optimization on Gradient Boosting. This is intentionally optional because it costs compute time.")
    n_iter=st.slider("Random-search iterations",4,16,8)
    if st.button("Run optimization",type="primary"):
        with st.spinner("Training and evaluating tuned model..."):
            tuned_bundle,_,_,table=build_training_bundle(tune=True,tuning_iter=n_iter)
            st.cache_resource.clear()
            st.success("Optimization complete.")
            st.dataframe(table,use_container_width=True,hide_index=True)
            st.json({"best_model":tuned_bundle["best_model"],"best_params":tuned_bundle.get("tuning_best_params",{})})
            st.info("Compare tuned RMSE/R² against the baseline leaderboard before adopting it.")
    if st.button("Run 5-fold cross-validation"):
        from src.models import get_models
        prep=__import__("src.preprocessing",fromlist=["preprocess_pipeline"]).preprocess_pipeline(df)
        X=engineer_features(prep["X_train_raw"]); sc=__import__("sklearn.preprocessing",fromlist=["StandardScaler"]).StandardScaler(); Xs=sc.fit_transform(X)
        cv=cross_validate_models(get_models(),Xs,prep["y_train"],5)
        st.dataframe(cv,use_container_width=True,hide_index=True)
    st.code("Random seed = 42\nTrain/test split = 80/20\nCV = shuffled 5-fold\nMetric objective = minimize RMSE / maximize R²",language="text")

elif page=="🔎 Explainability":
    # 17 feature importance, 18 derived feature transparency
    st.subheader("Explainability")
    model_name=st.selectbox("Explain model",list(bundle["models"].keys()))
    model=bundle["models"][model_name]
    if hasattr(model,"feature_importances_"):
        imp=pd.DataFrame({"Feature":bundle["feature_names"],"Importance":model.feature_importances_}).sort_values("Importance",ascending=False)
        st.bar_chart(imp.set_index("Feature").head(15))
        st.dataframe(imp,hide_index=True,use_container_width=True)
    else: st.info("This model does not expose tree feature_importances_; use the tree models for native importance.")
    st.subheader("Derived feature dictionary")
    derived=[x for x in bundle["feature_names"] if x not in FEATURE_NAMES]
    st.write(", ".join(derived))
    st.caption("Feature importance is model-specific and should not be interpreted as causal impact.")

elif page=="🗺️ Geo Explorer":
    # 19 geographic visualization
    st.subheader("Geographic price explorer")
    c1,c2=st.columns(2)
    with c1: lat=st.slider("Latitude",float(df.Latitude.min()),float(df.Latitude.max()),34.05)
    with c2: lon=st.slider("Longitude",float(df.Longitude.min()),float(df.Longitude.max()),-118.25)
    st.map(df[["Latitude","Longitude"]].rename(columns={"Latitude":"lat","Longitude":"lon"}).sample(min(3000,len(df)),random_state=42))
    st.metric("Nearest sample distance (approx.)",f"{np.sqrt(((df.Latitude-lat)**2+(df.Longitude-lon)**2).min()):.3f} degrees")
    st.caption("Map is an exploratory spatial view, not parcel-level geolocation or a guarantee of market value.")

elif page=="📈 Data Explorer":
    # 20 data preview, 21 stats, 22 correlation
    st.subheader("Dataset quality & exploration")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Rows",len(df)); c2.metric("Columns",df.shape[1]); c3.metric("Missing cells",int(df.isna().sum().sum())); c4.metric("Duplicate rows",int(df.duplicated().sum()))
    st.dataframe(df.describe().T,use_container_width=True)
    feature=st.selectbox("Distribution feature",list(df.columns))
    fig,ax=plt.subplots(figsize=(9,3.5)); ax.hist(df[feature].dropna(),bins=35); ax.set_title(f"Distribution: {feature}"); st.pyplot(fig)
    corr=df.corr(numeric_only=True)
    st.subheader("Correlation matrix")
    st.dataframe(corr.round(3),use_container_width=True)

elif page=="⚙️ Settings & QA":
    # 23 health checks, 24 reproducibility, 25 source transparency
    st.subheader("System QA")
    checks={
        "Model bundle loaded":bool(bundle.get("models")),
        "Best model selected":bool(bundle.get("best_model")),
        "Scaler feature count matches":len(bundle["feature_names"])==len(bundle["scaler"].mean_),
        "Required raw columns available":all(c in df.columns for c in FEATURE_NAMES+[TARGET_NAME]),
        "No NaN in training dataset":not df[FEATURE_NAMES+[TARGET_NAME]].isna().any().any(),
        "Prediction smoke test":np.isfinite(predict(bundle,pd.DataFrame([defaults]))[0]),
    }
    st.dataframe(pd.DataFrame({"Check":list(checks),"Status":["PASS" if v else "FAIL" for v in checks.values()]}),hide_index=True,use_container_width=True)
    st.subheader("Reproducibility")
    st.json({"random_seed":42,"raw_features":FEATURE_NAMES,"engineered_feature_count":len(bundle["feature_names"])-len(FEATURE_NAMES),"best_model":bundle["best_model"]})
    st.subheader("Data provenance")
    st.json(bundle["source_info"])
    st.download_button("Download model metadata",json.dumps({"best_model":bundle["best_model"],"features":bundle["feature_names"],"source":bundle["source_info"]},indent=2),"model_metadata.json","application/json")
