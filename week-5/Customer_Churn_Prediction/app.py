import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.predictor import validate_customer_input, transform_raw_customer, predict_from_features

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

@st.cache_resource
def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run 'python main.py' first.")
    return joblib.load(MODEL_PATH)

def score_customer(artifact, customer):
    validate_customer_input(customer)
    X = transform_raw_customer(
        customer,
        artifact["scaler"],
        artifact["feature_names"],
        artifact.get("numeric_feature_names"),
    )
    return predict_from_features(
        artifact["model"], X, artifact["feature_names"], artifact.get("threshold", 0.5)
    )

def text(v):
    return "" if pd.isna(v) else str(v)

st.title("📊 Customer Churn Prediction")
st.caption("Week 5 Machine Learning Project — interactive customer churn analysis")

try:
    artifact = load_artifact()
    model_loaded = True
except Exception as e:
    artifact = None
    model_loaded = False
    st.error("⚠️ Model could not be loaded.")
    st.code(str(e))
    st.info("Run `python main.py` once from the project folder, then refresh this page.")

with st.sidebar:
    st.header("Menu")
    page = st.radio("Select", ["🔮 Single Prediction", "📁 Batch Prediction", "📈 Model Dashboard", "ℹ️ About"])
    st.divider()
    if model_loaded:
        st.success("Model loaded")
        st.caption(f"Threshold: {artifact.get('threshold', 0.5):.3f}")
    else:
        st.warning("Model unavailable")

if page == "🔮 Single Prediction":
    st.subheader("Customer Information")
    st.write("Enter the customer details below and click **Predict Churn**.")

    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with c2:
        security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with c3:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=8, step=1)
        monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=10000.0, value=95.5, step=0.5)
        total = st.number_input("Total Charges", min_value=0.0, max_value=100000.0, value=764.0, step=1.0)

    customer = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple,
        "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": device, "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }

    if st.button("🔮 Predict Churn", type="primary", use_container_width=True, disabled=not model_loaded):
        try:
            result = score_customer(artifact, customer)
            probability = float(result["churn_probability"])
            st.session_state["last_result"] = result
            st.session_state["last_customer"] = customer
            a, b, c = st.columns(3)
            with a:
                st.metric("Prediction", result["prediction"])
            with b:
                st.metric("Churn Probability", f"{probability:.1%}")
            with c:
                st.metric("Risk Level", result["risk_level"])
            st.progress(probability)
            st.info(f"💡 Recommended action: **{result['recommended_action']}**")
            with st.expander("View submitted customer data"):
                st.json(customer)
        except Exception as e:
            st.error("Prediction failed")
            st.exception(e)

elif page == "📁 Batch Prediction":
    st.subheader("Batch Customer Prediction")
    st.write("Upload a CSV containing the same raw customer columns as the training dataset.")
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write(f"Rows loaded: **{len(df)}**")
            st.dataframe(df.head(20), use_container_width=True)
            if st.button("Run Batch Predictions", type="primary", disabled=not model_loaded):
                results = []
                errors = []
                for idx, row in df.iterrows():
                    customer = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
                    try:
                        r = score_customer(artifact, customer)
                        results.append({**row.to_dict(), **r})
                    except Exception as e:
                        errors.append({"row": int(idx), "error": str(e)})
                        results.append({**row.to_dict(), "prediction": "ERROR", "churn_probability": np.nan, "risk_level": "ERROR", "recommended_action": "Check input"})
                out = pd.DataFrame(results)
                st.success(f"Processed {len(out)} rows; {len(errors)} errors.")
                st.dataframe(out, use_container_width=True)
                st.download_button("⬇️ Download Predictions", out.to_csv(index=False).encode("utf-8"), "churn_predictions.csv", "text/csv")
                if errors:
                    with st.expander("View errors"):
                        st.json(errors)
        except Exception as e:
            st.error("Could not read the CSV")
            st.exception(e)

elif page == "📈 Model Dashboard":
    st.subheader("Model Performance Dashboard")
    results = ROOT / "outputs" / "results"
    metrics_file = results / "final_test_metrics.csv"
    summary_file = results / "best_model_summary.json"
    if metrics_file.exists():
        metrics = pd.read_csv(metrics_file)
        st.dataframe(metrics, use_container_width=True)
        numeric = metrics.select_dtypes(include=np.number).iloc[0]
        cols = st.columns(min(5, len(numeric)))
        for col, (name, value) in zip(cols, numeric.items()):
            col.metric(name.replace("_", " ").title(), f"{value:.3f}")
    else:
        st.warning("Run `python main.py` to generate model metrics.")
    if summary_file.exists():
        st.markdown("### Best model")
        st.json(json.loads(summary_file.read_text(encoding="utf-8")))
    st.markdown("### Available reports")
    files = sorted(results.glob("*")) if results.exists() else []
    for f in files:
        st.write(f"• {f.name}")

else:
    st.subheader("About this project")
    st.markdown("""
    **Customer Churn Prediction** is a supervised machine-learning application for identifying customers at risk of leaving.

    **Included features:** data validation, preprocessing, feature engineering, class-imbalance handling, model comparison, hyperparameter tuning, validation-only threshold selection, saved model artifact, single-customer prediction, batch CSV scoring, and an interactive dashboard.

    **Run commands:**
    ```powershell
    pip install -r requirements.txt
    python main.py
    python run_checks.py
    streamlit run app.py
    ```

    Then open **http://localhost:8501**.
    """)
