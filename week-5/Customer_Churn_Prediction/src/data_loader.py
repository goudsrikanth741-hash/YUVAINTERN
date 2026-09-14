"""
data_loader.py
==============
Loads the IBM Telco Customer Churn dataset for the Week 5 project.

Behaviour
---------
1. If a real dataset file is found at ``data/Telco-Customer-Churn.csv``
   (the standard IBM/Kaggle Telco Customer Churn export), it is loaded
   and used directly. This is the recommended path for a fully
   "actual-results" run.
2. If no such file is present (e.g. because this environment has no
   internet access to download the dataset), a synthetic dataset is
   generated locally. The synthetic generator matches the exact column
   schema of the real IBM Telco dataset and encodes the same qualitative
   relationships that are well documented for this dataset (e.g.
   month-to-month contracts, low tenure, fibre-optic internet and high
   monthly charges are associated with a higher churn probability;
   long-tenure customers on two-year contracts churn far less).

Any run using the synthetic fallback is clearly labelled as such in the
returned metadata dictionary and in the generated README/results, so
that synthetic runs are never confused with runs on the real dataset.

To use the real dataset:
    1. Download "WA_Fn-UseC_-Telco-Customer-Churn.csv" from the IBM/Kaggle
       Telco Customer Churn dataset.
    2. Rename or copy it to: data/Telco-Customer-Churn.csv
    3. Re-run `python main.py`.
"""

import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
REAL_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Telco-Customer-Churn.csv"))

CATEGORICAL_COLUMNS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
TARGET_COLUMN = "Churn"


def _generate_synthetic_telco(n_customers: int = 4000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a synthetic dataset that mirrors the schema and the
    documented qualitative churn drivers of the IBM Telco dataset.

    This is NOT the real dataset. It is used only as a runnable
    fallback when the real CSV is not available locally (e.g. no
    internet access to download it). All values are produced by a
    seeded random process, not copied from any real customer.
    """
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], n_customers)
    senior_citizen = rng.choice([0, 1], n_customers, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n_customers, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n_customers, p=[0.30, 0.70])

    tenure = rng.integers(0, 73, n_customers)  # months, matches real dataset range 0-72

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n_customers, p=[0.55, 0.21, 0.24]
    )
    phone_service = rng.choice(["Yes", "No"], n_customers, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No", "No phone service",
        rng.choice(["Yes", "No"], n_customers)
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], n_customers, p=[0.34, 0.44, 0.22]
    )

    def dependent_internet_feature():
        return np.where(
            internet_service == "No", "No internet service",
            rng.choice(["Yes", "No"], n_customers)
        )

    online_security = dependent_internet_feature()
    online_backup = dependent_internet_feature()
    device_protection = dependent_internet_feature()
    tech_support = dependent_internet_feature()
    streaming_tv = dependent_internet_feature()
    streaming_movies = dependent_internet_feature()

    paperless_billing = rng.choice(["Yes", "No"], n_customers, p=[0.59, 0.41])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n_customers, p=[0.34, 0.23, 0.22, 0.21]
    )

    # Base monthly charge depends on services subscribed
    base_charge = np.full(n_customers, 18.0)
    base_charge += np.where(phone_service == "Yes", 5.0, 0.0)
    base_charge += np.where(internet_service == "DSL", 25.0, 0.0)
    base_charge += np.where(internet_service == "Fiber optic", 45.0, 0.0)
    for feat in [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]:
        base_charge += np.where(feat == "Yes", rng.uniform(4, 9, n_customers), 0.0)
    monthly_charges = np.round(base_charge + rng.normal(0, 3.0, n_customers), 2)
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75)

    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 15, n_customers), 2)
    total_charges = np.clip(total_charges, 0, None)
    # New customers (tenure 0) commonly show blank/zero TotalCharges in the real dataset
    total_charges = np.where(tenure == 0, 0.0, total_charges)

    # --- Construct a churn probability from documented real-world drivers ---
    logit = np.full(n_customers, -1.6)
    logit += np.where(contract == "Month-to-month", 1.55, 0.0)
    logit += np.where(contract == "One year", 0.25, 0.0)
    logit += np.where(contract == "Two year", -1.1, 0.0)
    logit += np.where(internet_service == "Fiber optic", 0.55, 0.0)
    logit += np.where(internet_service == "No", -0.6, 0.0)
    logit += np.where(tech_support == "No", 0.35, 0.0)
    logit += np.where(online_security == "No", 0.35, 0.0)
    logit += np.where(paperless_billing == "Yes", 0.25, 0.0)
    logit += np.where(payment_method == "Electronic check", 0.45, 0.0)
    logit += np.where(senior_citizen == 1, 0.30, 0.0)
    logit += np.where(partner == "No", 0.15, 0.0)
    logit += np.where(dependents == "No", 0.15, 0.0)
    logit += -0.045 * tenure                       # long tenure reduces churn
    logit += 0.010 * (monthly_charges - 60.0)       # higher bills raise churn
    logit += rng.normal(0, 0.65, n_customers)       # unexplained noise

    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (rng.uniform(0, 1, n_customers) < churn_prob).astype(int)
    churn_labels = np.where(churn == 1, "Yes", "No")

    customer_id = [f"SYN-{i:05d}" for i in range(n_customers)]

    df = pd.DataFrame({
        "customerID": customer_id,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn_labels,
    })
    return df


def load_dataset(verbose: bool = True):
    """Load the Telco churn dataset.

    Returns
    -------
    df : pandas.DataFrame
    source_info : dict with keys {"source", "is_synthetic", "n_rows", "n_cols"}
    """
    if os.path.exists(REAL_DATA_PATH):
        df = pd.read_csv(REAL_DATA_PATH)
        # The real dataset stores TotalCharges as string with occasional blanks
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
        source_info = {
            "source": "real IBM Telco Customer Churn CSV (data/Telco-Customer-Churn.csv)",
            "is_synthetic": False,
            "n_rows": len(df),
            "n_cols": df.shape[1],
        }
    else:
        df = _generate_synthetic_telco()
        source_info = {
            "source": (
                "SYNTHETIC fallback dataset generated locally (no internet access to "
                "download the real IBM Telco CSV in this environment). Schema and "
                "documented churn-driver relationships match the real dataset. "
                "Place the real CSV at data/Telco-Customer-Churn.csv and re-run for "
                "results on the actual dataset."
            ),
            "is_synthetic": True,
            "n_rows": len(df),
            "n_cols": df.shape[1],
        }

    if verbose:
        print(f"[data_loader] Source: {source_info['source']}")
        print(f"[data_loader] Shape: {df.shape}")
        print(f"[data_loader] Churn rate: {(df[TARGET_COLUMN] == 'Yes').mean():.3f}")

    return df, source_info


if __name__ == "__main__":
    data, info = load_dataset()
    print(data.head())
    print(info)
