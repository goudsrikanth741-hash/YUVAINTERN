"""
data_loader.py
==============
Loads a California-Housing-style dataset for the Week 6 house price
research project.

Behaviour
---------
1. Tries `sklearn.datasets.fetch_california_housing()` first (the
   standard, real dataset: 20,640 California block groups, 8 numeric
   features, median house value target).
2. If that fails (e.g. no internet access to download it, as in this
   sandboxed environment), falls back to a synthetic dataset generated
   locally that matches the exact feature schema and the well-documented
   real-world relationships (e.g. median income is the strongest driver
   of median house value; latitude/longitude produce coastal price
   premiums).

The returned metadata dict clearly records which source was used, and
this project's README/results distinguish real-dataset runs from
synthetic-fallback runs.

To use the real dataset in an environment with internet access, simply
delete any cached negative result and re-run `python main.py` — no code
change is required, since `fetch_california_housing` is tried first.
"""

import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
CACHED_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "california_housing.csv")

FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
TARGET_NAME = "MedHouseVal"


def _generate_synthetic_california(n_samples: int = 5000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a synthetic dataset matching the California Housing schema
    and documented feature-target relationships (median income is by far
    the strongest predictor of median house value; there are coastal and
    Bay-Area / LA-area price premiums tied to latitude/longitude).

    This is NOT the real dataset - it is a locally-generated, seeded
    approximation used only because this environment cannot download the
    real data from OpenML/sklearn's remote source.
    """
    rng = np.random.default_rng(seed)

    med_inc = np.clip(rng.gamma(shape=3.0, scale=1.3, size=n_samples), 0.5, 15.0)
    house_age = np.clip(rng.normal(28, 12, n_samples), 1, 52)
    ave_rooms = np.clip(rng.normal(5.4, 1.5, n_samples) + 0.15 * med_inc, 1.5, 20)
    ave_bedrms = np.clip(ave_rooms * rng.uniform(0.18, 0.28, n_samples), 0.5, 5)
    population = np.clip(rng.normal(1400, 900, n_samples), 20, 8000)
    ave_occup = np.clip(rng.normal(3.0, 1.0, n_samples), 0.7, 12)

    # Latitude/longitude sampled to loosely mimic California's shape,
    # with denser sampling around the Bay Area and LA basin.
    cluster = rng.choice(["bay_area", "la_basin", "other"], size=n_samples, p=[0.30, 0.35, 0.35])
    latitude = np.empty(n_samples)
    longitude = np.empty(n_samples)
    for i, c in enumerate(cluster):
        if c == "bay_area":
            latitude[i] = rng.normal(37.7, 0.35)
            longitude[i] = rng.normal(-122.2, 0.35)
        elif c == "la_basin":
            latitude[i] = rng.normal(34.05, 0.4)
            longitude[i] = rng.normal(-118.3, 0.4)
        else:
            latitude[i] = rng.uniform(32.6, 41.9)
            longitude[i] = rng.uniform(-124.3, -114.3)

    coastal_premium = np.where(cluster != "other", 0.55, 0.0)

    # Median house value (in $100,000s, matching the real dataset's units)
    value = (
        0.42 * med_inc
        + 0.012 * (50 - np.abs(house_age - 35))
        + 0.05 * (ave_rooms - ave_bedrms)
        - 0.02 * ave_occup
        + coastal_premium
        + rng.normal(0, 0.35, n_samples)
        + 0.55
    )
    value = np.clip(value, 0.15, 5.00)  # real dataset target is capped at 5.00 ($500,000)

    df = pd.DataFrame({
        "MedInc": np.round(med_inc, 4),
        "HouseAge": np.round(house_age, 1),
        "AveRooms": np.round(ave_rooms, 4),
        "AveBedrms": np.round(ave_bedrms, 4),
        "Population": np.round(population, 0),
        "AveOccup": np.round(ave_occup, 4),
        "Latitude": np.round(latitude, 4),
        "Longitude": np.round(longitude, 4),
        "MedHouseVal": np.round(value, 4),
    })
    return df


def load_dataset(verbose: bool = True):
    """Load the California Housing dataset (real if downloadable, else synthetic).

    Returns
    -------
    df : pandas.DataFrame with feature columns + target column `MedHouseVal`
    source_info : dict describing which source was used
    """
    # 1. Try a locally cached CSV first (fastest, works fully offline once cached).
    if os.path.exists(CACHED_CSV_PATH):
        df = pd.read_csv(CACHED_CSV_PATH)
        source_info = {
            "source": "cached real California Housing CSV (data/california_housing.csv)",
            "is_synthetic": False,
        }
    else:
        # 2. Try scikit-learn's real dataset fetcher (requires internet on first use).
        try:
            from sklearn.datasets import fetch_california_housing
            bunch = fetch_california_housing(as_frame=True)
            df = bunch.frame.rename(columns={"MedHouseVal": TARGET_NAME})
            df.to_csv(CACHED_CSV_PATH, index=False)
            source_info = {
                "source": "real California Housing dataset via sklearn.datasets.fetch_california_housing",
                "is_synthetic": False,
            }
        except Exception as exc:
            # 3. Fall back to the synthetic generator, clearly labelled.
            df = _generate_synthetic_california()
            source_info = {
                "source": (
                    "SYNTHETIC fallback dataset generated locally (no internet access to "
                    f"download the real dataset in this environment; fetch failed with: "
                    f"{type(exc).__name__}). Schema and documented feature-target relationships "
                    "(income as the dominant price driver, coastal premiums) match the real "
                    "dataset. Delete data/california_housing.csv and re-run with internet access "
                    "to fetch the real data instead."
                ),
                "is_synthetic": True,
            }

    source_info["n_rows"] = len(df)
    source_info["n_cols"] = df.shape[1]

    if verbose:
        print(f"[data_loader] Source: {source_info['source']}")
        print(f"[data_loader] Shape: {df.shape}")
        print(f"[data_loader] Target range: [{df[TARGET_NAME].min():.3f}, {df[TARGET_NAME].max():.3f}] ($100k units)")

    return df, source_info


if __name__ == "__main__":
    data, info = load_dataset()
    print(data.head())
    print(info)
