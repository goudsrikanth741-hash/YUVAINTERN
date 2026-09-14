
"""Robust preprocessing with leakage-safe train/test splitting."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data_loader import FEATURE_NAMES, TARGET_NAME
RANDOM_SEED=42

def clean_data(df):
    df=df.copy().drop_duplicates().dropna()
    for col in ["AveRooms","AveBedrms","AveOccup","Population"]:
        q1,q99=df[col].quantile([.01,.99]); df[col]=df[col].clip(q1,q99)
    return df.reset_index(drop=True)

def preprocess_pipeline(df,test_size=.2,seed=RANDOM_SEED):
    df_clean=clean_data(df)
    X=df_clean[FEATURE_NAMES].copy(); y=df_clean[TARGET_NAME].values
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=seed)
    scaler=StandardScaler()
    X_train_scaled=pd.DataFrame(scaler.fit_transform(X_train),columns=FEATURE_NAMES,index=X_train.index)
    X_test_scaled=pd.DataFrame(scaler.transform(X_test),columns=FEATURE_NAMES,index=X_test.index)
    return {"df_clean":df_clean,"X_train":X_train_scaled,"X_test":X_test_scaled,
            "X_train_raw":X_train,"X_test_raw":X_test,"y_train":y_train,"y_test":y_test,
            "scaler":scaler}
