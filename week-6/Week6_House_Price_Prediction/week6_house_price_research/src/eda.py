"""
eda.py
======
Exploratory data analysis: distributions, correlation analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import FEATURE_NAMES, TARGET_NAME

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
RES_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "results")
sns.set_theme(style="whitegrid")


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe().T.round(4)


def plot_feature_distributions(df: pd.DataFrame, save_name="feature_distributions.png"):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_NAMES):
        sns.histplot(df[col], kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(col)
    plt.suptitle("Feature Distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, save_name), dpi=150)
    plt.close()


def plot_target_distribution(df: pd.DataFrame, save_name="target_distribution.png"):
    plt.figure(figsize=(6, 4.5))
    sns.histplot(df[TARGET_NAME], kde=True, color="darkorange", bins=40)
    plt.title("Target Distribution: Median House Value ($100,000s)")
    plt.xlabel("Median House Value")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, save_name), dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, save_name="correlation_heatmap.png"):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Correlation Analysis: Features vs Target")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, save_name), dpi=150)
    plt.close()
    return corr


def plot_geo_scatter(df: pd.DataFrame, save_name="geographic_price_distribution.png"):
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(df["Longitude"], df["Latitude"], c=df[TARGET_NAME],
                      cmap="viridis", s=8, alpha=0.6)
    plt.colorbar(sc, label="Median House Value ($100k)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Geographic Distribution of Median House Value")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, save_name), dpi=150)
    plt.close()
