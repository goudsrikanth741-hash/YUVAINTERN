"""
model.py
=========
Model-ready pipeline to validate that the preprocessing / feature
engineering produces usable inputs for a classifier.

Two complementary paths are provided:

1. CNN path (primary, TensorFlow/Keras)
   A small convolutional network trained directly on the preprocessed
   224x224 RGB image tensors. This is the "real" deep-learning path
   requested by the assignment. It is only used if TensorFlow is
   importable in the current environment.

2. Classical-ML baseline (automatic fallback, scikit-learn)
   A RandomForest classifier trained on the hand-crafted feature vectors
   produced by feature_extraction.py. This path has no heavy external
   dependency (scikit-learn is part of requirements.txt regardless), so
   it can always run and still genuinely demonstrates that the
   preprocessing + feature-engineering pipeline produces model-ready,
   class-separable inputs - which is the stated purpose of this section
   of the assignment.

main.py automatically tries the CNN path first and falls back to the
baseline path (printing a clear message) if TensorFlow is not available
in the current environment.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


# --------------------------------------------------------------------------
# 1. CNN path (TensorFlow / Keras)
# --------------------------------------------------------------------------

def build_cnn_model(input_shape: Tuple[int, int, int], num_classes: int):
    """
    Small, fast-to-train CNN. Depth is intentionally kept modest: the goal
    of this project is to validate the data pipeline, not to chase
    state-of-the-art accuracy.
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is not installed in this environment.")

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_cnn(model, X_train, y_train, X_val, y_val, class_weight=None,
              epochs: int = 10, batch_size: int = 32, seed: int = 42):
    tf.random.set_seed(seed)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        verbose=2,
    )
    return history


# --------------------------------------------------------------------------
# 2. Classical-ML baseline (scikit-learn, always runnable)
# --------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def build_baseline_classifier(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def train_baseline(clf, X_train: np.ndarray, y_train: np.ndarray,
                    scaler: StandardScaler = None) -> Tuple[RandomForestClassifier, StandardScaler]:
    if scaler is None:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
    else:
        X_train = scaler.transform(X_train)
    clf.fit(X_train, y_train)
    return clf, scaler


def predict_baseline(clf, scaler: StandardScaler, X):
    X = scaler.transform(X)
    return clf.predict(X)
