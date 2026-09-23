"""
train_model.py
----------------
Trains the Linear Regression model for predicting insurance charges.

This script reuses the SAME preprocessing approach as the original
Jupyter notebook (notebooks/insurance_linear_regression.ipynb):
    - sex:    female -> 1, male -> 0
    - smoker: yes -> 1, no -> 0
    - train_test_split with test_size=0.2, random_state=42
    - sklearn's LinearRegression (ordinary least squares)

NOTE ON REGION:
The original notebook dropped the "region" column entirely. Since the
web app's UI requires a region input (as specified in the project
brief), this script extends the notebook's preprocessing in the same
style: region is converted into numeric columns using one-hot
encoding (pd.get_dummies), which is the standard way to handle a
multi-category text column in a linear model. No other part of the
original modeling approach was changed.

Run this once to (re)generate model/model.pkl and model/metrics.json.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "insurance.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

REGION_CATEGORIES = ["northeast", "northwest", "southeast", "southwest"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw insurance data into the numeric feature matrix used
    for training/prediction. Mirrors the notebook's mapping for
    sex/smoker, and one-hot encodes region.
    """
    features = df.copy()

    # Same binary mapping as the original notebook
    features["sex"] = features["sex"].map({"female": 1, "male": 0})
    features["smoker"] = features["smoker"].map({"yes": 1, "no": 0})

    # One-hot encode region (extension of the notebook's approach,
    # since region was dropped there but is required by the app)
    region_dummies = pd.get_dummies(features["region"], prefix="region")
    for cat in REGION_CATEGORIES:
        col = f"region_{cat}"
        if col not in region_dummies.columns:
            region_dummies[col] = 0
    region_dummies = region_dummies[[f"region_{c}" for c in REGION_CATEGORIES]]

    features = features.drop(columns=["region"])
    features = pd.concat([features, region_dummies], axis=1)

    return features


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    insurance_data = pd.read_csv(DATA_PATH)

    X = insurance_data.drop(columns=["charges"])
    y = insurance_data["charges"]

    X = build_features(X)
    feature_columns = list(X.columns)

    # Same split as the notebook
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    n = X_test.shape[0]
    p = X_test.shape[1]
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    metrics = {
        "r2_score": round(float(r2), 4),
        "adjusted_r2_score": round(float(adjusted_r2), 4),
        "mean_squared_error": round(float(mse), 2),
        "root_mean_squared_error": round(float(rmse), 2),
        "test_set_size": int(n),
        "training_set_size": int(X_train.shape[0]),
        "dataset_size": int(len(insurance_data)),
        "num_features": int(p),
    }

    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(COLUMNS_PATH, "w") as f:
        json.dump(feature_columns, f, indent=2)

    print("Model trained and saved to", MODEL_PATH)
    print("Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
