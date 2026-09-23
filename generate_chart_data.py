"""
generate_chart_data.py
------------------------
Generates JSON data files used to render the two charts on the
frontend (Actual vs Predicted, and BMI vs Charges), using the SAME
trained model and test split as train_model.py.
"""

import json
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from train_model import DATA_PATH, MODEL_PATH, build_features

STATIC_DATA_DIR = os.path.join(os.path.dirname(__file__), "static", "data")


def main():
    os.makedirs(STATIC_DATA_DIR, exist_ok=True)

    insurance_data = pd.read_csv(DATA_PATH)
    X = insurance_data.drop(columns=["charges"])
    y = insurance_data["charges"]
    X_feat = build_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2, random_state=42
    )

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    actual_vs_predicted = [
        {"actual": round(float(a), 2), "predicted": round(float(p), 2)}
        for a, p in zip(y_test.tolist(), y_pred.tolist())
    ]

    bmi_vs_charges = [
        {
            "bmi": round(float(b), 2),
            "charges": round(float(c), 2),
            "smoker": s,
        }
        for b, c, s in zip(
            insurance_data["bmi"].tolist(),
            insurance_data["charges"].tolist(),
            insurance_data["smoker"].tolist(),
        )
    ]

    with open(os.path.join(STATIC_DATA_DIR, "actual_vs_predicted.json"), "w") as f:
        json.dump(actual_vs_predicted, f)

    with open(os.path.join(STATIC_DATA_DIR, "bmi_vs_charges.json"), "w") as f:
        json.dump(bmi_vs_charges, f)

    print("Chart data written to", STATIC_DATA_DIR)


if __name__ == "__main__":
    main()
