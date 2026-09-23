"""
app.py
------
Flask backend for the Insurance Charges Predictor.

Loads the already-trained Linear Regression model (model/model.pkl)
and serves:
    - the landing page
    - the prediction form page
    - a JSON prediction API used by static/js/script.js
    - the model performance metrics (from model/metrics.json)

The model is trained ONCE by running train_model.py (see README).
This app never retrains on request — it just loads the saved model,
which is the correct pattern for a production-style Flask ML app.
"""

import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from train_model import REGION_CATEGORIES, build_features

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
COLUMNS_PATH = os.path.join(BASE_DIR, "model", "feature_columns.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "insurance.csv")

app = Flask(__name__)

# ---------------------------------------------------------------
# Load the trained model + supporting artifacts ONCE at startup
# ---------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        "model/model.pkl not found. Run `python train_model.py` first "
        "to train and save the model."
    )

model = joblib.load(MODEL_PATH)

with open(METRICS_PATH) as f:
    METRICS = json.load(f)

with open(COLUMNS_PATH) as f:
    FEATURE_COLUMNS = json.load(f)

DATASET_SIZE = len(pd.read_csv(DATA_PATH))


def validate_input(data):
    """Validate and coerce raw form input. Returns (clean_dict, error_message)."""
    errors = []

    try:
        age = int(data.get("age"))
        if not (0 < age < 120):
            errors.append("Age must be between 1 and 119.")
    except (TypeError, ValueError):
        errors.append("Age must be a whole number.")
        age = None

    sex = data.get("sex")
    if sex not in ("male", "female"):
        errors.append("Sex must be 'male' or 'female'.")

    try:
        bmi = float(data.get("bmi"))
        if not (10 <= bmi <= 70):
            errors.append("BMI must be between 10 and 70.")
    except (TypeError, ValueError):
        errors.append("BMI must be a number.")
        bmi = None

    try:
        children = int(data.get("children"))
        if not (0 <= children <= 10):
            errors.append("Number of children must be between 0 and 10.")
    except (TypeError, ValueError):
        errors.append("Number of children must be a whole number.")
        children = None

    smoker = data.get("smoker")
    if smoker not in ("yes", "no"):
        errors.append("Smoker must be 'yes' or 'no'.")

    region = data.get("region")
    if region not in REGION_CATEGORIES:
        errors.append("Region must be one of: " + ", ".join(REGION_CATEGORIES))

    if errors:
        return None, errors

    clean = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    }
    return clean, None


@app.route("/")
def home():
    return render_template(
        "index.html",
        dataset_size=DATASET_SIZE,
        r2_score=METRICS["r2_score"],
    )


@app.route("/predict")
def predict_page():
    return render_template("predict.html", regions=REGION_CATEGORIES)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or request.form
    clean, errors = validate_input(data)

    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    # Build a single-row DataFrame with the raw columns, then apply
    # the SAME preprocessing pipeline used during training.
    row = pd.DataFrame([{
        "age": clean["age"],
        "sex": clean["sex"],
        "bmi": clean["bmi"],
        "children": clean["children"],
        "smoker": clean["smoker"],
        "region": clean["region"],
    }])

    features = build_features(row)
    # Ensure column order exactly matches training-time order
    features = features.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    prediction = float(model.predict(features)[0])
    prediction = max(prediction, 0)  # charges can't be negative

    return jsonify({
        "success": True,
        "prediction": round(prediction, 2),
        "input_summary": clean,
    })


@app.route("/api/metrics")
def api_metrics():
    return jsonify(METRICS)


if __name__ == "__main__":
    app.run(debug=True)
