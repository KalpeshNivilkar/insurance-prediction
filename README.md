# Insurance Charges Predictor

A small, full-stack machine learning web app that predicts medical
insurance charges using **Linear Regression**, built on top of an
existing Jupyter notebook and dataset.

---

## Project Overview

This project turns a Jupyter notebook ML experiment into a usable web
application. A user enters basic personal details — age, sex, BMI,
number of children, smoking status, and region — and the app returns
an estimated insurance charge, computed by a Linear Regression model
trained on real policyholder data.

The app is intentionally kept simple: a Flask backend loads a
pre-trained model and preprocessing logic, and a lightweight
HTML/CSS/JS frontend collects input and displays the prediction, model
performance, and two visualizations.

---

## Dataset

`data/insurance.csv` — **1,338 rows**, 7 columns:

| Column     | Type        | Description                              |
|------------|-------------|-------------------------------------------|
| `age`      | numeric     | Age of the primary beneficiary            |
| `sex`      | categorical | `male` / `female`                         |
| `bmi`      | numeric     | Body mass index                           |
| `children` | numeric     | Number of dependents covered              |
| `smoker`   | categorical | `yes` / `no`                              |
| `region`   | categorical | `northeast` / `northwest` / `southeast` / `southwest` |
| `charges`  | numeric     | **Target** — individual medical charges billed |

---

## Features Used by the Model

`age`, `sex`, `bmi`, `children`, `smoker`, `region` — all six input
fields on the prediction form are used by the model.

**Target variable:** `charges`.

---

## Machine Learning Algorithm

**Linear Regression** (`sklearn.linear_model.LinearRegression`), an
ordinary least squares model. It's a good fit here because:

- The target (`charges`) is a continuous number, not a category —
  exactly what regression is for.
- Charges trend fairly steadily with age, BMI, and especially smoking
  status, so a linear relationship captures a large share of the
  signal.
- It's fast to train, has no hyperparameters to tune, and its
  coefficients are directly interpretable — useful for a portfolio
  project you need to explain in an interview.

### How the model works (and how it was adapted from the notebook)

The original notebook (`notebooks/insurance_linear_regression.ipynb`)
does the following:

1. Loads `insurance.csv` with pandas.
2. Drops `charges` (target) and **`region`** from the feature set.
3. Maps `sex` → `{female: 1, male: 0}` and `smoker` → `{yes: 1, no: 0}`.
4. Splits the data 80/20 with `train_test_split(..., random_state=42)`.
5. Fits `LinearRegression()` and evaluates with R² and adjusted R².

**This app reuses that exact approach** — same algorithm, same
train/test split, same binary mapping for `sex` and `smoker` — via
`train_model.py`. The one adaptation: the notebook dropped `region`
entirely, but the app's UI (per the project brief) requires a region
input. So `region` is one-hot encoded (`region_northeast`,
`region_northwest`, `region_southeast`, `region_southwest`) using the
same "convert categories to numbers" philosophy as the notebook's
`sex`/`smoker` mapping — nothing else about the modeling approach was
changed. Re-running the original notebook's exact steps on this
dataset gives an R² of **0.781**; adding region the same way gives
**0.784** on this app's model — a negligible difference, confirming
the extension didn't change the model's behavior.

The model and its preprocessing are **trained once** (`train_model.py`)
and saved to `model/model.pkl` with `joblib`. Flask loads that file at
startup and never retrains on a request — it just runs new input
through the same `build_features()` function used during training,
then calls `model.predict()`.

---

## Technologies Used

- **Frontend:** HTML, CSS, JavaScript, [Chart.js](https://www.chartjs.org/) (via CDN) for charts
- **Backend:** Python, Flask
- **ML:** pandas, NumPy, scikit-learn, joblib
- **Dataset:** `insurance.csv` (1,338 records)

---

## Project Structure

```
insurance-prediction/
│
├── app.py                     # Flask app (routes + prediction API)
├── train_model.py             # Trains and saves the model (run once)
├── generate_chart_data.py     # Generates JSON used by the two charts
├── model/
│   ├── model.pkl              # Trained LinearRegression model
│   ├── metrics.json           # R², adjusted R², MSE, RMSE
│   └── feature_columns.json   # Exact feature column order
├── data/
│   └── insurance.csv
├── templates/
│   ├── index.html             # Landing page
│   └── predict.html           # Prediction form + results + charts
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   └── data/                  # Chart data (generated)
├── notebooks/
│   └── insurance_linear_regression.ipynb   # Original notebook, unchanged
├── requirements.txt
└── README.md
```

---

## Installation

```bash
cd insurance-prediction
python3 -m venv venv
source venv/bin/activate        # Window: \venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train the Model (first time only)

The repo ships with an already-trained `model/model.pkl`, but you can
regenerate it (and the chart data) from `insurance.csv` at any time:

```bash
python train_model.py
python generate_chart_data.py
```

## Run the Flask Application

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## Example Prediction

**Input:**

| Field    | Value     |
|----------|-----------|
| Age      | 27        |
| Sex      | Male      |
| BMI      | 42.1      |
| Children | 0         |
| Smoker   | Yes       |
| Region   | Southeast |

**Output:** *Estimated Insurance Charge: ~₹32,173*

For comparison, the same profile with `Smoker = No` predicts roughly
**₹4,600** — illustrating how heavily smoking status drives the
model's output, consistent with the BMI-vs-charges chart on the
prediction page.

---

## Model Evaluation

Computed on the 20% held-out test split (calculated, not invented):

| Metric | Value |
|---|---|
| R² Score | 0.7836 |
| Adjusted R² | 0.7760 |
| Mean Squared Error | 33,596,915.85 |
| Root Mean Squared Error | 5,796.28 |

An R² of ~0.78 means the model explains about 78% of the variance in
insurance charges from these six features. The remaining variance
likely comes from factors not in the dataset (e.g. pre-existing
conditions, specific treatments).

---

## Future Improvements

- Try non-linear models (Random Forest, Gradient Boosting) and compare R²/RMSE.
- Add polynomial or interaction features (e.g. `bmi × smoker`), since smoking appears to interact strongly with other features.
- Add confidence intervals around each prediction rather than a single point estimate.
- Persist user submissions to a small database to track prediction history.
- Add input auto-save and a "compare two profiles" view.
- Deploy behind Gunicorn + a proper WSGI setup for production use.

---

## Notes

- All amounts are displayed in ₹ (INR) purely for display formatting; the underlying model was trained directly on the `charges` column from `insurance.csv` with no currency conversion applied.
- This is an educational/portfolio project — predictions are estimates, not a substitute for an actual insurance quote.
