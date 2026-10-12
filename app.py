"""
app.py — Flask web application for the Human Development Index Predictor.
"""

import json
import math
import os
import pickle
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

# ---- Load trained artifacts once at startup ----
# If scikit-learn warns that the model was pickled with a different version,
# run `python model/train_model.py` once to rebuild it for your installation.
with open(BASE_DIR / "model" / "hdi_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(BASE_DIR / "model" / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(BASE_DIR / "model" / "metrics.json") as f:
    METRICS = json.load(f)

FEATURES = METRICS["features"]
USES_SCALER = bool(METRICS.get("uses_scaler", True))

FIELD_META = {
    "electricity_access": {
        "label": "Access to Electricity",
        "unit": "% of population", "min": 1.0, "max": 100.0, "step": 0.1, "default": 80.0,
    },
    "internet_users": {
        "label": "Internet Users",
        "unit": "% of population", "min": 0.0, "max": 100.0, "step": 0.1, "default": 40.0,
    },
    "water_access": {
        "label": "Access to Clean Drinking Water",
        "unit": "% of population", "min": 15.0, "max": 100.0, "step": 0.1, "default": 85.0,
    },
    "health_exp_gdp": {
        "label": "Health Expenditure",
        "unit": "% of GDP", "min": 1.0, "max": 30.0, "step": 0.1, "default": 6.0,
    },
    "female_labor_force": {
        "label": "Female Labor Force Participation",
        "unit": "% of female population", "min": 4.0, "max": 90.0, "step": 0.1, "default": 50.0,
    },
    "unemployment": {
        "label": "Unemployment Rate",
        "unit": "% of labor force", "min": 0.0, "max": 40.0, "step": 0.1, "default": 8.0,
    },
    "infant_mortality": {
        "label": "Infant Mortality Rate",
        "unit": "per 1,000 live births", "min": 1.0, "max": 185.0, "step": 0.1, "default": 25.0,
    },
    "fertility_rate": {
        "label": "Fertility Rate",
        "unit": "births per woman", "min": 0.7, "max": 8.0, "step": 0.01, "default": 2.8,
    },
    "urban_pop": {
        "label": "Urban Population",
        "unit": "% of total population", "min": 8.0, "max": 100.0, "step": 0.1, "default": 58.0,
    },
    "co2_per_capita": {
        "label": "CO2 Emissions",
        "unit": "metric tons per capita", "min": 0.0, "max": 210.0, "step": 0.1, "default": 5.0,
    },
}


def classify_hdi(score: float) -> str:
    if score >= 0.800:
        return "Very High"
    elif score >= 0.700:
        return "High"
    elif score >= 0.550:
        return "Medium"
    else:
        return "Low"


TIER_INFO = {
    "Very High": {
        "color": "#C9A24B",
        "message": "This profile places the country among the most developed "
                   "nations, with strong outcomes across health, education, "
                   "and income working together.",
    },
    "High": {
        "color": "#2E7D74",
        "message": "This profile reflects solid development with room to "
                   "close remaining gaps, typically in income or education "
                   "depth, to reach the top tier.",
    },
    "Medium": {
        "color": "#5B7DB1",
        "message": "This profile shows moderate development. Targeted "
                   "investment in healthcare, schooling, or income "
                   "generation could meaningfully lift outcomes.",
    },
    "Low": {
        "color": "#B4544A",
        "message": "This profile indicates significant developmental "
                   "challenges. Coordinated intervention across health, "
                   "education, and economic policy is likely needed.",
    },
}


@app.route("/")
def home():
    return render_template("index.html", metrics=METRICS)


def parse_inputs(form):
    """Parse and validate the form. Returns (values, error_message)."""
    values = {}
    for key, meta in FIELD_META.items():
        raw = form.get(key, "").strip()
        try:
            number = float(raw)
        except (TypeError, ValueError):
            return None, f"Please enter a valid number for {meta['label']}."
        if not math.isfinite(number):
            return None, f"{meta['label']} must be a finite number."
        if not (meta["min"] <= number <= meta["max"]):
            return None, (f"{meta['label']} must be between {meta['min']:g} "
                          f"and {meta['max']:g} {meta['unit']}.")
        values[key] = number
    return values, None


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", fields=FIELD_META, values=None)

    values, error = parse_inputs(request.form)
    if error:
        # Keep whatever the user typed so they don't have to start over.
        return render_template(
            "predict.html", fields=FIELD_META, values=request.form.to_dict(),
            error=error,
        ), 400

    # The model was fitted on plain arrays, so feed it a plain array too
    # (avoids scikit-learn's "X has feature names" warning).
    X_input = np.array([[values[f] for f in FEATURES]], dtype=float)
    if USES_SCALER:
        X_input = scaler.transform(X_input)
    raw_pred = float(model.predict(X_input)[0])
    hdi_score = float(np.clip(raw_pred, 0, 1))
    category = classify_hdi(hdi_score)
    tier = TIER_INFO[category]

    return render_template(
        "result.html",
        values=values,
        fields=FIELD_META,
        hdi_score=round(hdi_score, 3),
        category=category,
        tier=tier,
        gauge_pct=round(hdi_score * 100, 1),
    )


@app.route("/insights")
def insights():
    return render_template("insights.html", metrics=METRICS)


@app.route("/about")
def about():
    return render_template("about.html", metrics=METRICS)


if __name__ == "__main__":
    # Debug mode (which exposes an interactive console) is OFF by default and
    # the server only listens on localhost. Opt in with:  FLASK_DEBUG=1 python app.py
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="127.0.0.1", port=5000)
