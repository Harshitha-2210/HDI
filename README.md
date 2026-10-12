# HDI Predictor

A machine-learning web application that predicts the Human Development
Index (HDI) of a country from ten socioeconomic and infrastructure
indicators — electricity access, internet users, water access, health
expenditure, female labor force participation, unemployment, infant
mortality, fertility rate, urban population, and CO2 emissions — and
classifies the result into a development tier (Low, Medium, High, Very High).

## Project structure

```
HDI-main/
├── app.py                     # Flask application
├── requirements.txt
├── .gitignore
├── data/
│   ├── generate_dataset.py    # builds synthetic data/hdi_dataset_synthetic.csv
│   ├── hdi_dataset.csv        # real-world historical HDI data (4,484 rows, 193 countries)
│   └── hdi_dataset_synthetic.csv  # (optional) created by generate_dataset.py
├── notebooks/
│   └── eda.py                 # exploratory data analysis → static/images
├── model/
│   ├── train_model.py         # preprocessing + training + serialization
│   ├── hdi_model.pkl          # trained model artifact
│   ├── scaler.pkl             # fitted StandardScaler used by the model
│   └── metrics.json           # R², MAE, RMSE, feature importances
├── templates/                 # Jinja2 HTML templates
└── static/
    ├── css/style.css
    ├── js/main.js
    └── images/                # EDA figures (PNG)
```

## Setup

Requires Python 3.10 or newer. Run these from the project folder.

**Windows (PowerShell)**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python model\train_model.py
```

If PowerShell blocks the activate script, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python model/train_model.py
```

Your prompt should start with `(venv)` once the environment is active. The
`train_model.py` step takes under a minute; it rebuilds `model/hdi_model.pkl`
for the scikit-learn version you just installed (pickled models are tied to
the library version that created them).

## Rebuilding the EDA and model (optional)

The repo ships with EDA figures and a trained model. Retrain after installing
(see Setup) so the model matches your scikit-learn version. To regenerate the
visualizations and retrain manually:

```bash
python notebooks/eda.py           # rebuild static/images/*.png
python model/train_model.py       # retrain and re-pickle the model
```

To generate synthetic data instead (for reference):

```bash
python data/generate_dataset.py   # build synthetic data/hdi_dataset_synthetic.csv
```

## Run the app

```bash
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

The server listens on localhost only and runs with Flask debug mode **off**.
For local development with auto-reload, run `FLASK_DEBUG=1 python app.py`
(Windows PowerShell: `$env:FLASK_DEBUG=1; python app.py`). Never enable debug
mode on a machine reachable by other people.

## Pages

- **Home** – overview, live model metrics, and the tier legend
- **Predict** – enter a country's indicators and get an HDI prediction
- **Insights** – EDA charts (distributions, strip plots, heatmap, scatter
  plots) and model feature importances
- **Methodology** – how the pipeline was built and how the target is
  predicted from socioeconomic indicators

## About the dataset

The project uses real-world historical HDI data with 4,484 country-year observations.
The model trains on ten realistic socioeconomic and infrastructure indicators that
avoid target leakage and formula fitting. This approach keeps the project academically
defensible because the model learns from genuine correlations rather than deriving
HDI from its own component variables.

If you want to swap in a different dataset, keep the same ten feature names and the
`hdi` output (HDI score 0–1), then rerun `model/train_model.py`.

## Model

- **Algorithms compared:** Linear Regression (standardized) and Random Forest Regressor
- **Features:** electricity_access, internet_users, water_access, health_exp_gdp,
  female_labor_force, unemployment, infant_mortality, fertility_rate, urban_pop,
  co2_per_capita
- **Target:** hdi (continuous, 0–1)
- **Split:** ~80/20 **grouped by country** (`GroupShuffleSplit`). The dataset has
  ~23 yearly rows per country, so a plain random split would put the same
  country in both train and test and overstate accuracy (R² ≈ 0.99).
- **Imputation:** median, fitted on the training rows only
- **Evaluation:** R², MAE, RMSE on the held-out countries, plus 5-fold
  `GroupKFold` cross-validation. Current results: **R² ≈ 0.92, MAE ≈ 0.033**
  (5-fold CV on unseen countries); the single 39-country hold-out scores
  R² ≈ 0.96, MAE ≈ 0.026. Treat the 5-fold figure as the headline number.
- **Artifact selection:** the better model on the hold-out is saved to
  `model/hdi_model.pkl` (~18 MB)

### Known limitations

- Inputs are validated against the ranges seen in the training data
  (`FIELD_META` in `app.py`); predictions outside typical ranges are not
  reliable because a Random Forest cannot extrapolate.
- Electricity access carries ~64% of the forest's feature importance. Many
  countries sit at 100%, so for high-development countries the model leans on
  infant mortality and the other indicators.
- Pickled models must be loaded with the library versions in
  `requirements.txt` (scikit-learn 1.4.2). If you upgrade, rerun
  `python model/train_model.py`.
