Advanced Energy Forecasting Dashboard (XGBoost + LightGBM)

An end-to-end energy consumption forecasting system built with Python, Dash, XGBoost, and LightGBM.
Includes recursive forecasting, walk-forward validation, SHAP explainability, model versioning, and a REST API.

Features

Interactive Dash dashboard

Time-series visualization (energy consumption by zone)

Walk-forward CV for robust model evaluation

XGBoost & LightGBM model training

Recursive multi-step forecasting

Model versioning (manifest.json + timestamped models)

Exportable forecast CSV

REST API (/api/predict)

Optional SHAP explainability

Clean production-style code structure

 Project Structure
ml-energy-consumption-app/
│── app.py
│── requirements.txt
│── README.md
│── .gitignore
│── delhi_energy_2yr_zone.csv        (ignored by git)
│── models_adv/                      (generated automatically)
│── latest_forecast.csv              (generated automatically)
│── manifest.json                    (generated automatically)

 Installation & Setup
 1) Clone the repository
git clone https://github.com/sheebanadeem/ml-energy-consumption-app.git
cd ml-energy-consumption-app

 2) Create & activate virtual environment
Windows
python -m venv venv
.\venv\Scripts\activate

macOS / Linux
python3 -m venv venv
source venv/bin/activate

 3) Install dependencies
pip install -r requirements.txt

4) Add the dataset (required)

Place your CSV in the same folder as app.py:

delhi_energy_2yr_zone.csv


This file is intentionally excluded from GitHub using .gitignore.

 Run the Application
python app.py


Then open:

 http://127.0.0.1:8050/

1 How It Works
 Load data automatically

The app loads delhi_energy_2yr_zone.csv on startup (no upload needed).

2️ Train models

Click the Train Models button:

Computes lag, rolling & calendar features

Scales numerical data

Runs XGBoost walk-forward CV

Runs LightGBM walk-forward CV

Trains final models

Saves model version → models_adv/models_YYYYMMDD_HHMMSS.pkl

Saves details into manifest.json

3️ View evaluation plots

Leaderboard (RMSE, MAE)

SHAP feature importance (if installed)

4️ Generate forecasts

Click Forecast & Download:

Performs recursive forecasting

Saves latest_forecast.csv

Displays interactive forecast plot

API Usage
POST /api/predict
curl -X POST http://localhost:8050/api/predict \
-H "Content-Type: application/json" \
-d "{ \"features\": [{\"lag_1\":120, \"lag_24\":90, \"hour\":14}], \"model\":\"lgb\" }"

Response:
{
  "predictions": [3456.78]
}

Example Forecast Output
timestamp           forecast_mw
2023-01-01 01:00    3321.42
2023-01-01 02:00    3378.11
...

Download Endpoints
/download_model
/download_manifest
/download_latest_forecast


Each returns a downloadable file.

 Technologies Used

Python

Dash + Bootstrap

Plotly

XGBoost

LightGBM

Scikit-learn

Pandas / NumPy

SHAP (optional)

Joblib

REST API (Flask inside Dash)

 .gitignore Included

✔ Prevents committing:

virtual environments

large CSV dataset

model files

forecast CSV

pycache

 Future Improvements

MLflow tracking

Automated retraining (CRON / GitHub Actions)

Docker deployment

Add authentication to API endpoints

Use a database (Postgres / TimescaleDB) instead of CSV

 Author

Sheeba Nadeem
GitHub: https://github.com/sheebanadeem

Project: ml-energy-consumption-app

