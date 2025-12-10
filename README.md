<h1 align="center">⚡ Advanced Energy Forecasting — Dash Application</h1> <p align="center"> A complete end-to-end energy forecasting system using <strong>XGBoost</strong> and <strong>LightGBM</strong>, featuring a fully interactive <strong>Dash</strong> dashboard, walk-forward CV, recursive multi-step forecasting, model versioning, and downloadable artifacts. </p>
🚀 Features

Interactive Dash dashboard

Walk-forward cross-validation (time-series)

XGBoost + LightGBM training with automatic fallback for incompatible parameters

Recursive multi-step forecasting (24–336 hours)

SHAP optional explainability

Automatic model versioning

Downloadable:

Latest model

Manifest metadata

Forecast CSV

REST API endpoint: /api/predict

Loads delhi_energy_2yr_zone.csv directly from project folder (no upload needed)

📁 Repo Structure
ml-energy-consumption-app/
├─ app.py
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ delhi_energy_2yr_zone.csv   (local only, not committed)
└─ models_adv/                 (generated at runtime)

⚙️ Installation & Setup
1. Clone the repository
<div>
git clone https://github.com/sheebanadeem/ml-energy-consumption-app.git
cd ml-energy-consumption-app

</div>
2. Create & activate a virtual environment (Windows)
<div>
python -m venv venv
.\venv\Scripts\activate

</div>
macOS / Linux users:
<div>
python3 -m venv venv
source venv/bin/activate

</div>
3. Install dependencies
<div>
pip install -r requirements.txt

</div>
4. Ensure the dataset exists

Place your CSV file in the project folder:

delhi_energy_2yr_zone.csv

5. Run the Dash application
<div>
python app.py

</div>

Open in browser:

<div>
http://127.0.0.1:8050/

</div>
📊 Using the App (Quick Guide)
✔️ Train Models

Click Train Models

Walk-forward CV begins

Final models are saved in models_adv/

manifest.json updates automatically

✔️ Forecast

Choose forecast horizon

Click Forecast & Download

latest_forecast.csv is saved + downloadable

✔️ Visuals

Consumption graph

Leaderboard (RMSE, MAE)

SHAP summary (if installed)

Forecast plot

🔌 REST API — Predict Programmatically
Endpoint

POST /api/predict

Example payload
{
  "features": [
    {"lag_1": 120, "lag_24": 90, "hour": 14, "dayofweek": 2, "temp_c": 28}
  ],
  "model": "lgb"
}

Example cURL
<div>
curl -X POST http://localhost:8050/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"features\":[{\"lag_1\":120,\"lag_24\":90}],\"model\":\"lgb\"}"

</div>
🗂️ Model Artifacts

Saved automatically to models_adv/:

File	Description
models_<timestamp>.pkl	Full model bundle (xgb, lgb, scaler, features)
manifest.json	Metadata for versioning / lineage
latest_forecast.csv	Most recent generated forecast
🧾 Recommended .gitignore
<div>
venv/
__pycache__/
*.pyc
models_adv/
*.pkl
*.joblib
latest_forecast.csv
delhi_energy_2yr_zone.csv

</div>
📄 Resume-Ready Description (Copy & Paste)

Short (1–2 lines)

Built a production-grade energy forecasting system using XGBoost & LightGBM with walk-forward validation, SHAP explainability, recursive multi-step forecasting, and an interactive Dash dashboard.

Medium (3–4 lines)

Developed an advanced energy forecasting platform featuring automated feature scaling, walk-forward CV, XGBoost/LightGBM model selection, model versioning, and a REST API. Implemented recursive multi-step predictions, SHAP-based explainability, and a full Dash dashboard for visualization and forecasting export.

🔮 Future Enhancements

MLflow or W&B experiment tracking

Dockerization

Automated nightly retraining

Postgres/TimescaleDB time-series backend

Authentication for dashboard & API

👩‍💻 Author

Sheeba Nadeem
GitHub: https://github.com/sheebanadeem

<p align="center"><em>Want badges, screenshots, or an architecture diagram added? I can generate those too — just tell me!</em></p>
