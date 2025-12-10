<!-- README.md - HTML-wrapped, copy-button-ready -->

<h1 align="center">Advanced Energy Forecasting (Dash)</h1>

<p align="center">
  <strong>End-to-end energy consumption forecasting system</strong> using
  <code>XGBoost</code> & <code>LightGBM</code>, with a Dash dashboard, model
  versioning, forecasting export, and a REST prediction API.
</p>

---

## 🔎 Project Highlights

- Interactive **Dash** dashboard for visualization & forecasting  
- Walk-forward cross validation (time series)  
- XGBoost & LightGBM models (final training + CV)  
- Recursive multi-step forecasting (hourly horizon)  
- Model versioning + `manifest.json` metadata  
- Forecast CSV export & download endpoints  
- REST API endpoint: **`POST /api/predict`**  
- Optional SHAP explainability for feature importance  
- No upload UI — loads `delhi_energy_2yr_zone.csv` directly from project folder

---

## 📁 Repo structure

```text
ml-energy-consumption-app/
├─ app.py
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ delhi_energy_2yr_zone.csv  (place locally — not committed)
└─ models_adv/                (generated at runtime; ignored)
⚙️ Installation & Setup
<div>
bash
Copy code
# 1) clone the repository
git clone https://github.com/sheebanadeem/ml-energy-consumption-app.git
cd ml-energy-consumption-app
</div> <div>
bash
Copy code
# 2) create & activate a virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate
</div> <div>
bash
Copy code
# 2b) create & activate a virtual environment (macOS / Linux)
python3 -m venv venv
source venv/bin/activate
</div> <div>
bash
Copy code
# 3) install dependencies
pip install -r requirements.txt
</div>
📥 Add the dataset
Place your CSV file next to app.py:

<div>
text
Copy code
delhi_energy_2yr_zone.csv
</div>
The dataset is intentionally listed in .gitignore so it remains local. If you want to include a sample dataset for demo purposes, create a small sample_data.csv and commit that instead.

▶️ Run the app
<div>
bash
Copy code
python app.py
</div>
Open your browser:

<div>
text
Copy code
http://127.0.0.1:8050/
</div>
🧭 How to use (quick)
Open the app in your browser.

Select date range (or leave default).

Click Train Models to run walk-forward CV and final model fits.

Training status appears in the top-right (polling).

Models saved to models_adv/models_YYYYMMDD_HHMMSS.pkl and manifest.json updated.

Click Forecast & Download to generate next N-hour forecast — latest_forecast.csv saved.

Use /download_model, /download_manifest, /download_latest_forecast for downloads.

🔌 API — programmatic predictions
Endpoint

POST /api/predict

Payload

json
Copy code
{
  "features": [
    {"lag_1": 120, "lag_24": 90, "hour": 14, "dayofweek": 2, "temp_c": 30},
    {"lag_1": 110, "lag_24": 95, "hour": 15, "dayofweek": 2, "temp_c": 30}
  ],
  "model": "lgb"   // optional, "xgb" or "lgb"
}
cURL example

<div>
bash
Copy code
curl -X POST http://localhost:8050/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[{"lag_1":120,"lag_24":90,"hour":14}],"model":"lgb"}'
</div>
Response

json
Copy code
{"predictions":[3456.78]}
📊 Plots & explainability
Model leaderboard shows CV/holdout RMSE & MAE for XGBoost and LightGBM.

SHAP summary (if shap is installed) shows mean |SHAP value| for top features.
If SHAP is not installed or too slow on CPU, the app falls back to showing a message.

🧾 Saved artifacts
models_adv/models_<timestamp>.pkl → versioned model bundle (xgb, lgb, scaler, feature list)

models_adv/manifest.json → metadata for model runs (metrics, created_at, params)

models_adv/latest_forecast.csv → last generated forecast CSV

✅ .gitignore (recommended)
<div>
text
Copy code
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
*.log
.DS_Store
.vscode/
.idea/
models_adv/
*.pkl
*.joblib
latest_forecast.csv
delhi_energy_2yr_zone.csv
</div>
🛠 Troubleshooting & tips
If training seems stuck, check models_adv/train_status.json for current status.

If LightGBM/XGBoost complains about early-stopping kwargs, the app uses robust wrappers that retry compatible signatures.

If you accidentally committed the large CSV, remove it with:

<div>
bash
Copy code
git rm --cached delhi_energy_2yr_zone.csv
git commit -m "Remove dataset from repo"
git push
</div>
📝 Resume / Project description (copy-paste friendly)
Short (1 line)

Energy forecasting dashboard + API using XGBoost & LightGBM, walk-forward CV, recursive multi-step forecasting, and model versioning (Dash).

Medium (2 lines)

Developed an end-to-end energy consumption forecasting system featuring feature engineering, walk-forward CV, XGBoost & LightGBM models, SHAP explainability, a Dash dashboard, and a REST /api/predict endpoint. Implemented model versioning and downloadable forecasts for deployment-ready delivery.

🔮 Future improvements
Add MLflow or Weights & Biases for experiment tracking

Dockerize (Gunicorn + Nginx) for deployment

Add authentication & role-based access to the dashboard/API

Use TimescaleDB/Postgres for scalable time-series storage & scheduled retraining

Add unit tests and CI (GitHub Actions) for reproducibility



👩‍💻 Author
Sheeba Nadeem — GitHub: https://github.com/sheebanadeem

<p align="center"><em>If you want, I can also add badges, screenshots section (with image links), or a diagram (drawn as SVG) showing architecture — tell me which and I’ll update the README.</em></p> ```
I












ChatGPT 
