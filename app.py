# app.py
"""
Very-Advanced Energy Forecasting Dash app (cleaned)
- Uses local CSV: 'delhi_energy_2yr_zone.csv' (must be in same folder)
- Features:
    * XGBoost + LightGBM training with robust compatibility wrappers
    * Walk-forward CV
    * Optional Optuna tuning (if optuna installed)
    * Model versioning + manifest.json
    * Training status polling via dcc.Interval (poll_status writes to train-status-div)
    * SHAP explanation (if shap installed)
    * REST /api/predict endpoint
    * Download endpoints for model, manifest, forecast
    * NO upload functionality — reads CSV from disk only
"""
import os
import io
import json
import traceback
import logging
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb

# Optional libraries
try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Dash stack
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from flask import send_file, request, jsonify

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("energy_app")

# ---------------- Config ----------------
DATA_PATH = "delhi_energy_2yr_zone.csv"
MODEL_DIR = "models_adv"
os.makedirs(MODEL_DIR, exist_ok=True)
MANIFEST_PATH = os.path.join(MODEL_DIR, "manifest.json")
TRAIN_STATUS_PATH = os.path.join(MODEL_DIR, "train_status.json")
TARGET = "total_mw"

# ---------------- Utilities ----------------
def save_manifest_entry(entry):
    manifest = []
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r") as f:
                manifest = json.load(f)
        except Exception:
            manifest = []
    manifest.append(entry)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

def write_train_status(status: dict):
    try:
        with open(TRAIN_STATUS_PATH, "w") as f:
            json.dump(status, f, indent=2, default=str)
    except Exception:
        logger.exception("Failed to write train status")

def read_train_status():
    if os.path.exists(TRAIN_STATUS_PATH):
        try:
            with open(TRAIN_STATUS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

# ---------------- Data & features ----------------
def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}. Place CSV beside this script.")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def default_date_range(df):
    return df["timestamp"].min().date(), df["timestamp"].max().date()

def feature_columns(df):
    exclude = {"timestamp", TARGET, "season"}
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    preferred = [
        "lag_1","lag_24","lag_168","hour","dayofweek","is_weekend","month",
        "temp_c","humidity_pct","heat_index","roll_mean_24","roll_std_24"
    ]
    chosen = [c for c in preferred if c in cols] + [c for c in cols if c not in preferred]
    return chosen

# ---------------- Metrics ----------------
def evaluate(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = float(mean_absolute_error(y_true, y_pred))
    try:
        rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}

# ---------------- Robust fit helpers ----------------
def safe_xgb_fit(model, X_train, y_train, X_valid=None, y_valid=None, early_stopping_rounds=20):
    try:
        if X_valid is not None and y_valid is not None:
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=early_stopping_rounds, verbose=False)
        else:
            model.fit(X_train, y_train, verbose=False)
    except TypeError:
        logger.warning("safe_xgb_fit: early_stopping not supported; retrying without early_stopping_rounds")
        try:
            if X_valid is not None and y_valid is not None:
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            else:
                model.fit(X_train, y_train, verbose=False)
        except Exception:
            model.fit(X_train, y_train)
    return model

def safe_lgb_fit(model, X_train, y_train, X_valid=None, y_valid=None, early_stopping_rounds=30):
    # try several signature variants to be compatible across lightgbm versions
    try:
        if X_valid is not None and y_valid is not None:
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=early_stopping_rounds, verbose=False)
        else:
            model.fit(X_train, y_train, verbose=False)
        return model
    except TypeError as e:
        logger.warning("safe_lgb_fit: first signature failed (%s). Retrying without verbose.", str(e))
    try:
        if X_valid is not None and y_valid is not None:
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=early_stopping_rounds)
        else:
            model.fit(X_train, y_train)
        return model
    except TypeError as e:
        logger.warning("safe_lgb_fit: second signature failed (%s). Retrying without early_stopping_rounds.", str(e))
    try:
        if X_valid is not None and y_valid is not None:
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
        else:
            model.fit(X_train, y_train)
        return model
    except Exception as e:
        logger.warning("safe_lgb_fit: third attempt failed (%s). Final fallback to plain fit.", str(e))
    model.fit(X_train, y_train)
    return model

# ---------------- Cross-validation & tuning ----------------
def walk_forward_cv_score(model_type, X, y, n_splits=4, update_status=None):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []
    last_model = None
    fold = 0
    for train_idx, test_idx in tscv.split(X):
        fold += 1
        if update_status:
            update_status(f"CV fold {fold}/{n_splits}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if model_type == "xgb":
            m = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
                                 objective="reg:squarederror", random_state=42, verbosity=0, n_jobs=4)
            m = safe_xgb_fit(m, X_train, y_train, X_test, y_test, early_stopping_rounds=20)
        elif model_type == "lgb":
            m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=4)
            m = safe_lgb_fit(m, X_train, y_train, X_test, y_test, early_stopping_rounds=30)
        else:
            raise ValueError("unknown model_type")
        preds = m.predict(X_test)
        metrics_list.append(evaluate(y_test, preds))
        last_model = m
    avg = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
    return avg, last_model

def optuna_tune(model_type, X, y, n_trials=20, timeout=None):
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna is not installed.")
    def objective(trial):
        if model_type == "xgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
            m = xgb.XGBRegressor(**params, objective="reg:squarederror", random_state=42, verbosity=0, n_jobs=4)
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1200),
                "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
            }
            m = lgb.LGBMRegressor(**params, random_state=42, n_jobs=4)
        # simple time-split validation
        n = len(X)
        split = max(int(0.8 * n), 1)
        X_train, X_valid = X.iloc[:split], X.iloc[split:]
        y_train, y_valid = y.iloc[:split], y.iloc[split:]
        try:
            if model_type == "xgb":
                m = safe_xgb_fit(m, X_train, y_train, X_valid, y_valid, early_stopping_rounds=10)
            else:
                m = safe_lgb_fit(m, X_train, y_train, X_valid, y_valid, early_stopping_rounds=15)
            preds = m.predict(X_valid)
            return evaluate(y_valid, preds)["RMSE"]
        except Exception:
            return float("inf")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study

# ---------------- Recursive forecasting ----------------
def recursive_forecast_improved(df_recent, model, feat_cols, horizon=24):
    work = df_recent.copy().reset_index(drop=True)
    preds = []
    for h in range(horizon):
        X = work[feat_cols].iloc[[-1]].copy()
        pred = model.predict(X)[0]
        preds.append(pred)
        new_row = work.iloc[[-1]].copy()
        new_row[TARGET] = pred
        if "timestamp" in work.columns:
            new_ts = pd.to_datetime(new_row["timestamp"].iloc[0]) + pd.Timedelta(hours=1)
            new_row["timestamp"] = new_ts
            new_row["hour"] = new_ts.hour
            new_row["dayofweek"] = new_ts.dayofweek
            new_row["is_weekend"] = int(new_row["dayofweek"].iloc[0] >= 5)
            new_row["month"] = new_ts.month
        for lag in [1,24,168]:
            col = f"lag_{lag}"
            if col in work.columns:
                if lag == 1:
                    new_row[col] = work[TARGET].iloc[-1]
                else:
                    if len(work) >= lag:
                        new_row[col] = work[TARGET].iloc[-lag]
                    else:
                        new_row[col] = work[TARGET].iloc[-1]
        if "roll_mean_24" in work.columns:
            seq = pd.concat([work[TARGET], pd.Series([pred])], ignore_index=True)
            new_row["roll_mean_24"] = seq.shift(1).rolling(24, min_periods=1).mean().iloc[-1]
        if "roll_std_24" in work.columns:
            seq = pd.concat([work[TARGET], pd.Series([pred])], ignore_index=True)
            new_row["roll_std_24"] = seq.shift(1).rolling(24, min_periods=1).std().fillna(0).iloc[-1]
        work = pd.concat([work, new_row], ignore_index=True)
    last_ts = pd.to_datetime(df_recent["timestamp"].iloc[-1])
    forecast_index = [last_ts + pd.Timedelta(hours=i+1) for i in range(len(preds))]
    return pd.DataFrame({"timestamp": forecast_index, "forecast_mw": preds})

# ---------------- Dash app ----------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Energy Forecasting — Advanced"), md=8),
        dbc.Col(dbc.Button("Train Models", id="train-btn", color="primary"), md=2),
        dbc.Col(dbc.Button("Forecast & Download", id="forecast-btn", color="secondary"), md=2),
    ], align="center", className="my-2"),

    dbc.Row([
        dbc.Col(html.Label("Select date range:"), md=2),
        dbc.Col(dcc.DatePickerRange(id="date-range"), md=5),
        dbc.Col(html.Div(id="train-status-div", children="Train status: idle"), md=5)
    ], className="mb-2"),

    dbc.Row([
        dbc.Col(html.Label("Forecast horizon (hours)"), md=2),
        dbc.Col(dcc.Input(id="horizon", type="number", value=24, min=1, max=336), md=2),
        dbc.Col(html.Div([dbc.Button("Download latest model", id="dl-model-btn", color="info"),
                          html.A("Download manifest", href="/download_manifest", target="_blank", style={"marginLeft":"8px"})]), md=8)
    ], className="mb-2"),

    dbc.Row([ dbc.Col(dcc.Graph(id="ts-graph"), md=12) ]),
    dbc.Row([ dbc.Col(dcc.Graph(id="model-leaderboard"), md=6), dbc.Col(dcc.Graph(id="shap-summary"), md=6) ]),
    dbc.Row([ dbc.Col(dcc.Graph(id="forecast-plot"), md=12) ]),
    dbc.Row([ dbc.Col(html.Div(id="forecast-status-div"), md=12) ]),

    dcc.Store(id="store-data"),
    dcc.Store(id="store-model-path"),
    dcc.Interval(id="status-interval", interval=2*1000, n_intervals=0)
], fluid=True)

# ---------------- Callbacks ----------------
@app.callback(
    Output("store-data", "data"),
    Output("date-range", "start_date"),
    Output("date-range", "end_date"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date")
)
def load_data_store(s, e):
    df = load_data()
    default_start, default_end = default_date_range(df)
    if s is None or e is None:
        s, e = default_start, default_end
    mask = (df["timestamp"].dt.date >= pd.to_datetime(s).date()) & (df["timestamp"].dt.date <= pd.to_datetime(e).date())
    dff = df.loc[mask].copy()
    if dff.empty:
        dff = df.tail(24*30).copy()
    dff["timestamp"] = dff["timestamp"].astype(str)
    return dff.to_dict(orient="records"), str(s), str(e)

@app.callback(Output("ts-graph", "figure"), Input("store-data", "data"))
def update_ts(data):
    df = pd.DataFrame(data)
    if df.empty:
        fig = go.Figure(); fig.add_annotation(text="No data for selected range", showarrow=False, x=0.5, y=0.5)
        return fig
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["total_mw"], name="Total MW", fill="tozeroy"))
    for z in ["north_mw","south_mw","east_mw","west_mw"]:
        if z in df.columns:
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df[z], name=z, opacity=0.7))
    fig.update_layout(title="Consumption (selected range)", height=420)
    return fig

@app.callback(
    Output("store-model-path", "data"),
    Output("model-leaderboard", "figure"),
    Output("shap-summary", "figure"),
    Input("train-btn", "n_clicks"),
    State("store-data", "data"),
    prevent_initial_call=True
)
def train_models(n_clicks, data):
    ts_start = datetime.utcnow().isoformat()
    write_train_status({"status":"starting","started_at":ts_start,"message":"Starting training"})
    try:
        df = pd.DataFrame(data)
        if df.empty:
            write_train_status({"status":"error","message":"No data for training"})
            fig_err = go.Figure(); fig_err.add_annotation(text="No data for training", showarrow=False)
            return None, fig_err, fig_err

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        feats = feature_columns(df)
        numeric_feats = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_feats:
            write_train_status({"status":"error","message":"No numeric features"})
            raise ValueError("No numeric features available.")

        X = df[numeric_feats].reset_index(drop=True).fillna(0)
        y = df[TARGET].reset_index(drop=True)
        if len(X) < 50:
            write_train_status({"status":"error","message":"Insufficient rows"})
            raise ValueError("Not enough rows to train (need >= 50).")

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=numeric_feats)

        def update_status(msg):
            write_train_status({"status":"training","message":msg,"last_update":datetime.utcnow().isoformat()})

        n_splits = 4 if len(X_scaled) > 500 else 3
        update_status("Running XGBoost CV")
        xgb_metrics, _ = walk_forward_cv_score("xgb", X_scaled, y, n_splits=n_splits, update_status=update_status)
        update_status("Running LightGBM CV")
        lgb_metrics, _ = walk_forward_cv_score("lgb", X_scaled, y, n_splits=n_splits, update_status=update_status)

        tuning_results = None
        if OPTUNA_AVAILABLE:
            update_status("Optuna tuning (XGBoost quick)")
            try:
                study = optuna_tune("xgb", X_scaled, y, n_trials=10)
                tuning_results = {"xgb_best_params": study.best_params, "xgb_best_value": study.best_value}
            except Exception as e:
                logger.exception("Optuna tuning failed: %s", e)

        n = len(X_scaled)
        test_n = max(int(0.15 * n), 1)
        train_n = n - test_n
        X_train, X_test = X_scaled.iloc[:train_n], X_scaled.iloc[train_n:]
        y_train, y_test = y.iloc[:train_n], y.iloc[train_n:]

        update_status("Fitting XGBoost final")
        xgb_final = xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=6,
                                     objective="reg:squarederror", random_state=42, verbosity=0, n_jobs=4)
        xgb_final = safe_xgb_fit(xgb_final, X_train, y_train, X_test, y_test, early_stopping_rounds=25)

        update_status("Fitting LightGBM final")
        lgb_final = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.03, num_leaves=31, random_state=42, n_jobs=4)
        lgb_final = safe_lgb_fit(lgb_final, X_train, y_train, X_test, y_test, early_stopping_rounds=50)

        x_pred = xgb_final.predict(X_test)
        l_pred = lgb_final.predict(X_test)
        x_eval = evaluate(y_test, x_pred)
        l_eval = evaluate(y_test, l_pred)
        leaderboard = pd.DataFrame([x_eval, l_eval], index=["XGBoost", "LightGBM"])

        fig_board = go.Figure()
        fig_board.add_trace(go.Bar(x=leaderboard.index, y=leaderboard["RMSE"], name="RMSE"))
        fig_board.add_trace(go.Bar(x=leaderboard.index, y=leaderboard["MAE"], name="MAE"))
        fig_board.update_layout(title="Model leaderboard (lower is better)")

        # SHAP summary (optional)
        if SHAP_AVAILABLE:
            try:
                explainer = shap.Explainer(xgb_final)
                sample_X = X_scaled.tail(min(200, len(X_scaled)))
                shap_values = explainer(sample_X)
                shap_df = pd.DataFrame({
                    "feature": sample_X.columns,
                    "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
                }).sort_values("mean_abs_shap", ascending=False).head(20)
                fig_shap = px.bar(shap_df, x="mean_abs_shap", y="feature", orientation="h", title="SHAP mean |value| (top 20)")
                fig_shap.update_layout(yaxis={"autorange":"reversed"})
            except Exception as e:
                logger.exception("SHAP error: %s", e)
                fig_shap = go.Figure(); fig_shap.add_annotation(text=f"SHAP error: {str(e)}", showarrow=False)
        else:
            fig_shap = go.Figure(); fig_shap.add_annotation(text="SHAP not installed", showarrow=False)

        # Save models with versioned filename
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_filename = f"models_{ts}.pkl"
        save_path = os.path.join(MODEL_DIR, model_filename)
        models_obj = {
            "xgb": {"model": xgb_final, "feats": numeric_feats},
            "lgb": {"model": lgb_final, "feats": numeric_feats},
            "scaler": scaler,
            "created_at": ts
        }
        joblib.dump(models_obj, save_path)
        logger.info(f"Models saved to {save_path}")

        entry = {
            "filename": model_filename,
            "created_at": ts,
            "xgb_cv": xgb_metrics,
            "lgb_cv": lgb_metrics,
            "final_xgb": x_eval,
            "final_lgb": l_eval,
            "optuna": tuning_results
        }
        save_manifest_entry(entry)
        write_train_status({"status":"done","message":"Training completed","model_path":save_path,"last_update":datetime.utcnow().isoformat()})

        return save_path, fig_board, fig_shap

    except Exception as e:
        logger.exception("Error in train_models callback: %s", e)
        write_train_status({"status":"error","message":str(e),"trace":traceback.format_exc()})
        fig_err = go.Figure(); fig_err.add_annotation(text=f"Training error: {str(e)}", showarrow=False)
        return None, fig_err, fig_err

@app.callback(
    Output("forecast-plot", "figure"),
    Output("forecast-status-div", "children"),
    Input("forecast-btn", "n_clicks"),
    State("store-data", "data"),
    State("store-model-path", "data"),
    State("horizon", "value"),
    prevent_initial_call=True
)
def do_forecast(n_clicks, data, models_path_or_none, horizon):
    try:
        # Determine model to use (explicit path preferred)
        if isinstance(models_path_or_none, str) and os.path.exists(models_path_or_none):
            models_obj = joblib.load(models_path_or_none)
        else:
            files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith("models_") and f.endswith(".pkl")])
            if not files:
                fig = go.Figure(); fig.add_annotation(text="No trained model found (train first)", showarrow=False)
                return fig, "Forecast status: no trained model"
            models_obj = joblib.load(os.path.join(MODEL_DIR, files[-1]))

        df = pd.DataFrame(data)
        if df.empty:
            fig = go.Figure(); fig.add_annotation(text="No data available for forecasting", showarrow=False)
            return fig, "Forecast status: no data"

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        feats = models_obj["xgb"]["feats"]
        scaler = models_obj["scaler"]
        model = models_obj["lgb"]["model"]  # use LightGBM for forecasting by default

        recent = df.tail(300).copy()
        X_recent = recent[feats].fillna(0)
        X_recent_scaled = pd.DataFrame(scaler.transform(X_recent), columns=feats)
        recent_scaled = recent.copy()
        recent_scaled[feats] = X_recent_scaled

        forecast_df = recursive_forecast_improved(recent_scaled, model, feat_cols=feats, horizon=int(horizon))

        forecast_path = os.path.join(MODEL_DIR, "latest_forecast.csv")
        forecast_df.to_csv(forecast_path, index=False)

        fig = go.Figure()
        recent_x = recent["timestamp"].tail(24*7)
        recent_y = recent["total_mw"].tail(24*7)
        fig.add_trace(go.Scatter(x=recent_x, y=recent_y, name="Recent (7d)"))
        fig.add_trace(go.Scatter(x=forecast_df["timestamp"], y=forecast_df["forecast_mw"], name="Forecast", line=dict(dash="dash")))
        fig.update_layout(title=f"Next {horizon}h Forecast")

        status_div = html.Div([
            f"Forecast generated ({forecast_path}). ",
            html.A("Download CSV", href="/download_latest_forecast", target="_blank")
        ])
        return fig, status_div

    except Exception as e:
        logger.exception("Error in forecasting callback: %s", e)
        fig_err = go.Figure(); fig_err.add_annotation(text=f"Forecast error: {str(e)}", showarrow=False)
        return fig_err, f"Forecast status: error - {str(e)}"

@app.callback(
    Output("train-status-div", "children"),
    Input("status-interval", "n_intervals")
)
def poll_status(n):
    st = read_train_status()
    if not st:
        return "Train status: idle"
    status = st.get("status","idle")
    msg = st.get("message","")
    last = st.get("last_update", st.get("started_at",""))
    return html.Div([html.Div(f"Train status: {status}"), html.Div(msg), html.Div(f"Updated: {last}")])

# ---------------- Routes ----------------
@server.route("/download_latest_forecast")
def download_latest_forecast():
    path = os.path.join(MODEL_DIR, "latest_forecast.csv")
    if not os.path.exists(path):
        buf = io.StringIO()
        pd.DataFrame().to_csv(buf)
        buf.seek(0)
        return send_file(BytesIO(buf.getvalue().encode()), mimetype="text/csv", as_attachment=True, download_name="latest_forecast.csv")
    return send_file(path, as_attachment=True, download_name="latest_forecast.csv", mimetype="text/csv")

@server.route("/download_model")
def download_model():
    files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith("models_") and f.endswith(".pkl")])
    if not files:
        return ("No model available", 404)
    path = os.path.join(MODEL_DIR, files[-1])
    return send_file(path, as_attachment=True, download_name=files[-1])

@server.route("/download_manifest")
def download_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return send_file(BytesIO(json.dumps([]).encode()), mimetype="application/json", as_attachment=True, download_name="manifest.json")
    return send_file(MANIFEST_PATH, as_attachment=True, download_name="manifest.json", mimetype="application/json")

@server.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        payload = request.get_json(force=True)
        if not payload or "features" not in payload:
            return jsonify({"error":"invalid payload"}), 400
        features = pd.DataFrame(payload["features"])
        model_pref = payload.get("model","lgb")
        files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith("models_") and f.endswith(".pkl")])
        if not files:
            return jsonify({"error":"no model"}), 400
        models_obj = joblib.load(os.path.join(MODEL_DIR, files[-1]))
        feat_cols = models_obj["xgb"]["feats"]
        missing = [c for c in feat_cols if c not in features.columns]
        if missing:
            return jsonify({"error":"missing features", "missing":missing}), 400
        scaler = models_obj["scaler"]
        X = features[feat_cols].fillna(0)
        Xs = scaler.transform(X)
        if model_pref == "xgb":
            preds = models_obj["xgb"]["model"].predict(Xs).tolist()
        else:
            preds = models_obj["lgb"]["model"].predict(Xs).tolist()
        return jsonify({"predictions": preds})
    except Exception as e:
        logger.exception("api_predict failed: %s", e)
        return jsonify({"error": str(e)}), 500

# ---------------- Run ----------------
if __name__ == "__main__":
    logger.info("Starting advanced Dash app...")
    app.run(debug=True, port=8050)
