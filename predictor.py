"""
predictor.py
─────────────────────────────────────────────────────────────────────────────
Reads aggregated sales history from SQLite and fits a simple Linear-Regression
model to project revenue for the next 60 minutes.

Public API
──────────
    forecast_df = predict_next_hour(db_path="business_data.db")

Returns a pandas DataFrame with columns:
    timestamp   – future minute bucket (datetime)
    predicted   – projected sale_amount (float)
    lower       – lower bound of a 1-σ prediction interval
    upper       – upper bound

Can also be run stand-alone for a quick sanity-check:
    python predictor.py
"""

import sqlite3
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")

DB_PATH = "business_data.db"
FORECAST_MINUTES = 60
MIN_ROWS = 10          # need at least this many history points


# ── Data loading ────────────────────────────────────────────────────────────

def load_minutely_revenue(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Aggregate transactions into 1-minute revenue buckets.
    Returns a DataFrame indexed by minute with column 'revenue'.
    """
    conn = sqlite3.connect(db_path)
    query = """
        SELECT
            strftime('%Y-%m-%d %H:%M', timestamp) AS minute,
            SUM(sale_amount)                       AS revenue,
            COUNT(*)                               AS orders
        FROM transactions
        GROUP BY minute
        ORDER BY minute
    """
    df = pd.read_sql_query(query, conn, parse_dates=["minute"])
    conn.close()
    df = df.set_index("minute").asfreq("min", fill_value=0)
    df.index = pd.to_datetime(df.index)
    return df


# ── Feature engineering ─────────────────────────────────────────────────────

def make_features(index: pd.DatetimeIndex) -> np.ndarray:
    """
    Convert a DatetimeIndex into numeric features suitable for regression:
      - elapsed minutes from the first observed timestamp (trend)
      - sin/cos encodings of hour-of-day (daily seasonality)
      - sin/cos encodings of minute-of-hour (intra-hour seasonality)
    """
    t0 = index[0]
    elapsed = np.array([(ts - t0).total_seconds() / 60 for ts in index])

    hour_rad  = 2 * np.pi * index.hour / 24
    min_rad   = 2 * np.pi * index.minute / 60

    X = np.column_stack([
        elapsed,
        np.sin(hour_rad), np.cos(hour_rad),
        np.sin(min_rad),  np.cos(min_rad),
    ])
    return X


# ── Model ───────────────────────────────────────────────────────────────────

def build_model() -> Pipeline:
    """Degree-2 polynomial features + Ridge regression."""
    return Pipeline([
        ("poly",  PolynomialFeatures(degree=2, include_bias=False)),
        ("ridge", Ridge(alpha=1.0)),
    ])


# ── Main forecast function ──────────────────────────────────────────────────

def predict_next_hour(
    db_path: str = DB_PATH,
    forecast_minutes: int = FORECAST_MINUTES,
) -> pd.DataFrame:
    """
    Fit a regression model on historical minutely revenue and return a
    60-minute forecast with prediction intervals.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    forecast_minutes : int
        How many minutes ahead to forecast (default 60).

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, predicted, lower, upper
    """
    history = load_minutely_revenue(db_path)

    if len(history) < MIN_ROWS:
        # Not enough data yet — return a flat forecast at the global mean
        avg = history["revenue"].mean() if len(history) > 0 else 100.0
        future_idx = pd.date_range(
            start=datetime.utcnow(),
            periods=forecast_minutes,
            freq="min",
        )
        return pd.DataFrame({
            "timestamp": future_idx,
            "predicted": avg,
            "lower":     avg * 0.8,
            "upper":     avg * 1.2,
        })

    # ── Fit ────────────────────────────────────────────────────────────────
    X_hist = make_features(history.index)
    y_hist = history["revenue"].values

    model = build_model()
    model.fit(X_hist, y_hist)

    # Estimate residual std for prediction intervals
    y_pred_train = model.predict(X_hist)
    residuals     = y_hist - y_pred_train
    sigma         = residuals.std() if residuals.std() > 0 else y_hist.mean() * 0.1

    # ── Forecast index ─────────────────────────────────────────────────────
    last_ts   = history.index[-1]
    future_idx = pd.date_range(
        start=last_ts + timedelta(minutes=1),
        periods=forecast_minutes,
        freq="min",
    )

    X_future   = make_features(
        # Prepend last historical ts so elapsed offset is computed correctly
        pd.DatetimeIndex([history.index[0]] + list(future_idx))
    )[1:]  # drop the prepended anchor row

    y_future = model.predict(X_future)

    # Clip to non-negative values
    y_future = np.clip(y_future, 0, None)

    forecast_df = pd.DataFrame({
        "timestamp": future_idx,
        "predicted": np.round(y_future, 2),
        "lower":     np.round(np.clip(y_future - sigma, 0, None), 2),
        "upper":     np.round(y_future + sigma, 2),
    })

    return forecast_df


# ── Stand-alone smoke-test ──────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[predictor] Loading history from '{DB_PATH}' …")
    try:
        df = predict_next_hour()
        print(f"[predictor] Forecast for next {FORECAST_MINUTES} minutes:\n")
        print(df.to_string(index=False))
    except Exception as exc:
        print(f"[predictor] Error: {exc}")
        print("  → Make sure data_generator.py has been running for at least a minute.")
