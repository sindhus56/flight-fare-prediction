"""
flight_utils.py
---------------
Data cleaning, feature engineering, model loading and prediction helpers for the
Flight Price Prediction project (Developed by Sindhu S).

Everything in this file mirrors the steps of the Jupyter notebook
"PRCP-1025-Flight_Price_Prediction.ipynb" so that the Streamlit app behaves
exactly like the notebook:

  * cleaning      -> dropna, drop_duplicates, "No Info" -> "No info"
  * features      -> Duration_Minutes, Journey_Day/Month, Dep/Arrival Hour/Minute
  * encoding      -> Total_Stops mapped to 0-4, One-Hot (drop_first=True) for
                     Airline, Source, Destination, Additional_Info
  * split         -> train_test_split(test_size=0.2, random_state=42)
  * final model   -> Tuned RandomForestRegressor (GridSearchCV best params)

This module has NO Streamlit dependency, so it can be unit-tested on its own.
"""
from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Paths (put these files next to app.py)
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE_DIR / "Flight_Fare.xlsx",
    BASE_DIR / "data" / "Flight_Fare.xlsx",
]
MODEL_PATH = BASE_DIR / "flight_price_model.pkl"
FEATURES_PATH = BASE_DIR / "feature_columns.pkl"

# --------------------------------------------------------------------------- #
# Constants taken from the notebook
# --------------------------------------------------------------------------- #
STOP_MAPPING = {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}
STOP_LABELS = {v: k for k, v in STOP_MAPPING.items()}

# Best parameters found by GridSearchCV in the notebook
BEST_RF_PARAMS = dict(
    n_estimators=100,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
)

# "Final Model Performance Comparison" table from the notebook (test set)
NOTEBOOK_MODEL_RESULTS = pd.DataFrame(
    {
        "Model": [
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "Tuned Random Forest",
        ],
        "MAE": [1773.060050, 754.836757, 675.195388, 672.679340],
        "RMSE": [2656.287001, 1802.298118, 1489.327321, 1481.718330],
        "R2 Score": [0.661594, 0.844209, 0.893618, 0.894702],
    }
)

# Display names (the dataset spells Bangalore as "Banglore"; the model needs the
# original spelling, so we only change what the user *sees*).
CITY_DISPLAY = {"Banglore": "Bangalore"}


def city(name: str) -> str:
    """Pretty city name for the UI."""
    return CITY_DISPLAY.get(name, name)


# --------------------------------------------------------------------------- #
# Loading + cleaning
# --------------------------------------------------------------------------- #
def find_dataset() -> Path | None:
    """Return the first existing Flight_Fare.xlsx next to the app, else None."""
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    return None


def read_raw(source) -> pd.DataFrame:
    """Read the raw Excel file (path or file-like object)."""
    return pd.read_excel(source)


def convert_duration(duration: str) -> int:
    """'2h 50m' -> 170, '19h' -> 1140  (same function as the notebook)."""
    hours = 0
    minutes = 0
    if "h" in duration:
        hours = int(duration.split("h")[0])
    if "m" in duration:
        minutes = int(duration.split("h")[-1].replace("m", "").strip())
    return hours * 60 + minutes


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the notebook's `df_clean` (+ the engineered columns).
    Order of operations is identical to the notebook:
    dropna -> drop_duplicates -> fix 'No Info' -> feature extraction.
    """
    df = raw.copy()
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df["Additional_Info"] = df["Additional_Info"].replace("No Info", "No info")
    df.reset_index(drop=True, inplace=True)

    df["Duration_Minutes"] = df["Duration"].apply(convert_duration)

    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
    df["Journey_Day"] = df["Date_of_Journey"].dt.day
    df["Journey_Month"] = df["Date_of_Journey"].dt.month

    # "22:20" -> hour 22, minute 20   |   "01:10 22 Mar" -> hour 1, minute 10
    dep = df["Dep_Time"].astype(str).str.slice(0, 5)
    arr = df["Arrival_Time"].astype(str).str.slice(0, 5)
    df["Dep_Hour"] = dep.str.slice(0, 2).astype(int)
    df["Dep_Minute"] = dep.str.slice(3, 5).astype(int)
    df["Arrival_Hour"] = arr.str.slice(0, 2).astype(int)
    df["Arrival_Minute"] = arr.str.slice(3, 5).astype(int)
    return df


def build_ml_frame(df_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Reproduce the notebook's feature engineering -> (X, y)."""
    df_ml = df_clean.copy()
    df_ml.drop(
        columns=["Date_of_Journey", "Dep_Time", "Arrival_Time", "Duration", "Route"],
        inplace=True,
    )
    df_ml["Total_Stops"] = df_ml["Total_Stops"].map(STOP_MAPPING)
    for col in ["Airline", "Source", "Destination", "Additional_Info"]:
        df_ml = pd.get_dummies(df_ml, columns=[col], drop_first=True, dtype=int)
    X = df_ml.drop("Price", axis=1)
    y = df_ml["Price"]
    return X, y


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_feature_columns(fallback: list[str]) -> list[str]:
    """Use feature_columns.pkl from the notebook if present, else derive it."""
    if FEATURES_PATH.exists():
        try:
            cols = joblib.load(FEATURES_PATH)
            if isinstance(cols, (list, tuple)) and len(cols) > 0:
                return list(cols)
        except Exception:
            pass
    return list(fallback)


def load_or_train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Load `flight_price_model.pkl` (saved in the notebook). If the file is missing
    or cannot be loaded (e.g. scikit-learn version mismatch) train the same
    Tuned Random Forest again, using the notebook's best parameters.
    Returns (model, description_of_source).
    """
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            if getattr(model, "n_features_in_", None) == X_train.shape[1]:
                return model, f"Loaded saved model ({MODEL_PATH.name})"
        except Exception:
            pass
    model = RandomForestRegressor(**BEST_RF_PARAMS)
    model.fit(X_train, y_train)
    return model, "Trained at startup with the notebook's best parameters"


def prepare_bundle(df_clean: pd.DataFrame) -> dict:
    """Build features, split (same seed as notebook), get the model + metrics."""
    X, y = build_ml_frame(df_clean)
    feature_columns = load_feature_columns(list(X.columns))
    X = X.reindex(columns=feature_columns, fill_value=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model, source = load_or_train_model(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(mse**0.5),
        "r2": float(r2_score(y_test, y_pred)),
        "train_r2": float(model.score(X_train, y_train)),
    }
    importance = (
        pd.DataFrame({"Feature": feature_columns, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "model": model,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "importance": importance,
        "y_test": np.asarray(y_test),
        "y_pred": np.asarray(y_pred),
        "source": source,
    }


def pretty_feature(name: str) -> str:
    """'Airline_Jet Airways' -> 'Airline: Jet Airways', 'Dep_Hour' -> 'Dep Hour'."""
    for prefix in ("Airline", "Source", "Destination", "Additional_Info"):
        if name.startswith(prefix + "_"):
            label = "Info" if prefix == "Additional_Info" else prefix
            rest = name[len(prefix) + 1 :]
            if prefix in ("Source", "Destination"):
                rest = city(rest)
            return f"{label}: {rest}"
    return name.replace("_", " ")


# --------------------------------------------------------------------------- #
# Prediction (same logic as `predict_flight_price` in the notebook)
# --------------------------------------------------------------------------- #
def compute_arrival(dep_hour: int, dep_minute: int, duration_minutes: int):
    """Arrival clock time + how many days later the plane lands (0 = same day)."""
    base = datetime(2000, 1, 1, dep_hour, dep_minute)
    arr = base + timedelta(minutes=int(duration_minutes))
    return arr.hour, arr.minute, (arr.date() - base.date()).days


def build_features(feature_columns: list[str], rows: list[dict]) -> pd.DataFrame:
    """
    One-hot encode a list of flight dicts into the model's 36 feature columns.
    Each dict needs: airline, source, destination, total_stops, duration_minutes,
    journey_day, journey_month, dep_hour, dep_minute, arrival_hour,
    arrival_minute, additional_info.
    """
    X = pd.DataFrame(0, index=range(len(rows)), columns=feature_columns, dtype=int)
    for i, r in enumerate(rows):
        numeric = {
            "Total_Stops": r["total_stops"],
            "Duration_Minutes": r["duration_minutes"],
            "Journey_Day": r["journey_day"],
            "Journey_Month": r["journey_month"],
            "Dep_Hour": r["dep_hour"],
            "Dep_Minute": r["dep_minute"],
            "Arrival_Hour": r["arrival_hour"],
            "Arrival_Minute": r["arrival_minute"],
        }
        for col, val in numeric.items():
            if col in X.columns:
                X.at[i, col] = int(val)
        for prefix, value in (
            ("Airline_", r["airline"]),
            ("Source_", r["source"]),
            ("Destination_", r["destination"]),
            ("Additional_Info_", r["additional_info"]),
        ):
            col = prefix + str(value)
            if col in X.columns:  # dropped (first) category stays all-zero
                X.at[i, col] = 1
    return X


def predict_many(model, feature_columns: list[str], rows: list[dict]) -> np.ndarray:
    return model.predict(build_features(feature_columns, rows))


def predict_price(model, feature_columns: list[str], **kwargs) -> float:
    return float(predict_many(model, feature_columns, [kwargs])[0])


def make_row(
    airline,
    source,
    destination,
    travel_date: date,
    dep_hour,
    dep_minute,
    duration_minutes,
    total_stops,
    additional_info,
) -> dict:
    """Convenience: build the input dict for one flight (arrival is derived)."""
    ah, am, _ = compute_arrival(dep_hour, dep_minute, duration_minutes)
    return dict(
        airline=airline,
        source=source,
        destination=destination,
        total_stops=int(total_stops),
        duration_minutes=int(duration_minutes),
        journey_day=travel_date.day,
        journey_month=travel_date.month,
        dep_hour=int(dep_hour),
        dep_minute=int(dep_minute),
        arrival_hour=ah,
        arrival_minute=am,
        additional_info=additional_info,
    )


# --------------------------------------------------------------------------- #
# Data-driven helpers for the UI
# --------------------------------------------------------------------------- #
def route_map(df: pd.DataFrame) -> dict[str, list[str]]:
    """source -> destinations that actually exist in the data (most common first)."""
    out = {}
    for src, sub in df.groupby("Source"):
        out[src] = sub["Destination"].value_counts().index.tolist()
    return out


def route_subset(df: pd.DataFrame, source: str, destination: str) -> pd.DataFrame:
    return df[(df["Source"] == source) & (df["Destination"] == destination)]


def hhmm(minutes: float) -> str:
    minutes = int(round(minutes))
    return f"{minutes // 60}h {minutes % 60:02d}m"


def route_profile(df: pd.DataFrame, source: str, destination: str, airline: str | None = None) -> dict:
    """
    'Typical flight' on a route (medians from the dataset). Used by the Plan Better
    page, where the user does not enter duration/stops by hand.
    """
    sub = route_subset(df, source, destination)
    if airline is not None:
        a = sub[sub["Airline"] == airline]
        if len(a) >= 5:
            sub = a
    if sub.empty:
        sub = df
    dep_minutes = (sub["Dep_Hour"] * 60 + sub["Dep_Minute"]).median()
    stops = int(round(sub["Total_Stops"].map(STOP_MAPPING).median()))
    return dict(
        duration_minutes=int(round(sub["Duration_Minutes"].median())),
        total_stops=stops,
        dep_hour=int(dep_minutes // 60),
        dep_minute=int(dep_minutes % 60),
        n=len(sub),
    )


def quick_tips(df: pd.DataFrame) -> list[tuple[str, str]]:
    """A few insights computed from the dataset for the 'Plan Better' page."""
    tips: list[tuple[str, str]] = []

    month_avg = df.groupby("Journey_Month")["Price"].mean()
    if len(month_avg) >= 2:
        lo, hi = month_avg.idxmin(), month_avg.idxmax()
        tips.append(
            (
                "📅",
                f"{calendar.month_name[lo]} journeys were cheapest on average "
                f"(₹{month_avg[lo]:,.0f}) vs {calendar.month_name[hi]} (₹{month_avg[hi]:,.0f}).",
            )
        )

    stops_avg = df.assign(s=df["Total_Stops"].map(STOP_MAPPING)).groupby("s")["Price"].mean()
    if 0 in stops_avg.index and 1 in stops_avg.index:
        diff = stops_avg[1] - stops_avg[0]
        word = "cheaper" if diff < 0 else "costlier"
        tips.append(
            ("🛑", f"1-stop flights average ₹{abs(diff):,.0f} {word} than non-stop flights.")
        )

    bins = pd.cut(
        df["Dep_Hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["Night (00-06)", "Morning (06-12)", "Afternoon (12-18)", "Evening (18-24)"],
    )
    slot = df.groupby(bins, observed=True)["Price"].mean()
    if len(slot):
        tips.append(("🕒", f"{slot.idxmin()} departures have the lowest average fare (₹{slot.min():,.0f})."))

    air = df.groupby("Airline")["Price"].agg(["mean", "count"])
    air = air[air["count"] >= 100]
    if len(air):
        tips.append(("✈️", f"{air['mean'].idxmin()} has the lowest average fare among major airlines (₹{air['mean'].min():,.0f})."))
    return tips
