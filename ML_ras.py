"""Predict the normalized peak from time since the previous feeding and time of day."""

import ctypes
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

_LIBOMP = Path(__file__).resolve().parent / ".venv/lib/python3.9/site-packages/sklearn/.dylibs/libomp.dylib"
if _LIBOMP.exists():
    ctypes.CDLL(str(_LIBOMP))

import xgboost as xgb

FEATURES = Path("/Volumes/Expansion/features and curves 141026/features.parquet")
OUT = Path(__file__).resolve().parent / "ml_ras_peak.png"
PREDICTORS = ["minutes_since_previous", "time_of_day_minutes", "feeding_of_day"]
TARGET = "peak_height"
TEST_FRACTION = 0.2


def add_time_features(frame):
    """Gap since the previous feeding, clock time, and which feeding of the day."""
    ordered = frame.sort_values(["channel", "feeding_datetime"]).copy()
    gap = ordered.groupby("channel")["feeding_datetime"].diff()
    ordered["minutes_since_previous"] = gap.dt.total_seconds() / 60.0
    clock = ordered["feeding_datetime"]
    ordered["time_of_day_minutes"] = clock.dt.hour * 60 + clock.dt.minute + clock.dt.second / 60.0
    day = clock.dt.normalize()
    ordered["feeding_of_day"] = ordered.groupby(["channel", day]).cumcount() + 1
    return ordered.dropna(subset=["minutes_since_previous"])


def chronological_split(frame):
    """Hold out the last fraction of unique feeding times, all channels together."""
    times = np.sort(frame["feeding_datetime"].unique())
    cut = int(np.floor(len(times) * (1.0 - TEST_FRACTION)))
    cut = min(max(cut, 1), len(times) - 1)
    train = frame[frame["feeding_datetime"] < times[cut]]
    test = frame[frame["feeding_datetime"] >= times[cut]]
    return train, test


def regression_models():
    return [
        ("Linear regression", LinearRegression()),
        ("Ridge", Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])),
        (
            "k-nearest neighbors",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", KNeighborsRegressor(n_neighbors=7, weights="distance")),
                ]
            ),
        ),
        (
            "Random forest",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=3,
                random_state=0,
            ),
        ),
        (
            "Gradient boosting",
            GradientBoostingRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                random_state=0,
            ),
        ),
        (
            "Support vector regression",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", SVR(C=10.0, epsilon=0.1)),
                ]
            ),
        ),
        (
            "XGBoost",
            xgb.XGBRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=1.0,
                objective="reg:squarederror",
                random_state=0,
            ),
        ),
    ]


def score_model(name, model, train, test):
    model.fit(train[PREDICTORS], train[TARGET])
    predicted = model.predict(test[PREDICTORS])
    actual = test[TARGET].to_numpy()
    return {
        "name": name,
        "actual": actual,
        "predicted": predicted,
        "mae": mean_absolute_error(actual, predicted),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": r2_score(actual, predicted),
    }


def main():
    raw = pd.read_parquet(FEATURES, columns=["channel", "feeding_datetime", TARGET])
    frame = add_time_features(raw.dropna(subset=[TARGET]))
    train, test = chronological_split(frame)

    print(f"Rows {len(frame)}  train {len(train)}  test {len(test)}")

    results = [score_model(name, model, train, test) for name, model in regression_models()]
    results.sort(key=lambda row: row["rmse"])
    for row in results:
        print(f"{row['name']}: MAE {row['mae']:.4f}  RMSE {row['rmse']:.4f}  R2 {row['r2']:.4f}")

    ncols = 4
    nrows = int(np.ceil(len(results) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows))
    for ax, row in zip(axes.ravel(), results):
        actual = row["actual"]
        predicted = row["predicted"]
        ax.scatter(actual, predicted, s=16, color="#54278F", alpha=0.75)
        lo = float(min(actual.min(), predicted.min()))
        hi = float(max(actual.max(), predicted.max()))
        ax.plot([lo, hi], [lo, hi], color="0.4", linewidth=1.0)
        ax.set_title(f"{row['name']}\nR2 {row['r2']:.3f}")
        ax.set_xlabel("Actual normalized peak")
        ax.set_ylabel("Predicted normalized peak")
        ax.grid(True, color="0.75", alpha=0.4)
    for ax in axes.ravel()[len(results) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
