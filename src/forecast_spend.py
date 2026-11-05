"""Step 4: forecast hourly spend 7 days ahead and compare three models.

Days 1-7 are lag history, 8-17 train XGBoost, 18-21 calibrate the band, 22-28 are the test.
No input is newer than 168 hours before the hour being predicted, because a forecast made
a week ahead wouldn't have newer data.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TRAIN = (7 * 24, 17 * 24)
CALIBRATE = (17 * 24, 21 * 24)
TEST = (21 * 24, 28 * 24)
FEATURES = ["hour_of_day", "day_of_week", "lag_168", "avg_24_ending_168"]


def add_features(spend):
    df = spend.copy()
    df["hour_of_day"] = df.hour % 24
    df["day_of_week"] = (df.hour // 24) % 7     # the trace's day 1 is day_of_week 0, not a known weekday
    df["lag_168"] = df.spend.shift(168)
    df["avg_24_ending_168"] = df.spend.shift(168).rolling(24, min_periods=1).mean()
    return df


def between(df, span):
    return df[df.hour.between(span[0], span[1] - 1)]


def metrics(actual, predicted):
    error = actual - predicted
    return {"MAE ($/hour)": np.abs(error).mean(), "MAPE (%)": (np.abs(error) / actual).mean() * 100}


def main():
    RESULTS.mkdir(exist_ok=True)
    df = add_features(pd.read_parquet(ROOT / "data" / "spend_hourly.parquet"))
    train, calibrate, test = between(df, TRAIN), between(df, CALIBRATE), between(df, TEST)

    history = df[df.hour < TEST[0]]
    linear = np.polyval(np.polyfit(history.hour, history.spend, 1), test.hour)
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(train[FEATURES], train.spend)
    xgb = model.predict(test[FEATURES])

    table = pd.DataFrame({
        "Linear projection": metrics(test.spend, linear),
        "Seasonal naive": metrics(test.spend, test.lag_168),
        "XGBoost": metrics(test.spend, xgb),
    }).T
    table["Improvement over linear (%)"] = (1 - table["MAPE (%)"] / table.loc["Linear projection", "MAPE (%)"]) * 100
    table.round(2).to_csv(RESULTS / "forecast_metrics.csv", index_label="model")
    print(table.round(2).to_string())

    # The dashboard shows the best model on calibration days, so the test week plays no part in choosing.
    candidates = {"XGBoost": model.predict(calibrate[FEATURES]), "Seasonal naive": calibrate.lag_168.to_numpy()}
    best = min(candidates, key=lambda name: np.abs(calibrate.spend - candidates[name]).mean())
    forecast = xgb if best == "XGBoost" else test.lag_168.to_numpy()
    band = np.quantile(np.abs(calibrate.spend - candidates[best]), 0.9)
    inside = ((test.spend >= forecast - band) & (test.spend <= forecast + band)).mean()
    print(f"best on calibration: {best}; band +/-${band:,.0f}; {inside:.0%} of test hours inside")

    pd.DataFrame({
        "hour": test.hour, "actual": test.spend, "forecast": forecast,
        "low": forecast - band, "high": forecast + band, "model": best,
    }).round(2).to_csv(RESULTS / "forecast.csv", index=False)
    pd.Series({
        "model": best, "band_dollars_per_hour": round(float(band), 2), "test_hours_inside_band": round(float(inside), 4),
    }).to_csv(RESULTS / "forecast_band.csv", header=["value"], index_label="item")


if __name__ == "__main__":
    main()
