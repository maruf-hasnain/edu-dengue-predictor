import numpy as np
import pandas as pd

FEATURE_LABELS = {
    "rainfall_mm": "Current Week Rainfall (mm)",
    "avg_temp_c": "Current Avg Temp (°C)",
    "max_temp_c": "Current Max Temp (°C)",
    "min_temp_c": "Current Min Temp (°C)",
    "relative_humidity_pct": "Current Humidity (%)",
    "rainfall_lag_1w": "1-Week Lagged Rainfall",
    "rainfall_lag_2w": "2-Week Lagged Rainfall",
    "rainfall_lag_3w": "3-Week Lagged Rainfall",
    "rainfall_lag_4w": "4-Week Lagged Rainfall",
    "temp_lag_1w": "1-Week Lagged Temp",
    "temp_lag_2w": "2-Week Lagged Temp",
    "temp_lag_3w": "3-Week Lagged Temp",
    "temp_lag_4w": "4-Week Lagged Temp",
    "humidity_lag_1w": "1-Week Lagged Humidity",
    "humidity_lag_2w": "2-Week Lagged Humidity",
    "humidity_lag_3w": "3-Week Lagged Humidity",
    "humidity_lag_4w": "4-Week Lagged Humidity",
    "rainfall_roll_3w": "3-Week Cumulative Rainfall",
    "temp_roll_3w": "3-Week Avg Temperature",
    "humidity_roll_3w": "3-Week Avg Humidity",
    "temp_range_c": "Diurnal Temp Range",
    "week_sin": "Seasonal Cycle (Sin)",
    "week_cos": "Seasonal Cycle (Cos)",
    "water_logging_risk_index": "Ward Water-Logging Risk",
    "population_density_sqkm": "Ward Population Density",
    "sanitation_score": "Ward Sanitation Score",
    "rain_x_waterlogging_3w": "3W Rainfall × Water-Logging",
    "rain_x_density_3w": "3W Rainfall × Pop. Density",
    "rain_x_poor_sanitation_3w": "3W Rainfall × Poor Sanitation",
    "city_cat": "City Baseline",
    "ward_cat": "Ward Baseline",
}


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute ward-grouped 1-to-4-week weather lags and ward vulnerability interactions."""
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ward_id", "date"]).reset_index(drop=True)

    grouped = df.groupby("ward_id", group_keys=False)

    for lag in [1, 2, 3, 4]:
        df[f"rainfall_lag_{lag}w"] = grouped["rainfall_mm"].shift(lag)
        df[f"temp_lag_{lag}w"] = grouped["avg_temp_c"].shift(lag)
        df[f"humidity_lag_{lag}w"] = grouped["relative_humidity_pct"].shift(lag)

    df["rainfall_roll_3w"] = grouped["rainfall_mm"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=3).sum()
    )
    df["temp_roll_3w"] = grouped["avg_temp_c"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=3).mean()
    )
    df["humidity_roll_3w"] = grouped["relative_humidity_pct"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=3).mean()
    )

    df["temp_range_c"] = df["max_temp_c"] - df["min_temp_c"]
    week_of_year = df["date"].dt.isocalendar().week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * week_of_year / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * week_of_year / 52.0)

    df["rain_x_waterlogging_3w"] = df["rainfall_lag_3w"] * df["water_logging_risk_index"]
    df["rain_x_density_3w"] = df["rainfall_lag_3w"] * (df["population_density_sqkm"] / 10000.0)
    df["rain_x_poor_sanitation_3w"] = df["rainfall_lag_3w"] * (1.0 - df["sanitation_score"])

    return df


def classify_risk(predicted_cases: float) -> str:
    """Convert predicted weekly dengue cases into operational risk tiers."""
    if predicted_cases >= 10.0:
        return "Critical"
    if predicted_cases >= 5.0:
        return "High"
    if predicted_cases >= 2.0:
        return "Medium"
    return "Low"