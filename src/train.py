from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from src.features import build_features, classify_risk

DATA_PATH = Path("data/dengue.csv")
ARTIFACT_PATH = Path("artifacts/dengue_pipeline.joblib")
TWEEDIE_POWER = 1.3


def make_model(power: float = TWEEDIE_POWER) -> LGBMRegressor:
    return LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=power,
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )


def train_and_export() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH.resolve()}")

    print(f"Loading dataset from {DATA_PATH}...")
    raw_df = pd.read_csv(DATA_PATH)
    df_feat = build_features(raw_df).dropna().reset_index(drop=True)

    df_feat["city_cat"] = df_feat["city"].astype("category")
    df_feat["ward_cat"] = df_feat["ward_id"].astype("category")

    ignore_cols = {"date", "city", "ward_id", "dengue_cases"}
    feature_cols = [c for c in df_feat.columns if c not in ignore_cols]

    train_df = df_feat[df_feat["date"] < "2024-01-01"]
    val_df = df_feat[(df_feat["date"] >= "2024-01-01") & (df_feat["date"] < "2025-01-01")]
    train_val_df = df_feat[df_feat["date"] < "2025-01-01"]
    test_df = df_feat[df_feat["date"] >= "2025-01-01"]

    val_model = make_model(TWEEDIE_POWER)
    val_model.fit(train_df[feature_cols], train_df["dengue_cases"])
    val_preds = np.clip(val_model.predict(val_df[feature_cols]), 0, None)
    val_rmse = round(float(root_mean_squared_error(val_df["dengue_cases"], val_preds)), 3)
    print(f"Validation 2024 (power={TWEEDIE_POWER}) -> RMSE: {val_rmse}")

    final_model = make_model(TWEEDIE_POWER)
    final_model.fit(train_val_df[feature_cols], train_val_df["dengue_cases"])

    y_test = test_df["dengue_cases"]
    test_preds = np.clip(final_model.predict(test_df[feature_cols]), 0, None)

    test_mae = round(float(mean_absolute_error(y_test, test_preds)), 3)
    test_rmse = round(float(root_mean_squared_error(y_test, test_preds)), 3)
    test_r2 = round(float(r2_score(y_test, test_preds)), 3)

    actual_tiers = [classify_risk(v) for v in y_test]
    pred_tiers = [classify_risk(v) for v in test_preds]
    tier_acc = round(float(accuracy_score(actual_tiers, pred_tiers)) * 100.0, 1)

    print(
        f"Untouched 2025 Test -> MAE: {test_mae} | RMSE: {test_rmse} | "
        f"R2: {test_r2} | 4-Tier Accuracy: {tier_acc}%"
    )

    ward_metadata = (
        raw_df[
            [
                "city",
                "ward_id",
                "population_density_sqkm",
                "water_logging_risk_index",
                "sanitation_score",
            ]
        ]
        .drop_duplicates("ward_id")
        .set_index("ward_id")
        .to_dict(orient="index")
    )

    sorted_raw = raw_df.copy()
    sorted_raw["date"] = pd.to_datetime(sorted_raw["date"])
    sorted_raw = sorted_raw.sort_values(["ward_id", "date"])

    recent_history = {}
    for ward_id, group in sorted_raw.groupby("ward_id"):
        tail = group.tail(4).copy()
        tail["date"] = tail["date"].dt.strftime("%Y-%m-%d")
        recent_history[ward_id] = tail[
            [
                "date",
                "avg_temp_c",
                "max_temp_c",
                "min_temp_c",
                "rainfall_mm",
                "relative_humidity_pct",
                "dengue_cases",
            ]
        ].to_dict(orient="records")

    eval_df = test_df[["date", "ward_id", "dengue_cases"]].copy()
    eval_df["predicted_cases"] = np.round(test_preds, 2)
    weekly_eval = (
        eval_df.groupby("date", as_index=False)[["dengue_cases", "predicted_cases"]]
        .sum()
        .sort_values("date")
    )
    weekly_eval["date"] = weekly_eval["date"].dt.strftime("%Y-%m-%d")

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": final_model,
        "feature_cols": feature_cols,
        "city_categories": df_feat["city_cat"].cat.categories.tolist(),
        "ward_categories": df_feat["ward_cat"].cat.categories.tolist(),
        "ward_metadata": ward_metadata,
        "recent_history": recent_history,
        "weekly_test_chart": weekly_eval.to_dict(orient="records"),
        "metrics": {
            "model_name": f"LightGBM (Tweedie, p={TWEEDIE_POWER})",
            "val_2024_rmse": val_rmse,
            "test_2025_mae": test_mae,
            "test_2025_rmse": test_rmse,
            "test_2025_r2": test_r2,
            "test_2025_tier_accuracy_pct": tier_acc,
            "train_period": "2016-2023",
            "val_period": "2024",
            "test_period": "2025",
        },
    }
    joblib.dump(artifact, ARTIFACT_PATH)
    print(f"Saved model bundle -> {ARTIFACT_PATH}")


if __name__ == "__main__":
    train_and_export()