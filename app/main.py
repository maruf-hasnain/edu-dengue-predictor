from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.features import FEATURE_LABELS, build_features, classify_risk
from src.train import train_and_export

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "dengue_pipeline.joblib"

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

RISK_ADVISORIES = {
    "Low": "Routine vector surveillance and standard community sanitation checks recommended.",
    "Medium": "Increase larvicide inspection in stagnant water zones and monitor ward fever clinics.",
    "High": "Deploy rapid-response fogging teams, clear drainage blockages, and issue ward advisories.",
    "Critical": "Activate municipal outbreak protocol: emergency vector control, hospital surge readiness, and daily case tracking.",
}


class PredictRequest(BaseModel):
    ward_id: str = Field(..., description="Target municipal ward identifier")
    target_date: str = Field(..., description="Forecast week date in YYYY-MM-DD format")
    avg_temp_c: float = Field(..., ge=0.0, le=55.0)
    max_temp_c: float = Field(..., ge=0.0, le=60.0)
    min_temp_c: float = Field(..., ge=-5.0, le=50.0)
    rainfall_mm: float = Field(..., ge=0.0, le=1000.0)
    relative_humidity_pct: float = Field(..., ge=0.0, le=100.0)
    prior_3w_avg_rainfall_mm: Optional[float] = Field(default=None, ge=0.0, le=1000.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not ARTIFACT_PATH.exists():
        print(f"Artifact not found at {ARTIFACT_PATH}. Training pipeline automatically...")
        train_and_export()

    try:
        app.state.artifact = joblib.load(ARTIFACT_PATH)
        print(f"Loaded dengue pipeline artifact from {ARTIFACT_PATH}")
    except Exception as exc:
        print(
            f"Artifact load failed ({exc}); regenerating compatible artifact locally..."
        )
        train_and_export()
        app.state.artifact = joblib.load(ARTIFACT_PATH)
        print(f"Loaded regenerated dengue pipeline artifact from {ARTIFACT_PATH}")
    yield
    app.state.artifact = None


app = FastAPI(
    title="Urban Dengue Outbreak Early-Warning System",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_artifact(request: Request) -> dict[str, Any]:
    artifact = getattr(request.app.state, "artifact", None)
    if artifact is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")
    return artifact


def _format_feature_value(feature: str, raw_value: Any) -> str:
    if isinstance(raw_value, (int, float, np.integer, np.floating)):
        val = float(raw_value)
        if feature == "population_density_sqkm":
            return f"{int(round(val)):,}/km²"
        if "rainfall" in feature:
            return f"{val:.1f} mm"
        if "temp" in feature:
            return f"{val:.1f} °C"
        if "humidity" in feature:
            return f"{val:.1f}%"
        return f"{val:.2f}"
    return str(raw_value)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    artifact = _get_artifact(request)
    ward_metadata = artifact["ward_metadata"]
    metrics = artifact["metrics"]

    wards_by_city: dict[str, list[str]] = {}
    for ward_id in sorted(ward_metadata.keys()):
        city = ward_metadata[ward_id]["city"]
        wards_by_city.setdefault(city, []).append(ward_id)

    ward_list = sorted(ward_metadata.keys())
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "ward_list": ward_list,
            "wards_by_city": wards_by_city,
            "ward_metadata": ward_metadata,
            "metrics": metrics,
        },
    )


@app.get("/health")
async def health_check(request: Request):
    loaded = getattr(request.app.state, "artifact", None) is not None
    return {"status": "ok" if loaded else "loading", "artifact_loaded": loaded}


@app.get("/api/ward/{ward_id}")
async def get_ward_details(ward_id: str, request: Request):
    artifact = _get_artifact(request)
    ward_metadata = artifact["ward_metadata"]
    if ward_id not in ward_metadata:
        raise HTTPException(status_code=404, detail=f"Ward '{ward_id}' not found.")

    history = artifact["recent_history"].get(ward_id, [])
    if not history:
        raise HTTPException(
            status_code=404, detail=f"No recent history found for ward '{ward_id}'."
        )

    latest = history[-1]
    latest_date = datetime.strptime(latest["date"], "%Y-%m-%d")
    suggested_target_date = (latest_date + timedelta(days=7)).strftime("%Y-%m-%d")

    prior_3w = history[-3:]
    prior_3w_avg_rain = round(
        sum(float(row["rainfall_mm"]) for row in prior_3w) / len(prior_3w), 1
    )

    return {
        "ward_id": ward_id,
        "metadata": ward_metadata[ward_id],
        "defaults": {
            "target_date": suggested_target_date,
            "avg_temp_c": round(float(latest["avg_temp_c"]), 2),
            "max_temp_c": round(float(latest["max_temp_c"]), 2),
            "min_temp_c": round(float(latest["min_temp_c"]), 2),
            "rainfall_mm": round(float(latest["rainfall_mm"]), 2),
            "relative_humidity_pct": round(float(latest["relative_humidity_pct"]), 2),
            "prior_3w_avg_rainfall_mm": prior_3w_avg_rain,
        },
        "recent_history": history,
    }


@app.get("/api/validation-chart")
async def get_validation_chart(request: Request):
    artifact = _get_artifact(request)
    return {
        "metrics": artifact["metrics"],
        "series": artifact["weekly_test_chart"],
    }


@app.post("/api/predict")
async def predict_outbreak(payload: PredictRequest, request: Request):
    artifact = _get_artifact(request)
    ward_metadata = artifact["ward_metadata"]
    if payload.ward_id not in ward_metadata:
        raise HTTPException(status_code=404, detail=f"Ward '{payload.ward_id}' not found.")

    try:
        target_dt = datetime.strptime(payload.target_date, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail="target_date must be a valid date in YYYY-MM-DD format."
        ) from exc

    if payload.min_temp_c > payload.max_temp_c:
        raise HTTPException(
            status_code=422, detail="min_temp_c cannot be greater than max_temp_c."
        )

    history = artifact["recent_history"].get(payload.ward_id, [])
    if len(history) < 4:
        raise HTTPException(
            status_code=500,
            detail=f"Insufficient historical weeks for ward '{payload.ward_id}'.",
        )

    meta = ward_metadata[payload.ward_id]
    recent_4 = history[-4:]
    latest_hist = recent_4[-1]
    temp_shift = payload.avg_temp_c - float(latest_hist["avg_temp_c"])
    humidity_shift = payload.relative_humidity_pct - float(
        latest_hist["relative_humidity_pct"]
    )

    rows: list[dict[str, Any]] = []

    # Build a chronologically aligned 5-week window (t-4w, t-3w, t-2w, t-1w, target_date)
    # so build_features() computes exact 1-to-4-week lags for any simulated forecast date.
    # Delta-shifting preserves week-to-week variability while keeping baseline defaults exact.
    for idx, hist_row in enumerate(recent_4):
        weeks_back = 4 - idx
        row_date = (target_dt - timedelta(days=7 * weeks_back)).strftime("%Y-%m-%d")
        rainfall_val = float(hist_row["rainfall_mm"])
        if payload.prior_3w_avg_rainfall_mm is not None:
            if weeks_back in (1, 2, 3):
                rainfall_val = float(payload.prior_3w_avg_rainfall_mm)
            elif weeks_back == 4:
                rainfall_val = round(float(payload.prior_3w_avg_rainfall_mm) * 0.85, 2)

        rows.append(
            {
                "date": row_date,
                "city": meta["city"],
                "ward_id": payload.ward_id,
                "population_density_sqkm": meta["population_density_sqkm"],
                "water_logging_risk_index": meta["water_logging_risk_index"],
                "sanitation_score": meta["sanitation_score"],
                "avg_temp_c": round(float(hist_row["avg_temp_c"]) + temp_shift, 2),
                "max_temp_c": round(float(hist_row["max_temp_c"]) + temp_shift, 2),
                "min_temp_c": round(float(hist_row["min_temp_c"]) + temp_shift, 2),
                "rainfall_mm": rainfall_val,
                "relative_humidity_pct": round(
                    min(100.0, max(0.0, float(hist_row["relative_humidity_pct"]) + humidity_shift)),
                    2,
                ),
                "dengue_cases": float(hist_row.get("dengue_cases", 0.0)),
            }
        )

    rows.append(
        {
            "date": payload.target_date,
            "city": meta["city"],
            "ward_id": payload.ward_id,
            "population_density_sqkm": meta["population_density_sqkm"],
            "water_logging_risk_index": meta["water_logging_risk_index"],
            "sanitation_score": meta["sanitation_score"],
            "avg_temp_c": payload.avg_temp_c,
            "max_temp_c": payload.max_temp_c,
            "min_temp_c": payload.min_temp_c,
            "rainfall_mm": payload.rainfall_mm,
            "relative_humidity_pct": payload.relative_humidity_pct,
            "dengue_cases": 0.0,
        }
    )

    window_df = pd.DataFrame(rows)
    feat_df = build_features(window_df)

    feat_df["city_cat"] = pd.Categorical(
        feat_df["city"], categories=artifact["city_categories"]
    )
    feat_df["ward_cat"] = pd.Categorical(
        feat_df["ward_id"], categories=artifact["ward_categories"]
    )

    feature_cols = artifact["feature_cols"]
    X_row = feat_df.iloc[[-1]][feature_cols]

    model = artifact["model"]
    raw_pred = float(model.predict(X_row)[0])
    predicted_cases = round(max(0.0, raw_pred), 2)
    risk_tier = classify_risk(predicted_cases)

    contribs = model.predict(X_row, pred_contrib=True)[0]
    feature_contribs = contribs[:-1]
    base_value = round(float(contribs[-1]), 4)

    ranked = sorted(
        zip(feature_cols, feature_contribs),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )[:5]

    max_abs = max((abs(float(c)) for _, c in ranked), default=1.0) or 1.0
    top_drivers = []
    for feat_name, contrib_val in ranked:
        c_float = float(contrib_val)
        raw_feat_val = X_row.iloc[0][feat_name]
        top_drivers.append(
            {
                "feature": feat_name,
                "label": FEATURE_LABELS.get(feat_name, feat_name),
                "value": _format_feature_value(feat_name, raw_feat_val),
                "contribution": round(c_float, 4),
                "bar_pct": round((abs(c_float) / max_abs) * 100.0, 1),
                "direction": "increases_risk" if c_float >= 0 else "decreases_risk",
            }
        )

    target_row = feat_df.iloc[-1]
    return {
        "ward_id": payload.ward_id,
        "city": meta["city"],
        "target_date": payload.target_date,
        "predicted_cases": predicted_cases,
        "risk_tier": risk_tier,
        "advisory": RISK_ADVISORIES[risk_tier],
        "base_value": base_value,
        "ward_metadata": meta,
        "lag_summary": {
            "rainfall_lag_1w": round(float(target_row["rainfall_lag_1w"]), 1),
            "rainfall_lag_2w": round(float(target_row["rainfall_lag_2w"]), 1),
            "rainfall_lag_3w": round(float(target_row["rainfall_lag_3w"]), 1),
            "rainfall_roll_3w": round(float(target_row["rainfall_roll_3w"]), 1),
        },
        "top_drivers": top_drivers,
    }
