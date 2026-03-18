from pathlib import Path
from typing import List, Any, Dict
import json
import logging
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ── Logging setup ─────────────────────────────────────────────────────────────
# Pastikan folder logs ada
import os
os.makedirs("api/logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api/logs/churn_api.log")
    ]
)
logger = logging.getLogger(__name__)

# ── Paths & loading artefacts ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "api" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "churn_best_pipeline.joblib"
FEATURES_META_PATH = MODEL_DIR / "churn_features.json"

import urllib.request
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_URL = "https://github.com/jackq707/acmetel-churn-prediction-api/releases/download/v1.0.0/churn_best_pipeline.joblib"
if not BEST_MODEL_PATH.exists():
    logger.info(f"Downloading model from {MODEL_URL}...")
    urllib.request.urlretrieve(MODEL_URL, BEST_MODEL_PATH)
    logger.info("Model downloaded successfully.")

# Load pipeline (XGBoost champion model)
pipeline = joblib.load(BEST_MODEL_PATH)
logger.info(f"Loaded model from {BEST_MODEL_PATH}")

# Load features metadata (harus dibuat dari compare_models.py)
with open(FEATURES_META_PATH, "r") as f:
    features_meta = json.load(f)

ALL_FEATURES: List[str] = features_meta.get("all_features", [])
NUMERIC_FEATURES: List[str] = features_meta.get("numeric", [])
CATEGORICAL_FEATURES: List[str] = features_meta.get("categorical", [])

logger.info(f"Loaded {len(ALL_FEATURES)} features: {len(NUMERIC_FEATURES)} numeric, {len(CATEGORICAL_FEATURES)} categorical")

# ── API Key Security ──────────────────────────────────────────────────────────
API_KEYS = set(filter(None, os.getenv("API_KEYS", "acmetel-dev-key-2026").split(",")))
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key or api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key. Include X-API-Key header.")
    return api_key

# ── Rate Limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

# ── Pydantic models ───────────────────────────────────────────────────────────
class ChurnRequest(BaseModel):
    data: Dict[str, Any] = Field(
        ...,
        description="Dictionary of feature_name: value, must include all required features",
        example={
            "gender": "Female",
            "SeniorCitizen": 0,
            "tenure": 5,
            "Contract": "Month-to-month",
            "MonthlyCharges": 80.0,
            "TotalCharges_clean": 400.0
        }
    )

class ChurnBatchRequest(BaseModel):
    items: List[ChurnRequest]

class ChurnResponse(BaseModel):
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_flag: int = Field(..., ge=0, le=1)
    model_version: str = "xgboost-v1"

class ChurnBatchResponse(BaseModel):
    results: List[ChurnResponse]

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AcmeTel Churn Prediction API",
    dependencies=[Security(verify_api_key)],
    version="1.0.0",
    description="Production-ready API for customer churn prediction using XGBoost pipeline (ROC AUC: 0.8423)",
    tags_metadata=[
        {
            "name": "meta",
            "description": "Health checks and service metadata"
        },
        {
            "name": "prediction", 
            "description": "Churn prediction endpoints"
        }
    ]
)

# ── Helper functions ──────────────────────────────────────────────────────────
def validate_and_build_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Validate all required features present and build DataFrame"""
    missing_features = []
    for feat in ALL_FEATURES:
        if any(feat not in rec for rec in records):
            missing_features.append(feat)
    
    if missing_features:
        logger.warning(f"Missing features requested: {missing_features}")
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features in some records: {missing_features}"
        )
    
    df = pd.DataFrame(records)
    df = df.reindex(columns=ALL_FEATURES, fill_value=None)
    return df

def predict_proba_for_df(df: pd.DataFrame) -> List[float]:
    """Run inference on DataFrame using loaded pipeline"""
    proba = pipeline.predict_proba(df)[:, 1]
    return proba.tolist()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["meta"])
def health_check():
    """Service health check with model & features info"""
    return {
        "status": "ok",
        "model_loaded": True,
        "model_path": str(BEST_MODEL_PATH),
        "model_size_kb": BEST_MODEL_PATH.stat().st_size / 1024,
        "n_features": len(ALL_FEATURES),
        "numeric_features_count": len(NUMERIC_FEATURES),
        "categorical_features_count": len(CATEGORICAL_FEATURES),
        "features_sample": ALL_FEATURES[:5] + ["..."]
    }

@app.post("/predict", response_model=ChurnResponse, tags=["prediction"])
def predict(body: ChurnRequest):
    """Predict churn probability for single customer"""
    logger.info(f"Predict request: {len(body.data)} features")
    
    record = body.data
    missing = [f for f in ALL_FEATURES if f not in record]
    if missing:
        logger.warning(f"Missing features: {missing}")
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {missing}"
        )
    
    df = validate_and_build_dataframe([record])
    proba_list = predict_proba_for_df(df)
    prob = proba_list[0]
    
    churn_flag = 1 if prob >= 0.5 else 0
    logger.info(f"Prediction: prob={prob:.3f}, flag={churn_flag}")
    
    return ChurnResponse(
        churn_probability=round(prob, 4),
        churn_flag=churn_flag,
        model_version="xgboost-v1"
    )

@app.post("/predict_batch", response_model=ChurnBatchResponse, tags=["prediction"])
def predict_batch(body: ChurnBatchRequest):
    """Predict churn probability for batch of customers"""
    logger.info(f"Batch predict: {len(body.items)} customers")
    
    records = [item.data for item in body.items]
    if not records:
        raise HTTPException(status_code=400, detail="No items provided.")
    
    df = validate_and_build_dataframe(records)
    proba_list = predict_proba_for_df(df)
    
    results = []
    for i, prob in enumerate(proba_list):
        churn_flag = 1 if prob >= 0.5 else 0
        results.append(
            ChurnResponse(
                churn_probability=round(prob, 4),
                churn_flag=churn_flag,
                model_version="xgboost-v1"
            )
        )
    
    logger.info(f"Batch complete: {len(results)} predictions")
    return ChurnBatchResponse(results=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
