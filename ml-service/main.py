"""
FastAPI ML Microservice — Market Fit Predictor
Port: 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Dict, Any
import joblib
import json
import numpy as np
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE, "../models/market_fit_model.joblib")
SCALER_PATH = os.path.join(BASE, "../models/scaler.joblib")
META_PATH   = os.path.join(BASE, "../models/feature_meta.json")

# ─── Load artifacts ───────────────────────────────────────────────────────────
def load_artifacts():
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, META_PATH]):
        raise RuntimeError(
            "Model artifacts missing. Run `python generate_and_train.py` first."
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, scaler, meta

model, scaler, meta = load_artifacts()
FEATURE_COLUMNS  = meta["feature_columns"]
CATEGORY_FEATURES = meta["category_features"]
MAX_FEATURES     = meta["max_features"]
VALID_CATEGORIES = list(CATEGORY_FEATURES.keys())

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Market Fit Predictor — ML Service",
    description="Predicts Market Fit Score (0–100) for electronic products.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Schemas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    category: str
    features: Dict[str, float]

    @validator("category")
    def validate_category(cls, v):
        v = v.lower().strip().replace(" ", "_")
        if v not in VALID_CATEGORIES:
            raise ValueError(f"category must be one of {VALID_CATEGORIES}")
        return v

    @validator("features")
    def validate_features(cls, v, values):
        for key, val in v.items():
            if not isinstance(val, (int, float)):
                raise ValueError(f"Feature '{key}' must be numeric")
            if val < 0:
                raise ValueError(f"Feature '{key}' must be non-negative")
        return v

class PredictResponse(BaseModel):
    market_fit_score: float
    category: str
    features_used: Dict[str, float]

# ─── Helper ───────────────────────────────────────────────────────────────────
def build_feature_vector(category: str, features: Dict[str, float]) -> np.ndarray:
    """Convert {feature_name: value} → padded & one-hot feature array."""
    expected_keys = CATEGORY_FEATURES[category]
    # Raw numeric values (padded to MAX_FEATURES)
    raw = []
    for key in expected_keys:
        raw.append(features.get(key, 0.0))
    while len(raw) < MAX_FEATURES:
        raw.append(0.0)

    # One-hot for category
    cat_cols = [c for c in FEATURE_COLUMNS if c.startswith("cat_")]
    one_hot  = [1.0 if col == f"cat_{category}" else 0.0 for col in cat_cols]

    vector = raw + one_hot
    return np.array(vector, dtype=float).reshape(1, -1)

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "categories": VALID_CATEGORIES}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        vec = build_feature_vector(req.category, req.features)
        if vec.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(
                f"Feature vector length mismatch: got {vec.shape[1]}, "
                f"expected {len(FEATURE_COLUMNS)}"
            )
        vec_scaled = scaler.transform(vec)
        score = float(model.predict(vec_scaled)[0])
        score = round(max(0.0, min(100.0, score)), 2)
        return PredictResponse(
            market_fit_score=score,
            category=req.category,
            features_used=req.features,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/categories")
def get_categories():
    return {"categories": VALID_CATEGORIES, "features": CATEGORY_FEATURES}
