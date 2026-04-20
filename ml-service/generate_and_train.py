"""
Generate synthetic dataset and train RandomForestRegressor for Market Fit Prediction.
Run this FIRST before starting the FastAPI service.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import json

# ─── Config ───────────────────────────────────────────────────────────────────
DATASET_PATH = "../dataset/market_fit_data.csv"
MODEL_PATH   = "../models/market_fit_model.joblib"
SCALER_PATH  = "../models/scaler.joblib"
META_PATH    = "../models/feature_meta.json"
N_SAMPLES    = 5000
RANDOM_STATE = 42

os.makedirs("../dataset", exist_ok=True)
os.makedirs("../models",  exist_ok=True)

np.random.seed(RANDOM_STATE)

# ─── Feature definitions (max 5 features, padded to length 5) ─────────────────
CATEGORIES = {
    "smartphone": {
        "features": ["ram_gb", "storage_gb", "battery_mah", "camera_mp", "display_size"],
        "ranges":   [(2, 16), (32, 1024), (2000, 6000), (8, 200), (4.5, 7.0)],
        "weights":  [0.25, 0.15, 0.20, 0.25, 0.15],
    },
    "laptop": {
        "features": ["ram_gb", "storage_gb", "processor_ghz", "battery_wh"],
        "ranges":   [(4, 64), (128, 2048), (1.5, 5.5), (30, 100)],
        "weights":  [0.30, 0.20, 0.35, 0.15],
    },
    "smartwatch": {
        "features": ["battery_life_days", "display_size_mm", "resolution_px", "connectivity_count"],
        "ranges":   [(1, 14), (30, 55), (300, 500), (1, 6)],
        "weights":  [0.30, 0.20, 0.25, 0.25],
    },
    "washing_machine": {
        "features": ["capacity_kg", "spin_speed_rpm", "energy_rating", "water_consumption_l"],
        "ranges":   [(5, 15), (400, 1800), (1, 5), (30, 90)],
        "weights":  [0.25, 0.30, 0.30, 0.15],
    },
}

MAX_FEATURES = 5   # pad all vectors to this length

# ─── Synthetic score formula ────────────────────────────────────────────────
def compute_score(category, values):
    """Deterministic weighted score with some noise."""
    meta   = CATEGORIES[category]
    ranges = meta["ranges"]
    weights = meta["weights"]
    score = 0.0
    for i, (lo, hi) in enumerate(ranges):
        norm = (values[i] - lo) / (hi - lo)
        # Invert water_consumption (lower = better)
        if category == "washing_machine" and i == 3:
            norm = 1 - norm
        # Invert energy_rating only if higher means worse
        score += weights[i] * np.clip(norm, 0, 1)
    raw = score * 80 + 10                           # map [0,1] → [10,90]
    noise = np.random.normal(0, 4)
    return float(np.clip(raw + noise, 0, 100))

# ─── Generate dataset ──────────────────────────────────────────────────────
rows = []
per_cat = N_SAMPLES // len(CATEGORIES)

for cat, meta in CATEGORIES.items():
    ranges = meta["ranges"]
    for _ in range(per_cat):
        vals = [np.random.uniform(lo, hi) for lo, hi in ranges]
        score = compute_score(cat, vals)
        # Pad to MAX_FEATURES
        padded = vals + [0.0] * (MAX_FEATURES - len(vals))
        rows.append({
            "category": cat,
            **{f"f{i}": padded[i] for i in range(MAX_FEATURES)},
            "market_fit_score": score,
        })

df = pd.DataFrame(rows)
df.to_csv(DATASET_PATH, index=False)
print(f"Dataset saved → {DATASET_PATH}  ({len(df)} rows)")

# ─── Prepare features ──────────────────────────────────────────────────────
# One-hot encode category
cat_dummies = pd.get_dummies(df["category"], prefix="cat")
X = pd.concat([df[[f"f{i}" for i in range(MAX_FEATURES)]], cat_dummies], axis=1)
y = df["market_fit_score"]

feature_columns = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ─── Scale ─────────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─── Train ─────────────────────────────────────────────────────────────────
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=3,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model.fit(X_train_s, y_train)

preds = model.predict(X_test_s)
mae   = mean_absolute_error(y_test, preds)
r2    = r2_score(y_test, preds)
print(f"MAE: {mae:.2f}  |  R²: {r2:.4f}")

# ─── Save artifacts ────────────────────────────────────────────────────────
joblib.dump(model,  MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

meta_info = {
    "feature_columns": feature_columns,
    "categories": list(CATEGORIES.keys()),
    "category_features": {k: v["features"] for k, v in CATEGORIES.items()},
    "max_features": MAX_FEATURES,
}
with open(META_PATH, "w") as f:
    json.dump(meta_info, f, indent=2)

print(f"Model  saved → {MODEL_PATH}")
print(f"Scaler saved → {SCALER_PATH}")
print(f"Meta   saved → {META_PATH}")
print("✅ Training complete!")
