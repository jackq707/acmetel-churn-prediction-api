# src/models/save_features_meta.py
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

# Features dari preprocessing (sama seperti di train_model.py)
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges_clean"]
categorical_features = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", 
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
    "Contract", "PaperlessBilling", "PaymentMethod"
]
all_features = numeric_features + categorical_features

features_meta = {
    "all_features": all_features,
    "numeric": numeric_features,
    "categorical": categorical_features,
    "total_features": len(all_features),
    "created": "2026-03-11"
}

meta_path = MODEL_DIR / "churn_features.json"
with open(meta_path, "w") as f:
    json.dump(features_meta, f, indent=2)

print(f"Saved features metadata to {meta_path}")

