"""
train_model.py
==============
Fase 2 – Baseline training script untuk AcmeTel Churn Prediction.

Pipeline:
    raw CSV  →  preprocess  →  ColumnTransformer  →  LogisticRegression
                                (impute + scale)      (class_weight=balanced)

Output:
    models/churn_logreg_pipeline.joblib

Usage:
    python -m src.models.train_model
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Paths — naik 2 level dari src/models/ ke project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH    = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR    = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH   = MODEL_DIR / "churn_logreg_pipeline.joblib"


def load_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def preprocess_for_training(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """
    Membersihkan dan mempersiapkan DataFrame mentah untuk training.

    Steps:
        1. Drop kolom identifier (customerID).
        2. Encode target: Churn Yes/No → ChurnFlag 1/0.
        3. Convert TotalCharges ke numeric; isi NaN dengan 0 (customer baru, tenure=0).
        4. Pisahkan fitur (X) dan target (y).
        5. Identifikasi kolom numerik dan kategorikal secara otomatis.

    Returns:
        X                   : DataFrame fitur siap pakai.
        y                   : Series target (ChurnFlag).
        numeric_features    : List nama kolom numerik.
        categorical_features: List nama kolom kategorikal.
    """
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # 2) Target encoding: Churn -> ChurnFlag (1/0)
    df["ChurnFlag"] = df["Churn"].map({"Yes": 1, "No": 0})

    # 3) TotalCharges: convert to numeric, fill NaN (tenure=0) with 0
    df["TotalCharges_clean"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges_clean"] = df["TotalCharges_clean"].fillna(0)

    # 4) Feature / target split
    y = df["ChurnFlag"]

    # Drop kolom yang tidak akan jadi fitur
    drop_cols = ["Churn", "ChurnFlag", "TotalCharges", "tenure_bin"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 5) Identify numeric & categorical columns
    numeric_features     = ["tenure", "MonthlyCharges", "TotalCharges_clean"]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    return X, y, numeric_features, categorical_features


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """
    Membangun full sklearn Pipeline: preprocessing + model.

    Preprocessing:
        - Numerik  : SimpleImputer(median) → StandardScaler
        - Kategori : SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown='ignore')

    Model:
        LogisticRegression(class_weight='balanced', max_iter=1000)

    Returns:
        Pipeline siap di-fit.
    """
    # Numeric pipeline
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Model — n_jobs dihapus (deprecated sejak sklearn 1.8)
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
    )

    # Full pipeline
    model = Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])
    return model


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Mencetak metrik evaluasi lengkap ke stdout.

    Metrics: Accuracy, ROC AUC, Classification Report, Confusion Matrix.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print("=== Evaluation on Test Set ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC AUC  : {roc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("(Rows=Actual, Cols=Predicted | 0=No Churn, 1=Churn)")


def train_and_save() -> None:
    """Entry point: load data, train pipeline, evaluate, dan simpan model ke disk."""
    print(f"[1/6] Loading data from : {DATA_PATH}")
    df = load_raw_data(DATA_PATH)
    print(f"      Rows: {df.shape[0]:,}  |  Cols: {df.shape[1]}")

    print("[2/6] Preprocessing data...")
    X, y, numeric_features, categorical_features = preprocess_for_training(df)
    print(f"      Numeric features    : {numeric_features}")
    print(f"      Categorical features: {categorical_features}")
    print(f"      Target distribution :\n{y.value_counts().to_string()}")

    print("[3/6] Splitting train/test (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"      Train: {X_train.shape[0]:,} rows  |  Test: {X_test.shape[0]:,} rows")

    print("[4/6] Building pipeline (ColumnTransformer + LogisticRegression)...")
    model = build_pipeline(numeric_features, categorical_features)

    print("[5/6] Training model...")
    model.fit(X_train, y_train)
    print("      Training complete.")

    print("[6/6] Evaluating model on test set...")
    evaluate_model(model, X_test, y_test)

    print(f"\nSaving model pipeline to: {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved ({MODEL_PATH.stat().st_size / 1024:.1f} KB). Done!")


if __name__ == "__main__":
    train_and_save()
