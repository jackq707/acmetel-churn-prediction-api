"""
compare_models.py
=================
Fase 2.5 – Model Comparison untuk AcmeTel Churn Prediction.

Membandingkan: Logistic Regression vs Random Forest vs XGBoost
Kriteria pemilihan: ROC AUC (prioritas recall kelas Churn).

Output:
    models/churn_best_pipeline.joblib   <- model terbaik (champion)
    models/churn_logreg_pipeline.joblib <- baseline LogReg (referensi)

Usage:
    python -m src.models.compare_models
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
from xgboost import XGBClassifier


# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH    = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR    = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading & preprocessing ─────────────────────────────────────────────
def load_and_prepare(
    path: Path,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """
    Load CSV dan persiapkan X, y, serta daftar fitur untuk training.

    Steps: drop customerID, encode ChurnFlag, fix TotalCharges, pisah X/y.

    Returns:
        X, y, numeric_features, categorical_features
    """
    df = pd.read_csv(path)
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["ChurnFlag"]          = df["Churn"].map({"Yes": 1, "No": 0})
    df["TotalCharges_clean"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    y        = df["ChurnFlag"]
    drop_cols = ["Churn", "ChurnFlag", "TotalCharges", "tenure_bin"]
    X        = df.drop(columns=[c for c in drop_cols if c in df.columns])

    numeric_features     = ["tenure", "MonthlyCharges", "TotalCharges_clean"]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    return X, y, numeric_features, categorical_features


# ── Preprocessor (shared across all models) ───────────────────────────────────
def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Buat ColumnTransformer untuk preprocessing numerik dan kategorikal.

    Numeric  : SimpleImputer(median) + StandardScaler
    Categorical: SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown='ignore')

    Note: Gunakan clone() saat memakai preprocessor ini di dalam loop training
    agar setiap pipeline mendapat instance yang bersih (tidak shared fitted state).
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


# ── Model definitions ─────────────────────────────────────────────────────────
def get_models() -> dict:
    """
    Kembalikan dict kandidat model yang akan dibandingkan.

    - LogisticRegression : baseline, cepat, interpretable.
    - RandomForest       : ensemble, robust terhadap overfitting.
    - XGBoost            : gradient boosting, biasanya terbaik untuk tabular data.
    """
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            scale_pos_weight=5174 / 1869,   # ratio No Churn / Churn
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
    }


# ── Evaluate single model ─────────────────────────────────────────────────────
def evaluate(
    name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluasi satu pipeline dan cetak metrik lengkap ke stdout.

    Returns dict berisi Accuracy, ROC AUC, Recall/Precision/F1 untuk kelas Churn.
    """
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    results = {
        "Model"    : name,
        "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "ROC AUC"  : round(roc_auc_score(y_test, y_proba), 4),
        "Recall (Churn)"   : round(
            classification_report(y_test, y_pred, output_dict=True)["1"]["recall"], 4),
        "Precision (Churn)": round(
            classification_report(y_test, y_pred, output_dict=True)["1"]["precision"], 4),
        "F1 (Churn)"       : round(
            classification_report(y_test, y_pred, output_dict=True)["1"]["f1-score"], 4),
    }

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy        : {results['Accuracy']}")
    print(f"  ROC AUC         : {results['ROC AUC']}")
    print(f"  Recall  (Churn) : {results['Recall (Churn)']}")
    print(f"  Precision(Churn): {results['Precision (Churn)']}")
    print(f"  F1      (Churn) : {results['F1 (Churn)']}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"  (Rows=Actual, Cols=Predicted | 0=No Churn, 1=Churn)")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def compare_and_save() -> None:
    """Entry point: load data, train semua kandidat model, evaluasi, dan simpan yang terbaik."""
    print("=" * 55)
    print("  FASE 2.5 – Model Comparison")
    print("=" * 55)

    # Load data
    print(f"\n[1/4] Loading data...")
    X, y, num_feats, cat_feats = load_and_prepare(DATA_PATH)
    print(f"      {X.shape[0]:,} rows | {X.shape[1]} features")

    # Split
    print("[2/4] Splitting train/test (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    # Train & evaluate all models
    print("\n[3/4] Training & evaluating all models...")
    preprocessor = build_preprocessor(num_feats, cat_feats)
    models       = get_models()
    all_results  = []
    trained_pipelines = {}

    for name, clf in models.items():
        print(f"\n  → Training: {name}...")
        # clone() memastikan setiap pipeline dapat instance preprocessor
        # yang bersih — tidak ada shared fitted state antar model
        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", clf),
        ])
        pipeline.fit(X_train, y_train)
        results = evaluate(name, pipeline, X_test, y_test)
        all_results.append(results)
        trained_pipelines[name] = pipeline

    # Summary table
    print("\n\n" + "=" * 55)
    print("  SUMMARY – Model Comparison")
    print("=" * 55)
    df_results = pd.DataFrame(all_results).set_index("Model")
    print(df_results.to_string())

    # Pick best by ROC AUC
    best_name = df_results["ROC AUC"].idxmax()
    print(f"\n  ✅ Best model by ROC AUC: {best_name} ({df_results.loc[best_name, 'ROC AUC']})")

    # Save best model
    print("\n[4/4] Saving best model...")
    best_pipeline = trained_pipelines[best_name]
    best_path     = MODEL_DIR / "churn_best_pipeline.joblib"
    joblib.dump(best_pipeline, best_path)
    print(f"      Saved → {best_path} ({best_path.stat().st_size / 1024:.1f} KB)")

    # Also save logreg baseline for reference
    logreg_path = MODEL_DIR / "churn_logreg_pipeline.joblib"
    joblib.dump(trained_pipelines["Logistic Regression"], logreg_path)
    print(f"      Saved → {logreg_path} (baseline reference)")

    print("\nFase 2.5 complete. Siap lanjut ke Fase 3 (FastAPI)!")


if __name__ == "__main__":
    compare_and_save()
