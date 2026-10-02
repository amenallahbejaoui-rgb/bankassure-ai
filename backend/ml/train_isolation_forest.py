import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "fraud_features.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Load dataset
# ============================================================

df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")


# ============================================================
# 3. Define features
# ============================================================

TARGET = "is_fraud"

DROP_COLUMNS = [
    "transaction_id",
    "account_id",
    "customer_id",
    TARGET,
]

X = df.drop(columns=DROP_COLUMNS)
y = df[TARGET].astype(int)


categorical_features = [
    "transaction_type",
    "merchant_category",
    "country",
]

numeric_features = [
    column
    for column in X.columns
    if column not in categorical_features
]

print(f"Features: {X.shape[1]}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Numeric features: {len(numeric_features)}")


# ============================================================
# 4. Split (Isolation Forest trains on the majority class ONLY)
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42,
)

# Fit only on NORMAL (non-fraud) training samples -> true unsupervised anomaly setup
X_normal_train = X_train[y_train == 0]
print(f"\nFitting Isolation Forest on {len(X_normal_train)} normal training samples...")


# ============================================================
# 5. Preprocessing
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)


# ============================================================
# 6. Isolation Forest
# ============================================================

model = IsolationForest(
    n_estimators=300,
    max_samples="auto",
    contamination=0.05,
    max_features=1.0,
    bootstrap=False,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)


pipeline.fit(X_normal_train)


# ============================================================
# 7. Score and calibrate
# ============================================================

def to_anomaly_prob(raw_score, q_normal, q_anomaly):
    """Map IsolationForest raw score (-1..1ish) to calibrated 0..1 probability.

    Higher probability = more anomalous (fraud-like).
    """
    # raw_score: high = normal, low = anomaly
    s = -raw_score  # invert: high = anomaly

    p05 = q_normal   # below this -> normal
    p95 = q_anomaly  # above this -> anomaly

    out = (s - p05) / (p95 - p05 + 1e-9)
    return np.clip(out, 0.0, 1.0)


val_raw = pipeline.decision_function(X_val)
neg_val = -val_raw

q_normal = np.percentile(neg_val[y_val == 0], 95)
q_anomaly = np.percentile(neg_val[y_val == 1], 10)

print(f"\nCalibration on validation set:")
print(f"  95th percentile of -score on normal:   {q_normal:.4f}")
print(f"  10th percentile of -score on fraud:    {q_anomaly:.4f}")

anomaly_prob_val = to_anomaly_prob(val_raw, q_normal, q_anomaly)

# Quick sanity - correlation with fraud label
from sklearn.metrics import roc_auc_score, average_precision_score

print(f"  Anomaly-score ROC-AUC (val): {roc_auc_score(y_val, anomaly_prob_val):.4f}")
print(f"  Anomaly-score PR-AUC  (val): {average_precision_score(y_val, anomaly_prob_val):.4f}")


# ============================================================
# 8. Save
# ============================================================

model_path = MODEL_DIR / "fraud_isolation_forest.joblib"
metadata_path = MODEL_DIR / "fraud_isolation_forest_metadata.json"

joblib.dump(pipeline, model_path)

metadata = {
    "calibration_q_normal": float(q_normal),
    "calibration_q_anomaly": float(q_anomaly),
    "val_roc_auc": float(roc_auc_score(y_val, anomaly_prob_val)),
    "val_pr_auc": float(average_precision_score(y_val, anomaly_prob_val)),
}

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nModel saved to:    {model_path}")
print(f"Metadata saved to: {metadata_path}")
