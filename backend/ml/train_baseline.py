import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# --------------------------------------------------
# 3. Separate features and target
# --------------------------------------------------

TARGET = "is_fraud"

DROP_COLUMNS = [
    "transaction_id",
    "account_id",
    "customer_id",
    TARGET,
]

X = df.drop(columns=DROP_COLUMNS)
y = df[TARGET]


# --------------------------------------------------
# 4. Identify feature types
# --------------------------------------------------

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


# --------------------------------------------------
# 5. Train / validation / test split
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)

print("\nSplit sizes:")
print(f"Train:      {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test:       {len(X_test)}")


# --------------------------------------------------
# 6. Preprocessing
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)


# --------------------------------------------------
# 7. Logistic Regression
# --------------------------------------------------

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    C=1.0,
    solver="liblinear",
    random_state=42,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)


# --------------------------------------------------
# 8. Train
# --------------------------------------------------

print("\nTraining Logistic Regression...")

pipeline.fit(X_train, y_train)


# --------------------------------------------------
# 9. Threshold tuning on validation set
# --------------------------------------------------

def tune_threshold(y_true, probs, mode="f1", target_recall=0.60):
    """Scan decision thresholds; return best threshold and its metrics.

    mode:
      - "f1"           -> maximise F1-score on the positive class
      - "target_recall" -> lowest threshold where recall >= target_recall
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    best_score = -1.0
    best_threshold = 0.5
    best_row = None

    for t in thresholds:
        preds = (probs >= t).astype(int)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        if mode == "f1":
            if f1 > best_score:
                best_score = f1
                best_threshold = t
                best_row = (t, prec, rec, f1)
        elif mode == "target_recall":
            if rec >= target_recall:
                candidate_score = prec
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_threshold = t
                    best_row = (t, prec, rec, f1)

    if best_row is None:
        return tune_threshold(y_true, probs, mode="f1")

    return best_threshold, best_row


val_probabilities = pipeline.predict_proba(X_val)[:, 1]

print("\n===== THRESHOLD TUNING (validation set) =====")

best_f1_thr, (t_f1, p_f1, r_f1, f1_f1) = tune_threshold(
    y_val, val_probabilities, mode="f1"
)
print(
    f"[Best F1]     thr={t_f1:.3f}  "
    f"precision={p_f1:.4f}  recall={r_f1:.4f}  f1={f1_f1:.4f}"
)

best_r50_thr, (t_r50, p_r50, r_r50, f1_r50) = tune_threshold(
    y_val, val_probabilities, mode="target_recall", target_recall=0.50
)
print(
    f"[Recall>=0.50] thr={t_r50:.3f}  "
    f"precision={p_r50:.4f}  recall={r_r50:.4f}  f1={f1_r50:.4f}"
)

best_r60_thr, (t_r60, p_r60, r_r60, f1_r60) = tune_threshold(
    y_val, val_probabilities, mode="target_recall", target_recall=0.60
)
print(
    f"[Recall>=0.60] thr={t_r60:.3f}  "
    f"precision={p_r60:.4f}  recall={r_r60:.4f}  f1={f1_r60:.4f}"
)

DEFAULT_THRESHOLD = float(best_f1_thr)
print(f"\nDefault threshold (best F1): {DEFAULT_THRESHOLD:.3f}")


def evaluate_threshold(name, y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    print(f"\n===== {name} (threshold={threshold:.3f}) =====")
    print(classification_report(y_true, preds, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_true, probs):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_true, probs):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, preds))
    return preds


# --------------------------------------------------
# 10. Validation evaluation
# --------------------------------------------------

evaluate_threshold("VALIDATION RESULTS", y_val, val_probabilities, DEFAULT_THRESHOLD)


# --------------------------------------------------
# 11. Test evaluation
# --------------------------------------------------

test_probabilities = pipeline.predict_proba(X_test)[:, 1]
test_predictions = evaluate_threshold(
    "TEST RESULTS", y_test, test_probabilities, DEFAULT_THRESHOLD
)


# --------------------------------------------------
# 12. Save model + threshold metadata
# --------------------------------------------------

model_path = MODEL_DIR / "fraud_logistic_regression.joblib"
metadata_path = MODEL_DIR / "fraud_logistic_regression_metadata.json"

joblib.dump(pipeline, model_path)

metadata = {
    "default_threshold": DEFAULT_THRESHOLD,
    "threshold_best_f1": float(t_f1),
    "threshold_recall_0_50": float(t_r50),
    "threshold_recall_0_60": float(t_r60),
    "validation_best_f1": {
        "precision": float(p_f1),
        "recall": float(r_f1),
        "f1": float(f1_f1),
    },
    "validation_recall_0_50": {
        "precision": float(p_r50),
        "recall": float(r_r50),
        "f1": float(f1_r50),
    },
    "validation_recall_0_60": {
        "precision": float(p_r60),
        "recall": float(r_r60),
        "f1": float(f1_r60),
    },
    "test_roc_auc": float(roc_auc_score(y_test, test_probabilities)),
    "test_pr_auc": float(average_precision_score(y_test, test_probabilities)),
    "test_confusion_matrix": confusion_matrix(y_test, test_predictions).tolist(),
}

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nModel saved to:    {model_path}")
print(f"Metadata saved to: {metadata_path}")
