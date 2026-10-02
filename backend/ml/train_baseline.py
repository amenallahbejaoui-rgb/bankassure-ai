from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
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

# IDs are identifiers, not predictive features.
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
    max_iter=1000,
    class_weight="balanced",
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
# 9. Validation
# --------------------------------------------------

val_predictions = pipeline.predict(X_val)
val_probabilities = pipeline.predict_proba(X_val)[:, 1]

print("\n===== VALIDATION RESULTS =====")

print(
    classification_report(
        y_val,
        val_predictions,
        digits=4,
    )
)

print(
    f"ROC-AUC: {roc_auc_score(y_val, val_probabilities):.4f}"
)

print(
    f"PR-AUC:  {average_precision_score(y_val, val_probabilities):.4f}"
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, val_predictions))


# --------------------------------------------------
# 10. Final test evaluation
# --------------------------------------------------

test_predictions = pipeline.predict(X_test)
test_probabilities = pipeline.predict_proba(X_test)[:, 1]

print("\n===== TEST RESULTS =====")

print(
    classification_report(
        y_test,
        test_predictions,
        digits=4,
    )
)

print(
    f"ROC-AUC: {roc_auc_score(y_test, test_probabilities):.4f}"
)

print(
    f"PR-AUC:  {average_precision_score(y_test, test_probabilities):.4f}"
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))


# --------------------------------------------------
# 11. Save model
# --------------------------------------------------

model_path = MODEL_DIR / "fraud_logistic_regression.joblib"

joblib.dump(pipeline, model_path)

print(f"\nModel saved to: {model_path}")