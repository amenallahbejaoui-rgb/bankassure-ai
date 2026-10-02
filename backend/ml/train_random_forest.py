import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = "ml/data/fraud_features.csv"


# ============================================================
# 1. Load dataset
# ============================================================

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 2. Define features and target
# ============================================================

TARGET = "is_fraud"

DROP_COLUMNS = [
    "transaction_id",
    "account_id",
    "customer_id",
    TARGET,
]

X = df.drop(columns=DROP_COLUMNS)
y = df[TARGET]


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
# 3. Train / validation / test split
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

print("\nSplit sizes:")
print(f"Train:      {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test:       {len(X_test)}")


# ============================================================
# 4. Preprocessing
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)


# ============================================================
# 5. Random Forest
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            model,
        ),
    ]
)


# ============================================================
# 6. Train
# ============================================================

print("\nTraining Random Forest...")

pipeline.fit(
    X_train,
    y_train,
)


# ============================================================
# 7. Validation
# ============================================================

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
    f"ROC-AUC: "
    f"{roc_auc_score(y_val, val_probabilities):.4f}"
)

print(
    f"PR-AUC:  "
    f"{average_precision_score(y_val, val_probabilities):.4f}"
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_val,
        val_predictions,
    )
)


# ============================================================
# 8. Test
# ============================================================

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
    f"ROC-AUC: "
    f"{roc_auc_score(y_test, test_probabilities):.4f}"
)

print(
    f"PR-AUC:  "
    f"{average_precision_score(y_test, test_probabilities):.4f}"
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        test_predictions,
    )
)