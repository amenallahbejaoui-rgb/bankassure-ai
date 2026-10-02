from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "fraud_logistic_regression.joblib"
DATA_PATH = BASE_DIR / "data" / "fraud_features.csv"


# Load model and dataset
pipeline = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)


# Get preprocessing and model
preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]


# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

coefficients = model.coef_[0]


# Build feature importance table
importance = pd.DataFrame(
    {
        "feature": feature_names,
        "coefficient": coefficients,
        "absolute_coefficient": abs(coefficients),
    }
).sort_values(
    "absolute_coefficient",
    ascending=False,
)


print("\n===== TOP FEATURES =====\n")

print(
    importance[
        ["feature", "coefficient"]
    ].head(20).to_string(index=False)
)