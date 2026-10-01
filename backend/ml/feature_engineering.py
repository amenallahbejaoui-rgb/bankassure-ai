import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import pandas as pd
from sqlalchemy import text

from app.db.postgres import engine


OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "fraud_features.csv"


def load_transactions() -> pd.DataFrame:
    query = text(
        """
        SELECT
            t.id AS transaction_id,
            t.account_id,
            t.amount,
            t.currency,
            t.transaction_type,
            t.timestamp,
            t.merchant_category,
            t.country,
            t.is_international,
            t.merchant_risk,
            t.is_fraud,
            a.customer_id
        FROM transactions t
        JOIN accounts a
            ON t.account_id = a.id
        ORDER BY t.timestamp
        """
    )

    with engine.connect() as connection:
        return pd.read_sql(query, connection)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour

    df["day_of_week"] = df["timestamp"].dt.dayofweek

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["is_night"] = (
        (df["hour"] < 5)
        | (df["hour"] >= 23)
    ).astype(int)

    # --------------------------------------------------------
    # Customer historical behavior
    # --------------------------------------------------------

    # IMPORTANT:
    # shift(1) prevents the current transaction from
    # influencing its own historical features.
    df["customer_transaction_count"] = (
        df.groupby("customer_id")
        .cumcount()
    )

    df["customer_avg_amount"] = (
        df.groupby("customer_id")["amount"]
        .transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )

    # For customers whose first transaction has no
    # historical average, use the global average.
    global_average = df["amount"].mean()

    df["customer_avg_amount"] = (
        df["customer_avg_amount"]
        .fillna(global_average)
    )

    # --------------------------------------------------------
    # Amount behavior
    # --------------------------------------------------------

    df["amount_vs_customer_avg"] = (
        df["amount"]
        / df["customer_avg_amount"]
    )

    # --------------------------------------------------------
    # Rolling transaction frequency
    # --------------------------------------------------------

    transaction_times = (
        df.set_index("timestamp")
        .groupby("customer_id")["transaction_id"]
        .rolling("24h")
        .count()
        .reset_index(
            name="transactions_24h"
        )
    )

    df = df.merge(
        transaction_times,
        on=["customer_id", "timestamp"],
        how="left",
    )

    # The rolling count includes the current transaction,
    # so subtract one.
    df["transactions_24h"] = (
        df["transactions_24h"] - 1
    ).clip(lower=0)

    # --------------------------------------------------------
    # 7-day transaction frequency
    # --------------------------------------------------------

    transaction_times_7d = (
        df.set_index("timestamp")
        .groupby("customer_id")["transaction_id"]
        .rolling("7D")
        .count()
        .reset_index(
            name="transactions_7d"
        )
    )

    df = df.merge(
        transaction_times_7d,
        on=["customer_id", "timestamp"],
        how="left",
    )

    df["transactions_7d"] = (
        df["transactions_7d"] - 1
    ).clip(lower=0)

    # --------------------------------------------------------
    # International behavior
    # --------------------------------------------------------

    customer_international_count = (
        df.groupby("customer_id")[
            "is_international"
        ]
        .transform(
            lambda x: x.shift(1).expanding().sum()
        )
        .fillna(0)
    )

    df["customer_international_ratio"] = (
        customer_international_count
        / df["customer_transaction_count"].clip(
            lower=1
        )
    )

    # --------------------------------------------------------
    # Merchant risk
    # --------------------------------------------------------

    df["merchant_risk"] = (
        pd.to_numeric(
            df["merchant_risk"],
            errors="coerce",
        )
        .fillna(0)
    )

    # --------------------------------------------------------
    # Select ML features
    # --------------------------------------------------------

    feature_columns = [
        "transaction_id",
        "account_id",
        "customer_id",
        "amount",
        "merchant_risk",
        "is_international",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_night",
        "transaction_type",
        "merchant_category",
        "country",
        "customer_transaction_count",
        "customer_avg_amount",
        "amount_vs_customer_avg",
        "transactions_24h",
        "transactions_7d",
        "customer_international_ratio",
        "is_fraud",
    ]

    return df[feature_columns]


def main():
    print("Loading transactions...")

    df = load_transactions()

    print(
        f"Loaded {len(df):,} transactions."
    )

    print("Creating features...")

    features = create_features(df)

    print(
        f"Created {len(features.columns)} columns."
    )

    features.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 55)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 55)
    print(
        f"Rows:    {len(features):,}"
    )
    print(
        f"Columns: {len(features.columns)}"
    )
    print(
        f"Output:  {OUTPUT_FILE}"
    )
    print("=" * 55)

    print()
    print("Fraud distribution:")
    print(
        features["is_fraud"]
        .value_counts()
    )

    print()
    print("Feature preview:")
    print(
        features.head().to_string()
    )


if __name__ == "__main__":
    main()