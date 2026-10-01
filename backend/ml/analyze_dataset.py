from pathlib import Path

import pandas as pd


DATA_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "fraud_features.csv"
)


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)

    print("\n" + "=" * 60)
    print("BANKASSURE FRAUD DATASET VALIDATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # Basic information
    # ---------------------------------------------------------

    print("\n[1] Dataset shape")
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    print(df.columns.tolist())

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    print("\n[2] Missing values")

    missing = df.isna().sum()

    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values.")
    else:
        print(missing)

    # ---------------------------------------------------------
    # Fraud distribution
    # ---------------------------------------------------------

    print("\n[3] Fraud distribution")

    fraud_counts = df["is_fraud"].value_counts()

    print(fraud_counts)

    fraud_rate = df["is_fraud"].mean() * 100

    print(f"\nFraud rate: {fraud_rate:.2f}%")

    # ---------------------------------------------------------
    # Numeric statistics
    # ---------------------------------------------------------

    print("\n[4] Numeric statistics")

    numeric_columns = [
        "amount",
        "merchant_risk",
        "customer_avg_amount",
        "amount_vs_customer_avg",
        "transactions_24h",
        "transactions_7d",
        "customer_international_ratio",
    ]

    print(
        df[numeric_columns]
        .describe()
        .round(2)
        .to_string()
    )

    # ---------------------------------------------------------
    # Fraud vs normal
    # ---------------------------------------------------------

    print("\n[5] Fraud vs normal")

    comparison = (
        df.groupby("is_fraud")[
            [
                "amount",
                "merchant_risk",
                "amount_vs_customer_avg",
                "transactions_24h",
                "transactions_7d",
                "customer_international_ratio",
            ]
        ]
        .mean()
        .round(2)
    )

    print(comparison)

    # ---------------------------------------------------------
    # International transactions
    # ---------------------------------------------------------

    print("\n[6] International transactions")

    international = pd.crosstab(
        df["is_international"],
        df["is_fraud"],
        normalize="index",
    ) * 100

    international.columns = [
        "Normal %",
        "Fraud %",
    ]

    print(international.round(2))

    # ---------------------------------------------------------
    # Night transactions
    # ---------------------------------------------------------

    print("\n[7] Night transactions")

    night = pd.crosstab(
        df["is_night"],
        df["is_fraud"],
        normalize="index",
    ) * 100

    night.columns = [
        "Normal %",
        "Fraud %",
    ]

    print(night.round(2))

    # ---------------------------------------------------------
    # Transaction type
    # ---------------------------------------------------------

    print("\n[8] Transaction type")

    transaction_type = (
        df.groupby("transaction_type")["is_fraud"]
        .agg(
            transactions="count",
            frauds="sum",
            fraud_rate="mean",
        )
        .sort_values(
            "fraud_rate",
            ascending=False,
        )
    )

    transaction_type["fraud_rate"] *= 100

    print(
        transaction_type.round(2)
    )

    # ---------------------------------------------------------
    # Merchant category
    # ---------------------------------------------------------

    print("\n[9] Merchant category")

    merchant_category = (
        df.groupby("merchant_category")["is_fraud"]
        .agg(
            transactions="count",
            frauds="sum",
            fraud_rate="mean",
        )
        .sort_values(
            "fraud_rate",
            ascending=False,
        )
    )

    merchant_category["fraud_rate"] *= 100

    print(
        merchant_category.round(2)
    )

    # ---------------------------------------------------------
    # Country
    # ---------------------------------------------------------

    print("\n[10] Country")

    country = (
        df.groupby("country")["is_fraud"]
        .agg(
            transactions="count",
            frauds="sum",
            fraud_rate="mean",
        )
        .sort_values(
            "fraud_rate",
            ascending=False,
        )
    )

    country["fraud_rate"] *= 100

    print(country.round(2))

    # ---------------------------------------------------------
    # Correlations
    # ---------------------------------------------------------

    print("\n[11] Numeric correlations with fraud")

    correlation_columns = [
        "amount",
        "merchant_risk",
        "is_international",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_night",
        "customer_transaction_count",
        "customer_avg_amount",
        "amount_vs_customer_avg",
        "transactions_24h",
        "transactions_7d",
        "customer_international_ratio",
        "is_fraud",
    ]

    correlations = (
        df[correlation_columns]
        .corr()["is_fraud"]
        .drop("is_fraud")
        .sort_values(
            key=abs,
            ascending=False,
        )
    )

    print(
        correlations.round(4)
    )

    # ---------------------------------------------------------
    # Duplicate transactions
    # ---------------------------------------------------------

    print("\n[12] Duplicate transaction IDs")

    duplicates = df["transaction_id"].duplicated().sum()

    print(
        f"Duplicate IDs: {duplicates}"
    )

    # ---------------------------------------------------------
    # Final assessment
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()