import random
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import delete, func, select

from app.db.base import Base
from app.db.postgres import engine
from app.models.account import Account
from app.models.customer import Customer
from app.models.transaction import Transaction


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

NUM_CUSTOMERS = 1_000
MIN_ACCOUNTS_PER_CUSTOMER = 1
MAX_ACCOUNTS_PER_CUSTOMER = 4

NUM_TRANSACTIONS = 50_000

RESET_DATABASE = True

random.seed(SEED)


# ============================================================
# REFERENCE DATA
# ============================================================

FIRST_NAMES = [
    "Ahmed",
    "Mohamed",
    "Youssef",
    "Amine",
    "Ali",
    "Omar",
    "Karim",
    "Sami",
    "Mehdi",
    "Anis",
    "Sara",
    "Nour",
    "Mariem",
    "Aya",
    "Lina",
    "Ines",
    "Rania",
    "Salma",
    "Meriem",
    "Hana",
]

LAST_NAMES = [
    "Ben Ali",
    "Trabelsi",
    "Jaziri",
    "Mansouri",
    "Gharbi",
    "Khalfallah",
    "Haddad",
    "Ayari",
    "Kallel",
    "Brahmi",
    "Ben Amor",
    "Chaabane",
    "Dridi",
    "Saidi",
    "Tlili",
]

CITIES = [
    "Tunis",
    "Ariana",
    "Ben Arous",
    "Manouba",
    "Sousse",
    "Sfax",
    "Bizerte",
    "Nabeul",
    "Monastir",
    "Gabes",
]

COUNTRIES = [
    "Tunisia",
    "France",
    "Italy",
    "Germany",
    "Spain",
    "Turkey",
    "United Kingdom",
    "United States",
]

MERCHANTS = {
    "grocery": [
        "Carrefour",
        "Monoprix",
        "MG",
        "Aziza",
    ],
    "restaurant": [
        "Pizza Hut",
        "McDonalds",
        "KFC",
        "Local Restaurant",
    ],
    "shopping": [
        "Zara",
        "Decathlon",
        "LC Waikiki",
        "Online Store",
    ],
    "electronics": [
        "Samsung Store",
        "Apple Store",
        "MyTek",
        "Fnac",
    ],
    "travel": [
        "Booking",
        "Air France",
        "Tunisair",
        "Uber",
    ],
    "services": [
        "Internet Provider",
        "Electricity",
        "Water",
        "Telecom",
    ],
}

MERCHANT_CATEGORIES = list(MERCHANTS.keys())

TRANSACTION_TYPES = [
    "card_payment",
    "cash_withdrawal",
    "transfer",
    "online_payment",
    "direct_debit",
]


# ============================================================
# HELPERS
# ============================================================

def random_date_of_birth():
    start = datetime(1960, 1, 1)
    end = datetime(2005, 12, 31)

    days = (end - start).days

    return (start + timedelta(days=random.randint(0, days))).date()


def random_transaction_time():
    start = datetime(2025, 1, 1)
    end = datetime(2026, 9, 30)

    seconds = int((end - start).total_seconds())

    return start + timedelta(
        seconds=random.randint(0, seconds)
    )


def random_amount(transaction_type, is_fraud):
    if transaction_type == "cash_withdrawal":
        low = 20
        high = 1000
    elif transaction_type == "transfer":
        low = 50
        high = 5000
    else:
        low = 5
        high = 1500

    amount = random.uniform(low, high)

    # Fraudulent transactions are more likely
    # to have unusually large amounts.
    if is_fraud:
        amount *= random.uniform(1.5, 4.0)

    return round(amount, 2)


def random_country():
    # Most transactions happen domestically.
    if random.random() < 0.88:
        return "Tunisia"

    return random.choice(
        [
            country
            for country in COUNTRIES
            if country != "Tunisia"
        ]
    )


def calculate_merchant_risk(
    category,
    country,
    is_international,
    is_fraud,
):
    risk = {
        "grocery": 5,
        "restaurant": 8,
        "shopping": 15,
        "electronics": 25,
        "travel": 20,
        "services": 5,
    }[category]

    if is_international:
        risk += random.uniform(10, 25)

    if country not in ["Tunisia", "France", "Italy"]:
        risk += random.uniform(5, 20)

    if is_fraud:
        risk += random.uniform(20, 40)

    return round(min(risk, 99.99), 2)


def should_be_fraud(
    is_international,
    merchant_category,
    amount,
    merchant_risk,
    timestamp,
):
    score = 0.0

    if is_international:
        score += 0.12

    if merchant_category in [
        "electronics",
        "shopping",
        "travel",
    ]:
        score += 0.08

    if amount > 2000:
        score += 0.15

    if amount > 5000:
        score += 0.15

    if merchant_risk > 50:
        score += 0.20

    # Unusual night activity
    if timestamp.hour < 5 or timestamp.hour >= 23:
        score += 0.12

    # Base fraud probability
    score += 0.025

    return random.random() < min(score, 0.75)


# ============================================================
# RESET DATABASE
# ============================================================

def reset_database():
    print("Resetting database...")

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    print("Database reset complete.")


# ============================================================
# CUSTOMERS
# ============================================================

def generate_customers():
    print(f"Generating {NUM_CUSTOMERS:,} customers...")

    customers = []

    for i in range(1, NUM_CUSTOMERS + 1):
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)

        customers.append(
            {
                "customer_number": f"CUS-{i:06d}",
                "first_name": first_name,
                "last_name": last_name,
                "email": (
                    f"{first_name.lower()}."
                    f"{last_name.lower().replace(' ', '')}"
                    f"{i}@example.com"
                ),
                "phone": f"+216{random.randint(20000000, 99999999)}",
                "date_of_birth": random_date_of_birth(),
                "country": "Tunisia",
                "city": random.choice(CITIES),
                "created_at": datetime.utcnow(),
            }
        )

    with engine.begin() as connection:
        connection.execute(
            Customer.__table__.insert(),
            customers,
        )

    print("Customers inserted.")

    return customers


# ============================================================
# ACCOUNTS
# ============================================================

def generate_accounts():
    print("Generating accounts...")

    with engine.connect() as connection:
        customer_rows = connection.execute(
            Customer.__table__.select()
        ).mappings().all()

    accounts = []

    account_id = 1

    for customer in customer_rows:
        number_of_accounts = random.randint(
            MIN_ACCOUNTS_PER_CUSTOMER,
            MAX_ACCOUNTS_PER_CUSTOMER,
        )

        for _ in range(number_of_accounts):
            account_type = random.choice(
                [
                    "checking",
                    "savings",
                ]
            )

            balance = round(
                random.uniform(100, 25000),
                2,
            )

            accounts.append(
                {
                    "account_number": f"ACC-{account_id:08d}",
                    "customer_id": customer["id"],
                    "account_type": account_type,
                    "currency": "TND",
                    "balance": Decimal(str(balance)),
                    "status": "active",
                    "created_at": datetime.utcnow(),
                }
            )

            account_id += 1

    with engine.begin() as connection:
        connection.execute(
            Account.__table__.insert(),
            accounts,
        )

    with engine.connect() as connection:
        account_rows = connection.execute(
            Account.__table__.select()
        ).mappings().all()

    print(
        f"Accounts inserted: {len(account_rows):,}"
    )

    return account_rows


# ============================================================
# TRANSACTIONS
# ============================================================

def generate_transactions(accounts):
    print(
        f"Generating {NUM_TRANSACTIONS:,} transactions..."
    )

    transactions = []

    fraud_count = 0

    for i in range(1, NUM_TRANSACTIONS + 1):
        account = random.choice(accounts)

        transaction_type = random.choice(
            TRANSACTION_TYPES
        )

        timestamp = random_transaction_time()

        country = random_country()

        is_international = country != "Tunisia"

        merchant_category = random.choice(
            MERCHANT_CATEGORIES
        )

        merchant = random.choice(
            MERCHANTS[merchant_category]
        )

        # Temporary risk estimate before fraud label.
        merchant_risk = calculate_merchant_risk(
            merchant_category,
            country,
            is_international,
            False,
        )

        amount = random_amount(
            transaction_type,
            False,
        )

        is_fraud = should_be_fraud(
            is_international,
            merchant_category,
            amount,
            merchant_risk,
            timestamp,
        )

        if is_fraud:
            fraud_count += 1

            # Fraud transactions receive stronger
            # suspicious characteristics.
            amount = random_amount(
                transaction_type,
                True,
            )

            merchant_risk = calculate_merchant_risk(
                merchant_category,
                country,
                is_international,
                True,
            )

        transactions.append(
            {
                "transaction_number": f"TX-{i:09d}",
                "account_id": account["id"],
                "amount": Decimal(str(amount)),
                "currency": "TND",
                "transaction_type": transaction_type,
                "timestamp": timestamp,
                "merchant": merchant,
                "merchant_category": merchant_category,
                "country": country,
                "is_international": is_international,
                "merchant_risk": Decimal(
                    str(merchant_risk)
                ),
                "is_fraud": is_fraud,
                "status": "completed",
            }
        )

        # Insert in batches to avoid keeping
        # everything in one database operation.
        if len(transactions) >= 5_000:
            with engine.begin() as connection:
                connection.execute(
                    Transaction.__table__.insert(),
                    transactions,
                )

            transactions.clear()

            print(
                f"  {i:,}/{NUM_TRANSACTIONS:,} "
                "transactions inserted..."
            )

    if transactions:
        with engine.begin() as connection:
            connection.execute(
                Transaction.__table__.insert(),
                transactions,
            )

    print("Transactions inserted.")

    return fraud_count


# ============================================================
# SUMMARY
# ============================================================

def print_summary(fraud_count):
    with engine.connect() as connection:
        customer_count = connection.execute(
            select(func.count()).select_from(Customer.__table__)
        ).scalar()

        account_count = connection.execute(
            select(func.count()).select_from(Account.__table__)
        ).scalar()

        transaction_count = connection.execute(
            select(func.count()).select_from(Transaction.__table__)
        ).scalar()

    print()
    print("=" * 50)
    print("BANKASSURE SYNTHETIC DATA")
    print("=" * 50)
    print(f"Customers:     {customer_count:,}")
    print(f"Accounts:      {account_count:,}")
    print(f"Transactions:  {transaction_count:,}")
    print(f"Fraud:         {fraud_count:,}")
    print(
        f"Fraud rate:    "
        f"{fraud_count / transaction_count * 100:.2f}%"
    )
    print("=" * 50)


# ============================================================
# MAIN
# ============================================================

def main():
    if RESET_DATABASE:
        reset_database()

    generate_customers()

    accounts = generate_accounts()

    fraud_count = generate_transactions(
        accounts
    )

    print_summary(fraud_count)


if __name__ == "__main__":
    main()