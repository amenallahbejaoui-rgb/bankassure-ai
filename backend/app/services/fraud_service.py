"""Helpers to score a DB Transaction through the Model 1 Fraud Detection
ensemble (Logistic Regression baseline + XGBoost + Isolation Forest).

This module also computes simple customer/account velocity features on the
fly from the transactions table so the scorer gets the same feature family
it was trained on (customer_avg_amount, transactions_24h, transactions_7d,
transaction_frequency, etc.).  If these aggregates cannot be computed the
scorer falls back to sensible defaults - same as the public /fraud/score
endpoint.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
import sys
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.account import Account
from app.models.transaction import Transaction


# Allow `ml` sibling import regardless of cwd
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.fraud_scorer import FraudScoreResult, get_scorer  # noqa: E402


@dataclass
class CustomerProfile:
    transaction_count: int
    avg_amount: float
    tx_24h: float
    tx_7d: float
    international_ratio: float


def _load_customer_profile(
    db: Session,
    account_id: int,
    timestamp,
) -> CustomerProfile:
    """Pull velocity/avg features from the transaction history of the
    account's customer.  Graceful zero defaults if account/customer is
    unknown or history is empty.
    """
    defaults = CustomerProfile(
        transaction_count=0,
        avg_amount=0.0,
        tx_24h=0.0,
        tx_7d=0.0,
        international_ratio=0.0,
    )
    try:
        account = db.get(Account, account_id)
        if account is None:
            return defaults
        customer_id = account.customer_id

        # All transactions ever across all accounts of this customer
        all_stmt = (
            select(
                func.count(Transaction.id),
                func.coalesce(func.avg(Transaction.amount), 0),
                func.coalesce(
                    func.avg(func.cast(Transaction.is_international, type_=None)),
                    0,
                ),
            )
            .join(Account, Account.id == Transaction.account_id)
            .where(Account.customer_id == customer_id)
            .where(Transaction.timestamp < timestamp)
        )
        total_cnt, avg_amt, intl_ratio = db.execute(all_stmt).one()
        total_cnt = int(total_cnt or 0)
        avg_amt = float(avg_amt or 0.0)
        intl_ratio = float(intl_ratio or 0.0)

        cutoff_24 = timestamp - timedelta(hours=24)
        cutoff_7d = timestamp - timedelta(days=7)

        cnt_24 = db.scalar(
            select(func.count(Transaction.id))
            .join(Account, Account.id == Transaction.account_id)
            .where(Account.customer_id == customer_id)
            .where(Transaction.timestamp >= cutoff_24)
            .where(Transaction.timestamp < timestamp)
        ) or 0

        cnt_7d = db.scalar(
            select(func.count(Transaction.id))
            .join(Account, Account.id == Transaction.account_id)
            .where(Account.customer_id == customer_id)
            .where(Transaction.timestamp >= cutoff_7d)
            .where(Transaction.timestamp < timestamp)
        ) or 0

        return CustomerProfile(
            transaction_count=total_cnt,
            avg_amount=avg_amt,
            tx_24h=float(cnt_24),
            tx_7d=float(cnt_7d),
            international_ratio=intl_ratio,
        )
    except Exception:
        return defaults


def build_scorer_payload(
    *,
    transaction_id: str,
    amount: Decimal | float,
    timestamp,
    transaction_type: str,
    merchant: str | None,
    merchant_category: str,
    country: str,
    is_international: bool,
    merchant_risk: Decimal | float,
    account_id: int,
    db: Session | None = None,
    # Overrides / caller-provided raw features (highest priority)
    customer_avg_amount: Decimal | float | None = None,
    customer_transaction_count: int | None = None,
    transactions_24h: Decimal | float | None = None,
    transactions_7d: Decimal | float | None = None,
    customer_international_ratio: Decimal | float | None = None,
) -> dict[str, Any]:
    """Build the input dict expected by :class:`ml.fraud_scorer.FraudScorer`.

    Caller-provided velocity/avg features always win.  Otherwise, if a DB
    session is available we compute those features from transaction
    history; fall back to zeros otherwise.
    """
    profile: CustomerProfile | None = None
    if db is not None and (
        customer_avg_amount is None
        or transactions_24h is None
        or transactions_7d is None
        or customer_transaction_count is None
        or customer_international_ratio is None
    ):
        profile = _load_customer_profile(db, account_id, timestamp)

    _c_avg = customer_avg_amount
    _c_cnt = customer_transaction_count
    _t24 = transactions_24h
    _t7d = transactions_7d
    _ir = customer_international_ratio

    if profile is not None:
        if _c_avg is None:
            _c_avg = profile.avg_amount
        if _c_cnt is None:
            _c_cnt = profile.transaction_count
        if _t24 is None:
            _t24 = profile.tx_24h
        if _t7d is None:
            _t7d = profile.tx_7d
        if _ir is None:
            _ir = profile.international_ratio

    def _f(v, default=0.0) -> float:
        if v is None:
            return default
        if isinstance(v, Decimal):
            return float(v)
        return float(v)

    c_avg = _f(_c_avg, 0.0)
    amt_f = _f(amount, 0.0)

    return {
        "transaction_id": str(transaction_id),
        "amount": amt_f,
        "hour": int(timestamp.hour),
        "day_of_week": int(timestamp.weekday()),
        "country": str(country or "Unknown"),
        "merchant_category": str(merchant_category or "retail"),
        "merchant_risk": _f(merchant_risk, 0.0),
        "is_international": bool(is_international),
        "transaction_type": str(transaction_type or "card_payment"),
        "customer_avg_amount": c_avg,
        "customer_transaction_count": int(_c_cnt or 0),
        "transactions_24h": _f(_t24, 0.0),
        "transactions_7d": _f(_t7d, 0.0),
        "customer_international_ratio": _f(_ir, 0.0),
        # Passthrough so the API scorer does not drop merchant names
        "merchant": str(merchant or ""),
    }


def score_transaction_row(
    payload: dict[str, Any],
) -> FraudScoreResult:
    """Thin wrapper around the module-level scorer singleton."""
    return get_scorer().score_transaction(payload)


def apply_fraud_decision(
    transaction: Transaction,
    result: FraudScoreResult,
) -> None:
    """Mutate ``transaction`` row in-place with the scorer output and
    apply the status policy (REJECT -> "flagged").
    """
    transaction.fraud_score = float(result.fraud_probability)
    transaction.fraud_decision = result.decision
    try:
        # Postgres JSONB / SQLite text both accept a native list.
        transaction.fraud_reasons = list(result.reasons)
    except Exception:
        # Absolute last-resort fallback: text-JSON in a list box
        transaction.fraud_reasons = json.dumps(list(result.reasons))

    # Policy enforcement: any manual status the user set is overruled for
    # high-risk cases. REVIEW stays as whatever the caller asked ("pending
    # review" semantics) while the row still gets a "review" decision tag.
    if result.decision == "REJECT":
        transaction.status = "flagged"
        transaction.is_fraud = True
    elif result.decision == "REVIEW":
        if (transaction.status or "completed") == "completed":
            transaction.status = "review"


__all__ = [
    "CustomerProfile",
    "FraudScoreResult",
    "build_scorer_payload",
    "score_transaction_row",
    "apply_fraud_decision",
]
