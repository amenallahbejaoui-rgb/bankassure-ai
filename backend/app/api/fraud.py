from decimal import Decimal
from pathlib import Path
import sys

from fastapi import APIRouter, HTTPException, Query

from app.schemas.fraud import (
    FraudScoreDebugResponse,
    FraudScoreRequest,
    FraudScoreResponse,
)


# Make sure `ml` sibling package is importable regardless of CWD
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.fraud_scorer import get_scorer  # noqa: E402


router = APIRouter(
    prefix="/fraud",
    tags=["Fraud Detection"],
)


def _to_dict(req: FraudScoreRequest) -> dict:
    raw = req.model_dump()

    def _decimal_or(v, default):
        if v is None:
            return default
        if isinstance(v, Decimal):
            return float(v)
        return float(v)

    freq = _decimal_or(raw.get("transaction_frequency"), None)
    t24 = _decimal_or(raw.get("transactions_24h"), None)
    t7d = _decimal_or(raw.get("transactions_7d"), None)
    transactions_24h = t24 if t24 is not None else (freq if freq is not None else 0.0)
    transactions_7d = t7d if t7d is not None else (transactions_24h * 5.0)

    return {
        "transaction_id": raw.get("transaction_id"),
        "amount": _decimal_or(raw.get("amount"), 0.0),
        "hour": int(raw.get("hour", 12)),
        "day_of_week": int(raw.get("day_of_week", 1)),
        "country": raw.get("country", "Unknown"),
        "merchant_category": raw.get("merchant_category", "retail"),
        "merchant_risk": _decimal_or(raw.get("merchant_risk"), 0.0),
        "is_international": bool(raw.get("is_international", False)),
        "transaction_type": raw.get("transaction_type", "card_payment"),
        "customer_avg_amount": _decimal_or(raw.get("customer_avg_amount"), 0.0),
        "customer_transaction_count": int(raw.get("customer_transaction_count") or 0),
        "transactions_24h": float(transactions_24h),
        "transactions_7d": float(transactions_7d),
        "customer_international_ratio": _decimal_or(
            raw.get("customer_international_ratio"), 0.0
        ),
    }


@router.post(
    "/score",
    response_model=FraudScoreResponse,
    summary="Model 1 — Fraud Detection score a transaction",
    description=(
        "Returns {transaction_id, fraud_probability, decision} using the "
        "ensemble (Logistic Regression baseline + XGBoost + Isolation Forest "
        "anomaly gate). Decision tiers: APPROVE (<0.35), REVIEW (0.35-0.75), "
        "REJECT (>=0.75)."
    ),
)
def score_transaction(
    req: FraudScoreRequest,
    debug: bool = Query(
        False,
        description="If true, also return component scores and reason codes.",
    ),
):
    try:
        scorer = get_scorer()
        payload = _to_dict(req)
        result = scorer.score_transaction(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"scorer error: {e!r}") from e

    base = {
        "transaction_id": result.transaction_id,
        "fraud_probability": result.fraud_probability,
        "decision": result.decision,
    }

    if debug:
        return FraudScoreDebugResponse(
            **base,
            component_scores=result.component_scores,
            reasons=result.reasons,
        )
    return FraudScoreResponse(**base)
