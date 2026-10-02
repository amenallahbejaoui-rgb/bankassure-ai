from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class TransactionCreate(BaseModel):
    transaction_number: str
    account_id: int
    amount: Decimal
    currency: str
    transaction_type: str
    timestamp: datetime
    merchant: str
    merchant_category: str
    country: str
    is_international: bool = False
    merchant_risk: Decimal = 0
    is_fraud: bool = False
    status: str = "completed"


class TransactionResponse(TransactionCreate):
    id: int

    # --- Model 1 — Fraud Detection (server-populated, always read-only) ---
    fraud_score: Optional[float] = None
    fraud_decision: Optional[str] = None
    fraud_reasons: Optional[list[str]] = None

    model_config = ConfigDict(from_attributes=True)


class TransactionFraudReScoreResponse(BaseModel):
    id: int
    transaction_number: str
    fraud_probability: float
    decision: str
    reasons: list[str]
    component_scores: dict[str, Any]
