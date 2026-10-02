from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict


class FraudScoreRequest(BaseModel):
    """Raw transaction input for Model 1 - Fraud Detection.

    Maps to the 9 feature families described in the spec:
      amount, hour, location (country), merchant (merchant_category + merchant_risk),
      international (is_international), customer_avg_amount,
      transaction_frequency (-> transactions_24h / customer_transaction_count),
      merchant_risk, velocity (-> transactions_24h + transactions_7d).
    """

    transaction_id: str
    amount: Decimal
    hour: int
    day_of_week: Optional[int] = None

    country: str
    merchant_category: str
    merchant: Optional[str] = None

    merchant_risk: Decimal
    is_international: bool = False
    transaction_type: str = "card_payment"

    customer_avg_amount: Decimal
    customer_transaction_count: Optional[int] = 0

    transaction_frequency: Optional[Decimal] = None
    transactions_24h: Optional[Decimal] = None
    transactions_7d: Optional[Decimal] = None

    customer_international_ratio: Optional[Decimal] = None

    model_config = ConfigDict(extra="allow")


class FraudScoreResponse(BaseModel):
    """Exact contract required for Model 1 public output."""

    transaction_id: str
    fraud_probability: float
    decision: str


class FraudScoreDebugResponse(FraudScoreResponse):
    """Extended response for internal debugging / dashboard use."""

    component_scores: dict
    reasons: list[str]
