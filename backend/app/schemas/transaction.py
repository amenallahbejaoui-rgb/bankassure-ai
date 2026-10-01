from datetime import datetime
from decimal import Decimal

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

    model_config = ConfigDict(from_attributes=True)