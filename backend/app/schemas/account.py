from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class AccountCreate(BaseModel):
    account_number: str
    customer_id: int
    account_type: str
    currency: str
    balance: Decimal = 0
    status: str = "active"


class AccountResponse(AccountCreate):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)