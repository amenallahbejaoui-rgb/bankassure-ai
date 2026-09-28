from datetime import date, datetime

from pydantic import BaseModel, ConfigDict


class CustomerCreate(BaseModel):
    customer_number: str
    first_name: str
    last_name: str
    email: str
    phone: str
    date_of_birth: date
    country: str
    city: str


class CustomerResponse(CustomerCreate):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)