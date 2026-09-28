from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.customer import Customer
from app.schemas.customer import CustomerCreate, CustomerResponse


router = APIRouter(
    prefix="/customers",
    tags=["Customers"],
)


@router.post(
    "",
    response_model=CustomerResponse,
    status_code=201,
)
def create_customer(
    customer_data: CustomerCreate,
    db: Session = Depends(get_db),
):
    existing_customer = db.scalar(
        select(Customer).where(
            Customer.customer_number == customer_data.customer_number
        )
    )

    if existing_customer:
        raise HTTPException(
            status_code=409,
            detail="Customer number already exists",
        )

    existing_email = db.scalar(
        select(Customer).where(
            Customer.email == customer_data.email
        )
    )

    if existing_email:
        raise HTTPException(
            status_code=409,
            detail="Email already exists",
        )

    customer = Customer(**customer_data.model_dump())

    db.add(customer)
    db.commit()
    db.refresh(customer)

    return customer


@router.get(
    "",
    response_model=list[CustomerResponse],
)
def list_customers(
    db: Session = Depends(get_db),
):
    customers = db.scalars(
        select(Customer).order_by(Customer.id)
    ).all()

    return customers


@router.get(
    "/{customer_id}",
    response_model=CustomerResponse,
)
def get_customer(
    customer_id: int,
    db: Session = Depends(get_db),
):
    customer = db.get(Customer, customer_id)

    if customer is None:
        raise HTTPException(
            status_code=404,
            detail="Customer not found",
        )

    return customer