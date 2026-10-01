from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.account import Account
from app.models.customer import Customer
from app.schemas.account import AccountCreate, AccountResponse


router = APIRouter(
    prefix="/accounts",
    tags=["Accounts"],
)


@router.post(
    "",
    response_model=AccountResponse,
    status_code=201,
)
def create_account(
    account_data: AccountCreate,
    db: Session = Depends(get_db),
):
    customer = db.get(Customer, account_data.customer_id)

    if customer is None:
        raise HTTPException(
            status_code=404,
            detail="Customer not found",
        )

    existing_account = db.scalar(
        select(Account).where(
            Account.account_number == account_data.account_number
        )
    )

    if existing_account:
        raise HTTPException(
            status_code=409,
            detail="Account number already exists",
        )

    account = Account(**account_data.model_dump())

    db.add(account)
    db.commit()
    db.refresh(account)

    return account


@router.get(
    "",
    response_model=list[AccountResponse],
)
def list_accounts(
    db: Session = Depends(get_db),
):
    accounts = db.scalars(
        select(Account).order_by(Account.id)
    ).all()

    return accounts


@router.get(
    "/{account_id}",
    response_model=AccountResponse,
)
def get_account(
    account_id: int,
    db: Session = Depends(get_db),
):
    account = db.get(Account, account_id)

    if account is None:
        raise HTTPException(
            status_code=404,
            detail="Account not found",
        )

    return account


@router.get(
    "/customer/{customer_id}",
    response_model=list[AccountResponse],
)
def list_customer_accounts(
    customer_id: int,
    db: Session = Depends(get_db),
):
    customer = db.get(Customer, customer_id)

    if customer is None:
        raise HTTPException(
            status_code=404,
            detail="Customer not found",
        )

    accounts = db.scalars(
        select(Account)
        .where(Account.customer_id == customer_id)
        .order_by(Account.id)
    ).all()

    return accounts