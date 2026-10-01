from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.account import Account
from app.models.transaction import Transaction
from app.schemas.transaction import (
    TransactionCreate,
    TransactionResponse,
)


router = APIRouter(
    prefix="/transactions",
    tags=["Transactions"],
)


@router.post(
    "",
    response_model=TransactionResponse,
    status_code=201,
)
def create_transaction(
    transaction_data: TransactionCreate,
    db: Session = Depends(get_db),
):
    account = db.get(Account, transaction_data.account_id)

    if account is None:
        raise HTTPException(
            status_code=404,
            detail="Account not found",
        )

    existing_transaction = db.scalar(
        select(Transaction).where(
            Transaction.transaction_number
            == transaction_data.transaction_number
        )
    )

    if existing_transaction:
        raise HTTPException(
            status_code=409,
            detail="Transaction number already exists",
        )

    transaction = Transaction(
        **transaction_data.model_dump()
    )

    db.add(transaction)
    db.commit()
    db.refresh(transaction)

    return transaction


@router.get(
    "",
    response_model=list[TransactionResponse],
)
def list_transactions(
    db: Session = Depends(get_db),
):
    transactions = db.scalars(
        select(Transaction).order_by(Transaction.id)
    ).all()

    return transactions


@router.get(
    "/{transaction_id}",
    response_model=TransactionResponse,
)
def get_transaction(
    transaction_id: int,
    db: Session = Depends(get_db),
):
    transaction = db.get(Transaction, transaction_id)

    if transaction is None:
        raise HTTPException(
            status_code=404,
            detail="Transaction not found",
        )

    return transaction


@router.get(
    "/account/{account_id}",
    response_model=list[TransactionResponse],
)
def list_account_transactions(
    account_id: int,
    db: Session = Depends(get_db),
):
    account = db.get(Account, account_id)

    if account is None:
        raise HTTPException(
            status_code=404,
            detail="Account not found",
        )

    transactions = db.scalars(
        select(Transaction)
        .where(Transaction.account_id == account_id)
        .order_by(Transaction.timestamp.desc())
    ).all()

    return transactions