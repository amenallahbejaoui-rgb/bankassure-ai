from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.account import Account
from app.models.transaction import Transaction
from app.schemas.transaction import (
    TransactionCreate,
    TransactionFraudReScoreResponse,
    TransactionResponse,
)
from app.services.fraud_service import (
    apply_fraud_decision,
    build_scorer_payload,
    score_transaction_row,
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
    run_fraud_scoring: bool = True,
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

    # --- Model 1 — Fraud Detection live guardrail ---
    if run_fraud_scoring:
        try:
            payload = build_scorer_payload(
                transaction_id=transaction.transaction_number,
                amount=transaction.amount,
                timestamp=transaction.timestamp,
                transaction_type=transaction.transaction_type,
                merchant=transaction.merchant,
                merchant_category=transaction.merchant_category,
                country=transaction.country,
                is_international=transaction.is_international,
                merchant_risk=transaction.merchant_risk,
                account_id=transaction.account_id,
                db=db,
            )
            result = score_transaction_row(payload)
            apply_fraud_decision(transaction, result)
        except Exception as exc:  # pragma: no cover - scoring is best-effort
            # Never fail the create because a model is down.
            transaction.fraud_decision = "ERROR"
            transaction.fraud_reasons = [f"scorer_error:{type(exc).__name__}"]

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


@router.post(
    "/{transaction_id}/fraud/rescore",
    response_model=TransactionFraudReScoreResponse,
    summary="Re-score an existing transaction with the Model 1 ensemble",
)
def rescore_transaction_fraud(
    transaction_id: int,
    persist: bool = True,
    db: Session = Depends(get_db),
):
    """Run (or re-run) the Model 1 fraud scorer on an existing transaction.

    Useful after a model retrain or when analysts want to re-evaluate a
    case.  ``persist=True`` (default) also saves the updated scoring back
    onto the transaction row.
    """
    transaction = db.get(Transaction, transaction_id)
    if transaction is None:
        raise HTTPException(
            status_code=404,
            detail="Transaction not found",
        )

    payload = build_scorer_payload(
        transaction_id=transaction.transaction_number,
        amount=transaction.amount,
        timestamp=transaction.timestamp,
        transaction_type=transaction.transaction_type,
        merchant=transaction.merchant,
        merchant_category=transaction.merchant_category,
        country=transaction.country,
        is_international=transaction.is_international,
        merchant_risk=transaction.merchant_risk,
        account_id=transaction.account_id,
        db=db,
    )
    result = score_transaction_row(payload)

    if persist:
        apply_fraud_decision(transaction, result)
        db.commit()
        db.refresh(transaction)

    return TransactionFraudReScoreResponse(
        id=transaction.id,
        transaction_number=transaction.transaction_number,
        fraud_probability=result.fraud_probability,
        decision=result.decision,
        reasons=list(result.reasons),
        component_scores=result.component_scores,
    )


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