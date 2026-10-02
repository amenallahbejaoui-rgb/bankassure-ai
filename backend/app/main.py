from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from app.api.accounts import router as accounts_router
from app.api.customers import router as customers_router
from app.api.fraud import router as fraud_router
from app.db.postgres import (
    REQUIRED_COLUMNS,
    _get_missing,
    check_postgres,
    create_tables,
    engine,
    run_model_migrations,
)
from sqlalchemy import inspect
from app.models.customer import Customer
from app.api.transactions import router as transactions_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    yield


app = FastAPI(
    title="BankAssure AI",
    description="AI-powered banking and insurance intelligence platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(customers_router)
app.include_router(accounts_router)
app.include_router(transactions_router)
app.include_router(fraud_router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "bankassure-api",
    }


@app.get("/system/postgres")
def postgres_health():
    connected = check_postgres()

    return {
        "service": "postgresql",
        "status": "connected" if connected else "disconnected",
    }


@app.post(
    "/system/fraud-migrate",
    summary="Apply Model 1 fraud-column migrations to the transactions table",
)
def post_system_fraud_migrate(dry_run: bool = False) -> dict[str, Any]:
    """Apply idempotent ALTERs for fraud_score / fraud_decision /
    fraud_reasons columns.  ``dry_run=true`` tells you what WOULD happen
    without running any SQL.  Safe to call repeatedly."""
    try:
        inspector = inspect(engine)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"cannot connect to postgres: {exc!r}",
        ) from exc

    plan: dict[str, list[str]] = {}
    for table, columns in REQUIRED_COLUMNS.items():
        missing = _get_missing(inspector, table, columns.keys())
        plan[table] = [
            f"ADD COLUMN {col} {columns[col].get('postgresql', '?')}"
            for col in missing
        ]

    if dry_run:
        return {
            "status": "dry_run",
            "applied": False,
            "pending": plan,
        }

    applied_any = any(plan.values())
    try:
        run_model_migrations()
    except Exception as exc:  # pragma: no cover - surfaced as HTTP 500
        raise HTTPException(
            status_code=500,
            detail=f"migration failed: {exc!r}",
        ) from exc

    return {
        "status": "ok" if applied_any else "already_up_to_date",
        "applied": applied_any,
        "changes": plan,
    }