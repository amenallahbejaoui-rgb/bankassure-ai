from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.api.accounts import router as accounts_router
from app.api.customers import router as customers_router
from app.api.fraud import router as fraud_router
from app.db.postgres import check_postgres, create_tables
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