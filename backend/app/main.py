from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db.postgres import check_postgres, create_tables
from app.models.customer import Customer


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