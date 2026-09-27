from sqlalchemy import create_engine, text

from app.core.config import settings


DATABASE_URL = (
    f"postgresql+psycopg://"
    f"{settings.postgres_user}:{settings.postgres_password}"
    f"@{settings.postgres_host}:{settings.postgres_port}"
    f"/{settings.postgres_db}"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


def check_postgres() -> bool:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    return True