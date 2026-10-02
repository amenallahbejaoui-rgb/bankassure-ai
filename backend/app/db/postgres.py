from __future__ import annotations

import logging
from typing import Iterable

from sqlalchemy import Inspector, create_engine, inspect, text
from sqlalchemy.engine import Dialect

from app.core.config import settings
from app.db.base import Base
from app.models.account import Account
from app.models.customer import Customer
from app.models.transaction import Transaction


log = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Safe, online, idempotent ALTER migrations for Model 1 fraud columns.
#
# Design:
#   - Run AFTER Base.metadata.create_all(...) so brand-new tables get all
#     columns for free.
#   - For tables that already exist, use Inspector to check each expected
#     column, and only issue ALTER TABLE ADD COLUMN for the ones missing.
#   - Each entry declares dialect-specific SQL so we can add SQLite/text
#     fallbacks later if needed.  For now this project is Postgres-only.
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: dict[str, dict[str, dict[str, str]]] = {
    "transactions": {
        "fraud_score": {
            "postgresql": "DOUBLE PRECISION",
        },
        "fraud_decision": {
            "postgresql": "VARCHAR(20)",
        },
        "fraud_reasons": {
            "postgresql": "JSONB",
        },
    },
}


def _dialect_name(dialect: Dialect) -> str:
    name = dialect.name.lower()
    if "postgres" in name:
        return "postgresql"
    return name


def _get_missing(inspector: Inspector, table: str, columns: Iterable[str]) -> list[str]:
    try:
        existing = {c["name"] for c in inspector.get_columns(table)}
    except Exception:
        # Table doesn't exist yet; create_all() will handle it.  Nothing to
        # ALTER in that case.
        return []
    return [c for c in columns if c not in existing]


def run_model_migrations() -> None:
    """Apply any pending Model-1 ALTER migrations. Idempotent: safe to call
    on every server startup."""
    inspector = inspect(engine)
    dialect = engine.dialect
    dialect_name = _dialect_name(dialect)

    for table, columns in REQUIRED_COLUMNS.items():
        missing = _get_missing(inspector, table, columns.keys())
        if not missing:
            continue

        with engine.begin() as conn:
            for column_name in missing:
                spec = columns[column_name]
                if dialect_name not in spec:
                    log.warning(
                        "Skipping column %s.%s — no type spec for dialect %s",
                        table, column_name, dialect_name,
                    )
                    continue
                type_sql = spec[dialect_name]
                # NOTE: existence is already guaranteed-missing by
                # _get_missing() earlier, so we don't need "IF NOT EXISTS"
                # SQL syntax (which is not supported on SQLite / older DBs).
                stmt = text(
                    f"ALTER TABLE {table} ADD COLUMN "
                    f"{column_name} {type_sql}"
                )
                log.info("Auto-migrate: %s", stmt.text)
                conn.execute(stmt)


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
    try:
        run_model_migrations()
    except Exception as exc:  # pragma: no cover - best-effort safety net
        # Never crash the server startup because of a migration issue.
        # create_all() already guaranteed the core schema is usable; the
        # fraud columns are additive.  Log loudly so the operator sees it.
        log.error(
            "run_model_migrations() failed — fraud columns may be missing. "
            "Error: %r",
            exc,
        )


def _cli_print_migration_status() -> int:
    """Human-readable CLI for checking / running Model 1 migrations.

    Usage (from ``backend/`` directory with the venv activated)::

        python -m app.db.postgres dry-run
        python -m app.db.postgres apply
    """
    import sys

    args = sys.argv[1:]
    cmd = args[0] if args else "dry-run"
    inspector = inspect(engine)

    pending: dict[str, list[str]] = {}
    for table, columns in REQUIRED_COLUMNS.items():
        missing = _get_missing(inspector, table, columns.keys())
        pending[table] = missing

    if cmd == "dry-run" or cmd == "check":
        total = sum(len(v) for v in pending.values())
        if total == 0:
            print("[OK] Model 1 fraud columns are all present.  Nothing to do.")
            return 0
        print(f"[PENDING] {total} missing column(s):")
        for table, cols in pending.items():
            for col in cols:
                type_sql = REQUIRED_COLUMNS[table][col].get("postgresql", "?")
                print(f"  - {table}.{col}  ({type_sql})")
        print()
        print('Run:  python -m app.db.postgres apply')
        return 1

    if cmd == "apply":
        total = sum(len(v) for v in pending.values())
        if total == 0:
            print("[OK] Already up to date.  No changes applied.")
            return 0
        run_model_migrations()
        print(f"[APPLIED] {total} migration(s):")
        for table, cols in pending.items():
            for col in cols:
                print(f"  + {table}.{col}")
        return 0

    print(f"Unknown command: {cmd!r}.  Use 'dry-run' or 'apply'.")
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_cli_print_migration_status())
