from sqlalchemy.orm import Session

from app.db.postgres import engine


def get_db():
    with Session(engine) as session:
        yield session