from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

from config import settings

Path(settings.database_url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    echo=False,
)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
    # Enable WAL mode and busy timeout for concurrent access
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        conn.exec_driver_sql("PRAGMA busy_timeout=5000")


def get_session() -> Session:
    return Session(engine)
