from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.infra.db.base import Base


def _normalize_database_url(database_url: str) -> str:
    return database_url.replace("postgresql+psycopg", "postgresql+psycopg2")


from sqlalchemy.pool import StaticPool

@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    url = _normalize_database_url(settings.database_url)
    if url == "sqlite:///:memory:":
        engine = create_engine(
            url, 
            future=True, 
            connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
    else:
        engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    return engine


def create_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)
