from __future__ import annotations

from sqlalchemy import create_engine

from app.infra.db.base import Base
from app.infra.db import models  # noqa: F401


def upgrade(config, revision: str) -> None:
    if revision != "head":
        raise ValueError("The lightweight Alembic fallback only supports upgrading to head")

    url = config.get_main_option("sqlalchemy.url")
    if not url:
        raise ValueError("sqlalchemy.url must be configured before running upgrade")

    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
