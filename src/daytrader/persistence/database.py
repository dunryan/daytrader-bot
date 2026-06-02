"""Database engine + session management.

Thin wrapper around a SQLAlchemy engine bound to SQLite. Creates the schema on
first use and hands out sessions via a context manager that commits on success
and rolls back on error.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from daytrader.persistence.models import Base
from daytrader.utils.logging_setup import get_logger

logger = get_logger(__name__)


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):  # noqa: ANN001
    """Enable FK enforcement and WAL for better concurrency/durability."""
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()
    except Exception:  # noqa: BLE001 - non-sqlite engines in tests
        pass


class Database:
    """Owns the engine and session factory; creates tables on init."""

    def __init__(self, url: str, echo: bool = False) -> None:
        self.url = url
        # check_same_thread=False so the scheduler's worker threads can share it.
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, echo=echo, future=True, connect_args=connect_args)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        self.create_all()
        logger.info("Database ready at %s", url)

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Transactional session scope."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
