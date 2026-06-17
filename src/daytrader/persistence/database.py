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


#: Additive, idempotent column migrations for existing SQLite databases.
#: Fresh databases get these columns from ``create_all``; this list only
#: backfills schemas created before the column existed.
_SQLITE_MIGRATIONS: dict[str, dict[str, str]] = {
    "positions": {"atr_at_entry": "FLOAT", "initial_risk": "FLOAT"},
    "signals": {"meta_prob": "FLOAT"},
}


class Database:
    """Owns the engine and session factory; creates tables on init."""

    def __init__(self, url: str, echo: bool = False) -> None:
        self.url = url
        # check_same_thread=False so the scheduler's worker threads can share it.
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, echo=echo, future=True, connect_args=connect_args)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        self.create_all()
        self._migrate()
        logger.info("Database ready at %s", url)

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    def _migrate(self) -> None:
        """Apply additive column migrations (SQLite only, idempotent)."""
        if not self.url.startswith("sqlite"):
            return
        with self.engine.connect() as conn:
            for table, columns in _SQLITE_MIGRATIONS.items():
                existing = {
                    row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table})")
                }
                if not existing:
                    continue  # table missing entirely; create_all owns it
                for column, ddl_type in columns.items():
                    if column not in existing:
                        conn.exec_driver_sql(
                            f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"
                        )
                        logger.info("Migrated: added %s.%s", table, column)
            conn.commit()

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
