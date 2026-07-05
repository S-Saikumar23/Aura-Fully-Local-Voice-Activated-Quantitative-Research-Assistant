"""
AURA — Database Connection & Session Management.

Uses SQLAlchemy 2.0+ with PostgreSQL and the pgvector extension.
Provides engine creation, session factory, and table initialization.
Includes lazy engine creation and graceful fallback when DB is unavailable.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase

from config.settings import DATABASE_URL


# ---------------------------------------------------------------------------
# SQLAlchemy Base
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


# ---------------------------------------------------------------------------
# Lazy Engine & Session Factory
# ---------------------------------------------------------------------------
_engine = None
_SessionLocal = None
_db_available = None  # None = not checked yet


def _get_engine():
    """Lazily create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            echo=False,          # Set to True for SQL query debugging
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Reconnect on stale connections
        )
    return _engine


def _get_session_factory():
    """Lazily create the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=_get_engine(), autocommit=False, autoflush=False
        )
    return _SessionLocal


def check_db_connection() -> tuple[bool, str]:
    """
    Check if the database is reachable.

    Returns:
        Tuple of (is_connected, status_message).
    """
    global _db_available
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        _db_available = True
        return True, "PostgreSQL database is connected."
    except Exception as e:
        _db_available = False
        error_msg = str(e)
        if "refused" in error_msg.lower() or "connect" in error_msg.lower():
            return False, (
                "Cannot connect to PostgreSQL. Make sure the database server is running "
                "and the DATABASE_URL in .env is correct."
            )
        if "does not exist" in error_msg.lower():
            return False, (
                "Database 'aura_finance' does not exist. Create it with:\n"
                "  CREATE DATABASE aura_finance;"
            )
        return False, f"Database error: {error_msg}"


def is_db_available() -> bool:
    """Return whether the database is available (checks on first call)."""
    global _db_available
    if _db_available is None:
        _db_available, _ = check_db_connection()
    return _db_available


# ---------------------------------------------------------------------------
# Session Context Manager
# ---------------------------------------------------------------------------
@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Provide a transactional database session.

    Usage::

        with get_session() as session:
            trades = session.query(Trade).all()

    Automatically commits on success or rolls back on exception.
    """
    SessionLocal = _get_session_factory()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Database Initialization
# ---------------------------------------------------------------------------
def init_db() -> None:
    """
    Create all tables and enable the pgvector extension.

    Call this once on application startup or when setting up a new database.
    """
    engine = _get_engine()

    # Enable pgvector extension (requires PostgreSQL superuser or CREATE privilege)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Import models to register them with Base.metadata
    import finance.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    print("[DB] All tables created successfully.")
