"""
AURA — SQLAlchemy ORM Models.

Defines all database tables for the portfolio tracking and research
document storage system. Uses pgvector for embedding storage.
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Numeric,
    Text,
    DateTime,
    Date,
    CheckConstraint,
)
from pgvector.sqlalchemy import Vector

from finance.database import Base
from config.settings import EMBEDDING_DIM


class Trade(Base):
    """
    Individual trade records (buys and sells).

    Tracks entry/exit prices, quantities, and trade status.
    """
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Numeric(10, 2), nullable=False)
    exit_price = Column(Numeric(10, 2), nullable=True)
    trade_date = Column(DateTime, nullable=False, index=True)
    trade_type = Column(
        String(4),
        CheckConstraint("trade_type IN ('BUY', 'SELL')"),
        nullable=False,
    )
    status = Column(String(10), default="OPEN", index=True)

    def __repr__(self) -> str:
        return (
            f"<Trade(id={self.id}, symbol='{self.symbol}', "
            f"type='{self.trade_type}', qty={self.quantity}, "
            f"entry={self.entry_price}, status='{self.status}')>"
        )

    @property
    def realized_pnl(self) -> float | None:
        """Calculate realized P&L for closed trades."""
        if self.status != "CLOSED" or self.exit_price is None:
            return None
        if self.trade_type == "BUY":
            return float(self.exit_price - self.entry_price) * self.quantity
        else:  # SELL (short)
            return float(self.entry_price - self.exit_price) * self.quantity


class DailyPortfolioSnapshot(Base):
    """
    End-of-day portfolio valuations.

    Records total portfolio value and cash balance for each trading day.
    """
    __tablename__ = "daily_portfolio_snapshots"

    date = Column(Date, primary_key=True)
    total_value = Column(Numeric(15, 2), nullable=False)
    cash_balance = Column(Numeric(15, 2), nullable=False)

    def __repr__(self) -> str:
        return (
            f"<Snapshot(date={self.date}, value={self.total_value}, "
            f"cash={self.cash_balance})>"
        )


class PortfolioReturn(Base):
    """
    Daily portfolio returns.

    Stores the daily percentage return for performance analytics.
    """
    __tablename__ = "portfolio_returns"

    date = Column(Date, primary_key=True)
    daily_return = Column(Numeric(10, 6), nullable=True)

    def __repr__(self) -> str:
        return f"<Return(date={self.date}, return={self.daily_return})>"


class ResearchDocument(Base):
    """
    Ingested research document chunks with vector embeddings.

    Used by the RAG pipeline for semantic search over PDF content.
    """
    __tablename__ = "research_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=True, index=True)
    content = Column(Text, nullable=True)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=True)

    def __repr__(self) -> str:
        preview = (self.content[:50] + "...") if self.content and len(self.content) > 50 else self.content
        return f"<ResearchDoc(id={self.id}, file='{self.filename}', content='{preview}')>"
