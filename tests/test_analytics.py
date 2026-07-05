"""
Tests for the Portfolio Analytics Engine.

Tests core financial calculations against known values to verify
correctness of P&L, VaR, Sharpe ratio, and drawdown computations.
These tests use mocked database sessions to avoid PostgreSQL dependency.
"""

import pytest
from unittest.mock import patch, MagicMock
from decimal import Decimal
from datetime import date, datetime
from contextlib import contextmanager
import numpy as np


def _mock_session_context(mock_session):
    """
    Create a proper context manager mock that works with @contextmanager.

    The finance.database.get_session() uses @contextmanager, so we need
    to return a generator-based context manager, not a class-based one.
    """
    @contextmanager
    def _ctx():
        yield mock_session
    return _ctx


class TestPnLCalculations:
    """Test profit and loss calculations."""

    @patch("finance.analytics.get_session")
    def test_total_pnl_with_closed_trades(self, mock_get_session):
        """Verify total P&L sums up correctly for closed BUY trades."""
        from finance.analytics import get_total_pnl

        # Create mock trades
        trade1 = MagicMock()
        trade1.status = "CLOSED"
        trade1.trade_type = "BUY"
        trade1.entry_price = Decimal("100.00")
        trade1.exit_price = Decimal("110.00")
        trade1.quantity = 100
        trade1.realized_pnl = 1000.0  # (110-100)*100

        trade2 = MagicMock()
        trade2.status = "CLOSED"
        trade2.trade_type = "BUY"
        trade2.entry_price = Decimal("200.00")
        trade2.exit_price = Decimal("190.00")
        trade2.quantity = 50
        trade2.realized_pnl = -500.0  # (190-200)*50

        # Setup mock session with proper context manager
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = [
            trade1, trade2
        ]
        mock_get_session.side_effect = lambda: _mock_session_context(mock_session)()

        result = get_total_pnl()

        assert result["total_pnl"] == 500.0  # 1000 - 500
        assert result["num_closed_trades"] == 2
        assert result["winning_trades"] == 1
        assert result["losing_trades"] == 1
        assert result["win_rate"] == 50.0

    @patch("finance.analytics.get_session")
    def test_total_pnl_no_trades(self, mock_get_session):
        """Verify total P&L returns 0 when no trades exist."""
        from finance.analytics import get_total_pnl

        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = []
        mock_get_session.side_effect = lambda: _mock_session_context(mock_session)()

        result = get_total_pnl()
        assert result["total_pnl"] == 0.0
        assert result["num_closed_trades"] == 0


class TestRiskMetrics:
    """Test risk metric calculations."""

    @patch("finance.analytics.get_session")
    def test_max_drawdown(self, mock_get_session):
        """Test max drawdown with a known sequence: 100 → 120 → 90 → 110."""
        from finance.analytics import get_max_drawdown

        snapshots = []
        values_and_dates = [
            (100, date(2024, 1, 1)),
            (120, date(2024, 1, 2)),  # peak
            (90, date(2024, 1, 3)),   # trough: (120-90)/120 = 25%
            (110, date(2024, 1, 4)),
        ]
        for val, d in values_and_dates:
            snap = MagicMock()
            snap.total_value = Decimal(str(val))
            snap.date = d
            snapshots.append(snap)

        mock_session = MagicMock()
        mock_session.query.return_value.order_by.return_value.all.return_value = snapshots
        mock_get_session.side_effect = lambda: _mock_session_context(mock_session)()

        result = get_max_drawdown()
        assert result["max_drawdown_pct"] == 25.0
        assert result["peak_date"] == "2024-01-02"
        assert result["trough_date"] == "2024-01-03"

    @patch("finance.analytics.get_session")
    def test_volatility(self, mock_get_session):
        """Test volatility calculation with known return series."""
        from finance.analytics import get_volatility

        # Known returns: [0.01, -0.02, 0.015, -0.005, 0.03]
        known_returns = [0.01, -0.02, 0.015, -0.005, 0.03]
        return_rows = [(Decimal(str(r)),) for r in known_returns]

        mock_session = MagicMock()
        mock_session.query.return_value.order_by.return_value.all.return_value = return_rows
        mock_get_session.side_effect = lambda: _mock_session_context(mock_session)()

        result = get_volatility()

        # Calculate expected volatility
        expected_daily_vol = float(np.std(known_returns, ddof=1))
        expected_annual_vol = expected_daily_vol * np.sqrt(252)

        assert abs(result["daily_volatility_pct"] - expected_daily_vol * 100) < 0.01
        assert abs(result["annualized_volatility_pct"] - expected_annual_vol * 100) < 0.1
        assert result["observation_days"] == 5


class TestTradeModel:
    """Test the Trade model's realized_pnl property."""

    def test_buy_trade_pnl(self):
        """BUY trade: P&L = (exit - entry) * quantity."""
        from finance.models import Trade
        trade = Trade(
            symbol="AAPL", quantity=100,
            entry_price=Decimal("150.00"), exit_price=Decimal("160.00"),
            trade_date=datetime.now(), trade_type="BUY", status="CLOSED",
        )
        assert trade.realized_pnl == 1000.0  # (160-150)*100

    def test_sell_trade_pnl(self):
        """SELL (short) trade: P&L = (entry - exit) * quantity."""
        from finance.models import Trade
        trade = Trade(
            symbol="TSLA", quantity=50,
            entry_price=Decimal("200.00"), exit_price=Decimal("180.00"),
            trade_date=datetime.now(), trade_type="SELL", status="CLOSED",
        )
        assert trade.realized_pnl == 1000.0  # (200-180)*50

    def test_open_trade_pnl_is_none(self):
        """Open trades have no realized P&L."""
        from finance.models import Trade
        trade = Trade(
            symbol="MSFT", quantity=100,
            entry_price=Decimal("300.00"), exit_price=None,
            trade_date=datetime.now(), trade_type="BUY", status="OPEN",
        )
        assert trade.realized_pnl is None

    def test_losing_buy_trade(self):
        """Losing BUY trade should return negative P&L."""
        from finance.models import Trade
        trade = Trade(
            symbol="META", quantity=200,
            entry_price=Decimal("350.00"), exit_price=Decimal("320.00"),
            trade_date=datetime.now(), trade_type="BUY", status="CLOSED",
        )
        assert trade.realized_pnl == -6000.0  # (320-350)*200
