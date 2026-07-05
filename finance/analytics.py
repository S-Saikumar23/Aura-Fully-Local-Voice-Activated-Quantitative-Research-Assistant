"""
AURA — Portfolio Analytics Engine.

Provides quantitative analytics functions for the trading portfolio:
P&L, VaR, Sharpe ratio, max drawdown, volatility, and position summaries.
All calculations use data from the PostgreSQL database.
"""

from datetime import date, datetime
from decimal import Decimal

import numpy as np
from sqlalchemy import func, and_

from finance.database import get_session
from finance.models import Trade, DailyPortfolioSnapshot, PortfolioReturn


# ---------------------------------------------------------------------------
# Profit & Loss
# ---------------------------------------------------------------------------

def get_total_pnl() -> dict:
    """
    Calculate total realized P&L across all closed trades.

    Returns:
        Dict with total_pnl, num_trades, winning_trades, losing_trades.
    """
    with get_session() as session:
        closed_trades = (
            session.query(Trade)
            .filter(Trade.status == "CLOSED")
            .all()
        )

        # Process INSIDE the session to avoid DetachedInstanceError
        total_pnl = 0.0
        winners = 0
        losers = 0

        for trade in closed_trades:
            pnl = trade.realized_pnl
            if pnl is not None:
                total_pnl += pnl
                if pnl >= 0:
                    winners += 1
                else:
                    losers += 1

        num_closed = len(closed_trades)

    return {
        "total_pnl": round(total_pnl, 2),
        "num_closed_trades": num_closed,
        "winning_trades": winners,
        "losing_trades": losers,
        "win_rate": round(winners / max(num_closed, 1) * 100, 1),
    }


def get_symbol_pnl(symbol: str) -> dict:
    """
    Calculate realized P&L for a specific symbol.

    Args:
        symbol: Ticker symbol (e.g., "AAPL").

    Returns:
        Dict with symbol, pnl, num_trades, avg_pnl_per_trade.
    """
    symbol = symbol.upper()
    with get_session() as session:
        trades = (
            session.query(Trade)
            .filter(and_(Trade.symbol == symbol, Trade.status == "CLOSED"))
            .all()
        )

        if not trades:
            return {
                "symbol": symbol,
                "pnl": 0.0,
                "num_trades": 0,
                "avg_pnl_per_trade": 0.0,
                "message": f"No closed trades found for {symbol}.",
            }

        # Process INSIDE the session to avoid DetachedInstanceError
        total_pnl = sum(t.realized_pnl or 0.0 for t in trades)
        num_trades = len(trades)

    return {
        "symbol": symbol,
        "pnl": round(total_pnl, 2),
        "num_trades": num_trades,
        "avg_pnl_per_trade": round(total_pnl / num_trades, 2),
    }


def get_unrealized_pnl(current_prices: dict[str, float] | None = None) -> dict:
    """
    Calculate unrealized P&L for all open positions.

    Args:
        current_prices: Dict mapping symbol to current price. If None,
                        uses entry_price * 1.02 as a rough estimate.

    Returns:
        Dict with total unrealized P&L and per-symbol breakdown.
    """
    with get_session() as session:
        open_trades = (
            session.query(Trade)
            .filter(Trade.status == "OPEN")
            .all()
        )

        # Process INSIDE the session to avoid DetachedInstanceError
        breakdown = {}
        total_unrealized = 0.0
        num_open = len(open_trades)

        for trade in open_trades:
            current = (
                current_prices.get(trade.symbol, float(trade.entry_price) * 1.02)
                if current_prices
                else float(trade.entry_price) * 1.02
            )
            if trade.trade_type == "BUY":
                pnl = (current - float(trade.entry_price)) * trade.quantity
            else:
                pnl = (float(trade.entry_price) - current) * trade.quantity

            if trade.symbol not in breakdown:
                breakdown[trade.symbol] = 0.0
            breakdown[trade.symbol] += pnl
            total_unrealized += pnl

    return {
        "total_unrealized_pnl": round(total_unrealized, 2),
        "by_symbol": {k: round(v, 2) for k, v in breakdown.items()},
        "num_open_positions": num_open,
    }


# ---------------------------------------------------------------------------
# Risk Metrics
# ---------------------------------------------------------------------------

def get_max_drawdown() -> dict:
    """
    Calculate maximum drawdown from daily portfolio snapshots.

    Returns:
        Dict with max_drawdown (%), peak_date, trough_date.
    """
    with get_session() as session:
        snapshots = (
            session.query(DailyPortfolioSnapshot)
            .order_by(DailyPortfolioSnapshot.date)
            .all()
        )

        if len(snapshots) < 2:
            return {"max_drawdown_pct": 0.0, "message": "Insufficient data."}

        # Extract data INSIDE the session
        values = [float(s.total_value) for s in snapshots]
        dates = [s.date for s in snapshots]

    peak = values[0]
    peak_idx = 0
    max_dd = 0.0
    dd_peak_idx = 0
    dd_trough_idx = 0

    for i in range(1, len(values)):
        if values[i] > peak:
            peak = values[i]
            peak_idx = i

        dd = (peak - values[i]) / peak
        if dd > max_dd:
            max_dd = dd
            dd_peak_idx = peak_idx
            dd_trough_idx = i

    return {
        "max_drawdown_pct": round(max_dd * 100, 2),
        "peak_date": str(dates[dd_peak_idx]),
        "peak_value": values[dd_peak_idx],
        "trough_date": str(dates[dd_trough_idx]),
        "trough_value": values[dd_trough_idx],
    }


def get_var(confidence: float = 0.95) -> dict:
    """
    Calculate historical Value at Risk.

    Args:
        confidence: Confidence level (default 0.95 = 95%).

    Returns:
        Dict with VaR percentage and dollar amount.
    """
    with get_session() as session:
        returns = (
            session.query(PortfolioReturn.daily_return)
            .order_by(PortfolioReturn.date)
            .all()
        )
        latest = (
            session.query(DailyPortfolioSnapshot.total_value)
            .order_by(DailyPortfolioSnapshot.date.desc())
            .first()
        )

        # Extract values INSIDE the session
        return_values_raw = [r[0] for r in returns if r[0] is not None]
        portfolio_value = float(latest[0]) if latest else 1_000_000.0

    if len(return_values_raw) < 30:
        return {"var_pct": 0.0, "message": "Insufficient return data for VaR."}

    return_values = np.array([float(r) for r in return_values_raw])
    var_pct = float(np.percentile(return_values, (1 - confidence) * 100))

    return {
        "confidence": confidence,
        "var_pct": round(var_pct * 100, 4),
        "var_dollar": round(abs(var_pct) * portfolio_value, 2),
        "portfolio_value": round(portfolio_value, 2),
        "observation_days": len(return_values),
    }


def get_sharpe_ratio(risk_free_rate: float = 0.04) -> dict:
    """
    Calculate annualized Sharpe ratio.

    Args:
        risk_free_rate: Annual risk-free rate (default 4%).

    Returns:
        Dict with sharpe_ratio, annualized_return, annualized_volatility.
    """
    with get_session() as session:
        returns = (
            session.query(PortfolioReturn.daily_return)
            .order_by(PortfolioReturn.date)
            .all()
        )

        # Extract values INSIDE the session
        return_values_raw = [r[0] for r in returns if r[0] is not None]

    if len(return_values_raw) < 30:
        return {"sharpe_ratio": 0.0, "message": "Insufficient data."}

    return_values = np.array([float(r) for r in return_values_raw])

    daily_rf = risk_free_rate / 252  # Trading days in a year
    excess_returns = return_values - daily_rf

    mean_excess = float(np.mean(excess_returns))
    std_excess = float(np.std(excess_returns, ddof=1))

    if std_excess == 0:
        sharpe = 0.0
    else:
        sharpe = (mean_excess / std_excess) * np.sqrt(252)  # Annualize

    ann_return = float(np.mean(return_values)) * 252
    ann_vol = float(np.std(return_values, ddof=1)) * np.sqrt(252)

    return {
        "sharpe_ratio": round(float(sharpe), 4),
        "annualized_return_pct": round(ann_return * 100, 2),
        "annualized_volatility_pct": round(ann_vol * 100, 2),
        "risk_free_rate_pct": round(risk_free_rate * 100, 2),
    }


def get_volatility() -> dict:
    """
    Calculate annualized portfolio volatility.

    Returns:
        Dict with daily_vol and annualized_vol.
    """
    with get_session() as session:
        returns = (
            session.query(PortfolioReturn.daily_return)
            .order_by(PortfolioReturn.date)
            .all()
        )

        # Extract values INSIDE the session
        return_values_raw = [r[0] for r in returns if r[0] is not None]

    if len(return_values_raw) < 2:
        return {"annualized_volatility_pct": 0.0, "message": "Insufficient data."}

    return_values = np.array([float(r) for r in return_values_raw])

    daily_vol = float(np.std(return_values, ddof=1))
    ann_vol = daily_vol * np.sqrt(252)

    return {
        "daily_volatility_pct": round(daily_vol * 100, 4),
        "annualized_volatility_pct": round(float(ann_vol) * 100, 2),
        "observation_days": len(return_values),
    }


# ---------------------------------------------------------------------------
# Position & Trade Queries
# ---------------------------------------------------------------------------

def get_open_positions() -> list[dict]:
    """Return all currently open trades."""
    with get_session() as session:
        trades = (
            session.query(Trade)
            .filter(Trade.status == "OPEN")
            .order_by(Trade.symbol, Trade.trade_date)
            .all()
        )

        # Convert to dicts INSIDE the session to avoid DetachedInstanceError
        return [
            {
                "symbol": t.symbol,
                "trade_type": t.trade_type,
                "quantity": t.quantity,
                "entry_price": float(t.entry_price),
                "trade_date": str(t.trade_date.date()),
            }
            for t in trades
        ]


def get_trade_history(
    symbol: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int = 50,
) -> list[dict]:
    """
    Retrieve filtered trade history.

    Args:
        symbol: Optional symbol filter.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        limit: Maximum number of records (default 50).

    Returns:
        List of trade dicts ordered by date descending.
    """
    with get_session() as session:
        query = session.query(Trade)

        if symbol:
            query = query.filter(Trade.symbol == symbol.upper())
        if start_date:
            query = query.filter(Trade.trade_date >= datetime.combine(start_date, datetime.min.time()))
        if end_date:
            query = query.filter(Trade.trade_date <= datetime.combine(end_date, datetime.max.time()))

        trades = query.order_by(Trade.trade_date.desc()).limit(limit).all()

        # Convert to dicts INSIDE the session to avoid DetachedInstanceError
        return [
            {
                "id": t.id,
                "symbol": t.symbol,
                "trade_type": t.trade_type,
                "quantity": t.quantity,
                "entry_price": float(t.entry_price),
                "exit_price": float(t.exit_price) if t.exit_price else None,
                "trade_date": str(t.trade_date),
                "status": t.status,
                "realized_pnl": t.realized_pnl,
            }
            for t in trades
        ]


# ---------------------------------------------------------------------------
# Portfolio Summary
# ---------------------------------------------------------------------------

def get_portfolio_summary() -> dict:
    """
    Get a comprehensive portfolio summary combining all key metrics.

    Returns:
        Dict with all major analytics in one response.
    """
    with get_session() as session:
        latest = (
            session.query(DailyPortfolioSnapshot)
            .order_by(DailyPortfolioSnapshot.date.desc())
            .first()
        )
        # Extract values INSIDE the session
        current_value = float(latest.total_value) if latest else 0.0
        cash = float(latest.cash_balance) if latest else 0.0
        as_of = str(latest.date) if latest else "N/A"

    pnl = get_total_pnl()
    drawdown = get_max_drawdown()
    var = get_var()
    sharpe = get_sharpe_ratio()
    vol = get_volatility()
    open_pos = get_open_positions()

    return {
        "current_portfolio_value": current_value,
        "cash_balance": cash,
        "as_of_date": as_of,
        "realized_pnl": pnl,
        "max_drawdown": drawdown,
        "value_at_risk_95": var,
        "sharpe_ratio": sharpe,
        "volatility": vol,
        "open_positions_count": len(open_pos),
    }
