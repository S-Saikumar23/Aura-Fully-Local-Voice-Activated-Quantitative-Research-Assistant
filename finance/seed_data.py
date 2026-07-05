"""
AURA — Synthetic Portfolio Data Generator.

Generates realistic synthetic trading data for ~200 trades across 15 symbols
over a 2-year period, with daily portfolio snapshots and return series.
"""

import random
from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy import func

from finance.database import get_session
from finance.models import Trade, DailyPortfolioSnapshot, PortfolioReturn


# ---------------------------------------------------------------------------
# Symbol universe with realistic price ranges
# ---------------------------------------------------------------------------
SYMBOLS: dict[str, tuple[float, float]] = {
    "AAPL":  (150.0, 230.0),
    "MSFT":  (300.0, 450.0),
    "GOOGL": (120.0, 190.0),
    "TSLA":  (150.0, 400.0),
    "AMZN":  (120.0, 210.0),
    "NVDA":  (400.0, 950.0),
    "META":  (280.0, 530.0),
    "JPM":   (140.0, 210.0),
    "BAC":   (28.0, 42.0),
    "GS":    (310.0, 480.0),
    "XOM":   (95.0, 120.0),
    "CVX":   (140.0, 170.0),
    "PFE":   (25.0, 40.0),
    "JNJ":   (150.0, 175.0),
    "V":     (240.0, 300.0),
}

# Date range for synthetic data
START_DATE = datetime(2024, 1, 2)
END_DATE = datetime(2025, 12, 31)
INITIAL_CASH = 1_000_000.0


def _random_price(symbol: str, bias: float = 0.0) -> float:
    """Generate a random price within the symbol's range, with optional bias."""
    low, high = SYMBOLS[symbol]
    mid = (low + high) / 2
    price = random.gauss(mid + bias, (high - low) / 6)
    return round(max(low * 0.9, min(high * 1.1, price)), 2)


def _generate_trades() -> list[dict]:
    """Generate ~200 synthetic trades across the symbol universe."""
    trades = []
    current_date = START_DATE

    while len(trades) < 200 and current_date <= END_DATE:
        # 1-3 trades per trading day (weekdays only)
        if current_date.weekday() < 5:  # Monday–Friday
            num_trades = random.choices([0, 1, 2, 3], weights=[3, 5, 2, 1])[0]

            for _ in range(num_trades):
                symbol = random.choice(list(SYMBOLS.keys()))
                trade_type = random.choice(["BUY", "SELL"])
                quantity = random.choice([10, 25, 50, 100, 200, 500])
                entry_price = _random_price(symbol)

                # ~70% of trades are closed
                is_closed = random.random() < 0.70
                exit_price = None
                status = "OPEN"

                if is_closed:
                    # Exit price with small random P&L
                    pnl_pct = random.gauss(0.02, 0.08)  # slight positive bias
                    if trade_type == "BUY":
                        exit_price = round(entry_price * (1 + pnl_pct), 2)
                    else:
                        exit_price = round(entry_price * (1 - pnl_pct), 2)
                    exit_price = max(0.01, exit_price)
                    status = "CLOSED"

                trades.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "trade_date": current_date + timedelta(
                        hours=random.randint(9, 15),
                        minutes=random.randint(0, 59),
                    ),
                    "trade_type": trade_type,
                    "status": status,
                })

        current_date += timedelta(days=1)

    return trades[:200]  # Ensure exactly 200


def _generate_snapshots_and_returns() -> tuple[list[dict], list[dict]]:
    """
    Generate daily portfolio snapshots and return series.

    Simulates a random-walk portfolio value starting from INITIAL_CASH.
    """
    snapshots = []
    returns = []

    current_date = START_DATE
    portfolio_value = INITIAL_CASH
    cash_balance = INITIAL_CASH * 0.3  # 30% cash initially

    prev_value = portfolio_value

    while current_date <= END_DATE:
        if current_date.weekday() < 5:  # Weekdays only
            # Random daily return with slight positive drift
            daily_ret = random.gauss(0.0003, 0.012)  # ~7.5% annual return, ~19% vol
            portfolio_value = portfolio_value * (1 + daily_ret)
            cash_balance = cash_balance * (1 + random.gauss(0.0001, 0.001))

            snapshots.append({
                "date": current_date.date(),
                "total_value": round(portfolio_value, 2),
                "cash_balance": round(cash_balance, 2),
            })

            if prev_value > 0:
                ret = (portfolio_value - prev_value) / prev_value
                returns.append({
                    "date": current_date.date(),
                    "daily_return": round(ret, 6),
                })

            prev_value = portfolio_value

        current_date += timedelta(days=1)

    return snapshots, returns


def is_seeded() -> bool:
    """Check if the database already has trade data."""
    with get_session() as session:
        count = session.query(func.count(Trade.id)).scalar()
        return count > 0


def seed(force: bool = False) -> None:
    """
    Populate the database with synthetic portfolio data.

    Args:
        force: If True, clear existing data and re-seed.
    """
    if is_seeded() and not force:
        print("[SEED] Database already contains data. Use force=True to re-seed.")
        return

    print("[SEED] Generating synthetic portfolio data...")

    with get_session() as session:
        if force:
            session.query(PortfolioReturn).delete()
            session.query(DailyPortfolioSnapshot).delete()
            session.query(Trade).delete()
            print("[SEED] Cleared existing data.")

        # Generate and insert trades
        trades_data = _generate_trades()
        for td in trades_data:
            session.add(Trade(
                symbol=td["symbol"],
                quantity=td["quantity"],
                entry_price=Decimal(str(td["entry_price"])),
                exit_price=Decimal(str(td["exit_price"])) if td["exit_price"] else None,
                trade_date=td["trade_date"],
                trade_type=td["trade_type"],
                status=td["status"],
            ))
        print(f"[SEED] Inserted {len(trades_data)} trades.")

        # Generate and insert snapshots + returns
        snapshots_data, returns_data = _generate_snapshots_and_returns()

        for sd in snapshots_data:
            session.add(DailyPortfolioSnapshot(
                date=sd["date"],
                total_value=Decimal(str(sd["total_value"])),
                cash_balance=Decimal(str(sd["cash_balance"])),
            ))

        for rd in returns_data:
            session.add(PortfolioReturn(
                date=rd["date"],
                daily_return=Decimal(str(rd["daily_return"])),
            ))

        print(f"[SEED] Inserted {len(snapshots_data)} daily snapshots.")
        print(f"[SEED] Inserted {len(returns_data)} daily returns.")

    print("[SEED] Synthetic data seeding complete!")


if __name__ == "__main__":
    from finance.database import init_db
    init_db()
    seed(force=True)
