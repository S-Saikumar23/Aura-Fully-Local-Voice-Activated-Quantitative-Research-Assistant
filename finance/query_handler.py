"""
AURA — Portfolio Query Handler.

Maps natural-language portfolio questions to analytics functions and
formats the results into conversational responses via the LLM.
"""

import re

from finance import analytics
from core.llm import format_analytics_response


# ---------------------------------------------------------------------------
# Sub-intent patterns for portfolio queries
# ---------------------------------------------------------------------------
_QUERY_PATTERNS = {
    "total_pnl": [
        r"\btotal\s+p\s*&?\s*l\b", r"\btotal\s+profit\b", r"\btotal\s+loss\b",
        r"\boverall\s+p\s*&?\s*l\b", r"\bhow\s+much\s+(?:did\s+)?(?:i|we)\s+make\b",
        r"\boverall\s+(?:profit|loss|performance)\b",
    ],
    "symbol_pnl": [
        r"\bp\s*&?\s*l\s+(?:on|for)\s+(\w+)\b",
        r"\b(\w+)\s+p\s*&?\s*l\b",
        r"\bprofit\s+(?:on|for|from)\s+(\w+)\b",
        r"\bloss\s+(?:on|for|from)\s+(\w+)\b",
        r"\bhow\s+(?:much|did)\s+(?:i|we)\s+(?:make|lose)\s+(?:on|from|with)\s+(\w+)\b",
    ],
    "drawdown": [
        r"\bdrawdown\b", r"\bmax\s+drawdown\b", r"\bmaximum\s+drawdown\b",
        r"\bpeak\s+to\s+trough\b", r"\bbiggest\s+(?:decline|drop)\b",
    ],
    "var": [
        r"\bvar\b", r"\bvalue\s+at\s+risk\b", r"\brisk\s+(?:exposure|measure)\b",
    ],
    "sharpe": [
        r"\bsharpe\b", r"\bsharpe\s+ratio\b",
        r"\brisk[\s-]+adjusted\s+return\b",
    ],
    "volatility": [
        r"\bvolatility\b", r"\bvol\b", r"\bhow\s+(?:volatile|risky)\b",
        r"\bstandard\s+deviation\b",
    ],
    "open_positions": [
        r"\bopen\s+positions?\b", r"\bcurrent\s+(?:positions?|holdings?)\b",
        r"\bwhat\s+(?:am\s+i|are\s+we)\s+holding\b",
        r"\bactive\s+trades?\b",
    ],
    "trade_history": [
        r"\btrade\s+history\b", r"\brecent\s+trades?\b", r"\bpast\s+trades?\b",
        r"\btrading\s+history\b", r"\bshow\s+(?:me\s+)?trades?\b",
    ],
    "summary": [
        r"\bportfolio\s+summary\b", r"\boverall\s+summary\b",
        r"\bhow\s+is\s+my\s+portfolio\b", r"\bportfolio\s+overview\b",
        r"\bhow\s+(?:are\s+we|am\s+i)\s+doing\b",
        r"\bperformance\s+summary\b",
    ],
    "unrealized_pnl": [
        r"\bunrealized\b", r"\bunrealized\s+p\s*&?\s*l\b",
        r"\bpaper\s+(?:profit|loss|gains?)\b",
    ],
}

# Compile patterns
_COMPILED_QUERY_PATTERNS = {
    key: [re.compile(p, re.IGNORECASE) for p in patterns]
    for key, patterns in _QUERY_PATTERNS.items()
}


def _extract_symbol(text: str) -> str | None:
    """Extract a stock symbol from the query text."""
    # Common ticker symbols
    known_symbols = {
        "apple": "AAPL", "aapl": "AAPL",
        "microsoft": "MSFT", "msft": "MSFT",
        "google": "GOOGL", "googl": "GOOGL", "alphabet": "GOOGL",
        "tesla": "TSLA", "tsla": "TSLA",
        "amazon": "AMZN", "amzn": "AMZN",
        "nvidia": "NVDA", "nvda": "NVDA",
        "meta": "META", "facebook": "META",
        "jpmorgan": "JPM", "jpm": "JPM", "jp morgan": "JPM",
        "bank of america": "BAC", "bac": "BAC",
        "goldman sachs": "GS", "goldman": "GS", "gs": "GS",
        "exxon": "XOM", "xom": "XOM", "exxon mobil": "XOM",
        "chevron": "CVX", "cvx": "CVX",
        "pfizer": "PFE", "pfe": "PFE",
        "johnson": "JNJ", "jnj": "JNJ", "johnson and johnson": "JNJ",
        "visa": "V",
    }

    text_lower = text.lower()
    for name, symbol in known_symbols.items():
        if name in text_lower:
            return symbol

    # Try to find an uppercase ticker directly
    ticker_match = re.search(r"\b([A-Z]{1,5})\b", text)
    if ticker_match:
        potential = ticker_match.group(1)
        if potential in {"AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA",
                         "META", "JPM", "BAC", "GS", "XOM", "CVX", "PFE",
                         "JNJ", "V"}:
            return potential

    return None


def _identify_sub_intent(text: str) -> str:
    """Identify the specific portfolio query sub-intent."""
    scores = {}
    for key, patterns in _COMPILED_QUERY_PATTERNS.items():
        match_count = sum(1 for p in patterns if p.search(text))
        if match_count > 0:
            scores[key] = match_count

    if not scores:
        return "summary"  # Default to portfolio summary

    return max(scores, key=scores.get)  # type: ignore[arg-type]


def handle(query: str) -> str:
    """
    Handle a portfolio query by routing to the appropriate analytics function.

    Args:
        query: The user's natural-language portfolio question.

    Returns:
        A natural-language response with the requested analytics data.
    """
    sub_intent = _identify_sub_intent(query)
    print(f"[PORTFOLIO] Sub-intent: {sub_intent}")

    try:
        if sub_intent == "total_pnl":
            data = analytics.get_total_pnl()
            raw = (
                f"Total Realized P&L: ${data['total_pnl']:,.2f}\n"
                f"Closed Trades: {data['num_closed_trades']}\n"
                f"Winners: {data['winning_trades']}, Losers: {data['losing_trades']}\n"
                f"Win Rate: {data['win_rate']}%"
            )

        elif sub_intent == "symbol_pnl":
            symbol = _extract_symbol(query)
            if not symbol:
                return "I couldn't identify the stock symbol. Could you specify it? For example, 'What's my P&L on AAPL?'"
            data = analytics.get_symbol_pnl(symbol)
            if data.get("message"):
                return data["message"]
            raw = (
                f"Symbol: {data['symbol']}\n"
                f"Realized P&L: ${data['pnl']:,.2f}\n"
                f"Closed Trades: {data['num_trades']}\n"
                f"Avg P&L per Trade: ${data['avg_pnl_per_trade']:,.2f}"
            )

        elif sub_intent == "drawdown":
            data = analytics.get_max_drawdown()
            if data.get("message"):
                return data["message"]
            raw = (
                f"Maximum Drawdown: {data['max_drawdown_pct']}%\n"
                f"Peak: ${data['peak_value']:,.2f} on {data['peak_date']}\n"
                f"Trough: ${data['trough_value']:,.2f} on {data['trough_date']}"
            )

        elif sub_intent == "var":
            data = analytics.get_var()
            if data.get("message"):
                return data["message"]
            raw = (
                f"Value at Risk (95%): {data['var_pct']}% daily\n"
                f"Dollar VaR: ${data['var_dollar']:,.2f}\n"
                f"Portfolio Value: ${data['portfolio_value']:,.2f}\n"
                f"Based on {data['observation_days']} days of returns"
            )

        elif sub_intent == "sharpe":
            data = analytics.get_sharpe_ratio()
            if data.get("message"):
                return data["message"]
            raw = (
                f"Sharpe Ratio: {data['sharpe_ratio']}\n"
                f"Annualized Return: {data['annualized_return_pct']}%\n"
                f"Annualized Volatility: {data['annualized_volatility_pct']}%\n"
                f"Risk-Free Rate: {data['risk_free_rate_pct']}%"
            )

        elif sub_intent == "volatility":
            data = analytics.get_volatility()
            if data.get("message"):
                return data["message"]
            raw = (
                f"Daily Volatility: {data['daily_volatility_pct']}%\n"
                f"Annualized Volatility: {data['annualized_volatility_pct']}%\n"
                f"Based on {data['observation_days']} trading days"
            )

        elif sub_intent == "open_positions":
            positions = analytics.get_open_positions()
            if not positions:
                return "You have no open positions currently."
            lines = [f"You have {len(positions)} open positions:\n"]
            for p in positions[:15]:  # Limit to 15 for voice output
                lines.append(
                    f"  • {p['symbol']}: {p['trade_type']} {p['quantity']} shares "
                    f"@ ${p['entry_price']:.2f} (opened {p['trade_date']})"
                )
            if len(positions) > 15:
                lines.append(f"  ... and {len(positions) - 15} more.")
            raw = "\n".join(lines)

        elif sub_intent == "trade_history":
            symbol = _extract_symbol(query)
            trades = analytics.get_trade_history(symbol=symbol, limit=10)
            if not trades:
                return f"No trades found{' for ' + symbol if symbol else ''}."
            lines = [f"Recent trades{' for ' + symbol if symbol else ''}:\n"]
            for t in trades:
                pnl_str = f", P&L: ${t['realized_pnl']:,.2f}" if t['realized_pnl'] is not None else ""
                lines.append(
                    f"  • {t['trade_date'][:10]}: {t['trade_type']} {t['quantity']} "
                    f"{t['symbol']} @ ${t['entry_price']:.2f} [{t['status']}]{pnl_str}"
                )
            raw = "\n".join(lines)

        elif sub_intent == "unrealized_pnl":
            data = analytics.get_unrealized_pnl()
            raw = (
                f"Total Unrealized P&L: ${data['total_unrealized_pnl']:,.2f}\n"
                f"Open Positions: {data['num_open_positions']}\n"
                f"Breakdown by symbol:\n"
            )
            for sym, pnl in sorted(data["by_symbol"].items()):
                raw += f"  • {sym}: ${pnl:,.2f}\n"

        else:  # summary
            data = analytics.get_portfolio_summary()
            raw = (
                f"Portfolio Summary (as of {data['as_of_date']}):\n"
                f"Portfolio Value: ${data['current_portfolio_value']:,.2f}\n"
                f"Cash Balance: ${data['cash_balance']:,.2f}\n"
                f"Realized P&L: ${data['realized_pnl']['total_pnl']:,.2f}\n"
                f"Max Drawdown: {data['max_drawdown'].get('max_drawdown_pct', 0)}%\n"
                f"Sharpe Ratio: {data['sharpe_ratio'].get('sharpe_ratio', 0)}\n"
                f"Volatility: {data['volatility'].get('annualized_volatility_pct', 0)}%\n"
                f"Open Positions: {data['open_positions_count']}"
            )

        # Use LLM to format into natural language
        return format_analytics_response(query, raw)

    except Exception as e:
        print(f"[PORTFOLIO ERROR] {e}")
        return f"I encountered an error while analyzing your portfolio: {str(e)}"
