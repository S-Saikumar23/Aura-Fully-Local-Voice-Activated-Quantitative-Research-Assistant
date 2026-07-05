"""
Tests for the Intent Router.

Verifies that the keyword-based intent classifier correctly categorizes
various finance, research, sentiment, and system command queries.
"""

import pytest
from intent.router import classify, _keyword_classify, Intent


class TestKeywordClassification:
    """Test keyword-based intent classification (fast path)."""

    # --- PORTFOLIO_QUERY ---
    @pytest.mark.parametrize("query", [
        "What is my portfolio P&L?",
        "Show me the total profit and loss",
        "What's my drawdown?",
        "What is the value at risk?",
        "How volatile is my portfolio?",
        "Show me my Sharpe ratio",
        "List my open positions",
        "What trades did I make on Tesla?",
        "How much did I make on AAPL?",
        "Show me portfolio returns",
        "What's the total value?",
        "How many shares do I hold?",
        "What is my best performing trade?",
    ])
    def test_portfolio_queries(self, query):
        result = _keyword_classify(query)
        assert result == Intent.PORTFOLIO_QUERY, \
            f"Expected PORTFOLIO_QUERY for '{query}', got {result}"

    # --- RESEARCH_QA ---
    @pytest.mark.parametrize("query", [
        "What does the research report say about Apple?",
        "Summarize the PDF document",
        "What are the key findings in the paper?",
        "According to the report, what is the outlook?",
        "Find in the document about revenue growth",
        "What is the conclusion of the analysis report?",
    ])
    def test_research_queries(self, query):
        result = _keyword_classify(query)
        assert result == Intent.RESEARCH_QA, \
            f"Expected RESEARCH_QA for '{query}', got {result}"

    # --- SENTIMENT_ANALYSIS ---
    @pytest.mark.parametrize("query", [
        "What is the sentiment of the earnings call?",
        "Is the market outlook bullish or bearish?",
        "Analyze the sentiment of this report",
        "What's the market mood today?",
    ])
    def test_sentiment_queries(self, query):
        result = _keyword_classify(query)
        assert result == Intent.SENTIMENT_ANALYSIS, \
            f"Expected SENTIMENT_ANALYSIS for '{query}', got {result}"

    # --- SYSTEM_COMMAND ---
    @pytest.mark.parametrize("query", [
        "Lock my PC",
        "Take a screenshot",
        "Mute the volume",
        "What time is it?",
        "Open notepad",
        "Open command prompt",
    ])
    def test_system_commands(self, query):
        result = _keyword_classify(query)
        assert result == Intent.SYSTEM_COMMAND, \
            f"Expected SYSTEM_COMMAND for '{query}', got {result}"

    # --- GENERAL (no keyword match → returns None) ---
    @pytest.mark.parametrize("query", [
        "Hello how are you?",
        "Tell me a joke",
        "What is the meaning of life?",
    ])
    def test_general_queries_return_none(self, query):
        result = _keyword_classify(query)
        assert result is None, \
            f"Expected None (fallback to LLM) for '{query}', got {result}"


class TestIntentEnum:
    """Test Intent enum values."""

    def test_all_intents_exist(self):
        assert Intent.PORTFOLIO_QUERY.value == "portfolio_query"
        assert Intent.RESEARCH_QA.value == "research_qa"
        assert Intent.SENTIMENT_ANALYSIS.value == "sentiment_analysis"
        assert Intent.SYSTEM_COMMAND.value == "system_command"
        assert Intent.GENERAL.value == "general"

    def test_intent_count(self):
        assert len(Intent) == 5
