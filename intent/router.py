"""
AURA — Finance-Domain Intent Router.

Replaces the missing tanglish_bert_classifier module with a hybrid
keyword + LLM-based intent classification system for finance queries.
"""

import re
from enum import Enum

from core.llm import ask_llm


class Intent(Enum):
    """All supported intent categories."""
    PORTFOLIO_QUERY = "portfolio_query"
    RESEARCH_QA = "research_qa"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    SYSTEM_COMMAND = "system_command"
    GENERAL = "general"


# ---------------------------------------------------------------------------
# Keyword patterns for fast-path classification
# ---------------------------------------------------------------------------
_INTENT_PATTERNS: dict[Intent, list[str]] = {
    Intent.PORTFOLIO_QUERY: [
        r"\bp\s*&?\s*l\b", r"\bpnl\b", r"\bprofit\b", r"\bloss\b",
        r"\bdrawdown\b", r"\bmax\s+drawdown\b",
        r"\bvar\b", r"\bvalue\s+at\s+risk\b",
        r"\bvolatility\b", r"\bsharpe\b",
        r"\bportfolio\b", r"\btrades?\b", r"\bpositions?\b",
        r"\breturn(?:s)?\b", r"\bholdings?\b", r"\bhold\b", r"\bshares?\b",
        r"\bentry\s+price\b", r"\bexit\s+price\b",
        r"\bopen\s+positions?\b", r"\bclosed\s+trades?\b",
        r"\btotal\s+value\b", r"\bcash\s+balance\b",
        r"\bquantity\b", r"\bsymbol\b",
        r"\bhow\s+(?:much|many)\s+(?:did|have|do)\s+(?:i|we)\b",
        r"\bperformance\b", r"\bbest\s+(?:trade|stock|performing)\b",
        r"\bworst\s+(?:trade|stock|performing)\b",
    ],
    Intent.RESEARCH_QA: [
        r"\bresearch\b", r"\bpaper\b", r"\breport\b",
        r"\bpdf\b", r"\bdocument\b",
        r"\bwhat\s+does\s+the\s+(?:report|paper|document)\b",
        r"\baccording\s+to\b", r"\bfind\s+in\b",
        r"\bsummar(?:y|ize)\b", r"\banalysis\s+(?:report|paper)\b",
        r"\bkey\s+findings\b", r"\bconclusion\b",
    ],
    Intent.SENTIMENT_ANALYSIS: [
        r"\bsentiment\b", r"\bearnings?\s+call\b",
        r"\bbullish\b", r"\bbearish\b",
        r"\boutlook\b", r"\bmarket\s+mood\b",
        r"\bpositive\s+or\s+negative\b",
        r"\btone\b", r"\bfeeling\b.*\b(?:market|stock|report)\b",
        r"analyze\s+(?:the\s+)?sentiment",
    ],
    Intent.SYSTEM_COMMAND: [
        r"\bshutdown\b", r"\brestart\b",
        r"\bmute\b", r"\bunmute\b", r"\bvolume\b",
        r"\btime\b", r"\block\b",
        r"\bscreenshot\b", r"\bnotepad\b",
        r"\bcmd\b", r"\bcommand\s+prompt\b",
    ],
}

# Pre-compile all patterns for performance
_COMPILED_PATTERNS: dict[Intent, list[re.Pattern]] = {
    intent: [re.compile(p, re.IGNORECASE) for p in patterns]
    for intent, patterns in _INTENT_PATTERNS.items()
}


def _keyword_classify(text: str) -> Intent | None:
    """
    Attempt to classify intent using keyword patterns.

    Returns the intent with the most pattern matches, or None if
    no patterns match (ambiguous or unrecognized).
    """
    scores: dict[Intent, int] = {}
    cleaned = text.lower()

    for intent, patterns in _COMPILED_PATTERNS.items():
        match_count = sum(1 for p in patterns if p.search(cleaned))
        if match_count > 0:
            scores[intent] = match_count

    if not scores:
        return None

    # Return the intent with the highest match count
    best_intent = max(scores, key=scores.get)  # type: ignore[arg-type]

    # Require at least 1 match; if tie, return None for LLM fallback
    score_values = list(scores.values())
    max_score = max(score_values)
    if score_values.count(max_score) > 1:
        return None  # Ambiguous — let LLM decide

    return best_intent


def _llm_classify(text: str) -> Intent:
    """
    Use the Ollama LLM as a fallback intent classifier.

    Sends a structured classification prompt and parses the response.
    """
    classification_prompt = (
        "Classify the following user query into exactly ONE of these categories:\n"
        "1. PORTFOLIO_QUERY — asks about trades, P&L, drawdown, VaR, portfolio value, "
        "positions, returns, or any trading/investment data\n"
        "2. RESEARCH_QA — asks about content in research documents, PDFs, or reports\n"
        "3. SENTIMENT_ANALYSIS — asks about the sentiment, tone, or mood of "
        "financial text, earnings calls, or market outlook\n"
        "4. SYSTEM_COMMAND — asks to perform a system action like locking PC, "
        "screenshot, opening an app, muting, or volume control\n"
        "5. GENERAL — general conversation, greetings, or anything else\n\n"
        f'User query: "{text}"\n\n'
        "Respond with ONLY the category name (e.g., PORTFOLIO_QUERY). "
        "Do not include any other text."
    )

    response = ask_llm(
        classification_prompt,
        system_prompt="You are a precise intent classifier. Respond with only the category name.",
    )

    # Parse the LLM response
    response_clean = response.strip().upper().replace(" ", "_")

    intent_map = {
        "PORTFOLIO_QUERY": Intent.PORTFOLIO_QUERY,
        "RESEARCH_QA": Intent.RESEARCH_QA,
        "SENTIMENT_ANALYSIS": Intent.SENTIMENT_ANALYSIS,
        "SYSTEM_COMMAND": Intent.SYSTEM_COMMAND,
        "GENERAL": Intent.GENERAL,
    }

    # Try exact match first, then partial match
    if response_clean in intent_map:
        return intent_map[response_clean]

    for key, intent in intent_map.items():
        if key in response_clean:
            return intent

    return Intent.GENERAL  # Default fallback


def classify(text: str) -> Intent:
    """
    Classify user input into one of the supported intent categories.

    Uses a hybrid approach:
    1. Fast keyword-based matching for high-confidence classification.
    2. LLM-based fallback for ambiguous or unrecognized queries.

    Args:
        text: The user's transcribed or typed query.

    Returns:
        The classified Intent enum value.
    """
    # Step 1: Try keyword matching
    keyword_result = _keyword_classify(text)
    if keyword_result is not None:
        print(f"[INTENT] Keyword match: {keyword_result.value}")
        return keyword_result

    # Step 2: Fallback to LLM classification
    print("[INTENT] Keyword match inconclusive, using LLM classifier...")
    llm_result = _llm_classify(text)
    print(f"[INTENT] LLM classified: {llm_result.value}")
    return llm_result
