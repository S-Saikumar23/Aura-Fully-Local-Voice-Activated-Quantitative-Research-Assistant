"""
AURA — Local LLM Integration via Ollama.

Extracted from the original main.py. Provides chat completions through
Ollama with support for system prompts and RAG context injection.
Includes conversation history for multi-turn chat.
"""

import ollama

from config.settings import OLLAMA_MODEL

# Default system prompt (preserved from original Aura personality)
DEFAULT_SYSTEM_PROMPT = (
    "You are AURA, a quantitative research AI assistant specializing in finance. "
    "You are friendly, precise, and knowledgeable about trading, portfolio analytics, "
    "and financial research. Keep your answers concise and data-driven. "
    "When presenting numbers, use proper formatting (commas, percentages, decimals)."
)

# Original personality prompt for general chat
GENERAL_SYSTEM_PROMPT = (
    "You are AURA, an AI assistant who is friendly, interactive, and kind. "
    "You respond in a caring and polite tone. Keep your answers short, helpful, "
    "and make the user feel supported and respected."
)

# ---------------------------------------------------------------------------
# Conversation History (for multi-turn chat)
# ---------------------------------------------------------------------------
_conversation_history: list[dict] = []
_MAX_HISTORY = 10  # Keep last N exchanges to avoid context overflow


def clear_history() -> None:
    """Clear the conversation history."""
    global _conversation_history
    _conversation_history.clear()


def get_history() -> list[dict]:
    """Return the current conversation history."""
    return list(_conversation_history)


def _add_to_history(role: str, content: str) -> None:
    """Add a message to conversation history, trimming if needed."""
    global _conversation_history
    _conversation_history.append({"role": role, "content": content})
    # Keep only the last N*2 messages (N exchanges = N user + N assistant)
    if len(_conversation_history) > _MAX_HISTORY * 2:
        _conversation_history = _conversation_history[-_MAX_HISTORY * 2:]


# ---------------------------------------------------------------------------
# Ollama Health Check
# ---------------------------------------------------------------------------

def check_ollama_connection() -> tuple[bool, str]:
    """
    Check if Ollama server is running and the configured model is available.

    Returns:
        Tuple of (is_connected, status_message).
    """
    try:
        models_response = ollama.list()
        # ollama.list() returns a ListResponse object with a 'models' attribute
        available = []
        if hasattr(models_response, 'models'):
            available = [m.model for m in models_response.models]
        elif isinstance(models_response, dict):
            available = [m.get("name", "") for m in models_response.get("models", [])]

        if not available:
            return True, (
                f"Ollama is running but no models are installed. "
                f"Run: ollama pull {OLLAMA_MODEL}"
            )

        # Check if configured model is available (handle tag variations)
        model_base = OLLAMA_MODEL.split(":")[0]
        model_found = any(model_base in m for m in available)

        if model_found:
            return True, f"Ollama is running. Model '{OLLAMA_MODEL}' is available."
        else:
            return True, (
                f"Ollama is running but model '{OLLAMA_MODEL}' not found. "
                f"Available: {', '.join(available[:5])}. "
                f"Run: ollama pull {OLLAMA_MODEL}"
            )
    except Exception as e:
        error_msg = str(e)
        if "refused" in error_msg.lower() or "connect" in error_msg.lower():
            return False, (
                "Cannot connect to Ollama. Make sure Ollama is running. "
                "Start it with: ollama serve"
            )
        return False, f"Ollama error: {error_msg}"


# ---------------------------------------------------------------------------
# Response Parsing (handles both Pydantic objects and legacy dicts)
# ---------------------------------------------------------------------------

def _extract_response_content(response) -> str:
    """
    Extract the content string from an Ollama chat response.

    Handles both the modern Pydantic ChatResponse object (ollama >= 0.2)
    and the legacy dict format.
    """
    try:
        # Modern ollama library (>= 0.2): returns ChatResponse Pydantic object
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            return response.message.content.strip()
    except (AttributeError, TypeError):
        pass

    try:
        # Legacy dict format (ollama < 0.2)
        if isinstance(response, dict):
            return response.get("message", {}).get("content", "").strip()
    except (AttributeError, TypeError):
        pass

    # Last resort: try string conversion
    return str(response).strip() if response else ""


# ---------------------------------------------------------------------------
# Core LLM Functions
# ---------------------------------------------------------------------------

def ask_llm(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    use_history: bool = False,
) -> str:
    """
    Send a prompt to the Ollama LLM and return the response.

    Args:
        prompt: The user's question or instruction.
        system_prompt: Optional system prompt to set the LLM's behavior.
                       Defaults to GENERAL_SYSTEM_PROMPT.
        model: Optional model override. Defaults to OLLAMA_MODEL from config.
        use_history: If True, include conversation history for multi-turn chat.

    Returns:
        The LLM's response text, or a fallback message on failure.
    """
    print("Sending to AI...")
    _model = model or OLLAMA_MODEL
    _system = system_prompt or GENERAL_SYSTEM_PROMPT

    try:
        messages = [{"role": "system", "content": _system}]

        if use_history and _conversation_history:
            messages.extend(_conversation_history)

        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(
            model=_model,
            messages=messages,
        )

        reply = _extract_response_content(response)

        if not reply:
            return "I'm sorry, I couldn't generate a response."

        # Save to history if using multi-turn
        if use_history:
            _add_to_history("user", prompt)
            _add_to_history("assistant", reply)

        return reply

    except Exception as e:
        error_str = str(e)
        print(f"[LLM ERROR] {error_str}")

        if "refused" in error_str.lower() or "connect" in error_str.lower():
            return (
                "I can't connect to the Ollama server. "
                "Please make sure Ollama is running (ollama serve)."
            )
        if "not found" in error_str.lower() or "pull" in error_str.lower():
            return (
                f"The model '{_model}' is not installed. "
                f"Please run: ollama pull {_model}"
            )
        return "I'm sorry, I encountered an error connecting to the language model."


def ask_llm_with_context(
    query: str,
    context: str,
    system_prompt: str | None = None,
    model: str | None = None,
) -> str:
    """
    Send a query to the LLM with injected RAG context.

    This constructs a prompt that includes retrieved document context
    before the user's question, enabling grounded answers.

    Args:
        query: The user's question.
        context: Retrieved document text to ground the answer.
        system_prompt: Optional system prompt override.
        model: Optional model override.

    Returns:
        The LLM's context-aware response.
    """
    _system = system_prompt or DEFAULT_SYSTEM_PROMPT

    augmented_prompt = (
        f"Based on the following context from research documents:\n\n"
        f"---\n{context}\n---\n\n"
        f"Answer this question concisely and accurately: {query}\n"
        f"If the context doesn't contain relevant information, say so honestly."
    )

    return ask_llm(augmented_prompt, system_prompt=_system, model=model)


def format_analytics_response(
    query: str,
    analytics_data: str,
    model: str | None = None,
) -> str:
    """
    Use the LLM to format raw analytics data into a natural-language response.

    Args:
        query: The original user question.
        analytics_data: Raw data/metrics as a formatted string.
        model: Optional model override.

    Returns:
        A natural-language response incorporating the analytics data.
    """
    system = (
        "You are AURA, a quantitative finance assistant. "
        "Format the following portfolio analytics data into a clear, concise, "
        "natural-language response. Use proper number formatting. "
        "Be direct and professional."
    )

    prompt = (
        f"User asked: {query}\n\n"
        f"Here is the raw analytics data:\n{analytics_data}\n\n"
        f"Provide a clear, conversational answer."
    )

    return ask_llm(prompt, system_prompt=system, model=model)
