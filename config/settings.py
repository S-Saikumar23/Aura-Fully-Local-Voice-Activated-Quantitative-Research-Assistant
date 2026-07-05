"""
AURA Quantitative Research Assistant — Centralized Configuration

Loads all settings from .env file and exposes them as module-level constants.
Validates required settings on import.
"""

import os
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Resolve project root (directory containing this config/ package)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    print(f"[WARNING] No .env file found at {ENV_PATH}. Using system environment.")

# ---------------------------------------------------------------------------
# Pvporcupine Wake-Word
# ---------------------------------------------------------------------------
PORCUPINE_ACCESS_KEY: str = os.getenv("PORCUPINE_ACCESS_KEY", "")
PORCUPINE_KEYWORD_PATH: str = os.getenv("PORCUPINE_KEYWORD_PATH", "")

# ---------------------------------------------------------------------------
# Ollama LLM
# ---------------------------------------------------------------------------
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# ---------------------------------------------------------------------------
# Faster-Whisper STT
# ---------------------------------------------------------------------------
WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# ---------------------------------------------------------------------------
# Audio Settings
# ---------------------------------------------------------------------------
AUDIO_SAMPLE_RATE: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_RECORD_DURATION: int = int(os.getenv("AUDIO_RECORD_DURATION", "5"))
# Use absolute path in temp directory to avoid CWD issues
AUDIO_FILENAME: str = str(Path(tempfile.gettempdir()) / "aura_recording.wav")

# ---------------------------------------------------------------------------
# Screenshot save location
# ---------------------------------------------------------------------------
SCREENSHOT_DIR: Path = Path.home() / "Pictures"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# PostgreSQL + pgvector
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/aura_finance"
)

# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))

# ---------------------------------------------------------------------------
# Sentiment Model
# ---------------------------------------------------------------------------
SENTIMENT_MODEL: str = os.getenv(
    "SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESEARCH_PDF_DIR: Path = PROJECT_ROOT / os.getenv("RESEARCH_PDF_DIR", "data/research_pdfs")

# Ensure the research PDF directory exists
RESEARCH_PDF_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# TTS Settings
# ---------------------------------------------------------------------------
TTS_RATE: int = 160

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
_WARNINGS: list[str] = []

if not PORCUPINE_ACCESS_KEY:
    _WARNINGS.append(
        "PORCUPINE_ACCESS_KEY is not set. Wake-word detection will be disabled. "
        "Voice input will skip wake-word and record immediately."
    )
if not PORCUPINE_KEYWORD_PATH or not Path(PORCUPINE_KEYWORD_PATH).exists():
    _WARNINGS.append(
        f"PORCUPINE_KEYWORD_PATH is invalid or missing: '{PORCUPINE_KEYWORD_PATH}'. "
        "Wake-word detection will be disabled."
    )

for w in _WARNINGS:
    print(f"[CONFIG WARNING] {w}", file=sys.stderr)

# Convenience flag for checking Porcupine availability
PORCUPINE_AVAILABLE: bool = bool(
    PORCUPINE_ACCESS_KEY
    and PORCUPINE_KEYWORD_PATH
    and Path(PORCUPINE_KEYWORD_PATH).exists()
)
