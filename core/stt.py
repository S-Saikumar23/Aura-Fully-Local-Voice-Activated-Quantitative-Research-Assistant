"""
AURA -- Speech-to-Text via Faster-Whisper.

Extracted from the original main.py. Provides lazy-loaded Whisper model
initialization and audio file transcription.
"""

import os
import warnings

# Suppress the hf_xet download warning that clutters output
os.environ["HF_HUB_DISABLE_XET"] = "1"
warnings.filterwarnings("ignore", message=".*hf_xet.*")
warnings.filterwarnings("ignore", message=".*Xet.*")

from faster_whisper import WhisperModel

from config.settings import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    AUDIO_FILENAME,
)

# Lazy-loaded model instance
_whisper_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    """Return the Whisper model, initializing it on first call."""
    global _whisper_model
    if _whisper_model is None:
        print(f"[STT] Loading Whisper model '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE}...")
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        print("[STT] Whisper model loaded.")
    return _whisper_model


def transcribe(filename: str = AUDIO_FILENAME) -> str:
    """
    Transcribe an audio file to text using Faster-Whisper.

    Args:
        filename: Path to the WAV file to transcribe.

    Returns:
        Transcribed text (lowercased and stripped).
    """
    print("Transcribing...")
    model = _get_model()
    segments, _ = model.transcribe(filename)
    full_text = " ".join([segment.text for segment in segments]).lower().strip()
    print(f"You said: {full_text}")
    return full_text
