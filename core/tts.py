"""
AURA -- Text-to-Speech via pyttsx3.

Extracted from the original main.py. Provides a thread-safe TTS wrapper
using pyttsx3 with configurable speech rate. Handles engine lifecycle
and cross-thread COM issues on Windows.
"""

import sys
import threading
import pyttsx3

from config.settings import TTS_RATE

# Thread lock to prevent concurrent pyttsx3 access (it is not thread-safe)
_tts_lock = threading.Lock()

# Track which thread owns the engine (COM objects can't cross thread boundaries)
_engine: pyttsx3.Engine | None = None
_engine_thread_id: int | None = None


def _coinitialize():
    """Initialize COM on the current thread (Windows only).

    pyttsx3 uses COM (SAPI) on Windows, and COM must be initialized on
    each thread that uses it. QThread workers don't automatically do this.
    """
    if sys.platform == "win32":
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            # pythoncom not available -- pyttsx3 may still work via comtypes
            try:
                import comtypes
                comtypes.CoInitialize()
            except Exception:
                pass
        except Exception:
            pass


def _get_engine() -> pyttsx3.Engine:
    """
    Return the TTS engine, initializing or re-creating it as needed.

    pyttsx3 uses COM on Windows, which is thread-bound. If called from a
    different thread than the one that created the engine, we must recreate it.
    """
    global _engine, _engine_thread_id
    current_thread = threading.current_thread().ident

    if _engine is None or _engine_thread_id != current_thread:
        # Need to create a new engine for this thread
        try:
            if _engine is not None:
                try:
                    _engine.stop()
                except Exception:
                    pass
        except Exception:
            pass

        # Must initialize COM before creating engine on worker threads
        _coinitialize()

        _engine = pyttsx3.init()
        _engine.setProperty("rate", TTS_RATE)
        _engine_thread_id = current_thread

    return _engine


def speak(text: str) -> None:
    """
    Speak the given text aloud and print it to the console.

    This function is thread-safe -- concurrent calls are serialized.
    Handles engine crashes gracefully by recreating the engine.

    Args:
        text: The text to speak.
    """
    print(f"AURA: {text}")
    with _tts_lock:
        try:
            engine = _get_engine()
            engine.say(text)
            engine.runAndWait()
        except RuntimeError:
            # "run loop already started" or COM error -- recreate engine
            _force_reset_engine()
            try:
                engine = _get_engine()
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTS ERROR] Failed to speak after retry: {e}")
        except Exception as e:
            print(f"[TTS ERROR] {e}")
            _force_reset_engine()


def _force_reset_engine() -> None:
    """Force-reset the TTS engine by destroying the current instance."""
    global _engine, _engine_thread_id
    try:
        if _engine is not None:
            _engine.stop()
    except Exception:
        pass
    _engine = None
    _engine_thread_id = None
