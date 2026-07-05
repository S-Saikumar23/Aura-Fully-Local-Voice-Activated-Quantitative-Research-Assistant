"""
AURA — Wake-Word Detection via Pvporcupine.

Extracted from the original main.py. Listens for the "Hey Aura" hotword
using Picovoice Porcupine, with configurable access key and keyword path.
Falls back to immediate recording when Porcupine is not configured.
"""

import struct
from typing import Callable

import pyaudio
import pvporcupine

from config.settings import (
    PORCUPINE_ACCESS_KEY,
    PORCUPINE_KEYWORD_PATH,
    PORCUPINE_AVAILABLE,
)


def is_porcupine_configured() -> bool:
    """Check if Porcupine wake-word detection is properly configured."""
    return PORCUPINE_AVAILABLE


def listen_for_hotword(on_detected: Callable[[], None] | None = None) -> None:
    """
    Block until the "Hey Aura" wake word is detected.

    If Porcupine is not configured (missing access key or keyword path),
    this function returns immediately so voice recording can proceed
    without wake-word detection.

    Args:
        on_detected: Optional callback invoked when the hotword is detected.
                     If None, simply returns after detection.
    """
    if not PORCUPINE_AVAILABLE:
        print(
            "[HOTWORD] Porcupine not configured — skipping wake-word detection. "
            "Recording will start immediately."
        )
        if on_detected:
            on_detected()
        return

    porcupine = pvporcupine.create(
        access_key=PORCUPINE_ACCESS_KEY,
        keyword_paths=[PORCUPINE_KEYWORD_PATH],
    )

    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length,
    )

    print("Listening for 'Hey Aura'...")

    try:
        while True:
            pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm_unpacked = struct.unpack_from(
                "h" * porcupine.frame_length, pcm
            )
            result = porcupine.process(pcm_unpacked)
            if result >= 0:
                print("Hotword 'Hey Aura' detected!")
                if on_detected:
                    on_detected()
                break
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        porcupine.delete()
