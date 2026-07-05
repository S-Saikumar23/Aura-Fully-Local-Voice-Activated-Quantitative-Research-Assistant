"""
AURA — Audio recording and voice activity detection.

Extracted from the original main.py. Provides microphone recording with
noise reduction and WebRTC-based voice activity detection.
"""

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import noisereduce as nr
import webrtcvad

from config.settings import AUDIO_SAMPLE_RATE, AUDIO_RECORD_DURATION, AUDIO_FILENAME

# Initialize VAD with aggressiveness level 3 (most aggressive filtering)
_vad = webrtcvad.Vad(3)


def is_speech(audio_data: np.ndarray, samplerate: int = AUDIO_SAMPLE_RATE) -> bool:
    """
    Check if the audio data contains speech using WebRTC VAD.

    Args:
        audio_data: NumPy array of int16 audio samples.
        samplerate: Sample rate of the audio (must be 8000, 16000, 32000, or 48000).

    Returns:
        True if any 30ms frame contains speech.
    """
    audio_bytes = audio_data.tobytes()
    frame_duration = 30  # ms
    frame_size = int(samplerate * frame_duration / 1000) * 2  # 2 bytes per int16 sample
    for i in range(0, len(audio_bytes) - frame_size, frame_size):
        frame = audio_bytes[i : i + frame_size]
        if _vad.is_speech(frame, samplerate):
            return True
    return False


def record_audio(
    samplerate: int = AUDIO_SAMPLE_RATE,
    duration: int = AUDIO_RECORD_DURATION,
    filename: str = AUDIO_FILENAME,
) -> bool:
    """
    Record audio from the microphone, apply noise reduction, and save to WAV.

    Args:
        samplerate: Recording sample rate in Hz.
        duration: Recording duration in seconds.
        filename: Output WAV file path (defaults to absolute temp path).

    Returns:
        True if speech was detected in the recording, False otherwise.
    """
    print("\nListening...")
    try:
        recording = sd.rec(
            int(samplerate * duration),
            samplerate=samplerate,
            channels=1,
            dtype=np.int16,
        )
        sd.wait()
    except Exception as e:
        print(f"[AUDIO ERROR] Failed to record: {e}")
        return False

    audio_data = np.squeeze(recording)

    # Apply noise reduction
    reduced_noise = nr.reduce_noise(y=audio_data, sr=samplerate)

    # FIX: Clip before int16 cast to prevent overflow/wrapping artifacts
    final_audio = np.clip(reduced_noise, -32768, 32767).astype(np.int16)

    if is_speech(final_audio):
        wav.write(filename, samplerate, final_audio)
        print("Speech detected and recorded.")
        return True
    else:
        print("No speech detected.")
        return False
