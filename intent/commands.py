"""
AURA — System Commands.

Extracted from the original main.py. Contains all system-level actions
(lock PC, screenshot, volume control, app launching) and the fuzzy
command matcher.
"""

import os
import ctypes
from datetime import datetime
from ctypes import cast, POINTER
from typing import Callable

import pyautogui
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from fuzzywuzzy import fuzz

from core.tts import speak
from core.audio import record_audio
from core.stt import transcribe
from config.settings import SCREENSHOT_DIR


# ---------------------------------------------------------------------------
# System Action Functions (preserved from original main.py)
# ---------------------------------------------------------------------------

def lock_pc() -> None:
    """Lock the Windows workstation."""
    speak("Alright, locking your PC now.")
    ctypes.windll.user32.LockWorkStation()


def take_screenshot() -> None:
    """Capture a screenshot and save it to the Pictures directory."""
    speak("Sure, taking a screenshot for you.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = SCREENSHOT_DIR / f"aura_screenshot_{timestamp}.png"
    screenshot = pyautogui.screenshot()
    screenshot.save(str(filepath))
    print(f"Saved as {filepath}")
    speak(f"Screenshot saved to {filepath.name}")


def change_volume(action: str) -> None:
    """
    Change system volume.

    Args:
        action: One of "mute", "unmute", "up", "down".
    """
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))

        if action == "mute":
            volume.SetMute(1, None)
            speak("Muted the volume.")
        elif action == "unmute":
            volume.SetMute(0, None)
            speak("Unmuted the volume.")
        elif action == "up":
            current = volume.GetMasterVolumeLevelScalar()
            volume.SetMasterVolumeLevelScalar(min(current + 0.1, 1.0), None)
            speak("Volume increased.")
        elif action == "down":
            current = volume.GetMasterVolumeLevelScalar()
            volume.SetMasterVolumeLevelScalar(max(current - 0.1, 0.0), None)
            speak("Volume decreased.")
    except Exception as e:
        print(f"[VOLUME ERROR] {e}")
        speak("Sorry, I couldn't change the volume.")


def tell_time() -> None:
    """Speak the current time."""
    now = datetime.now()
    time_str = now.strftime("The time is %I:%M %p.")
    speak(time_str)


def open_app(command: str) -> None:
    """
    Open a system application.

    Args:
        command: One of "notepad", "cmd", "shutdown", "restart".
    """
    if command == "notepad":
        speak("Opening Notepad for you.")
        os.system("start notepad")
    elif command == "cmd":
        speak("Sure, launching Command Prompt.")
        os.system("start cmd")
    elif command == "shutdown":
        speak("Okay, shutting down your PC.")
        os.system("shutdown /s /t 1")
    elif command == "restart":
        speak("Restarting your PC now.")
        os.system("shutdown /r /t 1")
    else:
        speak("Sorry, I couldn't find a matching app to open.")


def confirm_and_execute(action_name: str, func: Callable) -> None:
    """
    Ask for voice confirmation before executing a dangerous action.

    Args:
        action_name: Description of the action (for the confirmation prompt).
        func: Callable to execute if the user confirms.
    """
    speak(f"Are you sure you want to {action_name}? Please say yes to confirm.")
    if record_audio():
        response = transcribe()
        if "yes" in response.lower():
            func()
        else:
            speak("No problem. I've cancelled the request.")


# ---------------------------------------------------------------------------
# Command Registry (preserved from original main.py)
# ---------------------------------------------------------------------------

COMMANDS: dict[str, Callable[[], None]] = {
    "lock my pc": lambda: lock_pc(),
    "take a screenshot": lambda: take_screenshot(),
    "open notepad": lambda: open_app("notepad"),
    "open cmd": lambda: open_app("cmd"),
    "command prompt": lambda: open_app("cmd"),
    "shutdown": lambda: confirm_and_execute(
        "shutdown the system", lambda: open_app("shutdown")
    ),
    "restart": lambda: confirm_and_execute(
        "restart the pc", lambda: open_app("restart")
    ),
    "what time is it now": lambda: tell_time(),
    "mute": lambda: change_volume("mute"),
    "unmute": lambda: change_volume("unmute"),
    "increase volume": lambda: change_volume("up"),
    "decrease volume": lambda: change_volume("down"),
    "tell me your name": lambda: speak(
        "Hi, I'm Aura. It's a pleasure to assist you."
    ),
    "love you": lambda: speak("Thank you! That means a lot to me."),
}


# ---------------------------------------------------------------------------
# Text Normalization (preserved from original main.py)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Remove filler words and normalize the input for command matching."""
    fillers = [
        "please", "can you", "do you", "could you", "would you",
        "will you", "hey aura", "aura", "kindly", "i want to",
    ]
    cleaned = text.lower().strip()
    for filler in fillers:
        cleaned = cleaned.replace(filler, "")
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Fuzzy Command Matcher (preserved from original main.py)
# ---------------------------------------------------------------------------

def match_command(text: str) -> Callable | None:
    """
    Match user text to a system command using fuzzy string matching.

    Args:
        text: The user's transcribed or typed query.

    Returns:
        A callable command if matched, or None if no match found.
    """
    cleaned_text = normalize_text(text)

    # Force exact keyword triggers
    if "command prompt" in cleaned_text or "cmd" in cleaned_text:
        return COMMANDS.get("open cmd")

    best_match = None
    highest_score = 0
    COMMAND_THRESHOLD = 70

    for phrase in COMMANDS:
        score = fuzz.token_set_ratio(cleaned_text, phrase)
        if score >= COMMAND_THRESHOLD and score > highest_score:
            best_match = phrase
            highest_score = score

    if best_match:
        print(f"Fuzzy Matched: {best_match} (score: {highest_score})")
        return COMMANDS[best_match]

    return None


def execute(text: str) -> bool:
    """
    Attempt to execute a system command from user text.

    Args:
        text: The user's transcribed or typed query.

    Returns:
        True if a command was matched and executed, False otherwise.
    """
    action = match_command(text)
    if action:
        action()
        return True
    else:
        speak("I couldn't find a matching system command, but I'm here to help with something else.")
        return False
