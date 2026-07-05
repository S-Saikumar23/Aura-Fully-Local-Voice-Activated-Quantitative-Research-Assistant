"""
AURA — Quantitative Research Assistant

Main entry point. Supports both CLI (voice) mode and GUI mode.

Usage:
    python main.py          # Launch GUI
    python main.py --cli    # Launch CLI with voice input
    python main.py --seed   # Seed database with synthetic data
    python main.py --init   # Initialize database only
"""

import sys
import argparse

from config.settings import OLLAMA_MODEL, PORCUPINE_AVAILABLE


def check_prerequisites() -> bool:
    """
    Run startup health checks for critical dependencies.

    Returns:
        True if all critical services are reachable.
    """
    all_ok = True

    # Check Ollama
    try:
        from core.llm import check_ollama_connection
        ollama_ok, ollama_msg = check_ollama_connection()
        if ollama_ok:
            print(f"[STARTUP] [OK] {ollama_msg}")
        else:
            print(f"[STARTUP] [FAIL] {ollama_msg}")
            all_ok = False
    except Exception as e:
        print(f"[STARTUP] [FAIL] Ollama check failed: {e}")
        all_ok = False

    # Check Database
    try:
        from finance.database import check_db_connection
        db_ok, db_msg = check_db_connection()
        if db_ok:
            print(f"[STARTUP] [OK] {db_msg}")
        else:
            print(f"[STARTUP] [FAIL] {db_msg}")
            all_ok = False
    except Exception as e:
        print(f"[STARTUP] [FAIL] Database check failed: {e}")
        all_ok = False

    # Check Porcupine (non-critical)
    if PORCUPINE_AVAILABLE:
        print("[STARTUP] [OK] Porcupine wake-word is configured.")
    else:
        print("[STARTUP] [WARN] Porcupine not configured -- voice will record immediately (no wake word).")

    return all_ok


def init_database():
    """Initialize the database and seed with synthetic data if empty."""
    from finance.database import init_db, is_db_available
    from finance.seed_data import seed, is_seeded

    if not is_db_available():
        print("[INIT] [FAIL] Database is not reachable. Please start PostgreSQL first.")
        print("[INIT]    Make sure DATABASE_URL in .env is correct.")
        return False

    print("[INIT] Initializing database...")
    init_db()

    if not is_seeded():
        print("[INIT] Database is empty, seeding with synthetic data...")
        seed()
    else:
        print("[INIT] Database already contains data.")

    return True


def run_cli():
    """Run AURA in CLI mode with voice input (original main loop)."""
    from core.hotword import listen_for_hotword
    from core.audio import record_audio
    from core.stt import transcribe
    from core.tts import speak
    from core.llm import ask_llm
    from intent.router import classify, Intent
    from intent.commands import execute as execute_command
    from finance.query_handler import handle as handle_portfolio_query
    from rag.qa_handler import answer_research_question
    from sentiment.analyzer import analyze_and_respond

    # Run startup checks
    print("\n" + "=" * 60)
    print("  AURA — Quantitative Research Assistant (CLI Mode)")
    print("=" * 60 + "\n")

    if not check_prerequisites():
        print("\n[WARNING] Some services are unavailable. AURA may have limited functionality.\n")

    # Initialize database
    if not init_database():
        print("[WARNING] Running without database. Portfolio features will be unavailable.\n")

    speak("AURA Quantitative Research Assistant is ready.")
    if PORCUPINE_AVAILABLE:
        speak("Say 'Hey Aura' to begin.")
    else:
        speak("Press Enter to start recording, or type your question.")

    print(f"[MAIN] Using LLM model: {OLLAMA_MODEL}")

    while True:
        try:
            if PORCUPINE_AVAILABLE:
                listen_for_hotword()
            else:
                input("\n[Press Enter to start recording, or type 'quit' to exit] ")

            speak("Hello! How may I assist you today?")

            if not record_audio():
                continue

            spoken_text = transcribe()

            # Exit commands
            if spoken_text in ["exit", "quit", "stop", "close"]:
                speak("Goodbye! Have a wonderful day ahead.")
                break

            # Classify intent
            intent = classify(spoken_text)
            print(f"[MAIN] Intent: {intent.value}")

            # Route to appropriate handler
            if intent == Intent.SYSTEM_COMMAND:
                execute_command(spoken_text)

            elif intent == Intent.PORTFOLIO_QUERY:
                speak("Let me check the portfolio data...")
                response = handle_portfolio_query(spoken_text)
                speak(response)

            elif intent == Intent.RESEARCH_QA:
                speak("Let me search through the research documents...")
                response = answer_research_question(spoken_text)
                speak(response)

            elif intent == Intent.SENTIMENT_ANALYSIS:
                speak("Analyzing sentiment...")
                response = analyze_and_respond(spoken_text)
                speak(response)

            else:  # GENERAL
                speak("Let me think about that for a moment...")
                response = ask_llm(spoken_text, use_history=True)
                speak(response)

        except KeyboardInterrupt:
            speak("Shutting down. Goodbye!")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            speak("I encountered an error. Please try again.")


def run_gui():
    """Launch the AURA GUI application."""
    # Initialize database before starting GUI
    print("\n" + "=" * 60)
    print("  AURA — Quantitative Research Assistant (GUI Mode)")
    print("=" * 60 + "\n")

    if not check_prerequisites():
        print("\n[WARNING] Some services are unavailable. AURA may have limited functionality.\n")

    if not init_database():
        print("[WARNING] Running without database. Portfolio features will be unavailable.\n")

    from aura_gui import run_gui as launch_gui
    launch_gui()


def main():
    """Parse arguments and launch the appropriate mode."""
    parser = argparse.ArgumentParser(
        description="AURA — Quantitative Research Assistant"
    )
    parser.add_argument(
        "--cli", action="store_true",
        help="Run in CLI mode with voice input instead of GUI",
    )
    parser.add_argument(
        "--seed", action="store_true",
        help="Seed the database with synthetic data and exit",
    )
    parser.add_argument(
        "--init", action="store_true",
        help="Initialize the database tables and exit",
    )
    parser.add_argument(
        "--force-seed", action="store_true",
        help="Force re-seed (deletes existing data)",
    )
    parser.add_argument(
        "--ingest", type=str, default=None,
        help="Path to a PDF file or directory to ingest",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Run health checks and exit",
    )

    args = parser.parse_args()

    # Health check only
    if args.check:
        print("\n" + "=" * 60)
        print("  AURA -- Health Check")
        print("=" * 60 + "\n")
        all_ok = check_prerequisites()
        print()
        if all_ok:
            print("[OK] All systems operational.")
        else:
            print("[FAIL] Some systems need attention.")
        return

    # Handle database commands
    if args.init:
        from finance.database import init_db, is_db_available
        if not is_db_available():
            print("[FAIL] Cannot connect to PostgreSQL. Check DATABASE_URL in .env")
            return
        init_db()
        print("Database initialized successfully.")
        return

    if args.seed or args.force_seed:
        from finance.database import init_db, is_db_available
        from finance.seed_data import seed
        if not is_db_available():
            print("[FAIL] Cannot connect to PostgreSQL. Check DATABASE_URL in .env")
            return
        init_db()
        seed(force=args.force_seed)
        return

    if args.ingest:
        from finance.database import init_db, is_db_available
        from pathlib import Path

        if not is_db_available():
            print("[FAIL] Cannot connect to PostgreSQL. Check DATABASE_URL in .env")
            return

        init_db()
        path = Path(args.ingest)

        if path.is_file() and path.suffix.lower() == ".pdf":
            from rag.pdf_ingester import ingest_pdf
            ingest_pdf(path)
        elif path.is_dir():
            from rag.pdf_ingester import ingest_all_pdfs
            ingest_all_pdfs(path)
        else:
            print(f"Error: '{args.ingest}' is not a valid PDF file or directory.")
        return

    # Launch application
    if args.cli:
        run_cli()
    else:
        run_gui()


if __name__ == "__main__":
    main()