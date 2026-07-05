"""
AURA -- Quantitative Research Assistant GUI.

Redesigned PyQt5 interface with:
- Chat tab: voice & text input, formatted conversation history
- Portfolio tab: key metrics dashboard, positions table
- Documents tab: PDF upload/ingestion, document list
- Startup health checks for Ollama and PostgreSQL
- Thread-safe TTS and proper worker lifecycle management
"""

import sys
import os
import warnings
from pathlib import Path

# Suppress hf_xet warning before any HuggingFace imports
os.environ["HF_HUB_DISABLE_XET"] = "1"
warnings.filterwarnings("ignore", message=".*hf_xet.*")
warnings.filterwarnings("ignore", message=".*Xet.*")

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QLineEdit, QTabWidget, QTableWidget, QTableWidgetItem,
    QFileDialog, QProgressBar, QFrame, QGridLayout, QSplitter,
    QHeaderView, QScrollArea, QGroupBox, QMessageBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QMimeData
from PyQt5.QtGui import QFont, QColor, QPalette, QDragEnterEvent, QDropEvent

# Import AURA modules
from core.audio import record_audio
from core.stt import transcribe
from core.tts import speak
from core.llm import ask_llm, check_ollama_connection, clear_history
from intent.router import classify, Intent
from intent.commands import execute as execute_command
from finance.query_handler import handle as handle_portfolio_query
from rag.qa_handler import answer_research_question
from sentiment.analyzer import analyze_and_respond
from config.settings import PORCUPINE_AVAILABLE, OLLAMA_MODEL


# =============================================================================
# Stylesheet — Premium Dark Finance Theme
# =============================================================================
STYLESHEET = """
    QWidget {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Segoe UI', 'Inter', sans-serif;
        font-size: 13px;
    }

    QTabWidget::pane {
        border: 1px solid #21262d;
        background-color: #0d1117;
        border-radius: 8px;
    }

    QTabBar::tab {
        background-color: #161b22;
        color: #8b949e;
        padding: 10px 24px;
        margin-right: 2px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        font-weight: bold;
        font-size: 13px;
    }

    QTabBar::tab:selected {
        background-color: #0d1117;
        color: #58a6ff;
        border-bottom: 2px solid #58a6ff;
    }

    QTabBar::tab:hover {
        background-color: #1c2128;
        color: #c9d1d9;
    }

    QPushButton {
        background-color: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
        padding: 8px 18px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 13px;
    }

    QPushButton:hover {
        background-color: #30363d;
        border-color: #58a6ff;
        color: #ffffff;
    }

    QPushButton:pressed {
        background-color: #1f6feb;
        color: white;
    }

    QPushButton:disabled {
        background-color: #161b22;
        color: #484f58;
        border-color: #21262d;
    }

    QPushButton#primaryBtn {
        background-color: #1f6feb;
        color: white;
        border: none;
    }

    QPushButton#primaryBtn:hover {
        background-color: #388bfd;
    }

    QPushButton#primaryBtn:disabled {
        background-color: #1a3a5c;
        color: #6e7681;
    }

    QPushButton#dangerBtn {
        background-color: #da3633;
        color: white;
        border: none;
    }

    QLineEdit {
        background-color: #161b22;
        color: #c9d1d9;
        border: 1px solid #30363d;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 14px;
    }

    QLineEdit:focus {
        border-color: #58a6ff;
    }

    QLineEdit:disabled {
        background-color: #0d1117;
        color: #484f58;
    }

    QTextEdit {
        background-color: #161b22;
        color: #c9d1d9;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 10px;
        font-size: 13px;
        line-height: 1.5;
    }

    QLabel {
        color: #c9d1d9;
    }

    QLabel#titleLabel {
        color: #58a6ff;
        font-size: 22px;
        font-weight: bold;
    }

    QLabel#metricLabel {
        color: #8b949e;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    QLabel#metricValue {
        color: #ffffff;
        font-size: 20px;
        font-weight: bold;
    }

    QLabel#positiveValue {
        color: #3fb950;
        font-size: 20px;
        font-weight: bold;
    }

    QLabel#negativeValue {
        color: #f85149;
        font-size: 20px;
        font-weight: bold;
    }

    QLabel#statusLabel {
        color: #8b949e;
        font-size: 12px;
        padding: 4px 8px;
    }

    QTableWidget {
        background-color: #161b22;
        color: #c9d1d9;
        border: 1px solid #21262d;
        border-radius: 6px;
        gridline-color: #21262d;
        font-size: 12px;
    }

    QTableWidget::item {
        padding: 6px 10px;
    }

    QTableWidget::item:selected {
        background-color: #1f6feb;
        color: white;
    }

    QHeaderView::section {
        background-color: #21262d;
        color: #8b949e;
        border: none;
        padding: 8px 10px;
        font-weight: bold;
        font-size: 11px;
        text-transform: uppercase;
    }

    QProgressBar {
        background-color: #21262d;
        border: none;
        border-radius: 4px;
        height: 6px;
        text-align: center;
    }

    QProgressBar::chunk {
        background-color: #1f6feb;
        border-radius: 4px;
    }

    QGroupBox {
        border: 1px solid #21262d;
        border-radius: 8px;
        margin-top: 12px;
        padding-top: 16px;
        font-weight: bold;
        color: #c9d1d9;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        left: 14px;
        padding: 0 6px;
    }

    QFrame#metricCard {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 16px;
    }

    QFrame#separator {
        background-color: #21262d;
        max-height: 1px;
    }

    QScrollArea {
        border: none;
    }
"""


# =============================================================================
# Worker Threads
# =============================================================================

class VoiceWorker(QThread):
    """Background thread for voice input processing."""
    update_text = pyqtSignal(str, str)  # (role, message)
    hotword_detected = pyqtSignal()
    status_changed = pyqtSignal(str)
    finished_processing = pyqtSignal()

    def run(self):
        try:
            if PORCUPINE_AVAILABLE:
                self.status_changed.emit("Listening for 'Hey Aura'...")
                from core.hotword import listen_for_hotword
                listen_for_hotword()
                self.hotword_detected.emit()
                self.status_changed.emit("Hotword detected! Recording...")
            else:
                self.status_changed.emit("Recording... (speak now)")
                self.hotword_detected.emit()

            # Show greeting as text only -- do NOT speak it via TTS
            # because the mic would pick up AURA's own speaker output
            self.update_text.emit("aura", "Listening... speak your question now.")

            import time
            time.sleep(0.3)  # Brief pause for audio buffer to settle

            if record_audio():
                self.status_changed.emit("📝 Transcribing...")
                text = transcribe()
                self.update_text.emit("user", text)

                if text in ["exit", "quit", "stop", "close"]:
                    self.update_text.emit("aura", "Goodbye! Have a wonderful day ahead.")
                    speak("Goodbye! Have a wonderful day ahead.")
                    self.finished_processing.emit()
                    return

                self._process_query(text)
            else:
                self.status_changed.emit("No speech detected. Ready.")
        except Exception as e:
            self.update_text.emit("aura", f"Error: {str(e)}")
            self.status_changed.emit(f"❌ Error: {str(e)[:50]}")
        finally:
            self.finished_processing.emit()

    def _process_query(self, text: str):
        """Route query through intent classifier and handle response."""
        self.status_changed.emit("🧠 Classifying intent...")
        intent = classify(text)

        if intent == Intent.SYSTEM_COMMAND:
            self.status_changed.emit("⚙️ Executing system command...")
            execute_command(text)
            self.update_text.emit("aura", "[System command executed]")

        elif intent == Intent.PORTFOLIO_QUERY:
            self.status_changed.emit("📊 Analyzing portfolio...")
            speak("Let me check the portfolio data...")
            self.update_text.emit("aura", "📊 Analyzing portfolio data...")
            response = handle_portfolio_query(text)
            speak(response)
            self.update_text.emit("aura", response)

        elif intent == Intent.RESEARCH_QA:
            self.status_changed.emit("📄 Searching research documents...")
            speak("Let me search through the research documents...")
            self.update_text.emit("aura", "📄 Searching research documents...")
            response = answer_research_question(text)
            speak(response)
            self.update_text.emit("aura", response)

        elif intent == Intent.SENTIMENT_ANALYSIS:
            self.status_changed.emit("📈 Analyzing sentiment...")
            speak("Analyzing sentiment...")
            self.update_text.emit("aura", "📈 Running sentiment analysis...")
            response = analyze_and_respond(text)
            speak(response)
            self.update_text.emit("aura", response)

        else:  # GENERAL
            self.status_changed.emit("💬 Thinking...")
            speak("Let me think about that...")
            response = ask_llm(text, use_history=True)
            speak(response)
            self.update_text.emit("aura", response)

        self.status_changed.emit("✅ Ready")


class TextQueryWorker(QThread):
    """Background thread for processing text input queries."""
    update_text = pyqtSignal(str, str)
    status_changed = pyqtSignal(str)
    finished_processing = pyqtSignal()

    def __init__(self, query: str, parent=None):
        super().__init__(parent)
        self.query = query

    def run(self):
        try:
            self.status_changed.emit("🧠 Processing query...")
            intent = classify(self.query)

            if intent == Intent.SYSTEM_COMMAND:
                self.status_changed.emit("⚙️ Executing system command...")
                execute_command(self.query)
                self.update_text.emit("aura", "[System command executed]")

            elif intent == Intent.PORTFOLIO_QUERY:
                self.status_changed.emit("📊 Analyzing portfolio...")
                self.update_text.emit("aura", "📊 Analyzing portfolio data...")
                response = handle_portfolio_query(self.query)
                self.update_text.emit("aura", response)

            elif intent == Intent.RESEARCH_QA:
                self.status_changed.emit("📄 Searching documents...")
                self.update_text.emit("aura", "📄 Searching research documents...")
                response = answer_research_question(self.query)
                self.update_text.emit("aura", response)

            elif intent == Intent.SENTIMENT_ANALYSIS:
                self.status_changed.emit("📈 Analyzing sentiment...")
                self.update_text.emit("aura", "📈 Running sentiment analysis...")
                response = analyze_and_respond(self.query)
                self.update_text.emit("aura", response)

            else:
                self.status_changed.emit("💬 Generating response...")
                response = ask_llm(self.query, use_history=True)
                self.update_text.emit("aura", response)

            self.status_changed.emit("✅ Ready")

        except Exception as e:
            self.update_text.emit("aura", f"❌ Error: {str(e)}")
            self.status_changed.emit("❌ Error occurred")
        finally:
            self.finished_processing.emit()


class PDFIngestWorker(QThread):
    """Background thread for PDF ingestion."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(str, int)  # (filename, chunk_count)
    error = pyqtSignal(str)

    def __init__(self, filepath: str, parent=None):
        super().__init__(parent)
        self.filepath = filepath

    def run(self):
        try:
            from rag.pdf_ingester import ingest_pdf
            self.progress.emit(f"Ingesting {Path(self.filepath).name}...")
            count = ingest_pdf(self.filepath)
            self.finished.emit(Path(self.filepath).name, count)
        except Exception as e:
            self.error.emit(str(e))


class PortfolioRefreshWorker(QThread):
    """Background thread for refreshing portfolio data."""
    data_ready = pyqtSignal(dict)
    positions_ready = pyqtSignal(list)
    error = pyqtSignal(str)

    def run(self):
        try:
            from finance.analytics import get_portfolio_summary, get_open_positions
            summary = get_portfolio_summary()
            positions = get_open_positions()
            self.data_ready.emit(summary)
            self.positions_ready.emit(positions)
        except Exception as e:
            self.error.emit(str(e))


# =============================================================================
# Metric Card Widget
# =============================================================================

class MetricCard(QFrame):
    """A styled card displaying a single metric with label and value."""

    def __init__(self, label: str, value: str = "--", parent=None):
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.setFixedHeight(100)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)

        self.label_widget = QLabel(label)
        self.label_widget.setObjectName("metricLabel")

        self.value_widget = QLabel(value)
        self.value_widget.setObjectName("metricValue")

        layout.addWidget(self.label_widget)
        layout.addWidget(self.value_widget)
        layout.addStretch()

    def set_value(self, value: str, positive: bool | None = None):
        """Update the displayed value with optional color coding."""
        self.value_widget.setText(value)
        if positive is True:
            self.value_widget.setObjectName("positiveValue")
        elif positive is False:
            self.value_widget.setObjectName("negativeValue")
        else:
            self.value_widget.setObjectName("metricValue")
        # Force style update
        self.value_widget.setStyleSheet(self.value_widget.styleSheet())
        self.style().unpolish(self.value_widget)
        self.style().polish(self.value_widget)


# =============================================================================
# Main GUI
# =============================================================================

class AuraGUI(QWidget):
    """Main application window for the AURA Quantitative Research Assistant."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AURA — Quantitative Research Assistant")
        self.setGeometry(200, 80, 1000, 700)
        self.setMinimumSize(800, 550)
        self.setStyleSheet(STYLESHEET)
        self.setAcceptDrops(True)

        self._init_ui()
        self._worker = None
        self._is_processing = False  # Guard against concurrent workers

        # Run startup health checks
        QTimer.singleShot(500, self._run_startup_checks)

    # -------------------------------------------------------------------------
    # Startup Health Checks
    # -------------------------------------------------------------------------

    def _run_startup_checks(self):
        """Run connectivity checks for Ollama and PostgreSQL on startup."""
        warnings = []

        # Check Ollama
        ollama_ok, ollama_msg = check_ollama_connection()
        if not ollama_ok:
            warnings.append(f"⚠️ Ollama: {ollama_msg}")
        else:
            print(f"[STARTUP] {ollama_msg}")
            if "not found" in ollama_msg.lower() or "not installed" in ollama_msg.lower():
                warnings.append(f"⚠️ {ollama_msg}")

        # Check Database
        try:
            from finance.database import check_db_connection
            db_ok, db_msg = check_db_connection()
            if not db_ok:
                warnings.append(f"⚠️ Database: {db_msg}")
            else:
                print(f"[STARTUP] {db_msg}")
        except Exception as e:
            warnings.append(f"⚠️ Database: {str(e)[:100]}")

        # Check Porcupine
        if not PORCUPINE_AVAILABLE:
            print("[STARTUP] Porcupine not configured — voice will record immediately (no wake word).")

        # Show warnings if any
        if warnings:
            warning_text = "\n\n".join(warnings)
            self._append_message(
                "aura",
                f"⚠️ **Startup Warnings:**\n\n{warning_text}\n\n"
                "Some features may not work. Check the README for setup instructions."
            )
            self._update_status("⚠️ Running with warnings")
        else:
            self._append_message(
                "aura",
                f"👋 Welcome! AURA is ready. Using model: **{OLLAMA_MODEL}**\n\n"
                "Try asking:\n"
                "• \"What's my portfolio P&L?\"\n"
                "• \"What is the max drawdown?\"\n"
                "• \"Summarize the research report\"\n"
                "• \"What's the sentiment on earnings?\""
            )
            self._update_status("✅ Ready")

    # -------------------------------------------------------------------------
    # UI Construction
    # -------------------------------------------------------------------------

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(10)

        # Title bar
        title_bar = QHBoxLayout()
        title = QLabel("⚡ AURA")
        title.setObjectName("titleLabel")
        subtitle = QLabel("Quantitative Research Assistant")
        subtitle.setStyleSheet("color: #8b949e; font-size: 14px; padding-top: 6px;")
        title_bar.addWidget(title)
        title_bar.addWidget(subtitle)
        title_bar.addStretch()

        self.status_label = QLabel("● Starting...")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setStyleSheet("color: #e3b341; font-size: 12px;")
        title_bar.addWidget(self.status_label)

        main_layout.addLayout(title_bar)

        # Separator
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.HLine)
        main_layout.addWidget(sep)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_chat_tab(), "💬 Chat")
        self.tabs.addTab(self._build_portfolio_tab(), "📊 Portfolio")
        self.tabs.addTab(self._build_documents_tab(), "📄 Documents")
        main_layout.addWidget(self.tabs)

    # ---- Chat Tab -----------------------------------------------------------

    def _build_chat_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Conversation display
        self.conversation = QTextEdit()
        self.conversation.setReadOnly(True)
        self.conversation.setPlaceholderText(
            "Start a conversation with AURA...\n\n"
            "Try asking:\n"
            '  • "What\'s my portfolio P&L?"\n'
            '  • "What is the max drawdown?"\n'
            '  • "Summarize the research report"\n'
            '  • "What\'s the sentiment on earnings?"'
        )
        layout.addWidget(self.conversation, 1)

        # Input area
        input_layout = QHBoxLayout()

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Type your question here or use the mic button...")
        self.text_input.returnPressed.connect(self._on_text_submit)
        input_layout.addWidget(self.text_input, 1)

        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("primaryBtn")
        self.send_btn.setFixedWidth(80)
        self.send_btn.clicked.connect(self._on_text_submit)
        input_layout.addWidget(self.send_btn)

        mic_label = "🎤 Voice" if PORCUPINE_AVAILABLE else "🎤 Record"
        self.mic_btn = QPushButton(mic_label)
        self.mic_btn.setFixedWidth(100)
        self.mic_btn.clicked.connect(self._on_voice_activate)
        input_layout.addWidget(self.mic_btn)

        # Clear history button
        self.clear_btn = QPushButton("🗑️")
        self.clear_btn.setFixedWidth(40)
        self.clear_btn.setToolTip("Clear conversation history")
        self.clear_btn.clicked.connect(self._on_clear_history)
        input_layout.addWidget(self.clear_btn)

        layout.addLayout(input_layout)

        return widget

    # ---- Portfolio Tab ------------------------------------------------------

    def _build_portfolio_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Refresh button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        refresh_btn = QPushButton("🔄 Refresh Data")
        refresh_btn.setObjectName("primaryBtn")
        refresh_btn.clicked.connect(self._refresh_portfolio)
        btn_layout.addWidget(refresh_btn)
        layout.addLayout(btn_layout)

        # Metrics grid
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(10)

        self.metric_value = MetricCard("PORTFOLIO VALUE")
        self.metric_pnl = MetricCard("REALIZED P&L")
        self.metric_drawdown = MetricCard("MAX DRAWDOWN")
        self.metric_sharpe = MetricCard("SHARPE RATIO")
        self.metric_volatility = MetricCard("VOLATILITY")
        self.metric_var = MetricCard("VALUE AT RISK (95%)")

        metrics_grid.addWidget(self.metric_value, 0, 0)
        metrics_grid.addWidget(self.metric_pnl, 0, 1)
        metrics_grid.addWidget(self.metric_drawdown, 0, 2)
        metrics_grid.addWidget(self.metric_sharpe, 1, 0)
        metrics_grid.addWidget(self.metric_volatility, 1, 1)
        metrics_grid.addWidget(self.metric_var, 1, 2)

        layout.addLayout(metrics_grid)

        # Positions table
        positions_label = QLabel("Open Positions")
        positions_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #58a6ff; margin-top: 8px;")
        layout.addWidget(positions_label)

        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(5)
        self.positions_table.setHorizontalHeaderLabels(
            ["Symbol", "Type", "Quantity", "Entry Price", "Date"]
        )
        self.positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.positions_table.setAlternatingRowColors(True)
        self.positions_table.setStyleSheet(
            self.positions_table.styleSheet()
            + "QTableWidget { alternate-background-color: #1c2128; }"
        )
        layout.addWidget(self.positions_table, 1)

        return widget

    # ---- Documents Tab ------------------------------------------------------

    def _build_documents_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Upload area
        upload_frame = QFrame()
        upload_frame.setObjectName("metricCard")
        upload_frame.setFixedHeight(120)
        upload_layout = QVBoxLayout(upload_frame)
        upload_layout.setAlignment(Qt.AlignCenter)

        drop_label = QLabel("📁 Drag & Drop PDF files here, or click Upload")
        drop_label.setAlignment(Qt.AlignCenter)
        drop_label.setStyleSheet("color: #8b949e; font-size: 14px;")
        upload_layout.addWidget(drop_label)

        upload_btn = QPushButton("📤 Upload PDF")
        upload_btn.setObjectName("primaryBtn")
        upload_btn.setFixedWidth(160)
        upload_btn.clicked.connect(self._on_upload_pdf)
        upload_layout.addWidget(upload_btn, alignment=Qt.AlignCenter)

        layout.addWidget(upload_frame)

        # Progress bar
        self.ingest_progress = QProgressBar()
        self.ingest_progress.setMaximum(0)  # Indeterminate
        self.ingest_progress.hide()
        layout.addWidget(self.ingest_progress)

        self.ingest_status = QLabel("")
        self.ingest_status.setStyleSheet("color: #8b949e;")
        layout.addWidget(self.ingest_status)

        # Document list
        docs_label = QLabel("Ingested Documents")
        docs_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #58a6ff;")
        layout.addWidget(docs_label)

        self.docs_table = QTableWidget()
        self.docs_table.setColumnCount(2)
        self.docs_table.setHorizontalHeaderLabels(["Filename", "Chunks"])
        self.docs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.docs_table, 1)

        # Refresh documents list
        refresh_docs_btn = QPushButton("🔄 Refresh List")
        refresh_docs_btn.clicked.connect(self._refresh_documents_list)
        layout.addWidget(refresh_docs_btn, alignment=Qt.AlignRight)

        return widget

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    def _on_text_submit(self):
        """Handle text input submission."""
        if self._is_processing:
            return  # Prevent concurrent queries

        query = self.text_input.text().strip()
        if not query:
            return

        self.text_input.clear()
        self._append_message("user", query)
        self._set_processing(True)

        self._worker = TextQueryWorker(query)
        self._worker.update_text.connect(self._append_message)
        self._worker.status_changed.connect(self._update_status)
        self._worker.finished_processing.connect(self._on_processing_done)
        self._worker.start()

    def _on_voice_activate(self):
        """Start voice input processing."""
        if self._is_processing:
            return  # Prevent concurrent workers

        self._set_processing(True)

        self._worker = VoiceWorker()
        self._worker.update_text.connect(self._append_message)
        self._worker.hotword_detected.connect(
            lambda: self._update_status("🔊 Hotword detected! Listening...")
        )
        self._worker.status_changed.connect(self._update_status)
        self._worker.finished_processing.connect(self._on_processing_done)
        self._worker.start()

    def _on_processing_done(self):
        """Re-enable input controls after processing."""
        self._set_processing(False)
        self.text_input.setFocus()

    def _set_processing(self, is_processing: bool):
        """Enable/disable input controls based on processing state."""
        self._is_processing = is_processing
        self.text_input.setEnabled(not is_processing)
        self.send_btn.setEnabled(not is_processing)
        self.mic_btn.setEnabled(not is_processing)

    def _on_clear_history(self):
        """Clear conversation display and LLM history."""
        self.conversation.clear()
        clear_history()
        self._append_message("aura", "Conversation history cleared. How can I help you?")

    def _on_upload_pdf(self):
        """Open file dialog to select a PDF for ingestion."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select PDF", str(Path.home()), "PDF Files (*.pdf)"
        )
        if filepath:
            self._ingest_pdf(filepath)

    def _ingest_pdf(self, filepath: str):
        """Start PDF ingestion in background thread."""
        self.ingest_progress.show()
        self.ingest_status.setText(f"Ingesting {Path(filepath).name}...")

        worker = PDFIngestWorker(filepath)
        worker.progress.connect(lambda msg: self.ingest_status.setText(msg))
        worker.finished.connect(self._on_ingest_done)
        worker.error.connect(self._on_ingest_error)
        # Keep reference to prevent GC
        self._ingest_worker = worker
        worker.start()

    def _on_ingest_done(self, filename: str, chunk_count: int):
        """Handle completed PDF ingestion."""
        self.ingest_progress.hide()
        if chunk_count > 0:
            self.ingest_status.setText(
                f"✅ Successfully ingested '{filename}' ({chunk_count} chunks)"
            )
        else:
            self.ingest_status.setText(
                f"ℹ️ '{filename}' was already ingested."
            )
        self._refresh_documents_list()

    def _on_ingest_error(self, error: str):
        """Handle PDF ingestion error."""
        self.ingest_progress.hide()
        self.ingest_status.setText(f"❌ Error: {error}")

    def _refresh_portfolio(self):
        """Refresh portfolio metrics and positions table."""
        self._update_status("📊 Loading portfolio data...")
        worker = PortfolioRefreshWorker()
        worker.data_ready.connect(self._update_metrics)
        worker.positions_ready.connect(self._update_positions_table)
        worker.error.connect(
            lambda e: self._update_status(f"❌ Portfolio error: {e[:50]}")
        )
        self._portfolio_worker = worker
        worker.start()

    def _refresh_documents_list(self):
        """Refresh the documents table with current ingested docs."""
        try:
            from rag.pdf_ingester import get_ingested_documents
            docs = get_ingested_documents()

            self.docs_table.setRowCount(len(docs))
            for row, doc in enumerate(docs):
                self.docs_table.setItem(row, 0, QTableWidgetItem(doc["filename"]))
                self.docs_table.setItem(row, 1, QTableWidgetItem(str(doc["chunks"])))
        except Exception as e:
            print(f"[GUI] Error refreshing documents: {e}")

    # -------------------------------------------------------------------------
    # UI Updates
    # -------------------------------------------------------------------------

    def _append_message(self, role: str, message: str):
        """Append a message to the chat conversation display."""
        if role == "user":
            styled = (
                f'<div style="margin: 8px 0; padding: 10px 14px; '
                f'background-color: #1f6feb; color: white; '
                f'border-radius: 12px 12px 4px 12px; '
                f'max-width: 85%; margin-left: auto; text-align: right;">'
                f'<b>You:</b> {message}</div>'
            )
        else:
            styled = (
                f'<div style="margin: 8px 0; padding: 10px 14px; '
                f'background-color: #21262d; color: #c9d1d9; '
                f'border-radius: 12px 12px 12px 4px; '
                f'max-width: 85%;">'
                f'<b>AURA:</b> {message}</div>'
            )
        self.conversation.append(styled)

        # Auto-scroll to bottom
        scrollbar = self.conversation.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _update_status(self, status: str):
        """Update the status label."""
        self.status_label.setText(status)

        # Color code based on status
        if "error" in status.lower() or "❌" in status:
            self.status_label.setStyleSheet("color: #f85149; font-size: 12px;")
        elif "ready" in status.lower() or "✅" in status:
            self.status_label.setStyleSheet("color: #3fb950; font-size: 12px;")
        elif "⚠️" in status:
            self.status_label.setStyleSheet("color: #e3b341; font-size: 12px;")
        else:
            self.status_label.setStyleSheet("color: #58a6ff; font-size: 12px;")

    def _update_metrics(self, summary: dict):
        """Update portfolio metric cards with fresh data."""
        value = summary.get("current_portfolio_value", 0)
        self.metric_value.set_value(f"${value:,.2f}")

        pnl = summary.get("realized_pnl", {}).get("total_pnl", 0)
        self.metric_pnl.set_value(
            f"${pnl:+,.2f}",
            positive=pnl >= 0 if pnl != 0 else None,
        )

        dd = summary.get("max_drawdown", {}).get("max_drawdown_pct", 0)
        self.metric_drawdown.set_value(f"-{dd:.2f}%", positive=False if dd > 5 else None)

        sharpe = summary.get("sharpe_ratio", {}).get("sharpe_ratio", 0)
        self.metric_sharpe.set_value(
            f"{sharpe:.3f}",
            positive=sharpe > 0.5 if sharpe != 0 else None,
        )

        vol = summary.get("volatility", {}).get("annualized_volatility_pct", 0)
        self.metric_volatility.set_value(f"{vol:.2f}%")

        var = summary.get("value_at_risk_95", {}).get("var_dollar", 0)
        self.metric_var.set_value(f"${var:,.2f}", positive=False)

        self._update_status("✅ Portfolio data loaded")

    def _update_positions_table(self, positions: list):
        """Populate the positions table with open positions data."""
        self.positions_table.setRowCount(len(positions))
        for row, pos in enumerate(positions):
            self.positions_table.setItem(row, 0, QTableWidgetItem(pos["symbol"]))
            self.positions_table.setItem(row, 1, QTableWidgetItem(pos["trade_type"]))
            self.positions_table.setItem(row, 2, QTableWidgetItem(str(pos["quantity"])))
            self.positions_table.setItem(
                row, 3, QTableWidgetItem(f"${pos['entry_price']:.2f}")
            )
            self.positions_table.setItem(row, 4, QTableWidgetItem(pos["trade_date"]))

    # -------------------------------------------------------------------------
    # Drag and Drop
    # -------------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Accept PDF file drags."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(".pdf"):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event: QDropEvent):
        """Handle dropped PDF files."""
        for url in event.mimeData().urls():
            filepath = url.toLocalFile()
            if filepath.lower().endswith(".pdf"):
                self.tabs.setCurrentIndex(2)  # Switch to Documents tab
                self._ingest_pdf(filepath)


# =============================================================================
# Application Entry Point
# =============================================================================

def run_gui():
    """Launch the AURA GUI application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Set dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(13, 17, 23))
    palette.setColor(QPalette.WindowText, QColor(201, 209, 217))
    palette.setColor(QPalette.Base, QColor(22, 27, 34))
    palette.setColor(QPalette.AlternateBase, QColor(28, 33, 40))
    palette.setColor(QPalette.Text, QColor(201, 209, 217))
    palette.setColor(QPalette.Button, QColor(33, 38, 45))
    palette.setColor(QPalette.ButtonText, QColor(201, 209, 217))
    palette.setColor(QPalette.Highlight, QColor(31, 111, 235))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = AuraGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_gui()
