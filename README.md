# AURA — Voice-Activated Quantitative Research Assistant

A 100% local AI-powered finance research assistant that answers natural-language questions about your trading portfolio and research documents via **voice** or **text**. No cloud APIs required.

## ✨ Features

- **🎤 Voice-Activated**: Say "Hey Aura" to start a conversation (Pvporcupine wake-word)
- **📊 Portfolio Analytics**: P&L, VaR, Sharpe ratio, max drawdown, volatility, and position tracking
- **📄 Research RAG**: Upload PDF reports and ask questions — answers are grounded in document content
- **📈 Sentiment Analysis**: Analyze the sentiment of earnings calls and financial reports
- **🧠 Local LLM**: All inference runs locally via Ollama (no OpenAI/cloud APIs)
- **💬 Text & Voice Input**: Type queries or use voice through the PyQt5 GUI
- **⚡ System Commands**: Lock PC, take screenshots, control volume, and more
- **🔄 Conversation History**: Multi-turn chat — AURA remembers context from previous messages
- **🏥 Health Checks**: Startup diagnostics for Ollama, PostgreSQL, and Porcupine

## 🏗️ Architecture

```
Voice/Text → Intent Router → Handler → Ollama LLM → TTS/Display
                  │
                  ├── PORTFOLIO_QUERY  → Analytics Engine → PostgreSQL
                  ├── RESEARCH_QA      → RAG Pipeline → pgvector
                  ├── SENTIMENT        → Transformer Model
                  ├── SYSTEM_COMMAND   → System Actions
                  └── GENERAL          → LLM Chat (with history)
```

## 📋 Prerequisites

- **Python 3.11+**
- **PostgreSQL 15+** with the [pgvector](https://github.com/pgvector/pgvector) extension
- **Ollama** with a model installed (e.g., `ollama pull qwen2.5:3b`)
- **Pvporcupine** access key ([get one free](https://picovoice.ai/)) — *optional, voice works without it*

### Hardware Requirements

| RAM | Recommended Ollama Model | Notes |
|-----|-------------------------|-------|
| 8 GB | `qwen2.5:3b`, `gemma2:2b` | Best for laptops |
| 16 GB | `llama3.2:3b`, `mistral:7b` | Good balance |
| 32 GB | `llama3.1:8b`, `mixtral:8x7b` | Best quality |

## 🚀 Setup

### 1. Install Dependencies

```bash
cd Aura-main
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings:
#   - OLLAMA_MODEL (choose based on your RAM)
#   - DATABASE_URL
#   - PORCUPINE_ACCESS_KEY (optional)
```

### 3. Install & Start Ollama

```bash
# Install Ollama from https://ollama.ai
# Pull the recommended model:
ollama pull qwen2.5:3b

# Make sure Ollama is running:
ollama serve
```

### 4. Set Up PostgreSQL

```sql
CREATE DATABASE aura_finance;
\c aura_finance
CREATE EXTENSION vector;
```

### 5. Initialize & Seed Database

```bash
python main.py --init       # Create tables
python main.py --seed       # Generate synthetic portfolio data
```

### 6. Health Check (Optional)

```bash
python main.py --check      # Verify Ollama, PostgreSQL, and Porcupine
```

### 7. Launch

```bash
# GUI mode (default)
python main.py

# CLI mode (voice only)
python main.py --cli
```

## 📂 Project Structure

```
Aura-main/
├── main.py                    # Entry point (CLI & GUI launcher)
├── aura_gui.py                # PyQt5 GUI (Chat, Portfolio, Documents tabs)
├── conftest.py                # Pytest configuration
├── config/
│   └── settings.py            # Centralized .env configuration
├── core/
│   ├── audio.py               # Microphone recording & VAD
│   ├── stt.py                 # Speech-to-text (Faster-Whisper)
│   ├── tts.py                 # Text-to-speech (pyttsx3)
│   ├── hotword.py             # Wake-word detection (Pvporcupine)
│   └── llm.py                 # LLM integration (Ollama) with history
├── intent/
│   ├── router.py              # Intent classification (keyword + LLM)
│   └── commands.py            # System commands (lock, volume, etc.)
├── finance/
│   ├── database.py            # SQLAlchemy engine & sessions (lazy init)
│   ├── models.py              # ORM models (Trade, Snapshot, Return, Doc)
│   ├── seed_data.py           # Synthetic data generator
│   ├── analytics.py           # Portfolio analytics engine
│   └── query_handler.py       # NL query → analytics → response
├── rag/
│   ├── pdf_ingester.py        # PDF → chunks → embeddings → pgvector
│   ├── retriever.py           # Semantic similarity search
│   └── qa_handler.py          # RAG question answering
├── sentiment/
│   └── analyzer.py            # Transformer-based sentiment analysis
├── data/
│   └── research_pdfs/         # Drop PDF files here
└── tests/
    ├── test_analytics.py      # Portfolio math tests
    ├── test_intent_router.py  # Intent classification tests
    └── test_rag.py            # RAG pipeline tests
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📊 Usage Examples

### Voice
> "Hey Aura... What's my portfolio P&L?"
> "Hey Aura... What is the max drawdown?"
> "Hey Aura... Summarize the research report on Apple"
> "Hey Aura... What's the sentiment of the latest earnings call?"

### Text (via GUI)
Type directly into the chat input box — no wake word needed.

### CLI Commands
```bash
# Run health checks
python main.py --check

# Ingest a single PDF
python main.py --ingest path/to/report.pdf

# Ingest all PDFs in a directory
python main.py --ingest data/research_pdfs/

# Force re-seed database
python main.py --force-seed
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Wake Word | Pvporcupine (optional) |
| Speech-to-Text | Faster-Whisper (CPU/int8) |
| Text-to-Speech | pyttsx3 |
| LLM | Ollama (qwen2.5:3b recommended for 8GB RAM) |
| Database | PostgreSQL + pgvector |
| ORM | SQLAlchemy 2.0 |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Sentiment | cardiffnlp/twitter-roberta-base-sentiment |
| GUI | PyQt5 |
| PDF Parsing | PyPDF2 |

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Cannot connect to Ollama" | Run `ollama serve` in a terminal |
| "Model not found" | Run `ollama pull qwen2.5:3b` |
| "Cannot connect to PostgreSQL" | Start PostgreSQL and check DATABASE_URL in `.env` |
| "No speech detected" | Speak louder or increase `AUDIO_RECORD_DURATION` in `.env` |
| Voice input not working | Porcupine is optional — voice will record immediately without it |
| App is slow on first launch | Models are downloading (~500MB total). This is a one-time operation. |

## 📝 License

This project is for educational and research purposes.
# Aura-Fully-Local-Voice-Activated-Quantitative-Research-Assistant
