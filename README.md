# CS Department Chatbot

An AI-powered FAQ chatbot for a college Computer Science department, built with a local LLM (via [Ollama](https://ollama.com)) and Retrieval-Augmented Generation (RAG) over department documents. Runs entirely on your own machine — no API keys, no cost.

Originally built as a PyQt desktop app, then rebuilt as a Flask web app with streaming responses and a usage analytics dashboard.

> **Note on the content:** The department facts and sample documents in this repo (courses, faculty, placement stats, etc.) are illustrative placeholder data for demonstration purposes, not real figures from any specific institution.

---

## ✨ Features

- **Real LLM-powered answers**, not hardcoded keyword matching — runs a local model via Ollama, so it's free and private
- **RAG (Retrieval-Augmented Generation)** — answers are grounded in real `.txt` documents you drop into a `documents/` folder, retrieved by semantic similarity (embeddings + cosine similarity), not just keyword search
- **Streaming responses** — answers appear word-by-word in the browser, like ChatGPT, instead of one long wait
- **Source citations** — every RAG-backed answer shows which document it pulled from
- **Analytics dashboard** — SQLite-backed logging of every question, with a `/analytics` page showing usage by topic, daily volume, and average response time
- **Graceful fallback** — if Ollama isn't running, the bot still works using simple keyword-matched FAQ answers instead of crashing
- **Two interfaces** — a Flask web app (primary) and the original PyQt5 desktop app, sharing the same core logic

---

## 🖼️ Screenshots

*(Add a screenshot or short GIF of the chat interface here, and one of the analytics dashboard — drag them into this README on GitHub, or reference an `assets/` folder.)*

```
![Chat interface](assets/screenshot-chat.png)
![Analytics dashboard](assets/screenshot-analytics.png)
```

---

## 🏗️ Architecture

```
User's browser
      │
      ▼
  app.py (Flask)  ──────►  rag_engine.py (RAGChatEngine)
      │                          │
      │                          ├─► Ollama (local LLM + embeddings)
      │                          └─► documents/*.txt (knowledge base)
      │
      ▼
  db.py (SQLite) ──► chatbot.db (interaction log)
```

- **`rag_engine.py`** — the "brain": loads FAQ data, chunks and embeds documents, retrieves relevant context per question, calls the local LLM, and falls back to keyword matching if the LLM is unavailable. No GUI code — shared by both interfaces.
- **`app.py`** — Flask server: serves the chat UI, exposes a streaming `/api/chat/stream` endpoint, and an `/analytics` dashboard.
- **`db.py`** — SQLite logging and analytics queries.
- **`chatbot.py`** — the original PyQt5 desktop version (uses its own copy of the same logic, run on a background thread so the UI doesn't freeze).

---

## 🚀 Getting Started

### 1. Install [Ollama](https://ollama.com/download) and pull the models
```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```
Make sure Ollama is running in the background (it usually auto-starts after install).

### 2. Set up the Python environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 3. Run the web app
```bash
python app.py
```
Your browser will open automatically to `http://127.0.0.1:5000`.

### 4. (Optional) Run the desktop version
```bash
pip install PyQt5
python chatbot.py
```

---

## 📁 Adding Your Own Knowledge Base

Drop any `.txt` files into the `documents/` folder. They're automatically chunked, embedded, and searched at question time — no code changes needed. Replace the placeholder `department_info.txt` with real course catalogs, handbooks, or faculty lists whenever you have them.

---

## 🧪 Running Tests

```bash
pip install -r requirements-dev.txt
pytest
```

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **LLM:** Ollama (local inference — `llama3.2:1b` for chat, `nomic-embed-text` for embeddings)
- **RAG:** Custom chunking + cosine-similarity retrieval (no external vector DB required)
- **Database:** SQLite
- **Desktop UI:** PyQt5
- **Testing:** pytest

---

## 📄 License

See [LICENSE](LICENSE).