"""
app.py
------
Web version of the CS Department Chatbot. Run with:
    python app.py
Then open http://localhost:5000 in your browser.

Uses the same RAG/LLM logic as the desktop app (see rag_engine.py),
so it needs Ollama running locally the same way the desktop app did.
"""

import json
import os
import time
import threading
import webbrowser
from flask import Flask, render_template, request, jsonify, session, Response
from rag_engine import RAGChatEngine
import db
import uuid

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-this-for-real-deployment"

db.init_db()

# One shared engine instance; the knowledge base is loaded once at startup
engine = RAGChatEngine()

# Per-session conversation history, kept in memory (server restarts clear it)
conversations = {}


@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html", ai_mode=engine.ollama_available)


@app.route("/api/chat", methods=["POST"])
def chat():
    """Non-streaming endpoint (kept for simplicity / API use).
    The web UI uses /api/chat/stream instead for a faster-feeling reply."""
    data = request.get_json()
    message = (data or {}).get("message", "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    session_id = session.get("session_id", "default")
    history = conversations.setdefault(session_id, [])

    start_time = time.time()
    answer, sources = engine.get_response(message, history)
    response_time_ms = int((time.time() - start_time) * 1000)
    db.log_interaction(session_id, message, answer, sources, response_time_ms)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    conversations[session_id] = history[-20:]

    return jsonify({"answer": answer, "sources": sources})


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """Streaming endpoint: sends newline-delimited JSON objects as the
    answer is generated, so the browser can show it word-by-word."""
    data = request.get_json()
    message = (data or {}).get("message", "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    session_id = session.get("session_id", "default")
    history = conversations.setdefault(session_id, [])
    history_snapshot = list(history)  # safe copy for the generator to read

    def generate():
        full_answer = ""
        sources = []
        start_time = time.time()
        for event in engine.stream_response(message, history_snapshot):
            if event["type"] == "token":
                full_answer += event["text"]
            elif event["type"] == "done":
                sources = event.get("sources", [])
            yield json.dumps(event) + "\n"

        response_time_ms = int((time.time() - start_time) * 1000)
        db.log_interaction(session_id, message, full_answer, sources, response_time_ms)

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": full_answer})
        conversations[session_id] = history[-20:]

    return Response(generate(), mimetype="text/plain")


@app.route("/analytics")
def analytics():
    stats = db.get_stats()
    return render_template("analytics.html", stats=stats)


@app.route("/api/clear", methods=["POST"])
def clear():
    session_id = session.get("session_id", "default")
    conversations[session_id] = []
    return jsonify({"status": "cleared"})


def open_browser():
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    # Flask's debug reloader runs this file twice (once as a watcher
    # process, once as the real server) - WERKZEUG_RUN_MAIN is only set
    # on the real one, so this check stops the browser opening twice.
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Timer(1.0, open_browser).start()

    app.run(debug=True, port=5000, threaded=True)