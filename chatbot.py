import sys
import json
import requests
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QTextEdit, QLineEdit, QPushButton,
                             QComboBox, QGroupBox, QFrame, QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, QDateTime, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette, QTextCursor, QIcon, QPixmap

print("Current working directory:", os.getcwd())
print("Files in this directory:", os.listdir())

# ============================================================
# LLM CONFIGURATION (Ollama - free, runs locally, no API key)
# ============================================================
# 1. Install Ollama from https://ollama.com/download
# 2. Run once in a terminal:  ollama pull llama3.2
# 3. Make sure the Ollama app/service is running in the background
# No key needed. If Ollama isn't running or the model isn't pulled,
# the bot automatically falls back to the old keyword-matching FAQ
# logic so the app still works either way.
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "llama3.2:1b"  # smaller & faster than llama3.2 (3b) — run: ollama pull llama3.2:1b
RESPONSE_MAX_TOKENS = 120     # caps answer length for faster replies
OLLAMA_EMBED_MODEL = "nomic-embed-text"  # run: ollama pull nomic-embed-text

# ============================================================
# RAG CONFIGURATION
# ============================================================
# Any .txt files placed in the "documents" folder get chunked,
# embedded, and searched at question time. Add real department
# documents here later (course catalog, handbook, etc.) — just
# save them as plain .txt files in this folder.
DOCUMENTS_FOLDER = "documents"
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # overlap between chunks so context isn't cut mid-thought
TOP_K_CHUNKS = 3        # how many chunks to pull in per question


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)


class ResponseWorker(QThread):
    """Runs the (potentially slow) LLM/RAG call on a background thread
    so the PyQt GUI never freezes while waiting for a response."""
    finished = pyqtSignal(str, list)  # (answer_text, source_chunk_labels)

    def __init__(self, chatbot, message):
        super().__init__()
        self.chatbot = chatbot
        self.message = message

    def run(self):
        answer, sources = self.chatbot.get_response(self.message)
        self.finished.emit(answer, sources)


class MessageWidget(QWidget):
    def __init__(self, text, is_user=True, timestamp=None, sources=None, is_typing=False):
        super().__init__()
        self.is_user = is_user
        self.timestamp = timestamp or QDateTime.currentDateTime().toString("hh:mm AP")
        self.is_typing = is_typing

        outer_layout = QHBoxLayout()
        outer_layout.setContentsMargins(8, 4, 8, 4)
        self.setLayout(outer_layout)

        # Avatar (simple circular badge)
        avatar = QLabel("🎓" if not is_user else "🧑")
        avatar.setFixedSize(36, 36)
        avatar.setAlignment(Qt.AlignCenter)
        avatar_color = "#0052CC" if not is_user else "#4caf50"
        avatar.setStyleSheet(f"""
            background-color: {avatar_color};
            color: white;
            border-radius: 18px;
            font-size: 16px;
        """)

        # Bubble containing the message
        bubble = QFrame()
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(14, 10, 14, 10)
        bubble.setLayout(bubble_layout)
        bubble.setMaximumWidth(650)

        message_label = QLabel(text)
        message_label.setObjectName("message_text")
        message_label.setWordWrap(True)
        message_label.setTextFormat(Qt.RichText)
        message_label.setStyleSheet("color: #1a1a1a; font-size: 16px; font-weight: normal; background: transparent;")
        bubble_layout.addWidget(message_label)

        # Source citation footer (only for bot messages answered via RAG)
        if sources:
            sources_label = QLabel("📄 Based on: " + "; ".join(sources))
            sources_label.setWordWrap(True)
            sources_label.setStyleSheet("color: #6b7280; font-size: 11px; font-style: italic; font-weight: normal; background: transparent; padding-top: 4px;")
            bubble_layout.addWidget(sources_label)

        time_label = QLabel(self.timestamp)
        time_label.setAlignment(Qt.AlignRight)
        time_label.setStyleSheet("color: #9ca3af; font-size: 11px; font-weight: normal; background: transparent; padding-top: 2px;")
        bubble_layout.addWidget(time_label)

        if is_user:
            bubble.setStyleSheet("""
                QFrame {
                    background-color: #dcf8c6;
                    border-radius: 14px;
                }
            """)
            outer_layout.addStretch()
            outer_layout.addWidget(bubble)
            outer_layout.addWidget(avatar)
        else:
            bubble.setStyleSheet("""
                QFrame {
                    background-color: #ffffff;
                    border-radius: 14px;
                    border: 1px solid #e5e7eb;
                }
            """)
            outer_layout.addWidget(avatar)
            outer_layout.addWidget(bubble)
            outer_layout.addStretch()


class ChatBot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SECAB INSTITUTE OF ENGINEERING AND TECHNOLOGY VIJAYPUR")
        self.setStyleSheet("color:black;font-size:20px;font-weight:bold;")
        self.setGeometry(200, 200, 800, 600)

        # Initialize FAQ database (still used for quick-question dropdown
        # and as a fallback / context source for the LLM)
        self.faq_data = self.load_faq_data()

        # Keep a short running conversation so the LLM has context
        # (list of {"role": "user"/"assistant", "content": "..."})
        self.conversation_history = []

        # Check once at startup whether Ollama is reachable, so the
        # header can show an accurate AI mode status
        self.ollama_available = self.check_ollama_available()

        # RAG: load and embed the knowledge base documents once at startup
        self.knowledge_chunks = []     # list of chunk text
        self.knowledge_sources = []    # list of source filenames, same order as chunks
        self.knowledge_embeddings = [] # list of embedding vectors, same order
        if self.ollama_available:
            self.build_knowledge_base()

        # Track the background worker thread for the current in-flight request
        self.active_worker = None

        self.setup_ui()

        # Load any previously saved chat history from last session
        self.load_chat_history()

    def check_ollama_available(self):
        """Quick check if the local Ollama server is up and responding."""
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def embed_text(self, text):
        """Get an embedding vector for a piece of text from Ollama."""
        try:
            resp = requests.post(
                OLLAMA_EMBED_URL,
                json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("embedding")
        except Exception as e:
            print(f"Embedding failed: {e}")
            return None

    def build_knowledge_base(self):
        """Read every .txt file in the documents folder, chunk it,
        and embed each chunk so it can be searched later."""
        if not os.path.isdir(DOCUMENTS_FOLDER):
            print(f"No '{DOCUMENTS_FOLDER}' folder found — skipping RAG setup.")
            return

        for filename in os.listdir(DOCUMENTS_FOLDER):
            if not filename.lower().endswith(".txt"):
                continue
            filepath = os.path.join(DOCUMENTS_FOLDER, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"Could not read {filename}: {e}")
                continue

            for chunk in chunk_text(text):
                embedding = self.embed_text(chunk)
                if embedding:
                    self.knowledge_chunks.append(chunk)
                    self.knowledge_sources.append(os.path.splitext(filename)[0].replace("_", " ").title())
                    self.knowledge_embeddings.append(embedding)

        print(f"RAG knowledge base ready: {len(self.knowledge_chunks)} chunks loaded.")

    def retrieve_relevant_chunks(self, query, top_k=TOP_K_CHUNKS):
        """Find the most relevant knowledge base chunks for a question.
        Returns a list of (chunk_text, source_label) tuples."""
        if not self.knowledge_chunks:
            return []

        query_embedding = self.embed_text(query)
        if not query_embedding:
            return []

        scored = [
            (cosine_similarity(query_embedding, emb), chunk, source)
            for chunk, source, emb in zip(self.knowledge_chunks, self.knowledge_sources, self.knowledge_embeddings)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(chunk, source) for score, chunk, source in scored[:top_k]]

    def load_faq_data(self):
        """Load FAQ questions and answers"""
        return {
            "admission": {
                "questions": [
                    "Where is Admission cell Located?",
                    "Where is Principal Office?"
                ],
                "answers": {
                    "default": "It is located at the Enterance of the College."
                }
            },
            "courses": {
                "questions": [
                    "What are  the basic languages required?",
                    "What programming languages are taught?"
                ],
                "answers": {
                    "default": "Python, Java, and C++."
                }
            },
            "Department": {
                "questions": [
                    "Where is Computer Science Department located?",
                    "Where is HOD Cabin Located?",
                ],
                "answers": {
                    "default": "It is Located in Block 4."
                }
            },
            "college": {
                "questions": [
                    "What are department in our college?",
                    "How many department are there in our college?"
                ],
                "answers": {
                    "default": "There are 6 department there are EEE,EC,CSE,AIML,MECH,CIVIL."
                }
            },
            "careers": {
                "questions": [
                    "What job opportunities are there?",
                    "What's the average starting salary?"
                ],
                "answers": {
                    "default": "Tech initator and many startups. The average starting salary is 3LPA with near 75% placement rate."
                }
            }
        }

    def build_system_prompt(self, retrieved_chunks=None):
        """Turn the FAQ data + retrieved document chunks into context
        the LLM can draw on."""
        facts = []
        for category, data in self.faq_data.items():
            facts.append(f"- {category}: {data['answers']['default']}")
        facts_text = "\n".join(facts)

        prompt = (
            "You are the official CS Department FAQ chatbot for SECAB Institute "
            "of Engineering and Technology, Vijaypur. Answer student questions "
            "about admissions, courses, the department, faculty, and career "
            "opportunities. Be friendly, concise (2-4 sentences), and stay on "
            "topic. If a question is unrelated to the college or department, "
            "politely say you can only help with CS department topics.\n\n"
            "STYLE RULES (follow exactly):\n"
            "- Start your answer with the actual information, not a lead-in.\n"
            "- Never repeat the college or department's full name in your "
            "answer — the student already knows where they study.\n"
            "- Never restate or rephrase the student's question.\n"
            "- Keep it short and direct, like a knowledgeable senior student "
            "answering a junior's question in the hallway.\n\n"
            "Example of the WRONG style (too formal, repeats the question):\n"
            "Q: What are 3rd year subjects?\n"
            "A: In the Computer Science and Engineering department at SECAB "
            "Institute of Engineering and Technology, 3rd year students are "
            "exposed to the following subjects: ...\n\n"
            "Example of the RIGHT style (direct, no preamble):\n"
            "Q: What are 3rd year subjects?\n"
            "A: 3rd year covers Database Management Systems, Operating "
            "Systems, Computer Networks, Software Engineering, and Web "
            "Technologies.\n\n"
            "Known facts about the department:\n"
            f"{facts_text}\n"
        )

        if retrieved_chunks:
            context_text = "\n---\n".join(chunk for chunk, source in retrieved_chunks)
            prompt += (
                "\nRelevant excerpts from department documents:\n"
                f"{context_text}\n\n"
                "Prefer information from these excerpts when they're relevant. "
            )

        prompt += (
            "\nIf the answer isn't in the known facts or excerpts, you may "
            "answer generally and helpfully, but don't invent specific "
            "numbers, names, or dates."
        )
        return prompt

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Header
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_widget.setStyleSheet("""
                                    background-color:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #1e3c72,stop:1 #2a5298);
                                    border-radius:8px 8px 0 0;
                                    """)

        header_layout.setContentsMargins(15, 8, 15, 8)
        header_layout.setSpacing(15)

        # College logo (left side)
        logo_label = QLabel()
        logo_pixmap = QPixmap("logo.png").scaled(55, 55, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(logo_pixmap)
        logo_label.setFixedSize(55, 55)

        # Title text
        title_label = QLabel("Computer Science Department Chatbot")
        title_label.setStyleSheet("color:white;font-size:22px;font-weight:bold;")
        title_label.setAlignment(Qt.AlignVCenter)

        # Small status label so you can see whether AI mode is active.
        # We check this at startup by pinging Ollama - see check_ollama_available().
        mode_text = "AI mode: ON (Ollama)" if self.ollama_available else "AI mode: OFF (Ollama not running)"
        mode_label = QLabel(mode_text)
        mode_label.setStyleSheet("color:#c7d7ff;font-size:13px;font-weight:normal;")

        header_layout.addWidget(logo_label)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(mode_label)

        clear_button = QPushButton("Clear Chat")
        clear_button.clicked.connect(self.clear_chat)
        clear_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(255,255,255,0.15);
                color: white;
                font-size: 13px;
                font-weight: normal;
                border-radius: 6px;
                padding: 6px 12px;
                border: 1px solid rgba(255,255,255,0.3);
            }
            QPushButton:hover {
                background-color: rgba(255,255,255,0.25);
            }
        """)
        header_layout.addWidget(clear_button)

        layout.addWidget(header_widget)

        # Chat area
        self.chat_area = QScrollArea()
        self.chat_area.setWidgetResizable(True)
        self.chat_area_widget = QWidget()
        self.chat_area_layout = QVBoxLayout(self.chat_area_widget)
        self.chat_area.setWidget(self.chat_area_widget)
        self.chat_area.setStyleSheet("background-color:#f5f6fa; border: none;")
        self.chat_area_layout.setSpacing(2)
        layout.addWidget(self.chat_area)

        # Quick questions
        quick_questions_group = QGroupBox("Quick Questions")
        quick_questions_layout = QHBoxLayout(quick_questions_group)
        quick_questions_group.setStyleSheet("""
            QGroupBox{
                font-size:20px;
                font-weight:bold;
                padding:8px;
                border:1px solid black;
                border-radius:10px;
                margin-top:10px;
            }
        """)

        self.quick_question_combo = QComboBox()
        self.quick_question_combo.setMinimumHeight(50)
        self.quick_question_combo.setMinimumWidth(800)
        self.quick_question_combo.setStyleSheet("""
            QComboBox{
                    font-size:20px;
                    padding:10px 15px;
                    border-radius:10px;
                    border:1px solid #999;
                    background-color:#e6f0ff;
                    color:#1e3c72;
                    font-weight:bold;
                }
                QComboBox::drop-down{
                        width:30px;
                }
            """)
        self.quick_question_combo.addItem("Select a common question...")

        # Add questions from all categories
        for category, data in self.faq_data.items():
            for question in data["questions"]:
                self.quick_question_combo.addItem(question)

        quick_questions_layout.addWidget(self.quick_question_combo)

        ask_button = QPushButton("Ask This")
        ask_button.clicked.connect(self.ask_quick_question)
        ask_button.setMinimumHeight(50)
        ask_button.setMinimumWidth(80)
        ask_button.setStyleSheet("""
            QPushButton{
                background-color:#0052CC;
                color: white;
                font-size:22px;
                font-weight:bold;
                border-radius:10px;
                padding:10px 25px;
                border:none;
            }
            QPushButton:hover{
                background-color:#003E99;
            }
        """)
        quick_questions_layout.addWidget(ask_button, stretch=1)

        layout.addWidget(quick_questions_group)

        # Input area
        input_frame = QFrame()
        input_layout = QHBoxLayout(input_frame)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your question about the CS department...")
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)

        send_button = QPushButton("Send")
        send_button.clicked.connect(self.send_message)
        send_button.setStyleSheet("background-color: #007bff; color: white; font-weight: bold;")
        input_layout.addWidget(send_button)

        layout.addWidget(input_frame)
        self.setStyleSheet("color:black;font-size:22px;font-weight:bold;")

    def ask_quick_question(self):
        question = self.quick_question_combo.currentText()
        if question != "Select a common question...":
            self.add_user_message(question)
            self.process_message(question)
            self.quick_question_combo.setCurrentIndex(0)

    def send_message(self):
        message = self.input_field.text().strip()
        if message:
            self.add_user_message(message)
            self.input_field.clear()
            self.process_message(message)

    def add_user_message(self, text):
        message_widget = MessageWidget(text, is_user=True)
        self.chat_area_layout.addWidget(message_widget)
        self.scroll_to_bottom()

    def add_bot_message(self, text, sources=None, save=True):
        message_widget = MessageWidget(text, is_user=False, sources=sources)
        self.chat_area_layout.addWidget(message_widget)
        self.scroll_to_bottom()

    def show_typing_indicator(self):
        self.typing_widget = MessageWidget("Typing...", is_user=False, is_typing=True)
        self.chat_area_layout.addWidget(self.typing_widget)
        self.scroll_to_bottom()

    def remove_typing_indicator(self):
        if getattr(self, "typing_widget", None) is not None:
            self.chat_area_layout.removeWidget(self.typing_widget)
            self.typing_widget.deleteLater()
            self.typing_widget = None

    def scroll_to_bottom(self):
        scrollbar = self.chat_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ============================================================
    # LLM CALL
    # ============================================================
    def call_llm(self, message, retrieved_chunks):
        """Call the local Ollama server for a real, context-aware answer.
        Returns None if the call fails, so the caller can fall back
        to the old keyword-matching logic."""
        messages = [{"role": "system", "content": self.build_system_prompt(retrieved_chunks)}]
        # include the last few turns for context
        messages.extend(self.conversation_history[-6:])
        messages.append({"role": "user", "content": message})

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_predict": RESPONSE_MAX_TOKENS},
                },
                timeout=60,  # local models can be slower than a cloud API
            )
            response.raise_for_status()
            data = response.json()
            answer = data["message"]["content"].strip()

            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": answer})
            return answer
        except Exception as e:
            print(f"LLM call failed, falling back to FAQ matching: {e}")
            return None

    def get_response(self, message):
        """Compute the bot's answer for a message. Runs on a background
        thread (see ResponseWorker) so the GUI never freezes on a slow
        Ollama call. Returns (answer_text, source_labels)."""
        message_lower = message.lower().strip()
        words = message_lower.split()
        word_count = len(words)

        # Fast path: skip RAG + LLM entirely for simple greetings/thanks.
        # Checks whole words only (not substrings) so "think" doesn't
        # falsely match "hi", etc.
        if word_count <= 4 and any(word in words for word in ['hello', 'hi', 'hey', 'greetings']):
            return "Hello! How can I help you with Computer Science department information today?", []
        if word_count <= 5 and any(word.strip('.,!?') in ['thank', 'thanks', 'thankyou'] or 'appreciate' in word for word in words):
            return "You're welcome! Is there anything else you'd like to know about our Computer Science department?", []

        if self.ollama_available:
            retrieved_chunks = self.retrieve_relevant_chunks(message)
            llm_answer = self.call_llm(message, retrieved_chunks)
            if llm_answer:
                source_labels = sorted(set(source for chunk, source in retrieved_chunks))
                return llm_answer, source_labels

        # ---- Fallback: original keyword-matching logic ----
        response = None
        if any(word in message_lower for word in ['admission', 'apply', 'office', 'application', 'gpa', 'deadline']):
            response = self.faq_data["admission"]["answers"]["default"]
        elif any(word in message_lower for word in ['course', 'class', 'programming', 'languages', 'python', 'java']):
            response = self.faq_data["courses"]["answers"]["default"]
        elif any(word in message_lower for word in ['computer science', 'department', 'cabin', 'block']):
            response = self.faq_data["Department"]["answers"]["default"]
        elif any(word in message_lower for word in ['many', 'college', 'cse']):
            response = self.faq_data["college"]["answers"]["default"]
        elif any(word in message_lower for word in ['job', 'career', 'salary', 'hire', 'placement']):
            response = self.faq_data["careers"]["answers"]["default"]
        elif any(word in message_lower for word in ['thank', 'thanks', 'appreciate']):
            response = "You're welcome! Is there anything else you'd like to know about our Computer Science department?"

        if response:
            return response, []
        return ("I'm sorry, I don't have an answer for that. "
                "Please ask something related to the CS Department."), []

    def process_message(self, message):
        """Kick off a background worker to fetch the response, so the
        UI stays responsive (shows a typing indicator meanwhile)."""
        self.show_typing_indicator()
        self.input_field.setEnabled(False)

        self.active_worker = ResponseWorker(self, message)
        self.active_worker.finished.connect(self.on_response_ready)
        self.active_worker.start()

    def on_response_ready(self, answer, sources):
        self.remove_typing_indicator()
        self.add_bot_message(answer, sources=sources if sources else None)
        self.input_field.setEnabled(True)
        self.input_field.setFocus()

    def closeEvent(self, event):
        """Save chat history when closing"""
        try:
            chat_history = []
            for i in range(self.chat_area_layout.count()):
                widget = self.chat_area_layout.itemAt(i).widget()
                if isinstance(widget, MessageWidget) and not widget.is_typing:
                    label = widget.findChild(QLabel, "message_text")
                    if label:
                        chat_history.append({
                            "text": label.text(),
                            "is_user": widget.is_user,
                            "timestamp": widget.timestamp
                        })

            with open("chat_history.json", "w") as f:
                json.dump(chat_history, f)

        except Exception as e:
            print(f"Error saving chat history: {e}")

        event.accept()

    def load_chat_history(self):
        """Restore the previous session's chat, or show a welcome
        message if there's nothing saved yet."""
        history = []
        if os.path.exists("chat_history.json"):
            try:
                with open("chat_history.json", "r") as f:
                    history = json.load(f)
            except Exception as e:
                print(f"Could not load previous chat history: {e}")

        if history:
            for entry in history:
                widget = MessageWidget(entry["text"], is_user=entry["is_user"], timestamp=entry.get("timestamp"))
                self.chat_area_layout.addWidget(widget)
            self.scroll_to_bottom()
        else:
            self.add_bot_message("Hello! I'm the CS Department FAQ chatbot. I can answer questions about admissions, courses, faculty, and career opportunities. How can I help you today?")

    def clear_chat(self):
        """Wipe the chat window and start a fresh conversation."""
        while self.chat_area_layout.count():
            item = self.chat_area_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.conversation_history = []
        self.add_bot_message("Chat cleared. How can I help you today?")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    window = ChatBot()
    window.show()

    sys.exit(app.exec_())