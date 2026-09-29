"""
rag_engine.py
--------------
The "brain" of the CS Department chatbot: FAQ data, document-based RAG
retrieval, and the Ollama LLM call. No GUI code here at all — this is
shared by both the desktop app (chatbot.py) and the web app (app.py).

Speed optimizations applied:
1. Uses llama3.2:1b (smaller/faster) instead of the 3B model by default.
   Run: ollama pull llama3.2:1b
2. Caps response length (num_predict) so answers don't ramble.
3. Supports streaming (stream_response) so the web UI can show words
   as they're generated instead of waiting for the full answer.
4. Skips the embedding + RAG search step entirely for simple messages
   like greetings/thanks, answering those instantly.
"""

import os
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "llama3.2:1b"        # smaller & faster than llama3.2 (3b)
OLLAMA_EMBED_MODEL = "nomic-embed-text"
RESPONSE_MAX_TOKENS = 120           # caps answer length for faster replies

DOCUMENTS_FOLDER = "documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_CHUNKS = 3


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
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


class RAGChatEngine:
    """Holds the FAQ data + embedded knowledge base, and answers
    questions using RAG + a local Ollama model, falling back to
    keyword matching if Ollama isn't available."""

    def __init__(self):
        self.faq_data = self._load_faq_data()
        self.knowledge_chunks = []
        self.knowledge_sources = []
        self.knowledge_embeddings = []
        self.ollama_available = self._check_ollama_available()
        if self.ollama_available:
            self._build_knowledge_base()

    # ---------- setup ----------
    def _load_faq_data(self):
        return {
            "admission": {
                "questions": ["Where is Admission cell Located?", "Where is Principal Office?"],
                "answers": {"default": "It is located at the Enterance of the College."}
            },
            "courses": {
                "questions": ["What are the basic languages required?", "What programming languages are taught?"],
                "answers": {"default": "Python, Java, and C++."}
            },
            "Department": {
                "questions": ["Where is Computer Science Department located?", "Where is HOD Cabin Located?"],
                "answers": {"default": "It is Located in Block 4."}
            },
            "college": {
                "questions": ["What are department in our college?", "How many department are there in our college?"],
                "answers": {"default": "There are 6 department there are EEE,EC,CSE,AIML,MECH,CIVIL."}
            },
            "careers": {
                "questions": ["What job opportunities are there?", "What's the average starting salary?"],
                "answers": {"default": "Tech initator and many startups. The average starting salary is 3LPA with near 75% placement rate."}
            }
        }

    def _check_ollama_available(self):
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def _embed_text(self, text):
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

    def _build_knowledge_base(self):
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
                embedding = self._embed_text(chunk)
                if embedding:
                    self.knowledge_chunks.append(chunk)
                    self.knowledge_sources.append(os.path.splitext(filename)[0].replace("_", " ").title())
                    self.knowledge_embeddings.append(embedding)
        print(f"RAG knowledge base ready: {len(self.knowledge_chunks)} chunks loaded.")

    # ---------- retrieval ----------
    def _retrieve_relevant_chunks(self, query, top_k=TOP_K_CHUNKS):
        if not self.knowledge_chunks:
            return []
        query_embedding = self._embed_text(query)
        if not query_embedding:
            return []
        scored = [
            (cosine_similarity(query_embedding, emb), chunk, source)
            for chunk, source, emb in zip(self.knowledge_chunks, self.knowledge_sources, self.knowledge_embeddings)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(chunk, source) for score, chunk, source in scored[:top_k]]

    def _build_system_prompt(self, retrieved_chunks):
        facts = [f"- {category}: {data['answers']['default']}" for category, data in self.faq_data.items()]
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

    def _build_messages(self, message, conversation_history, retrieved_chunks):
        messages = [{"role": "system", "content": self._build_system_prompt(retrieved_chunks)}]
        messages.extend(conversation_history[-6:])
        messages.append({"role": "user", "content": message})
        return messages

    def _call_llm(self, message, conversation_history, retrieved_chunks):
        """Non-streaming call — waits for the full answer."""
        messages = self._build_messages(message, conversation_history, retrieved_chunks)
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_predict": RESPONSE_MAX_TOKENS},
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except Exception as e:
            print(f"LLM call failed, falling back to FAQ matching: {e}")
            return None

    def _call_llm_stream(self, message, conversation_history, retrieved_chunks):
        """Streaming call — yields text tokens as they're generated."""
        messages = self._build_messages(message, conversation_history, retrieved_chunks)
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": True,
                "options": {"num_predict": RESPONSE_MAX_TOKENS},
            },
            timeout=60,
            stream=True,
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            token = data.get("message", {}).get("content", "")
            if token:
                yield token
            if data.get("done"):
                break

    # ---------- shortcuts for simple messages (skip RAG + LLM entirely) ----------
    def _try_simple_response(self, message):
        message_lower = message.lower().strip()
        words = message_lower.split()
        word_count = len(words)
        if word_count <= 4 and any(word in words for word in ['hello', 'hi', 'hey', 'greetings']):
            return "Hello! How can I help you with Computer Science department information today?"
        if word_count <= 5 and any(word.strip('.,!?') in ['thank', 'thanks', 'thankyou'] or 'appreciate' in word for word in words):
            return "You're welcome! Is there anything else you'd like to know about our Computer Science department?"
        return None

    def _fallback_response(self, message):
        """Keyword-matched FAQ answer, used when Ollama is unavailable
        or the LLM call fails."""
        message_lower = message.lower()
        response = None
        if any(w in message_lower for w in ['admission', 'apply', 'office', 'application', 'gpa', 'deadline']):
            response = self.faq_data["admission"]["answers"]["default"]
        elif any(w in message_lower for w in ['course', 'class', 'programming', 'languages', 'python', 'java']):
            response = self.faq_data["courses"]["answers"]["default"]
        elif any(w in message_lower for w in ['computer science', 'department', 'cabin', 'block']):
            response = self.faq_data["Department"]["answers"]["default"]
        elif any(w in message_lower for w in ['many', 'college', 'cse']):
            response = self.faq_data["college"]["answers"]["default"]
        elif any(w in message_lower for w in ['job', 'career', 'salary', 'hire', 'placement']):
            response = self.faq_data["careers"]["answers"]["default"]
        return response or "I'm sorry, I don't have an answer for that. Please ask something related to the CS Department."

    # ---------- public API ----------
    def get_response(self, message, conversation_history=None):
        """Non-streaming: returns (answer_text, source_labels_list)."""
        conversation_history = conversation_history or []

        simple = self._try_simple_response(message)
        if simple is not None:
            return simple, []

        if self.ollama_available:
            retrieved_chunks = self._retrieve_relevant_chunks(message)
            llm_answer = self._call_llm(message, conversation_history, retrieved_chunks)
            if llm_answer:
                source_labels = sorted(set(source for chunk, source in retrieved_chunks))
                return llm_answer, source_labels

        return self._fallback_response(message), []

    def stream_response(self, message, conversation_history=None):
        """Streaming generator: yields dicts of either
        {"type": "token", "text": "..."} for each piece of the answer, or
        {"type": "done", "sources": [...]} once finished.
        Simple messages and fallback answers are yielded as one token."""
        conversation_history = conversation_history or []

        simple = self._try_simple_response(message)
        if simple is not None:
            yield {"type": "token", "text": simple}
            yield {"type": "done", "sources": []}
            return

        if self.ollama_available:
            retrieved_chunks = self._retrieve_relevant_chunks(message)
            source_labels = sorted(set(source for chunk, source in retrieved_chunks))
            got_any = False
            try:
                for token in self._call_llm_stream(message, conversation_history, retrieved_chunks):
                    got_any = True
                    yield {"type": "token", "text": token}
            except Exception as e:
                print(f"Streaming LLM call failed, falling back to FAQ matching: {e}")
                got_any = False
            if got_any:
                yield {"type": "done", "sources": source_labels}
                return

        yield {"type": "token", "text": self._fallback_response(message)}
        yield {"type": "done", "sources": []}