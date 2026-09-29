"""
Tests for rag_engine.py.

These mock all Ollama HTTP calls, so the test suite runs anywhere
(including GitHub Actions CI) without needing Ollama actually installed
or running.
"""

import sys
import os
from unittest.mock import patch, MagicMock

# Allow importing rag_engine.py from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_engine import RAGChatEngine, chunk_text, cosine_similarity


# ---------- pure function tests ----------

def test_chunk_text_splits_long_text():
    text = "a" * 1200
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)


def test_chunk_text_short_text_single_chunk():
    text = "short piece of text"
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert chunks == [text]


def test_cosine_similarity_identical_vectors():
    v = [1, 2, 3]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1, 0], [0, 1]) == 0


def test_cosine_similarity_zero_vector_no_crash():
    assert cosine_similarity([0, 0], [1, 1]) == 0


# ---------- engine tests (Ollama unavailable, so pure fallback logic) ----------

@patch("rag_engine.requests.get")
def make_offline_engine(mock_get):
    """Helper: build an engine where Ollama is unreachable, so it only
    uses the keyword-matching fallback and simple-message shortcuts."""
    mock_get.side_effect = Exception("connection refused")
    return RAGChatEngine()


def test_engine_reports_ollama_unavailable_when_unreachable():
    engine = make_offline_engine()
    assert engine.ollama_available is False


def test_simple_greeting_shortcut():
    engine = make_offline_engine()
    answer = engine._try_simple_response("hi")
    assert answer is not None
    assert "hello" in answer.lower()


def test_greeting_substring_bug_regression():
    """Regression test: 'think' contains 'hi' as a substring but is NOT
    a greeting. This previously caused the bot to answer with a canned
    hello instead of a real answer. Should return None (not a greeting)."""
    engine = make_offline_engine()
    assert engine._try_simple_response("i think chemistry also") is None


def test_thanks_shortcut():
    engine = make_offline_engine()
    answer = engine._try_simple_response("thanks a lot")
    assert answer is not None
    assert "welcome" in answer.lower()


def test_fallback_response_matches_courses_keyword():
    engine = make_offline_engine()
    answer = engine._fallback_response("what programming languages are taught?")
    assert "python" in answer.lower() or "java" in answer.lower()


def test_fallback_response_matches_careers_keyword():
    engine = make_offline_engine()
    answer = engine._fallback_response("what is the average starting salary?")
    assert "lpa" in answer.lower() or "salary" in answer.lower()


def test_fallback_response_unknown_topic():
    engine = make_offline_engine()
    answer = engine._fallback_response("what's the weather like today?")
    assert "don't have an answer" in answer.lower()


def test_get_response_uses_fallback_when_ollama_unavailable():
    engine = make_offline_engine()
    answer, sources = engine.get_response("what courses are offered?")
    assert isinstance(answer, str) and len(answer) > 0
    assert sources == []  # no RAG sources used when Ollama is down