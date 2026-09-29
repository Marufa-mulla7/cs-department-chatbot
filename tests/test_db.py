"""
Tests for db.py — uses a temporary SQLite file so it never touches
your real chatbot.db.
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import db


def setup_temp_db(monkeypatch):
    """Point db.DB_PATH at a temp file for the duration of a test."""
    tmp_path = tempfile.mktemp(suffix=".db")
    monkeypatch.setattr(db, "DB_PATH", tmp_path)
    db.init_db()
    return tmp_path


def test_categorize_courses():
    assert db.categorize("what programming languages are taught?") == "Courses"


def test_categorize_admissions():
    assert db.categorize("how do I apply for admission?") == "Admissions"


def test_categorize_careers():
    assert db.categorize("what is the average salary after graduation?") == "Careers"


def test_categorize_unknown_falls_back_to_other():
    assert db.categorize("what's the capital of France?") == "Other"


def test_log_and_read_interaction(monkeypatch):
    setup_temp_db(monkeypatch)
    db.log_interaction("session-1", "what courses are offered?", "Python, Java, C++", [], 250)

    stats = db.get_stats()
    assert stats["total_questions"] == 1
    assert stats["recent"][0]["question"] == "what courses are offered?"


def test_stats_empty_db_does_not_crash(monkeypatch):
    setup_temp_db(monkeypatch)
    stats = db.get_stats()
    assert stats["total_questions"] == 0
    assert stats["avg_response_ms"] == 0
    assert stats["recent"] == []


def test_rag_used_percentage(monkeypatch):
    setup_temp_db(monkeypatch)
    db.log_interaction("s1", "q1", "a1", ["Department Info"], 100)
    db.log_interaction("s1", "q2", "a2", [], 100)

    stats = db.get_stats()
    assert stats["total_questions"] == 2
    assert stats["rag_used_count"] == 1
    assert stats["rag_used_pct"] == 50