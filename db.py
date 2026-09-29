"""
db.py
-----
SQLite storage for chat interactions + simple analytics helpers.
Keeps a permanent record of every question asked and answer given,
independent of the in-memory per-session conversation history in app.py.
"""

import sqlite3
import time
from datetime import datetime, timedelta
from contextlib import contextmanager

DB_PATH = "chatbot.db"

# Same rough categories used by the FAQ/RAG fallback logic, used here
# just to group questions into a meaningful "top topics" analytic.
CATEGORY_KEYWORDS = {
    "Admissions": ["admission", "apply", "application", "gpa", "deadline"],
    "Courses": ["course", "class", "programming", "languages", "python", "java", "subject"],
    "Department": ["department", "cabin", "block", "hod"],
    "College": ["college", "cse", "many"],
    "Careers": ["job", "career", "salary", "hire", "placement"],
    "Greeting / Thanks": ["hello", "hi", "hey", "thank", "thanks"],
}


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                used_rag INTEGER NOT NULL DEFAULT 0,
                sources TEXT,
                response_time_ms INTEGER,
                created_at TEXT NOT NULL
            )
        """)


def categorize(question):
    q = question.lower()
    words = q.split()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in words for kw in keywords):
            return category
    return "Other"


def log_interaction(session_id, question, answer, sources, response_time_ms):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO interactions
               (session_id, question, answer, used_rag, sources, response_time_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                question,
                answer,
                1 if sources else 0,
                ", ".join(sources) if sources else "",
                response_time_ms,
                datetime.utcnow().isoformat(),
            ),
        )


def get_stats():
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM interactions").fetchone()["c"]

        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        today_count = conn.execute(
            "SELECT COUNT(*) AS c FROM interactions WHERE created_at >= ?", (today_start,)
        ).fetchone()["c"]

        avg_time_row = conn.execute(
            "SELECT AVG(response_time_ms) AS avg_ms FROM interactions WHERE response_time_ms IS NOT NULL"
        ).fetchone()
        avg_response_ms = round(avg_time_row["avg_ms"]) if avg_time_row["avg_ms"] else 0

        rag_used_count = conn.execute(
            "SELECT COUNT(*) AS c FROM interactions WHERE used_rag = 1"
        ).fetchone()["c"]

        recent = conn.execute(
            "SELECT question, answer, created_at, used_rag FROM interactions ORDER BY id DESC LIMIT 15"
        ).fetchall()

        all_questions = conn.execute("SELECT question FROM interactions").fetchall()

    # Categorize in Python (simpler than SQL for this small scale)
    category_counts = {}
    for row in all_questions:
        cat = categorize(row["question"])
        category_counts[cat] = category_counts.get(cat, 0) + 1
    top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

    # Last 7 days of volume, for a simple trend
    daily_counts = []
    with get_db() as conn:
        for i in range(6, -1, -1):
            day = (datetime.utcnow() - timedelta(days=i)).date()
            day_start = day.isoformat()
            day_end = (day + timedelta(days=1)).isoformat()
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM interactions WHERE created_at >= ? AND created_at < ?",
                (day_start, day_end),
            ).fetchone()["c"]
            daily_counts.append({"date": day.strftime("%b %d"), "count": count})

    return {
        "total_questions": total,
        "today_count": today_count,
        "avg_response_ms": avg_response_ms,
        "rag_used_count": rag_used_count,
        "rag_used_pct": round((rag_used_count / total) * 100) if total else 0,
        "top_categories": top_categories,
        "recent": [dict(r) for r in recent],
        "daily_counts": daily_counts,
    }