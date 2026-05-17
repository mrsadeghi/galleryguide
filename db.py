"""
Gallery Guide — SQLite conversation store
Stores sessions and messages for sidebar history.
"""

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DATABASE_PATH", "./gallery_guide.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL DEFAULT 'New conversation',
                language    TEXT NOT NULL DEFAULT 'en',
                created_at  INTEGER NOT NULL,
                updated_at  INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role        TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content     TEXT NOT NULL,
                sources     TEXT,
                created_at  INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);
        """)
    print(f"DB initialized at {DB_PATH}")


# ── Sessions ──────────────────────────────────────────────────────────

def create_session(session_id: str, language: str = "en") -> dict:
    now = int(time.time())
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, language, created_at, updated_at) VALUES (?,?,?,?)",
            (session_id, language, now, now),
        )
    return {"id": session_id, "title": "New conversation", "language": language}


def update_session_title(session_id: str, title: str):
    now = int(time.time())
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET title=?, updated_at=? WHERE id=?",
            (title[:80], now, session_id),
        )


def get_sessions(limit: int = 30) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, language, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))


# ── Messages ──────────────────────────────────────────────────────────

def save_message(session_id: str, role: str, content: str, sources: list = None):
    now = int(time.time())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, sources, created_at) VALUES (?,?,?,?,?)",
            (session_id, role, content, json.dumps(sources) if sources else None, now),
        )
        conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id))


def get_messages(session_id: str, limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, sources FROM messages WHERE session_id=? ORDER BY created_at ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    result = []
    for r in rows:
        msg = {"role": r["role"], "content": r["content"], "sources": [], "hasArtwork": False}
        if r["sources"]:
            try:
                sources = json.loads(r["sources"])
                msg["sources"] = sources
                msg["hasArtwork"] = (
                    r["role"] == "assistant" and
                    len(sources) > 0 and
                    bool(sources[0].get("image_url"))
                )
            except Exception:
                pass
        result.append(msg)
    return result


# Initialize on import
init_db()
