"""
Per-user memory, preferences, and global admin memory.
SQLite-backed user data; global memory from a text file.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class User:
    id: int
    provider: str
    provider_uid: str
    display_name: str
    email: str
    created_at: str


def _db_path(root: str) -> str:
    base = os.path.join(root, "temp")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "users.db")


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT NOT NULL,
            provider_uid TEXT NOT NULL,
            display_name TEXT DEFAULT '',
            email TEXT DEFAULT '',
            password_hash TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(provider, provider_uid)
        );
        CREATE TABLE IF NOT EXISTS user_memory (
            user_id INTEGER PRIMARY KEY,
            memory_text TEXT DEFAULT '',
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS user_prefs (
            user_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value TEXT,
            PRIMARY KEY (user_id, key),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE INDEX IF NOT EXISTS idx_users_provider_uid ON users(provider, provider_uid);
    """)


def get_or_create_user(
    root: str,
    *,
    provider: str,
    provider_uid: str,
    display_name: str = "",
    email: str = "",
    password_hash: Optional[str] = None,
) -> User:
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        cur = conn.execute(
            "SELECT id, provider, provider_uid, display_name, email, created_at FROM users WHERE provider = ? AND provider_uid = ?",
            (provider, provider_uid),
        )
        row = cur.fetchone()
        if row:
            return User(id=row[0], provider=row[1], provider_uid=row[2], display_name=row[3] or "", email=row[4] or "", created_at=row[5] or "")
        conn.execute(
            "INSERT INTO users (provider, provider_uid, display_name, email, password_hash) VALUES (?, ?, ?, ?, ?)",
            (provider, provider_uid, display_name, email, password_hash),
        )
        conn.commit()
        uid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        cur2 = conn.execute("SELECT id, provider, provider_uid, display_name, email, created_at FROM users WHERE id = ?", (uid,))
        r = cur2.fetchone()
        return User(id=r[0], provider=r[1], provider_uid=r[2], display_name=r[3] or "", email=r[4] or "", created_at=r[5] or "")
    finally:
        conn.close()


def get_user_by_email_password(root: str, email: str, password_hash: str) -> Optional[User]:
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        cur = conn.execute(
            "SELECT id, provider, provider_uid, display_name, email, created_at FROM users WHERE provider = 'email' AND email = ? AND password_hash = ?",
            (email, password_hash),
        )
        row = cur.fetchone()
        if row:
            return User(id=row[0], provider=row[1], provider_uid=row[2], display_name=row[3] or "", email=row[4] or "", created_at=row[5] or "")
        return None
    finally:
        conn.close()


def load_user_memory(root: str, user_id: int) -> str:
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        cur = conn.execute("SELECT memory_text FROM user_memory WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        return (row[0] or "").strip() if row else ""
    finally:
        conn.close()


def save_user_memory(root: str, user_id: int, text: str) -> None:
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        conn.execute(
            "INSERT INTO user_memory (user_id, memory_text, updated_at) VALUES (?, ?, datetime('now')) ON CONFLICT(user_id) DO UPDATE SET memory_text = excluded.memory_text, updated_at = datetime('now')",
            (user_id, (text or "").strip()),
        )
        conn.commit()
    finally:
        conn.close()


def get_user_pref(root: str, user_id: int, key: str) -> Optional[str]:
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        cur = conn.execute("SELECT value FROM user_prefs WHERE user_id = ? AND key = ?", (user_id, key))
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def set_user_pref(root: str, user_id: int, key: str, value: str) -> None:
    path = _db_path(root)
    conn = sqlite3.connect(path)
    _init_db(conn)
    try:
        conn.execute(
            "INSERT INTO user_prefs (user_id, key, value) VALUES (?, ?, ?) ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value",
            (user_id, key, value),
        )
        conn.commit()
    finally:
        conn.close()


def global_memory_path(root: str) -> str:
    base = os.path.join(root, "temp")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "global_memory.txt")


def load_global_memory(root: str) -> str:
    path = global_memory_path(root)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def save_global_memory(root: str, text: str) -> None:
    path = global_memory_path(root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write((text or "").strip())


def trusted_sources_path(root: str) -> str:
    base = os.path.join(root, "temp", "imagination1")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "trusted_sources.txt")


def load_trusted(root: str, starter_domains: Optional[List[str]] = None) -> List[str]:
    path = trusted_sources_path(root)
    if not os.path.exists(path):
        domains = starter_domains or [
            "reuters.com", "apnews.com", "bbc.com", "npr.org", "pbs.org",
            "nytimes.com", "washingtonpost.com", "wsj.com", "bloomberg.com", "theguardian.com",
            "who.int", "cdc.gov", "nih.gov", "ncbi.nlm.nih.gov", "nasa.gov", "noaa.gov", "usgs.gov",
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(domains) + "\n")
    with open(path, "r", encoding="utf-8") as f:
        return [x.strip().lower() for x in f if x.strip()]
