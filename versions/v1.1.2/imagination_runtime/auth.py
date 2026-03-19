"""
Multi-provider auth: HuggingFace OAuth (Gradio built-in), email/password (SQLite+bcrypt),
and stubs for Google, GitHub, Apple.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Tuple

from .users import get_or_create_user, get_user_by_email_password, User


def hash_password(password: str) -> str:
    """Hash password with bcrypt. Falls back to simple hash if bcrypt unavailable."""
    try:
        import bcrypt
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    except ImportError:
        import hashlib
        return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored hash."""
    try:
        import bcrypt
        return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
    except (ImportError, ValueError):
        import hashlib
        return hashlib.sha256(password.encode("utf-8")).hexdigest() == stored_hash


def login_email_password(root: str, email: str, password: str) -> Optional[User]:
    """Authenticate via email/password. Returns User or None."""
    pw_hash = hash_password(password)
    return get_user_by_email_password(root, email, pw_hash)


def signup_email_password(
    root: str,
    email: str,
    password: str,
    display_name: str = "",
) -> Tuple[Optional[User], Optional[str]]:
    """
    Create new user with email/password. Returns (User, None) on success,
    or (None, error_message) on failure.
    """
    provider = "email"
    provider_uid = email.strip().lower()
    if not provider_uid or "@" not in provider_uid:
        return (None, "Invalid email")
    if not password or len(password) < 6:
        return (None, "Password must be at least 6 characters")

    path = os.path.join(root, "temp", "users.db")
    if os.path.exists(path):
        import sqlite3
        conn = sqlite3.connect(path)
        cur = conn.execute("SELECT id FROM users WHERE provider = ? AND provider_uid = ?", (provider, provider_uid))
        if cur.fetchone():
            conn.close()
            return (None, "Email already registered. Use login instead.")

    pw_hash = hash_password(password)
    user = get_or_create_user(
        root,
        provider=provider,
        provider_uid=provider_uid,
        display_name=display_name or email.split("@")[0],
        email=email,
        password_hash=pw_hash,
    )
    return (user, None)


def resolve_user_from_oauth_profile(profile: Any, root: str) -> Optional[User]:
    """
    Given a gr.OAuthProfile (from HF LoginButton), get or create user.
    Returns User or None.
    """
    if profile is None:
        return None
    name = getattr(profile, "name", None) or getattr(profile, "username", None) or ""
    email = getattr(profile, "email", None) or ""
    uid = getattr(profile, "sub", None) or getattr(profile, "id", None) or str(id(profile))
    provider = "huggingface"
    user = get_or_create_user(
        root,
        provider=provider,
        provider_uid=str(uid),
        display_name=name,
        email=email,
    )
    return user


# Google, GitHub, Apple require OAuth client setup. Stubs for UI:
AUTH_PROVIDERS = {
    "huggingface": {"label": "Sign in with Hugging Face", "available": True},
    "google": {"label": "Sign in with Google", "available": False, "note": "Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET"},
    "github": {"label": "Sign in with GitHub", "available": False, "note": "Set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET"},
    "apple": {"label": "Sign in with Apple", "available": False, "note": "Coming soon"},
    "email": {"label": "Email / Password", "available": True},
}
