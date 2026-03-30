"""
Hidden internal system instructions. Loaded from file and injected into
the system prompt. Never exposed to the user.
"""
from __future__ import annotations

import os


def internal_instructions_path(root: str) -> str:
    base = os.path.join(root, "temp")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "internal_instructions.txt")


def load_internal_instructions(root: str) -> str:
    path = internal_instructions_path(root)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except (IOError, OSError):
            pass
    return ""


def inject_internal_instructions(base_prompt: str, root: str) -> str:
    """
    Prepend internal instructions to the base system prompt.
    If no internal instructions file exists, returns base_prompt unchanged.
    """
    internal = load_internal_instructions(root)
    if not internal:
        return base_prompt
    return f"{internal}\n\n{base_prompt}"
