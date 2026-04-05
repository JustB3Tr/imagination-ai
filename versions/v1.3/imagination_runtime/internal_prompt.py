"""
Hidden internal system instructions. Loaded from file and injected into
the system prompt. Never exposed to the user.

A fixed attribution block is always prepended first so it wins over stale
global_memory.txt and generic training priors when users ask who built the app.
"""
from __future__ import annotations

import os

# Always first in the assembled system prompt (before optional internal_instructions.txt).
ATTRIBUTION_FIRST_BLOCK = """[AUTHORITATIVE — THIS DEPLOYMENT ONLY]
Brady McCauley (age 14) built and maintains this Imagination chat app and UI. He is the author of this instance—not Hugging Face, OpenAI, Google, Anthropic, Meta, or any other company as the maker of *this product*.
Downloading model weights from Hugging Face (or elsewhere) is only hosting; it is not who built this application.
If asked who made you, who built this, who created Imagination, or who owns this assistant, you must answer with Brady McCauley as the developer of this deployment. Do not substitute a corporate author. Only if the question is narrowly about which company pretrained the base language model may you name that company separately, and you must still credit Brady for Imagination itself.""".strip()


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


def tutor_math_policy_path(root: str) -> str:
    """Shipped tutor policy next to package root (v1.3) or under temp/."""
    pkg = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tutor_math_policy.txt")
    if os.path.isfile(pkg):
        return pkg
    base = os.path.join(root, "temp")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "tutor_math_policy.txt")


def load_tutor_math_policy(root: str) -> str:
    path = tutor_math_policy_path(root)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except (OSError, UnicodeDecodeError):
            pass
    return ""


def inject_internal_instructions(base_prompt: str, root: str) -> str:
    """
    Prepend ATTRIBUTION_FIRST_BLOCK, then optional temp/internal_instructions.txt,
    then the rest of the system prompt.
    """
    internal = load_internal_instructions(root)
    parts = [ATTRIBUTION_FIRST_BLOCK]
    if internal:
        parts.append(internal)
    parts.append(base_prompt)
    return "\n\n".join(parts)
