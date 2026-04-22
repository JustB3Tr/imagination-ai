"""Lightweight runtime flags for HTTP health endpoints (avoid duplicating RUNTIME checks)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def v13_line_version() -> str:
    """Semver from ``versions/v1.3/VERSION`` next to this package (Colab + local)."""
    try:
        p = Path(__file__).resolve().parent.parent / "VERSION"
        if p.is_file():
            s = p.read_text(encoding="utf-8", errors="replace").strip()
            return s or "unknown"
    except (OSError, TypeError, ValueError):
        pass
    return "unknown"


def health_version_dict() -> Dict[str, Any]:
    v = v13_line_version()
    return {"backend_version": v, "v13": v}


def vision_health_dict() -> Dict[str, Any]:
    """
    Current main-model vision stack (from ``imagination_v1_3.RUNTIME``).

    - ``vision_mode``: ``unloaded`` | ``text_only`` | ``clip_projector`` | ``native_vlm`` | ``unknown``
    - ``clip_projector_active``: True only when CLIP + trained projector shim is loaded.
    """
    try:
        from imagination_v1_3 import RUNTIME
    except Exception:
        return {"vision_mode": "unknown", "clip_projector_active": False}

    if RUNTIME.main_model is None or RUNTIME.main_tokenizer is None:
        return {"vision_mode": "unloaded", "clip_projector_active": False}
    if not RUNTIME.main_is_vlm:
        return {"vision_mode": "text_only", "clip_projector_active": False}
    proc = RUNTIME.main_processor
    if proc is not None and getattr(proc, "is_clip_projector", False):
        return {"vision_mode": "clip_projector", "clip_projector_active": True}
    return {"vision_mode": "native_vlm", "clip_projector_active": False}
