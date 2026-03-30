"""
Append-only JSONL log of main-model turns for SFT / preference data (prompt, trace, answer).

Files live under IMAGINATION_ROOT/temp/training_exports/.

Disable with IMAGINATION_TRAINING_LOG=0 (or false/no/off).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List

_APPEND_LOCK = Lock()


def training_log_enabled() -> bool:
    v = (os.getenv("IMAGINATION_TRAINING_LOG") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def main_training_log_path(root: str) -> str:
    base = os.path.join(root, "temp", "training_exports")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "main_model_turns.jsonl")


def _serialize_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for m in messages or []:
        role = m.get("role") or ""
        content = m.get("content") or ""
        if isinstance(role, str) and isinstance(content, str):
            out.append({"role": role, "content": content})
    return out


def append_main_model_training_turn(
    root: str,
    *,
    messages: List[Dict[str, str]],
    thinking_md: str,
    trace_summary: str,
    step_lines: List[str],
    thinking_collapsed_html: str,
    use_web: bool,
    web_trigger_reason: str,
    source_cards: List[Dict[str, Any]],
    answer: str,
    user_message: str,
) -> None:
    if not training_log_enabled():
        return
    path = main_training_log_path(root)
    record: Dict[str, Any] = {
        "schema": "imagination_main_turn_v1",
        "app_version": "1.2",
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "chat_main",
        "user_message": user_message,
        "messages": _serialize_messages(messages),
        "trace": {
            "collapsed_summary": trace_summary,
            "step_lines": list(step_lines),
            "thinking_collapsed_html": thinking_collapsed_html,
            "thinking_path_system_text": (thinking_md or "").strip(),
            "use_web": use_web,
            "web_trigger_reason": web_trigger_reason,
            "source_cards": list(source_cards),
        },
        "answer": answer or "",
    }
    line = json.dumps(record, ensure_ascii=False)
    with _APPEND_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
