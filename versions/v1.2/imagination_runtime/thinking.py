"""
System-prompt hints for web vs no-web turns, plus HTML for the in-chat collapsible trace UI.
"""
from __future__ import annotations

import html
from typing import Any, Dict, List


def _esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def build_thinking_path(
    *,
    use_web: bool,
    reason: str,
    query: str,
    source_cards: List[Dict[str, Any]],
    conversation_turns: int,
) -> str:
    """
    Short system-prompt hint for the model only. Detailed search/source trace lives in the UI;
    do not repeat step labels or source lists here (avoids the model echoing them in replies).
    """
    _ = (reason, query, source_cards)  # retained for API compatibility with callers
    if use_web:
        return (
            "Web search was used for this turn; excerpts may appear below as retrieved notes. "
            "Use them for factual grounding when relevant. "
            "Answer in plain, concise prose only: do not narrate search steps, label steps, "
            "evaluate sources in the reply, use inline [1]/[2] citations, list URLs, or add a "
            '"Sources used" section—the app UI already shows that trace.'
        )
    lines = ["No web search for this turn; answer from model and conversation context."]
    if conversation_turns > 0:
        lines.append(f"Prior context: {conversation_turns} user/assistant turn(s) in history.")
    return " ".join(lines)


def build_thinking_path_no_web(conversation_turns: int = 0) -> str:
    """Short system hint when no web search is used."""
    return build_thinking_path(
        use_web=False,
        reason="",
        query="",
        source_cards=[],
        conversation_turns=conversation_turns,
    )


def build_thinking_html_open(
    *,
    step_lines: List[str],
    summary_label: str = "Thinking",
) -> str:
    """
    Collapsible block shown while reasoning (open). Uses safe HTML only.
    """
    body = "".join(f'<div class="thinking-step">{_esc(line)}</div>' for line in step_lines)
    return (
        f'<details class="thinking-block" open><summary class="thinking-summary">'
        f'<span class="thinking-summary-text">{_esc(summary_label)}</span>'
        f'<span class="thinking-dots" aria-hidden="true"><span></span><span></span><span></span></span>'
        f"</summary><div class=\"thinking-body\">{body}</div></details>"
    )


def build_thinking_html_collapsed(*, summary: str, step_lines: List[str]) -> str:
    """Collapsed details after answer; user can expand to see steps."""
    body = "".join(f'<div class="thinking-step">{_esc(line)}</div>' for line in step_lines)
    return (
        f'<details class="thinking-block thinking-done"><summary class="thinking-summary">'
        f"{_esc(summary)}</summary><div class=\"thinking-body\">{body}</div></details>"
    )


def compose_assistant_display(thinking_html: str, answer_part: str) -> str:
    """Visible assistant bubble: thinking block + answer (markdown/plain)."""
    ap = answer_part or ""
    if not ap.strip():
        return thinking_html
    return f"{thinking_html}\n\n{ap}"


def friendly_web_decision_reason(reason: str) -> str:
    """User-facing line for why we searched."""
    r = (reason or "").lower()
    if "pattern" in r or "matched" in r:
        return "This looks like a question that needs up-to-date information — searching the web."
    if "manual" in r:
        return "Searching the web as requested."
    return "Checking whether a web search would help…"
